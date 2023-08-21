from typing import Tuple
import open3d as o3d
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss_utils import norm_l2_loss, chamfer_loss, pseudo_chamfer_loss, hausdorff_loss, curvature_loss, kNN_smoothing_loss, _get_kappa_ori, _get_kappa_adv

def get_normal_vector(points):
    """Calculate the normal vector.

    Args:
        points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 3].
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.squeeze(0).detach().cpu().numpy())
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    normal_vec = torch.FloatTensor(pcd.normals).cuda().unsqueeze(0)
    return normal_vec

def proj_surface(x, n):
    beta = (x * n).sum((1,2)) / (n * n).sum((1,2))
    alpha = x - beta[:, np.newaxis, np.newaxis] * n
    return alpha

def norm(x):
    res = (x ** 2).sum((1, 2)).sqrt()
    res[res == 0] = 1e-12
    return res

def gram_schmidt(g, delta):
    res = [g]
    for d in delta:
        alpha = d
        for b in res:
            alpha = proj_surface(alpha, b)
        if (alpha > 1e-12).any():
            res.append(alpha / norm(alpha)[:, np.newaxis, np.newaxis])
    return res[1:]

def get_spin_axis_matrix(normal_vec):
    """Calculate the spin-axis matrix.

    Args:
        normal_vec (torch.cuda.FloatTensor): the normal vectors for all N points, [1, N, 3].
    """
    _, N, _ = normal_vec.shape
    x = normal_vec[:,:,0] # [1, N]
    y = normal_vec[:,:,1] # [1, N]
    z = normal_vec[:,:,2] # [1, N]
    assert abs(normal_vec).max() <= 1
    u = torch.zeros(1, N, 3, 3).cuda()
    denominator = torch.sqrt(1-z**2) # \sqrt{1-z^2}, [1, N]
    u[:,:,0,0] = y / denominator
    u[:,:,0,1] = - x / denominator
    u[:,:,0,2] = 0.
    u[:,:,1,0] = x * z / denominator
    u[:,:,1,1] = y * z / denominator
    u[:,:,1,2] = - denominator
    u[:,:,2] = normal_vec
    # revision for |z| = 1, boundary case.
    pos = torch.where(abs(z ** 2 - 1) < 1e-4)[1]
    u[:,pos,0,0] = 1 / np.sqrt(2)
    u[:,pos,0,1] = - 1 / np.sqrt(2)
    u[:,pos,0,2] = 0.
    u[:,pos,1,0] = z[:,pos] / np.sqrt(2)
    u[:,pos,1,1] = z[:,pos] / np.sqrt(2)
    u[:,pos,1,2] = 0.
    u[:,pos,2,0] = 0.
    u[:,pos,2,1] = 0.
    u[:,pos,2,2] = z[:,pos]
    return u.data

def get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix):
    """Calculate the spin-axis matrix.

    Args:
        new_points (torch.cuda.FloatTensor): the transformed point cloud with N points, [1, N, 3].
        spin_axis_matrix (torch.cuda.FloatTensor): the rotate matrix for transformation, [1, N, 3, 3].
        translation_matrix (torch.cuda.FloatTensor): the offset matrix for transformation, [1, N, 3, 3].
    """
    inputs = torch.matmul(spin_axis_matrix.transpose(-1, -2), new_points.unsqueeze(-1)) # U^T P', [1, N, 3, 1]
    inputs = inputs - translation_matrix.unsqueeze(-1) # P = U^T P' - (P \cdot N) N, [1, N, 3, 1]
    inputs = inputs.squeeze(-1) # P, [1, N, 3]
    return inputs


def get_transformed_point_cloud(points, normal_vec):
    """Calculate the spin-axis matrix.

    Args:
        points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 3].
        normal_vec (torch.cuda.FloatTensor): the normal vectors for all N points, [1, N, 3].
    """
    intercept = torch.mul(points, normal_vec).sum(-1, keepdim=True) # P \cdot N, [1, N, 1]
    spin_axis_matrix = get_spin_axis_matrix(normal_vec) # U, [1, N, 3, 3]
    translation_matrix = torch.mul(intercept, normal_vec).data # (P \cdot N) N, [1, N, 3]
    new_points = points + translation_matrix #  P + (P \cdot N) N, [1, N, 3]
    new_points = new_points.unsqueeze(-1) # P + (P \cdot N) N, [1, N, 3, 1]
    new_points = torch.matmul(spin_axis_matrix, new_points) # P' = U (P + (P \cdot N) N), [1, N, 3, 1]
    new_points = new_points.squeeze(-1).data # P', [1, N, 3]
    return new_points, spin_axis_matrix, translation_matrix


class boundary_projectuion(nn.Module) :

    def __init__(self, args) -> None:
        super(boundary_projectuion, self).__init__()

        self.l2_weight = args.l2_weight
        self.hd_weight = args.hd_weight
        self.cd_weight = args.cd_weight
        self.curv_weight = args.curv_weight
        self.curv_loss_knn = args.curv_loss_knn

        self.step_size = args.step_size
        self.stage2_step_size = args.stage2_steps

        self.epoch = 0
        self.max_steps = args.max_steps

        self.output_losses = None
        self.output_points = None
        
        self.in_out = False
        self.exponential_step = args.exponential_step

    
    def get_loss(self, points, ori_points, normal_vec, loss_type = "l2"):
        if loss_type == "l2":
            loss = norm_l2_loss(points, ori_points)
        elif loss_type == "curv":
            ori_kappa = _get_kappa_ori(ori_points.transpose(1, 2), normal_vec.transpose(1, 2), self.curv_loss_knn)
            adv_kappa, normal_curr_iter = _get_kappa_adv(points.transpose(1, 2), ori_points.transpose(1, 2), normal_vec.transpose(1, 2), self.curv_loss_knn)
            loss = curvature_loss(points.transpose(1, 2), ori_points.transpose(1, 2), adv_kappa, ori_kappa).mean()
        elif loss_type == "hd":
            loss = hausdorff_loss(points.transpose(1, 2), ori_points.transpose(1, 2))
        elif loss_type == "cd":
            loss = pseudo_chamfer_loss(points.transpose(1, 2), ori_points.transpose(1, 2))
        else:
            raise NotImplementedError

        return loss

    
    def forward(self):
        pass

class boundary_projection_4(boundary_projectuion):

    # threshold = [2e-3, 5e-5, 5e-5, 1e-10], optim_seq = ["l2", "curv", "hd", "cd"]

    def __init__(self, args, threshold = [2e-3, 5e-5, 5e-5, 1e-10], optim_seq = ["l2", "curv", "hd", "cd"]) -> None:
        super(boundary_projection_4, self).__init__(args)

        self.gamma = 0.060
        self.initial_lr = 0.03

        self.learning_rate = self.initial_lr
        self.grad_lr = self.initial_lr


        
        self.epoch = 0
        self.stage = 1
        self.losses_buffer = torch.ones(len(optim_seq)).cuda() * 1e5

        self.threshold = threshold
        self.optim_seq = optim_seq

        self.output_losses = torch.ones(len(optim_seq)).cuda() * 1e5

    def loss_cal(self, points, ori_points, normal_vec) -> Tuple[torch.Tensor, list]:

        losses = torch.zeros_like(self.losses_buffer)
        deltas = []

        for optim_idx in range(len(self.optim_seq)):
            optim_term = self.optim_seq[optim_idx]
            loss = self.get_loss(points, ori_points, normal_vec, optim_term)
            if optim_idx < self.stage:
                loss.backward()
                delta = points.grad.detach().clone()
                points.grad.zero_()
                deltas.append(delta)
                
            losses[optim_idx] = loss

        return losses, deltas

    def forward(self, points, ori_points, normal_vec, g_hat, logits, target):
        points = points.detach()
        points.requires_grad = True

        losses, deltas = self.loss_cal(points, ori_points, normal_vec)
        alpha = gram_schmidt(g_hat, deltas)
        alpha_hat = alpha[-1]

        if logits.argmax(1).item() != target.item():

            self.in_out = True

            if (self.output_losses >= losses).all():
                self.output_losses = losses
                self.output_points = points.detach().clone()

            if self.exponential_step:
                z = points - alpha_hat * self.learning_rate
            else:
                z = points - alpha_hat * self.stage2_step_size

            
            
        else:

            self.in_out = False

            # self.stage = 1

            if self.exponential_step:
                z = points - g_hat * self.grad_lr
            else:
                z = points - g_hat * self.step_size
                # self.step_size = self.step_size * 0.5
                # self.stage2_step_size = self.stage2_step_size * 0.5

        points = z.detach().clone()

        self.update_stage(losses)
        self.update_step_size()
        

        return points

    def update_step_size(self):
        self.learning_rate  = self.learning_rate  * (1 - self.gamma)
        self.grad_lr  = self.grad_lr  * (1 - self.gamma)

    def update_stage(self, losses):
        if self.stage < len(self.threshold) and torch.abs(losses[self.stage - 1] - self.losses_buffer[self.stage - 1]).item() < self.threshold[self.stage - 1]:
            self.stage = self.stage + 1
            self.learning_rate = self.initial_lr
            self.grad_lr = self.initial_lr
        self.losses_buffer = losses


class boundary_projection_4_si(boundary_projection_4):

    def __init__(self, args, threshold = [2e-3, 1e-10], optim_seq = ["l2", "curv", "hd"]) -> None:
        super(boundary_projection_4_si, self).__init__(args, threshold, optim_seq)

    def loss_cal(self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec) -> Tuple[torch.Tensor, list, torch.Tensor]:

        losses = torch.zeros_like(self.losses_buffer)
        deltas = []

        points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)

        for optim_idx in range(len(self.optim_seq)):
            optim_term = self.optim_seq[optim_idx]
            loss = self.get_loss(points, ori_points, normal_vec, optim_term)
            if optim_idx < self.stage:
                loss.backward(retain_graph=True)
                delta = new_points.grad.detach().clone()
                new_points.grad.zero_()
                delta[:,:,2] = 0.
                deltas.append(delta)

                
            losses[optim_idx] = loss


        return losses, deltas, points

    def forward(self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec, g_hat, logits, target):

        losses, deltas, points = self.loss_cal(new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec)
        alpha = gram_schmidt(g_hat, deltas)
        alpha_hat = alpha[-1]

        if logits.argmax(1).item() != target.item():

            self.in_out = True

            if (self.output_losses >= losses).all():
                self.output_losses = losses
                self.output_points = points.detach().clone()

            if self.exponential_step:
                z = new_points - alpha_hat * self.learning_rate
            else:
                z = new_points - alpha_hat * self.stage2_step_size

        else:

            self.in_out = False

            self.stage = 1

            if self.exponential_step:
                z = new_points - g_hat * self.grad_lr
            else:
                z = new_points - g_hat * self.step_size


        new_points = z.detach().clone()
        points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)

        self.update_stage(losses)
        self.update_step_size()
        

        return points


class boundary_projection_3(boundary_projectuion):

    def __init__(self, args, optim_seq = ["l2", "curv"]) -> None:
        super(boundary_projection_3, self).__init__(args)

        self.gamma = 0.060
        self.initial_lr = 0.03
        self.learning_rate = self.initial_lr

        self.grad_lr = self.initial_lr
        
        self.epoch = 0
        self.optim_seq = optim_seq

        self.output_losses = torch.ones(len(optim_seq)).cuda() * 1e5

    def loss_cal(self, points, ori_points, normal_vec) -> Tuple[torch.Tensor, torch.Tensor]:

        losses = torch.zeros_like(self.output_losses).cuda()
        deltas = []

        for optim_idx in range(len(self.optim_seq)):
            optim_term = self.optim_seq[optim_idx]
            loss = self.get_loss(points, ori_points, normal_vec, optim_term)
            loss.backward()
            delta = points.grad.detach().clone()
            deltas.append(delta)
            losses[optim_idx] = loss

            points.grad.zero_()

        return losses, deltas

    def forward(self, points, ori_points, normal_vec, g_hat, logits, target):
        points = points.detach()
        points.requires_grad = True

        losses, deltas = self.loss_cal(points, ori_points, normal_vec) # tensor[m, 1] list[1, n, 3]
        alpha = gram_schmidt(g_hat, deltas)
        
        alpha_hat = torch.stack(alpha).sum(0)

        if logits.argmax(1).item() != target.item():

            self.in_out = True

            if (self.output_losses >= losses).all():
                self.output_losses = losses
                self.output_points = points.detach().clone()

            if self.exponential_step:
                z = points - alpha_hat * self.learning_rate
            else:
                z = points - alpha_hat * self.step_size
            
        else:

            self.in_out = False

            if self.exponential_step:
                z = points - g_hat * self.grad_lr
            else:
                z = points - g_hat * self.step_size

        points = z.detach().clone()

        self.update_step_size()

        return points

    def update_step_size(self):
        self.learning_rate  = self.learning_rate  * (1 - self.gamma)
        self.grad_lr  = self.grad_lr  * (1 - self.gamma)

class boundary_projection_3_si(boundary_projection_3):

    def __init__(self, args, optim_seq = ["l2"]) -> None:
        super(boundary_projection_3_si, self).__init__(args, optim_seq)

    def loss_cal(self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec) -> Tuple[torch.Tensor, list, torch.Tensor]:

        losses = torch.zeros_like(self.output_losses).cuda()
        deltas = [] 

        points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)

        for optim_idx in range(len(self.optim_seq)):
            optim_term = self.optim_seq[optim_idx]
            loss = self.get_loss(points, ori_points, normal_vec, optim_term)

            loss.backward(retain_graph=True) 
            delta = new_points.grad.detach().clone() 
            new_points.grad.zero_() 
            delta[:,:,2] = 0. 
            deltas.append(delta)
 
                
            losses[optim_idx] = loss


        return losses, deltas, points

    def forward(self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec, g_hat, logits, target):

        losses, deltas, points = self.loss_cal(new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec)
        alpha = gram_schmidt(g_hat, deltas)
        alpha_hat = np.array(alpha).sum(0)

        if logits.argmax(1).item() != target.item():

            self.in_out = True

            if (self.output_losses >= losses).all():
                self.output_losses = losses
                self.output_points = points.detach().clone()

            if self.exponential_step:
                z = new_points - alpha_hat * self.learning_rate
            else:
                z = new_points - alpha_hat * self.stage2_step_size

        else:

            self.in_out = False

            self.stage = 1

            if self.exponential_step:
                z = new_points - g_hat * self.grad_lr
            else:
                z = new_points - g_hat * self.step_size
                # self.step_size = self.step_size * 0.5
                # self.stage2_step_size = self.stage2_step_size * 0.5

        new_points = z.detach().clone()
        points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)

        self.update_step_size()
        

        return points

class boundary_projection_2(boundary_projectuion):

    def __init__(self, args, weights = [0.5, 0.5], optim_seq = ["l2", "curv"]) -> None:
        super(boundary_projection_2, self).__init__(args)

        self.gamma = 0.090
        self.initial_lr = 0.03
        self.weights = weights
        self.optim_seq = optim_seq

        self.init()


    def init(self): 
        self.learning_rate = self.initial_lr
        self.grad_lr = self.initial_lr
        self.epoch = 0
        self.output_losses = torch.ones(len(self.optim_seq)).cuda() * 1e5
        self.output_points = None

    def loss_cal(self, points, ori_points, normal_vec) -> Tuple[torch.Tensor, torch.Tensor]:

        losses = torch.zeros_like(self.output_losses).cuda()
        deltas = torch.zeros_like(points).cuda()

        for optim_idx in range(len(self.optim_seq)):
            optim_term = self.optim_seq[optim_idx]
            loss = self.get_loss(points, ori_points, normal_vec, optim_term)
            loss.backward()
            delta = points.grad.detach().clone()
            
            deltas += self.weights[optim_idx] * delta / norm(delta)[:, np.newaxis, np.newaxis]
            losses[optim_idx] = loss

            points.grad.zero_()

        return losses, deltas

    def forward(self, points, ori_points, normal_vec, g_hat, logits, target):
        points = points.detach()
        points.requires_grad = True

        losses, deltas = self.loss_cal(points, ori_points, normal_vec)
        alpha = proj_surface(deltas, g_hat) 
        alpha_hat = alpha / norm(alpha)

        if logits.argmax(1).item() != target.item():

            self.in_out = True

            if (self.output_losses >= losses).all():
                self.output_losses = losses.detach().clone()
                self.output_points = points.detach().clone()

            if self.exponential_step:
                z = points - alpha_hat * self.learning_rate
            else:
                z = points - alpha_hat * self.stage2_step_size
            
        else:

            self.in_out = False

            if self.exponential_step:
                z = points - g_hat * self.grad_lr
            else:
                z = points - g_hat * self.step_size

        points = z.detach().clone()

        self.update_step_size()

        return points

    def update_step_size(self):
        self.learning_rate  = self.learning_rate  * (1 - self.gamma)
        self.grad_lr  = self.grad_lr  * (1 - self.gamma)


class boundary_projection_2_si(boundary_projection_2):

    def __init__(self, args, weights = [0.5, 0.5], optim_seq = ["l2", "curv"]) -> None:
        super(boundary_projection_2_si, self).__init__(args, weights, optim_seq)

    def loss_cal(self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec) -> Tuple[torch.Tensor, list]:

        losses = torch.zeros_like(self.output_losses).cuda()
        deltas = torch.zeros_like(new_points).cuda()

        points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)

        for optim_idx in range(len(self.optim_seq)):
            optim_term = self.optim_seq[optim_idx]
            loss = self.get_loss(points, ori_points, normal_vec, optim_term)
            loss.backward(retain_graph=True)
            delta = new_points.grad.detach().clone()
            new_points.grad.zero_()

            deltas += self.weights[optim_idx] * delta / norm(delta)[:, np.newaxis, np.newaxis]
            deltas[:,:,2] = 0.
                
            losses[optim_idx] = loss

        return losses, deltas, points

    def forward(self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec, g_hat, logits, target):

        losses, deltas, points = self.loss_cal(new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec)
        alpha = proj_surface(deltas, g_hat)
        alpha_hat = alpha / norm(alpha)

        if logits.argmax(1).item() != target.item():

            self.in_out = True

            if (self.output_losses >= losses).all():
                self.output_losses = losses
                self.output_points = points.detach().clone()

            if self.exponential_step:
                z = new_points - alpha_hat * self.learning_rate
            else:
                z = new_points - alpha_hat * self.stage2_step_size

        else:

            self.in_out = False

            self.stage = 1

            if self.exponential_step:
                z = new_points - g_hat * self.grad_lr
            else:
                z = new_points - g_hat * self.step_size
                # self.step_size = self.step_size * 0.5
                # self.stage2_step_size = self.stage2_step_size * 0.5

        new_points = z.detach().clone()
        points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)

        self.update_step_size()
        

        return points
    

class boundary_projection_1(boundary_projectuion):

    def __init__(self, args) -> None:
        super(boundary_projection_1, self).__init__(args)

    def forward(self, points, ori_points, normal_vec, g_hat, logits, target):
        raise NotImplementedError


class boundary_projection_1_si(boundary_projection_1):

    def __init__(self, args) -> None:
        super(boundary_projection_1_si, self).__init__(args)

    def loss_cal(self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec) -> Tuple[torch.Tensor, list]:

        losses = torch.zeros_like(self.output_losses).cuda()
        deltas = torch.zeros_like(new_points).cuda()

        points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)

        optim_term = "l2"
        loss = self.get_loss(points, ori_points, normal_vec, optim_term)
        loss.backward()
        delta = new_points.grad.detach().clone()

        deltas = delta
        deltas[:,:,2] = 0.
            
        losses = loss

        return losses, deltas, points

    def forward(self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec, g_hat, logits, target):

        loss, delta, points = self.loss_cal(new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec)

        delta_norm = (delta ** 2).sum((1,2)).sqrt()
        delta_norm[delta_norm == 0] = 1e-12
        
        r = (delta * g_hat).sum((1,2)) / delta_norm

        gamma = self.gamma_min + self.epoch / (self.max_steps + 1) * (self.gamma_max - self.gamma_min)

        if logits.argmax(1).item() != target.item():
            if (self.output_losses >= loss).all():
                self.output_losses = loss
                self.output_points = points.detach().clone()
            
            epsilon = gamma * delta_norm
            v_star = ori_points + r[:, None, None] * g_hat
            yi_vstar_norm = ((points - v_star) ** 2).sum((1,2)).sqrt()
            yi_vstar_norm[yi_vstar_norm == 0] = 1e-9

            tmp = (points - v_star) / yi_vstar_norm[:, None, None]
            tmp = tmp * torch.sqrt(torch.max(torch.zeros_like(r), epsilon ** 2 - r ** 2))[:, None, None]
            z = v_star + tmp
            points = z.detach()
            if torch.isnan(points).sum().item() != 0:
                assert False, "Out NAN Occured!!!"
            
        else:
            epsilon = delta_norm / gamma
            tmp = (r + torch.sqrt(epsilon ** 2 - delta_norm ** 2 + r ** 2))
            z = points - tmp[:, None, None] * g_hat
            
            points = z.detach()
            if torch.isnan(points).sum().item() != 0:
                assert False, "In NAN Occured!!!"

        new_points = z.detach().clone()
        points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)

        self.update_step_size()
        

        return points
    
    
class boundary_projection_query_si(boundary_projection_3):

    def __init__(self, args, optim_seq = ["l2"]) -> None:
        super(boundary_projection_query_si, self).__init__(args, optim_seq)

    def loss_cal(self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec) -> Tuple[torch.Tensor, list, torch.Tensor]:

        losses = torch.zeros_like(self.output_losses).cuda()
        deltas = [] 

        points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)

        for optim_idx in range(len(self.optim_seq)):
            optim_term = self.optim_seq[optim_idx]
            loss = self.get_loss(points, ori_points, normal_vec, optim_term)

            loss.backward(retain_graph=True) 
            delta = new_points.grad.detach().clone() 
            new_points.grad.zero_() 
            delta[:,:,2] = 0. 
            deltas.append(delta)
                
            losses[optim_idx] = loss


        return losses, deltas, points
    
    def gradient_map_project(self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec, g_hat):
        losses, deltas, points = self.loss_cal(new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec)
        alpha = gram_schmidt(g_hat, deltas)
        alpha_hat = np.array(alpha).sum(0)
        
        return alpha_hat

    def forward(self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec, g_hat, logits, target):

        losses, deltas, points = self.loss_cal(new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec)
        alpha = gram_schmidt(g_hat, deltas)
        alpha_hat = np.array(alpha).sum(0)

        if logits.argmax(1).item() != target.item():

            self.in_out = True

            if (self.output_losses >= losses).all():
                self.output_losses = losses
                self.output_points = points.detach().clone()

            if self.exponential_step:
                z = new_points - alpha_hat * self.learning_rate
            else:
                z = new_points - alpha_hat * self.stage2_step_size

        else:

            self.in_out = False

            self.stage = 1

            if self.exponential_step:
                z = new_points - g_hat * self.grad_lr
            else:
                z = new_points - g_hat * self.step_size


        new_points = z.detach().clone()
        points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)

        self.update_step_size()
        

        return points