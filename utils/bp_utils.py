from typing import Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common_utils import _normalize
from utils.loss_utils import (_get_kappa_adv, _get_kappa_ori, curvature_loss,
                              hausdorff_loss, local_curvature_loss,
                              loss_wrapper, norm_l2_loss, pseudo_chamfer_loss)
from utils.siadv_utils import *


def proj_surface(x, n):
    """
    Args:
        x: [B, n, 3]
        n: [B, n, 3]
    """
    beta = (x * n).sum((1, 2)) / (n * n).sum((1, 2))
    alpha = x - beta[:, None, None] * n
    return alpha


def gram_schmidt(g, delta):
    """
    Args:
        g: [B, n, 3]
        delta: list of [B, n, 3]
    Return:
        list[B, n, 3]
    """
    res = [g]
    for d in delta:
        alpha = d
        for b in res:
            alpha = proj_surface(alpha, b)

        res.append(_normalize(alpha, dim=(1, 2)))
    return res[1:]


class boundary_projectuion(nn.Module):

    def __init__(
        self,
        l2_weight: float,
        hd_weight: float,
        cd_weight: float,
        curv_weight: float,
        curv_loss_knn: int,
        step_size: float,
        stage2_steps: float,
        max_steps: int,
        exponential_step: bool,
    ) -> None:
        super(boundary_projectuion, self).__init__()

        self.l2_weight = l2_weight
        self.hd_weight = hd_weight
        self.cd_weight = cd_weight
        self.curv_weight = curv_weight
        self.curv_loss_knn = curv_loss_knn

        self.step_size = step_size
        self.stage2_step_size = stage2_steps

        self.epoch = 0
        self.max_steps = max_steps

        self.output_losses = None
        self.output_points = None

        self.in_out = False
        self.exponential_step = exponential_step

        self.prepare_optim_loss()

    def prepare_optim_loss(self):
        self.optims = []
        if self.l2_weight != 0.0:
            self.optims.append(
                loss_wrapper(norm_l2_loss, channel_first=False,
                             keep_batch=True)
            )

        if self.hd_weight != 0.0:
            self.optims.append(
                loss_wrapper(hausdorff_loss, channel_first=False,
                             keep_batch=True)
            )

        if self.curv_weight != 0.0:
            self.optims.append(
                loss_wrapper(
                    local_curvature_loss,
                    channel_first=False,
                    keep_batch=True,
                    need_normal=True,
                )
            )

        if self.cd_weight != 0.0:
            self.optims.append(
                loss_wrapper(pseudo_chamfer_loss,
                             channel_first=False, keep_batch=True)
            )

    def get_loss(
        self, points: torch.Tensor, ori_points: torch.Tensor, normal_vec: torch.Tensor
    ) -> list[torch.Tensor]:
        loss = []
        for optim in self.optims:
            if optim.need_normal:
                loss.append(optim(points, ori_points, normal_vec))
            else:
                loss.append(optim(points, ori_points))

        return loss

    # import line_profiler

    # @line_profiler.profile
    # def get_loss(self, points, ori_points, normal_vec, loss_type="l2"):
    #     if loss_type == "l2":
    #         loss = norm_l2_loss(points, ori_points)
    #     elif loss_type == "curv":
    #         ori_kappa = _get_kappa_ori(
    #             ori_points.transpose(1, 2),
    #             normal_vec.transpose(1, 2),
    #             self.curv_loss_knn,
    #         )
    #         adv_kappa, normal_curr_iter = _get_kappa_adv(
    #             points.transpose(1, 2),
    #             ori_points.transpose(1, 2),
    #             normal_vec.transpose(1, 2),
    #             self.curv_loss_knn,
    #         )
    #         loss = curvature_loss(
    #             points.transpose(1, 2), ori_points.transpose(1, 2), adv_kappa, ori_kappa
    #         ).mean()
    #     elif loss_type == "hd":
    #         loss = hausdorff_loss(points.transpose(1, 2), ori_points.transpose(1, 2))
    #     elif loss_type == "cd":
    #         loss = pseudo_chamfer_loss(
    #             points.transpose(1, 2), ori_points.transpose(1, 2)
    #         )
    #     else:
    #         raise NotImplementedError

    #     return loss

    def forward(self):
        pass


class boundary_projection_4(boundary_projectuion):

    # threshold = [2e-3, 5e-5, 5e-5, 1e-10], optim_seq = ["l2", "curv", "hd", "cd"]

    def __init__(
        self,
        args,
        threshold=[2e-3, 5e-5, 5e-5, 1e-10],
        optim_seq=["l2", "curv", "hd", "cd"],
    ) -> None:
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
        self.learning_rate = self.learning_rate * (1 - self.gamma)
        self.grad_lr = self.grad_lr * (1 - self.gamma)

    def update_stage(self, losses):
        if (
            self.stage < len(self.threshold)
            and torch.abs(
                losses[self.stage - 1] - self.losses_buffer[self.stage - 1]
            ).item()
            < self.threshold[self.stage - 1]
        ):
            self.stage = self.stage + 1
            self.learning_rate = self.initial_lr
            self.grad_lr = self.initial_lr
        self.losses_buffer = losses


class boundary_projection_4_si(boundary_projection_4):

    def __init__(
        self, args, threshold=[2e-3, 1e-10], optim_seq=["l2", "curv", "hd"]
    ) -> None:
        super(boundary_projection_4_si, self).__init__(
            args, threshold, optim_seq)

    def loss_cal(
        self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec
    ) -> Tuple[torch.Tensor, list, torch.Tensor]:

        losses = torch.zeros_like(self.losses_buffer)
        deltas = []

        points = get_original_point_cloud(
            new_points, spin_axis_matrix, translation_matrix
        )

        for optim_idx in range(len(self.optim_seq)):
            optim_term = self.optim_seq[optim_idx]
            loss = self.get_loss(points, ori_points, normal_vec, optim_term)
            if optim_idx < self.stage:
                loss.backward(retain_graph=True)
                delta = new_points.grad.detach().clone()
                new_points.grad.zero_()
                delta[:, :, 2] = 0.0
                deltas.append(delta)

            losses[optim_idx] = loss

        return losses, deltas, points

    def forward(
        self,
        new_points,
        spin_axis_matrix,
        translation_matrix,
        ori_points,
        normal_vec,
        g_hat,
        logits,
        target,
    ):

        losses, deltas, points = self.loss_cal(
            new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec
        )
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
        points = get_original_point_cloud(
            new_points, spin_axis_matrix, translation_matrix
        )

        self.update_stage(losses)
        self.update_step_size()

        return points


class boundary_projection_3(boundary_projectuion):

    def __init__(
        self,
        l2_weight: float,
        hd_weight: float,
        cd_weight: float,
        curv_weight: float,
        curv_loss_knn: int,
        step_size: float,
        stage2_steps: float,
        max_steps: int,
        exponential_step: bool,
    ) -> None:
        super(boundary_projection_3, self).__init__(
            l2_weight,
            hd_weight,
            cd_weight,
            curv_weight,
            curv_loss_knn,
            step_size,
            stage2_steps,
            max_steps,
            exponential_step,
        )

        self.gamma = 0.060
        self.initial_lr = 0.03
        self.learning_rate = self.initial_lr

        self.grad_lr = self.initial_lr

        self.epoch = 0

        self.output_losses = None

    def reset(self, batch_size: int):
        self.learning_rate = self.initial_lr
        self.grad_lr = self.initial_lr

        self.output_losses = torch.ones((batch_size, self.optims.__len__())).cuda() * 1e5
        self.output_points = None

    def get_delta(
        self, points: torch.Tensor, ori_points: torch.Tensor, normal_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        points = points.detach()
        points.requires_grad = True

        deltas = []

        loss_list = self.get_loss(points, ori_points, normal_vec)

        for loss_mask in range(loss_list.__len__()):
            loss = loss_list[loss_mask].sum()

            if points.grad is not None:
                points.grad.zero_()

            if loss_mask != loss_list.__len__() - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            deltas.append(points.grad.detach().clone())

        return torch.stack(loss_list).transpose(0, 1), deltas

    def forward(self, points, ori_points, normal_vec, g_hat, logits, target):
        losses, deltas = self.get_delta(
            points, ori_points, normal_vec
        )  # list[B, n, 3]

        alpha = gram_schmidt(g_hat, deltas)  # list[B, n, 3]

        alpha_hat = torch.stack(alpha).sum(0)  # [B, n, 3]

        pred_labels = logits.argmax(dim=-1)  # [B]
        in_out_mask = pred_labels != target  # [B]

        # self.in_out = in_out_mask  # [B]，用于记录每个样本是否 attack 成功

        update_mask = torch.logical_and(
            in_out_mask, (self.output_losses >= losses).all(-1))

        # # 更新 output_losses 和 output_points
        self.output_losses = torch.where(update_mask[:, None], losses, self.output_losses)
        
        if self.output_points is None:
            self.output_points = points
        
        self.output_points = torch.where(
            update_mask[:, None, None],
            points.detach().clone(),
            self.output_points
        )
        # print(in_out_mask)
        if self.exponential_step:
            z = torch.where(
                in_out_mask[:, None, None],
                points - alpha_hat * self.learning_rate,
                points - g_hat * self.grad_lr
            )
        else:
            z = torch.where(
                in_out_mask[:, None, None],
                points - alpha_hat * self.step_size,
                points - g_hat * self.step_size
            )

        # if logits.argmax(-1).item() != target.item():

        #     self.in_out = True

        #     if (self.output_losses >= losses).all():
        #         self.output_losses = losses
        #         self.output_points = points.detach().clone()

        #     if self.exponential_step:
        #         z = points - alpha_hat * self.learning_rate
        #     else:
        #         z = points - alpha_hat * self.step_size

        # else:

        #     self.in_out = False

        #     if self.exponential_step:
        #         z = points - g_hat * self.grad_lr
        #     else:
        #         z = points - g_hat * self.step_size

        points = z.detach().clone()

        self.update_step_size()

        return points

    def update_step_size(self):
        self.learning_rate = self.learning_rate * (1 - self.gamma)
        self.grad_lr = self.grad_lr * (1 - self.gamma)


class boundary_projection_3_si(boundary_projection_3):

    def __init__(self, args, optim_seq=["l2"]) -> None:
        super(boundary_projection_3_si, self).__init__(args, optim_seq)

    def loss_cal(
        self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec
    ) -> Tuple[torch.Tensor, list, torch.Tensor]:

        losses = torch.zeros_like(self.output_losses).cuda()
        deltas = []

        points = get_original_point_cloud(
            new_points, spin_axis_matrix, translation_matrix
        )

        for optim_idx in range(len(self.optim_seq)):
            optim_term = self.optim_seq[optim_idx]
            loss = self.get_loss(points, ori_points, normal_vec, optim_term)

            loss.backward(retain_graph=True)
            delta = new_points.grad.detach().clone()
            new_points.grad.zero_()
            delta[:, :, 2] = 0.0
            deltas.append(delta)

            losses[optim_idx] = loss

        return losses, deltas, points

    def forward(
        self,
        new_points,
        spin_axis_matrix,
        translation_matrix,
        ori_points,
        normal_vec,
        g_hat,
        logits,
        target,
    ):

        losses, deltas, points = self.loss_cal(
            new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec
        )
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
        points = get_original_point_cloud(
            new_points, spin_axis_matrix, translation_matrix
        )

        self.update_step_size()

        return points


class boundary_projection_2(boundary_projectuion):

    def __init__(self, args, weights=[0.5, 0.5], optim_seq=["l2", "curv"]) -> None:
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

    def loss_cal(
        self, points, ori_points, normal_vec
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        losses = torch.zeros_like(self.output_losses).cuda()
        deltas = torch.zeros_like(points).cuda()

        for optim_idx in range(len(self.optim_seq)):
            optim_term = self.optim_seq[optim_idx]
            loss = self.get_loss(points, ori_points, normal_vec, optim_term)
            loss.backward()
            delta = points.grad.detach().clone()

            deltas += (
                self.weights[optim_idx] * delta /
                norm(delta)[:, np.newaxis, np.newaxis]
            )
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
        self.learning_rate = self.learning_rate * (1 - self.gamma)
        self.grad_lr = self.grad_lr * (1 - self.gamma)


class boundary_projection_2_si(boundary_projection_2):

    def __init__(self, args, weights=[0.5, 0.5], optim_seq=["l2", "curv"]) -> None:
        super(boundary_projection_2_si, self).__init__(
            args, weights, optim_seq)

    def loss_cal(
        self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec
    ) -> Tuple[torch.Tensor, list]:

        losses = torch.zeros_like(self.output_losses).cuda()
        deltas = torch.zeros_like(new_points).cuda()

        points = get_original_point_cloud(
            new_points, spin_axis_matrix, translation_matrix
        )

        for optim_idx in range(len(self.optim_seq)):
            optim_term = self.optim_seq[optim_idx]
            loss = self.get_loss(points, ori_points, normal_vec, optim_term)
            loss.backward(retain_graph=True)
            delta = new_points.grad.detach().clone()
            new_points.grad.zero_()

            deltas += (
                self.weights[optim_idx] * delta /
                norm(delta)[:, np.newaxis, np.newaxis]
            )
            deltas[:, :, 2] = 0.0

            losses[optim_idx] = loss

        return losses, deltas, points

    def forward(
        self,
        new_points,
        spin_axis_matrix,
        translation_matrix,
        ori_points,
        normal_vec,
        g_hat,
        logits,
        target,
    ):

        losses, deltas, points = self.loss_cal(
            new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec
        )
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
        points = get_original_point_cloud(
            new_points, spin_axis_matrix, translation_matrix
        )

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

    def loss_cal(
        self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec
    ) -> Tuple[torch.Tensor, list]:

        losses = torch.zeros_like(self.output_losses).cuda()
        deltas = torch.zeros_like(new_points).cuda()

        points = get_original_point_cloud(
            new_points, spin_axis_matrix, translation_matrix
        )

        optim_term = "l2"
        loss = self.get_loss(points, ori_points, normal_vec, optim_term)
        loss.backward()
        delta = new_points.grad.detach().clone()

        deltas = delta
        deltas[:, :, 2] = 0.0

        losses = loss

        return losses, deltas, points

    def forward(
        self,
        new_points,
        spin_axis_matrix,
        translation_matrix,
        ori_points,
        normal_vec,
        g_hat,
        logits,
        target,
    ):

        loss, delta, points = self.loss_cal(
            new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec
        )

        delta_norm = (delta**2).sum((1, 2)).sqrt()
        delta_norm[delta_norm == 0] = 1e-12

        r = (delta * g_hat).sum((1, 2)) / delta_norm

        gamma = self.gamma_min + self.epoch / (self.max_steps + 1) * (
            self.gamma_max - self.gamma_min
        )

        if logits.argmax(1).item() != target.item():
            if (self.output_losses >= loss).all():
                self.output_losses = loss
                self.output_points = points.detach().clone()

            epsilon = gamma * delta_norm
            v_star = ori_points + r[:, None, None] * g_hat
            yi_vstar_norm = ((points - v_star) ** 2).sum((1, 2)).sqrt()
            yi_vstar_norm[yi_vstar_norm == 0] = 1e-9

            tmp = (points - v_star) / yi_vstar_norm[:, None, None]
            tmp = (
                tmp
                * torch.sqrt(torch.max(torch.zeros_like(r), epsilon**2 - r**2))[
                    :, None, None
                ]
            )
            z = v_star + tmp
            points = z.detach()
            if torch.isnan(points).sum().item() != 0:
                assert False, "Out NAN Occured!!!"

        else:
            epsilon = delta_norm / gamma
            tmp = r + torch.sqrt(epsilon**2 - delta_norm**2 + r**2)
            z = points - tmp[:, None, None] * g_hat

            points = z.detach()
            if torch.isnan(points).sum().item() != 0:
                assert False, "In NAN Occured!!!"

        new_points = z.detach().clone()
        points = get_original_point_cloud(
            new_points, spin_axis_matrix, translation_matrix
        )

        self.update_step_size()

        return points


class boundary_projection_query_si(boundary_projection_3):

    def __init__(self, args, optim_seq=["l2"]) -> None:
        super(boundary_projection_query_si, self).__init__(args, optim_seq)

    def loss_cal(
        self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec
    ) -> Tuple[torch.Tensor, list, torch.Tensor]:

        losses = torch.zeros_like(self.output_losses).cuda()
        deltas = []

        points = get_original_point_cloud(
            new_points, spin_axis_matrix, translation_matrix
        )

        for optim_idx in range(len(self.optim_seq)):
            optim_term = self.optim_seq[optim_idx]
            loss = self.get_loss(points, ori_points, normal_vec, optim_term)

            loss.backward(retain_graph=True)
            delta = new_points.grad.detach().clone()
            new_points.grad.zero_()
            delta[:, :, 2] = 0.0
            deltas.append(delta)

            losses[optim_idx] = loss

        return losses, deltas, points

    def gradient_map_project(
        self,
        new_points,
        spin_axis_matrix,
        translation_matrix,
        ori_points,
        normal_vec,
        g_hat,
    ):
        losses, deltas, points = self.loss_cal(
            new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec
        )
        alpha = gram_schmidt(g_hat, deltas)
        alpha_hat = np.array(alpha).sum(0)

        return alpha_hat

    def forward(
        self,
        new_points,
        spin_axis_matrix,
        translation_matrix,
        ori_points,
        normal_vec,
        g_hat,
        logits,
        target,
    ):

        losses, deltas, points = self.loss_cal(
            new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec
        )
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
        points = get_original_point_cloud(
            new_points, spin_axis_matrix, translation_matrix
        )

        self.update_step_size()

        return points


# __all__ = {
#     "bp3": boundary_projection_3,
#     "bp3_si": boundary_projection_3_si
# }
