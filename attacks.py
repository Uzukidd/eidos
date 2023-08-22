import os
import sys
import numpy as np
import importlib
import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from utils.loss_utils import norm_l2_loss, pseudo_chamfer_loss, hausdorff_loss, curvature_loss, _get_kappa_ori, _get_kappa_adv
from utils.bp_utils import *

from baselines import *
from attacker import geoA3_attack, GSDA_attack


from models import build_model_from_cfg, load_point_bert
import os
from utils.config_pointbert import *
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model/classifier'))



class PointCloudAttack(object):
    def __init__(self, args):
        """Shape-invariant Adversarial Attack for 3D Point Clouds.
        """
        self.args = args
        self.device = args.device
        self.recall = []

        self.initial_const = args.initial_const
        self.eps = args.eps
        self.normal = args.normal
        self.step_size = args.step_size
        self.num_class = args.num_class
        self.max_steps = args.max_steps
        self.top5_attack = args.top5_attack
        self.batch_size = args.batch_size
        self.l2_weight = args.l2_weight
        self.hd_weight = args.hd_weight
        self.cd_weight = args.cd_weight
        self.curv_weight = args.curv_weight
        self.binary_max_steps = args.binary_max_steps
        self.curv_loss_knn = args.curv_loss_knn
        self.num_class = args.num_class
        self.bp_version = args.bp_version
        self.stage2_steps = args.stage2_steps

        assert args.transfer_attack_method is None or args.query_attack_method is None
        assert not args.transfer_attack_method is None or not args.query_attack_method is None
        self.attack_method = args.transfer_attack_method if args.query_attack_method is None else args.query_attack_method

        self.build_models()
        self.defense_method = args.defense_method
        self.pre_head = None
        if not args.defense_method is None:
            self.pre_head = self.get_defense_head(args.defense_method)

    def get_delta(self, points, ori_points, normal_vec, reg_type = "l2"):
        points = points.detach()
        points.requires_grad = True
        if reg_type == "l2":
            loss = self.get_loss(points, ori_points, normal_vec, reg_type)
            loss.backward()
            delta = points.grad.detach() * 0.5
        elif reg_type == "curv":
            loss = self.get_loss(points, ori_points, normal_vec, reg_type)
            loss.backward()
            delta = points.grad.detach()
        elif reg_type == "hd":
            loss = self.get_loss(points, ori_points, normal_vec, reg_type)
            loss.backward()
            delta = points.grad.detach()
        elif reg_type == "cd":
            loss = self.get_loss(points, ori_points, normal_vec, reg_type)
            loss.backward()
            delta = points.grad.detach()
        else:
            raise NotImplementedError

        return delta.detach(), loss.detach()


    def get_delta_si(self, new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec, reg_type = "l2"):
        new_points = new_points.detach()
        new_points.requires_grad = True
        points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
        if reg_type == "l2":
            loss = self.get_loss(points, ori_points, normal_vec, reg_type)
            loss.backward()
            delta = new_points.grad.detach() * 0.5
        elif reg_type == "curv":
            loss = self.get_loss(points, ori_points, normal_vec, reg_type)
            loss.backward()
            delta = new_points.grad.detach()
        elif reg_type == "hd":
            loss = self.get_loss(points, ori_points, normal_vec, reg_type)
            loss.backward()
            delta = new_points.grad.detach()
        elif reg_type == "cd":
            loss = self.get_loss(points, ori_points, normal_vec, reg_type)
            loss.backward()
            delta = new_points.grad.detach()
        else:
            raise NotImplementedError

    
        return delta.detach(), loss.detach()

    def get_delta_gsda(self, points, factor, ori_points, normal_vec, reg_type = "l2"):
        if reg_type == "l2":
            loss = self.get_loss(points, ori_points, normal_vec, reg_type)
            loss.backward(retain_graph=True)
            delta = factor.grad.detach() * 0.5
            factor.grad.zero_()
        elif reg_type == "curv":
            loss = self.get_loss(points, ori_points, normal_vec, reg_type)
            loss.backward(retain_graph=True)
            delta = factor.grad.detach()
            factor.grad.zero_()
        elif reg_type == "hd":
            loss = self.get_loss(points, ori_points, normal_vec, reg_type)
            loss.backward(retain_graph=True)
            delta = factor.grad.detach()
            factor.grad.zero_()
        elif reg_type == "cd":
            loss = self.get_loss(points, ori_points, normal_vec, reg_type)
            loss.backward(retain_graph=True)
            delta = factor.grad.detach()
            factor.grad.zero_()
        else:
            raise NotImplementedError

        return delta.detach(), loss.detach()


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


    def build_models(self):
        """Build white-box surrogate model and black-box target model.
        """
        if self.args.surrogate_model == "point_transformer":
            wb_classifier = self.build_models_aux()
        else:
            # load white-box surrogate models
            MODEL = importlib.import_module(self.args.surrogate_model)
            wb_classifier = MODEL.get_model(
                self.num_class,
                normal_channel=self.normal
            )
            wb_classifier = wb_classifier.to(self.args.device)
            wb_classifier = self.load_models(wb_classifier, self.args.surrogate_model)
        
        if self.args.target_model == "point_transformer":
            classifier = self.build_models_aux()
        else:
            # load black-box target models
            MODEL = importlib.import_module(self.args.target_model)
            classifier = MODEL.get_model(
                self.num_class,
                normal_channel=self.normal
            )
            classifier = classifier.to(self.args.device)
            classifier = self.load_models(classifier, self.args.target_model)
        # set eval
        self.wb_classifier = wb_classifier.eval()
        self.classifier = classifier.eval()
        

    def build_models_aux(self):
        """Build point_transformer.
        """
        CKPT_PATH = "./checkpoint/ModelNet40/PointTransformer_ModelNet1024points.pth"
        MODEL_CFG_PATH = "./cfgs/ModelNet_models/PointTransformer.yaml"
        
        config = cfg_from_yaml_file(MODEL_CFG_PATH)
        base_model = build_model_from_cfg(config.model)
        load_point_bert(base_model, CKPT_PATH)

        return base_model.cuda()

    def load_models(self, classifier, model_name):
        """Load white-box surrogate model and black-box target model.
        """
        model_path = os.path.join('./checkpoint/' + self.args.dataset, model_name)
        if os.path.exists(model_path + '.pth'):
            checkpoint = torch.load(model_path + '.pth')
        elif os.path.exists(model_path + '.t7'):
            checkpoint = torch.load(model_path + '.t7')
        elif os.path.exists(model_path + '.tar'):
            checkpoint = torch.load(model_path + '.tar')
        else:
            raise NotImplementedError

        try:
            if 'model_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state_dict'])
            elif 'model_state' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state'])
            elif 'state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['state_dict'])
            else:
                classifier.load_state_dict(checkpoint)
        except:
            classifier = nn.DataParallel(classifier)
            classifier.load_state_dict(checkpoint)
        return classifier

    def gsda_boundary_projection_2(self, ori_points, _x, v, factor, mask, normal_vec, target, output_points, output_reg, initial_lr, gamma, i):
        factor = factor.detach()
        factor.requires_grad = True
        gft = factor * mask + _x
        points = torch.einsum('bij,bjk->bik', v, gft)
        
        reg_loss = torch.zeros_like(output_reg).cuda()
        delta = torch.zeros(points.size()).cuda()

        if self.l2_weight != 0.0:
            delta_l2, reg_loss_l2 = self.get_delta_gsda(points, factor, ori_points, None, "l2")
            delta_l2_norm = (delta_l2 ** 2).sum((1,2)).sqrt()
            delta_l2_norm[delta_l2_norm == 0] = 1e-12
            delta += self.l2_weight * delta_l2 / delta_l2_norm[:, None, None]
            reg_loss[:, 0] = reg_loss_l2
        
        if self.hd_weight != 0.0:
            delta_hd, reg_loss_hd = self.get_delta_gsda(points, factor, ori_points, None, "hd")
            delta_hd_norm = (delta_hd ** 2).sum((1,2)).sqrt()
            delta_hd_norm[delta_hd_norm == 0] = 1e-12
            delta += self.hd_weight * delta_hd / delta_hd_norm[:, None, None]
            reg_loss[:, 1] = reg_loss_hd

        if self.cd_weight != 0.0:
            delta_cd, reg_loss_cd = self.get_delta_gsda(points, factor, ori_points, None, "cd")
            delta_cd_norm = (delta_cd ** 2).sum((1,2)).sqrt()
            delta_cd_norm[delta_cd_norm == 0] = 1e-12
            delta += self.cd_weight * delta_cd / delta_cd_norm[:, None, None]
            reg_loss[:, 2] = reg_loss_cd
        
        if self.curv_weight != 0.0:
            delta_curv, reg_loss_curv = self.get_delta_gsda(points, factor, ori_points, normal_vec, "curv")
            delta_curv_norm = (delta_curv ** 2).sum((1,2)).sqrt()
            delta_curv_norm[delta_curv_norm == 0] = 1e-12
            delta += self.curv_weight * delta_curv / delta_curv_norm[:, None, None]
            reg_loss[:, 3] = reg_loss_curv


        if not self.defense_method is None:
            logits = self.wb_classifier(self.pre_head(points.transpose(1, 2)))
        else:
            logits = self.wb_classifier(points.transpose(1, 2))

        logits = F.log_softmax(logits)
        loss = logits[:, target.item()]
        self.wb_classifier.zero_grad()
        loss.backward()

        g = factor.grad.detach()

        g_norm = (g ** 2).sum((1,2)).sqrt()
        g_norm[g_norm == 0] = 1e-12
        g_hat = g / g_norm[:, None, None]

        alpha_hat = delta
        initial_lr = initial_lr * (1 - gamma)

        if logits.argmax(1).item() != target.item():
            if (reg_loss <= output_reg).all():
                output_points = points.detach()
                output_reg = reg_loss
            
            z = factor - alpha_hat * initial_lr
            factor = z.detach()
            if torch.isnan(factor).sum().item() != 0:
                assert False, "Out NAN Occured!!!"  
            
        else:

            z = factor - g_hat * self.step_size
            factor = z.detach()
            if torch.isnan(factor).sum().item() != 0:
                assert False, "In NAN Occured!!!"


        return factor, output_points, output_reg, initial_lr

    def CWLoss(self, logits, target, kappa=0, tar=False, num_classes=40):
        """Carlini & Wagner attack loss. 

        Args:
            logits (torch.cuda.FloatTensor): the predicted logits, [1, num_classes].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
        target_one_hot = Variable(torch.eye(num_classes).type(torch.cuda.FloatTensor)[target.long()].cuda())

        real = torch.sum(target_one_hot*logits, 1)
        if not self.top5_attack:
            ### top-1 attack
            other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
        else:
            ### top-5 attack
            other = torch.topk((1-target_one_hot)*logits - (target_one_hot*10000), 5)[0][:, 4]
        kappa = torch.zeros_like(other).fill_(kappa)

        if tar:
            return torch.sum(torch.max(other-real, kappa))
        else :
            return torch.sum(torch.max(real-other, kappa))


    def run(self, points, target):
        """Main attack method.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        if self.attack_method == 'ifgm_si_adv':
            return self.shape_invariant_ifgm(points, target)
        elif self.attack_method == 'ifgm_bp_ours':
            return self.shape_invariant_ifgm_bp_mod2(points, target)
        elif self.attack_method == 'ifgm_si_adv_query':
            return self.shape_invariant_query_attack(points, target)
        elif self.attack_method == 'ifgm_bp_ours_query':
            return self.shape_invariant_ifgm_bp_query_attack(points, target)
        elif self.attack_method == 'ifgm_si_bp':
            return self.shape_invariant_ifgm_si_bp(points, target)
        elif self.attack_method == "ifgm_bp":
            return self.ifgm_bp(points, target)
        elif self.attack_method == 'geoa3':
            return self.geoa3_attack(points, target)
        elif self.attack_method == 'gsda':
            return self.gsda_attack(points, target)
        elif self.attack_method == 'gsda_bp':
            return self.gsda_attack_bp(points, target)
        elif self.attack_method == 'simba':
            return self.simba_attack(points, target)
        elif self.attack_method == 'simbapp':
            return self.simbapp_attack(points, target)
        else:
            NotImplementedError


    def get_defense_head(self, method):
        """Set the pre-processing based defense module.

        Args:
            method (str): defense method name.
        """
        if method == 'sor':
            pre_head = SORDefense(k=2, alpha=1.1)
        elif method == 'srs':
            pre_head = SRSDefense(drop_num=500)
        elif method == 'dupnet':
            pre_head = DUPNet(sor_k=2, sor_alpha=1.1, npoint=1024, up_ratio=4)
        else:
            raise NotImplementedError
        return pre_head

    def predict(self, points):
        logits = None
        points = points[:,:,:3].data # P, [1, N, 3]
        points = points.contiguous()
        with torch.no_grad():
            points = points.transpose(1, 2) # P, [1, 3, N]
            if not self.defense_method is None:
                if not self.args.query_attack_method is None:
                    logits = self.classifier(self.pre_head(points))
                else:
                    logits = self.wb_classifier(self.pre_head(points))
            else:
                if not self.args.query_attack_method is None:
                    logits = self.classifier(points)
                else:
                    logits = self.wb_classifier(points)

            logits = logits.argmax(1)
        
        return logits

    def ifgm_bp(self, points, target):
        """Black-box I-FGSM based on shape-invariant sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        normal_vec = points[:,:,-3:].detach() # N, [1, N, 3]
        normal_vec = normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True)) # N, [1, N, 3]
        points = points[:,:,:3].detach() # P, [1, N, 3]
        ori_points = points.detach()
        clip_func = ClipPointsLinf(budget=self.eps)# * np.sqrt(3*1024))

        gamma_max = 1.0
        gamma_min = 0.05

        stage_2 = False

        output_points = points.detach()
        output_reg = 1e10

        for i in range(self.max_steps):
            if not stage_2:
                points = points.detach()
                points.requires_grad = True

                # get white-box gradients
                if not self.defense_method is None:
                    logits = self.wb_classifier(self.pre_head(points.transpose(1, 2)))
                else:
                    logits = self.wb_classifier(points.transpose(1, 2))
                
                loss = self.CWLoss(logits, target, kappa=0., tar=False, num_classes=self.num_class)
                # logits = F.log_softmax(logits)
                # loss = logits[:, target.item()]
                self.wb_classifier.zero_grad()
                loss.backward()

                grad = points.grad.detach()

                # L2
                norm = torch.sum(grad ** 2, dim=[1, 2]) ** 0.5
                norm[norm == 0] = 1e-12

                gamma = gamma_min + i / (self.max_steps + 1) * (gamma_max - gamma_min)
                
                points = points - gamma * self.step_size * np.sqrt(3*1024) * grad / (norm[:, np.newaxis, np.newaxis])
                points = clip_func(points, ori_points)

                if not self.defense_method is None:
                    logits = self.wb_classifier(self.pre_head(points.transpose(1, 2)))
                else:
                    logits = self.wb_classifier(points.transpose(1, 2))
                
                logits = logits.argmax(1).item()

                delta, reg_loss = self.get_delta(points, ori_points, None, "l2")
                output_points = points.detach()
                output_reg = reg_loss.item()

                if logits != target.item():
                    stage_2 = True

            else:
                points = points.detach()
                points.requires_grad = True

                delta, reg_loss = self.get_delta(points, ori_points, None, "l2")

                if not self.defense_method is None:
                    logits = self.wb_classifier(self.pre_head(points.transpose(1, 2)))
                else:
                    logits = self.wb_classifier(points.transpose(1, 2))

                logits = F.log_softmax(logits)
                loss = logits[:, target.item()]
                self.wb_classifier.zero_grad()
                loss.backward()

                g = points.grad.detach()

                g_norm = (g ** 2).sum((1,2)).sqrt()
                delta_norm = (delta ** 2).sum((1,2)).sqrt()
                g_norm[g_norm == 0] = 1e-12
                delta_norm[delta_norm == 0] = 1e-12
                g_hat = g / g_norm[:, None, None]
                
                r = (delta * g_hat).sum((1,2)) / delta_norm

                gamma = gamma_min + i / (self.max_steps + 1) * (gamma_max - gamma_min)

                if logits.argmax(1).item() != target.item():
                    if reg_loss.item() < output_reg:
                        output_points = points.detach()
                        output_reg = reg_loss.item()
                    
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

        with torch.no_grad():
            adv_points = output_points.detach()
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.transpose(1, 2)))
            else:
                adv_logits = self.classifier(points.transpose(1, 2))
            adv_target = adv_logits.argmax(1).item()

        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1

        del normal_vec, grad
        return adv_points, adv_target, (adv_logits.argmax(1) != target).sum().item()

    def geoa3_attack(self, points, target):
        data = points
        data = (data[0][np.newaxis, :, 0:3], data[0][np.newaxis, :, 3:6], target)
        new_data = [None, None, None, None]
        new_data[0] = torch.zeros(size=(data[0].shape[0], 1, data[0].shape[1], data[0].shape[2]))
        new_data[0][:, np.newaxis, :, :] = data[0]

        new_data[1] = torch.zeros(size=(data[0].shape[0], 1, data[0].shape[1], data[0].shape[2]))
        new_data[1][:, np.newaxis, :, :] = data[1]

        new_data[2] = torch.zeros(size=(data[2].shape[0], 1))
        new_data[2][:, np.newaxis] = data[2]

        new_data[3] = torch.zeros(size=(data[2].shape[0], 1))
        ten_label_indexes = [0, 2, 4, 5, 8, 22, 30, 33, 35, 37]

        for bs in range(new_data[0].shape[0]):
            target_labels = []
            label_index = np.random.randint(len(ten_label_indexes) - 1, size=1)[0]
            if ten_label_indexes[label_index] >= new_data[2][bs][0]:
                label = ten_label_indexes[label_index + 1]
            else:
                label = ten_label_indexes[label_index]
            

            target_labels.append(label)

            target_labels = torch.from_numpy(np.array(target_labels)).long()
            new_data[3][bs] = target_labels

        data = (new_data[0].float().contiguous(), new_data[1].float().contiguous(), new_data[2].long(), new_data[3].long())
        if not self.defense_method is None:
            defense = self.pre_head
        else:
            defense = None
        adv_pc, targeted_label, attack_success_indicator, best_attack_step, loss, cd_loss, hd_loss, cur_loss = geoA3_attack.attack(self.wb_classifier, defense, data, self.args)
        return adv_pc.permute(0, 2, 1), targeted_label, None

    def gsda_attack(self, points, target):
        data = points
        data = (data[0][np.newaxis, :, 0:3], data[0][np.newaxis, :, 3:6], target)
        new_data = [None, None, None, None]
        new_data[0] = torch.zeros(size=(data[0].shape[0], 1, data[0].shape[1], data[0].shape[2]))
        new_data[0][:, np.newaxis, :, :] = data[0]

        new_data[1] = torch.zeros(size=(data[0].shape[0], 1, data[0].shape[1], data[0].shape[2]))
        new_data[1][:, np.newaxis, :, :] = data[1]

        new_data[2] = torch.zeros(size=(data[2].shape[0], 1))
        new_data[2][:, np.newaxis] = data[2]

        new_data[3] = torch.zeros(size=(data[2].shape[0], 1))
        ten_label_indexes = [0, 2, 4, 5, 8, 22, 30, 33, 35, 37]

        for bs in range(new_data[0].shape[0]):
            target_labels = []
            label_index = np.random.randint(len(ten_label_indexes) - 1, size=1)[0]
            if ten_label_indexes[label_index] >= new_data[2][bs][0]:
                label = ten_label_indexes[label_index + 1]
            else:
                label = ten_label_indexes[label_index]
            
            # label = new_data[2][bs][0]
            target_labels.append(label)

            target_labels = torch.from_numpy(np.array(target_labels)).long()
            new_data[3][bs] = target_labels

        data = (new_data[0].float().contiguous(), new_data[1].float().contiguous(), new_data[2].long(), new_data[3].long())
        if not self.defense_method is None:
            defense = self.pre_head
        else:
            defense = None
        adv_pc, targeted_label, attack_success_indicator, best_attack_step, loss = GSDA_attack.attack(self.wb_classifier, defense, data, self.args)

        return adv_pc.permute(0, 2, 1), targeted_label, None

    def gsda_attack_bp(self, points, target):
        data = points
        data = (data[0][np.newaxis, :, 0:3], data[0][np.newaxis, :, 3:6], target)
        new_data = [None, None, None, None]
        new_data[0] = torch.zeros(size=(data[0].shape[0], 1, data[0].shape[1], data[0].shape[2]))
        new_data[0][:, np.newaxis, :, :] = data[0]

        new_data[1] = torch.zeros(size=(data[0].shape[0], 1, data[0].shape[1], data[0].shape[2]))
        new_data[1][:, np.newaxis, :, :] = data[1]

        new_data[2] = torch.zeros(size=(data[2].shape[0], 1))
        new_data[2][:, np.newaxis] = data[2]

        new_data[3] = torch.zeros(size=(data[2].shape[0], 1))
        ten_label_indexes = [0, 2, 4, 5, 8, 22, 30, 33, 35, 37]

        for bs in range(new_data[0].shape[0]):
            target_labels = []
            label_index = np.random.randint(len(ten_label_indexes) - 1, size=1)[0]
            if ten_label_indexes[label_index] >= new_data[2][bs][0]:
                label = ten_label_indexes[label_index + 1]
            else:
                label = ten_label_indexes[label_index]
            
            target_labels.append(label)

            target_labels = torch.from_numpy(np.array(target_labels)).long()
            new_data[3][bs] = target_labels

        data = (new_data[0].float().contiguous(), new_data[1].float().contiguous(), new_data[2].long(), new_data[3].long())
        if not self.defense_method is None:
            defense = self.pre_head
        else:
            defense = None
        adv_pc, targeted_label, attack_success_indicator, best_attack_step, loss = GSDA_attack.attack_bp(self.wb_classifier, defense, data, self.args, self)

        return adv_pc, targeted_label, None

    def shape_invariant_ifgm(self, points, target):
        """Black-box I-FGSM based on shape-invariant sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        normal_vec = points[:,:,-3:].data # N, [1, N, 3]
        normal_vec = normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True)) # N, [1, N, 3]
        points = points[:,:,:3].data # P, [1, N, 3]
        ori_points = points.data
        clip_func = ClipPointsLinf(budget=self.eps)# * np.sqrt(3*1024))

        for i in range(self.max_steps):
            # P -> P', detach()
            new_points, spin_axis_matrix, translation_matrix = get_transformed_point_cloud(points, normal_vec)
            new_points = new_points.detach()

            new_points.requires_grad = True
            # P' -> P
            points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
            points = points.transpose(1, 2) # P, [1, 3, N]
            # get white-box gradients
            if not self.defense_method is None:
                logits = self.wb_classifier(self.pre_head(points))
            else:
                logits = self.wb_classifier(points)
            loss = self.CWLoss(logits, target, kappa=0., tar=False, num_classes=self.num_class)
            self.wb_classifier.zero_grad()
            loss.backward()
            # print(loss.item(), logits.max(1)[1], target)
            grad = new_points.grad.data # g, [1, N, 3]
            grad[:,:,2] = 0.

            # update P', P and N
            # # Linf
            # new_points = new_points - self.step_size * torch.sign(grad)
            # L2
            norm = torch.sum(grad ** 2, dim=[1, 2]) ** 0.5
            
            new_points = new_points - self.step_size * np.sqrt(3*1024) * grad / (norm[:, None, None] + 1e-9)
            
            points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix) # P, [1, N, 3]
            points = clip_func(points, ori_points)

            normal_vec = get_normal_vector(points) # N, [1, N, 3]

        with torch.no_grad():
            adv_points = points.data
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.transpose(1, 2).detach()))
            else:
                adv_logits = self.classifier(points.transpose(1, 2).detach())
            adv_target = adv_logits.data.max(1)[1]

        
        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1

        del normal_vec, grad, new_points, spin_axis_matrix, translation_matrix
        return adv_points, adv_target, (adv_logits.data.max(1)[1] != target).sum().item()

    def shape_invariant_ifgm_bp_mod2(self, points, target):
        """Black-box I-FGSM based on shape-invariant sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        normal_vec = points[:,:,-3:].data # N, [1, N, 3]
        normal_vec = normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True)) # N, [1, N, 3]
        points = points[:,:,:3].data # P, [1, N, 3]
        ori_points = points.data
        clip_func = ClipPointsLinf(budget=self.eps)# * np.sqrt(3*1024))

        si_grad_required = False

        if self.bp_version == "bp4":
            bp = boundary_projection_4(self.args)
        elif self.bp_version == "bp4_si":
            bp = boundary_projection_4_si(self.args)
            si_grad_required = True
        elif self.bp_version == "bp3":
            bp_optims = []
            if self.l2_weight != 0.0:
                bp_optims.append("l2")

            if self.hd_weight != 0.0:
                bp_optims.append("hd")

            if self.curv_weight != 0.0:
                bp_optims.append("curv")
        
            if self.cd_weight != 0.0:
                bp_optims.append("cd")
            bp = boundary_projection_3(self.args, bp_optims)
        elif self.bp_version == "bp3_si":
            bp_optims = []
            if self.l2_weight != 0.0:
                bp_optims.append("l2")

            if self.hd_weight != 0.0:
                bp_optims.append("hd")

            if self.curv_weight != 0.0:
                bp_optims.append("curv")
        
            if self.cd_weight != 0.0:
                bp_optims.append("cd")
            bp = boundary_projection_3_si(self.args, bp_optims)
            si_grad_required = True
        elif self.bp_version == "bp2":
            bp_weights = []
            bp_optims = []
            if self.l2_weight != 0.0:
                bp_weights.append(self.l2_weight)
                bp_optims.append("l2")

            if self.hd_weight != 0.0:
                bp_weights.append(self.hd_weight)
                bp_optims.append("hd")

            if self.curv_weight != 0.0:
                bp_weights.append(self.curv_weight)
                bp_optims.append("curv")
        
            if self.cd_weight != 0.0:
                bp_weights.append(self.cd_weight)
                bp_optims.append("cd")

            bp = boundary_projection_2(self.args, weights = bp_weights, optim_seq = bp_optims)
        elif self.bp_version == "bp2_si":
            bp_weights = []
            bp_optims = []
            if self.l2_weight != 0.0:
                bp_weights.append(self.l2_weight)
                bp_optims.append("l2")

            if self.hd_weight != 0.0:
                bp_weights.append(self.hd_weight)
                bp_optims.append("hd")

            if self.curv_weight != 0.0:
                bp_weights.append(self.curv_weight)
                bp_optims.append("curv")
        
            if self.cd_weight != 0.0:
                bp_weights.append(self.cd_weight)
                bp_optims.append("cd")

            bp = boundary_projection_2_si(self.args, weights = bp_weights, optim_seq = bp_optims)
            si_grad_required = True
        elif self.bp_version == "bp1_si":
            bp = boundary_projection_1_si(self.args)
            si_grad_required = True
        
        stage2 = False

        output_points = points
        output_reg = torch.ones((points.size(0), 4)).cuda() * 1e10

        for i in range(self.max_steps):
            # print(self.max_steps)
            bp.epoch = i

            if not stage2:
                # P -> P', detach()
                new_points, spin_axis_matrix, translation_matrix = get_transformed_point_cloud(points, normal_vec)
                new_points = new_points.detach()
                new_points.requires_grad = True
                # P' -> P
                points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
                points = points.transpose(1, 2) # P, [1, 3, N]
                # get white-box gradients
                if not self.defense_method is None:
                    logits = self.wb_classifier(self.pre_head(points))
                else:
                    logits = self.wb_classifier(points)
                loss = self.CWLoss(logits, target, kappa=0., tar=False, num_classes=self.num_class)
                self.wb_classifier.zero_grad()
                loss.backward()
                # print(loss.item(), logits.max(1)[1], target)
                grad = new_points.grad.data # g, [1, N, 3]
                grad[:,:,2] = 0.
                # update P', P and N
                # # Linf
                # new_points = new_points - self.step_size * torch.sign(grad)
                # L2
                norm = torch.sum(grad ** 2, dim=[1, 2]) ** 0.5
                new_points = new_points - self.step_size * np.sqrt(3*1024) * grad / (norm[:, None, None] + 1e-9)
                points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix) # P, [1, N, 3]
                points = clip_func(points, ori_points)

                points = points.detach()

                normal_vec = get_normal_vector(points) # N, [1, N, 3]

                if not self.defense_method is None:
                    logits = self.wb_classifier(self.pre_head(points.transpose(1, 2)))
                else:
                    logits = self.wb_classifier(points.transpose(1, 2))
                
                logits = logits.argmax(-1).item()
                stage2 = logits != target.item()
                reg_loss = torch.zeros_like(output_reg).cuda()

                if self.l2_weight != 0.0:
                    reg_loss_l2 = self.get_loss(points, ori_points, None, "l2")
                    reg_loss[:, 0] = reg_loss_l2
                
                if self.hd_weight != 0.0:
                    reg_loss_hd = self.get_loss(points, ori_points, None, "hd")
                    reg_loss[:, 1] = reg_loss_hd

                if self.cd_weight != 0.0:
                    reg_loss_cd = self.get_loss(points, ori_points, None, "cd")
                    reg_loss[:, 2] = reg_loss_cd
                
                if self.curv_weight != 0.0:
                    reg_loss_curv = self.get_loss(points,ori_points, normal_vec, "curv")
                    reg_loss[:, 3] = reg_loss_curv


                if logits != target.item() and (reg_loss <= output_reg).all():
                    output_points = points.detach()
                    output_reg = reg_loss
                
            else:

                points = points.detach()
                points.requires_grad = True

                if not si_grad_required:

                    if not self.defense_method is None:
                        logits = self.wb_classifier(self.pre_head(points.transpose(1, 2)))
                    else:
                        logits = self.wb_classifier(points.transpose(1, 2))

                    logits = F.log_softmax(logits)
                    loss = logits[:, target.item()]
                    self.wb_classifier.zero_grad()
                    loss.backward()

                    g = points.grad.detach()

                    g_norm = (g ** 2).sum((1,2)).sqrt()
                    g_norm[g_norm == 0] = 1e-12
                    g_hat = g / g_norm[:, None, None]


                    points = bp(points, ori_points, normal_vec, g_hat, logits, target)

                    normal_vec = get_normal_vector(points)
                
                else:

                    new_points, spin_axis_matrix, translation_matrix = get_transformed_point_cloud(points, normal_vec)
                    new_points = new_points.detach()
                    new_points.requires_grad = True

                    points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
                    if not self.defense_method is None:
                        logits = self.wb_classifier(self.pre_head(points.transpose(1, 2)))
                    else:
                        logits = self.wb_classifier(points.transpose(1, 2))

                    logits = F.log_softmax(logits)
                    loss = logits[:, target.item()]
                    self.wb_classifier.zero_grad()
                    loss.backward()

                    g = new_points.grad.detach().clone()
                    g[:, :, 2] = 0.

                    new_points.grad.zero_()

                    g_norm = (g ** 2).sum((1,2)).sqrt()
                    g_norm[g_norm == 0] = 1e-12
                    g_hat = g / g_norm[:, None, None]

                    normal_vec = torch.zeros_like(normal_vec)
                    normal_vec[:, :, 2] = 1

                    points = bp(new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec, g_hat, logits, target)

                    normal_vec = get_normal_vector(points)
                    
        with torch.no_grad():
            if not bp.output_points is None:
                adv_points = bp.output_points.clone()
            else:
                adv_points = points.clone()
            
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(adv_points.transpose(1, 2)))
            else:
                adv_logits = self.classifier(adv_points.transpose(1, 2))
            adv_target = adv_logits.argmax(-1)

        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1

        del normal_vec, grad, new_points, spin_axis_matrix, translation_matrix
        return adv_points, adv_target, (adv_logits.data.max(1)[1] != target).sum().item()
    
    def shape_invariant_ifgm_si_bp(self, points, target):
        """Black-box I-FGSM based on shape-invariant sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        normal_vec = points[:,:,-3:].data # N, [1, N, 3]
        normal_vec = normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True)) # N, [1, N, 3]
        points = points[:,:,:3].data # P, [1, N, 3]
        ori_points = points.data
        clip_func = ClipPointsLinf(budget=self.eps)# * np.sqrt(3*1024))
        gamma_max = 1.0
        gamma_min = 0.05
        stage2 = False

        output_points = points
        output_reg = 1e10

        for i in range(self.max_steps):

            if not stage2:
                # P -> P', detach()
                new_points, spin_axis_matrix, translation_matrix = get_transformed_point_cloud(points, normal_vec)
                new_points = new_points.detach()
                new_points.requires_grad = True
                # P' -> P
                points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
                points = points.transpose(1, 2) # P, [1, 3, N]
                # get white-box gradients
                if not self.defense_method is None:
                    logits = self.wb_classifier(self.pre_head(points))
                else:
                    logits = self.wb_classifier(points)
                loss = self.CWLoss(logits, target, kappa=0., tar=False, num_classes=self.num_class)
                self.wb_classifier.zero_grad()
                loss.backward()
                # print(loss.item(), logits.max(1)[1], target)
                grad = new_points.grad.data # g, [1, N, 3]
                grad[:,:,2] = 0.
                # update P', P and N
                # # Linf
                # new_points = new_points - self.step_size * torch.sign(grad)
                # L2
                norm = torch.sum(grad ** 2, dim=[1, 2]) ** 0.5
                new_points = new_points - self.step_size * np.sqrt(3*1024) * grad / (norm[:, None, None] + 1e-9)
                points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix) # P, [1, N, 3]
                points = clip_func(points, ori_points)

                points = points.detach()

                normal_vec = get_normal_vector(points) # N, [1, N, 3]
                if not self.defense_method is None:
                    logits = self.wb_classifier(self.pre_head(points.transpose(1, 2)))
                else:
                    logits = self.wb_classifier(points.transpose(1, 2))
                
                logits = logits.argmax(-1).item()

                stage2 = logits != target.item()

                reg_loss = self.get_loss(points, ori_points, normal_vec, "l2")

                if logits != target.item() and reg_loss < output_reg:
                    output_points = points.detach()
                    output_reg = reg_loss

            else:

                # P -> P'
                new_points, spin_axis_matrix, translation_matrix = get_transformed_point_cloud(points, normal_vec)
                new_points = new_points.detach()
                new_points.requires_grad = True
                # P' -> P
                points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)

                new_ori_points = ori_points + translation_matrix
                new_ori_points = new_ori_points.unsqueeze(-1)
                new_ori_points = torch.matmul(spin_axis_matrix, new_ori_points)
                new_ori_points = new_ori_points.squeeze(-1).data

                normal_vec = torch.zeros_like(new_ori_points)
                normal_vec[:, :, 2] = 1

                delta, reg_loss = self.get_delta_si(points,spin_axis_matrix, translation_matrix, ori_points, None, "l2")

                if not self.defense_method is None:
                    logits = self.wb_classifier(self.pre_head(points.transpose(1, 2)))
                else:
                    logits = self.wb_classifier(points.transpose(1, 2))

                logits = F.log_softmax(logits)
                loss = logits[:, target.item()]
                self.wb_classifier.zero_grad()
                loss.backward()

                g = new_points.grad.detach()

                g_norm = (g ** 2).sum((1,2)).sqrt()
                delta_norm = (delta ** 2).sum((1,2)).sqrt()
                g_norm[g_norm == 0] = 1e-12
                delta_norm[delta_norm == 0] = 1e-12
                g_hat = g / g_norm[:, None, None]
                
                r = (delta * g_hat).sum((1,2)) / delta_norm

                gamma = gamma_min + i / (self.max_steps + 1) * (gamma_max - gamma_min)

                if logits.argmax(1).item() != target.item():
                    if reg_loss.item() < output_reg:
                        output_points = points.detach()
                        output_reg = reg_loss.item()
                    
                    epsilon = gamma * delta_norm
                    v_star = new_ori_points + r[:, None, None] * g_hat
                    yi_vstar_norm = ((new_points - v_star) ** 2).sum((1,2)).sqrt()
                    yi_vstar_norm[yi_vstar_norm == 0] = 1e-9

                    tmp = (new_points - v_star) / yi_vstar_norm[:, None, None]
                    tmp = tmp * torch.sqrt(torch.max(torch.zeros_like(r), epsilon ** 2 - r ** 2))[:, None, None]
                    z = v_star + tmp
                    new_points = z.detach()
                    if torch.isnan(new_points).sum().item() != 0:
                        assert False, "Out NAN Occured!!!"
                    
                else:
                    epsilon = delta_norm / gamma
                    tmp = (r + torch.sqrt(epsilon ** 2 - delta_norm ** 2 + r ** 2))
                    z = new_points - tmp[:, None, None] * g_hat
                    
                    new_points = z.detach()
                    if torch.isnan(new_points).sum().item() != 0:
                        assert False, "In NAN Occured!!!"

                points = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
                normal_vec = get_normal_vector(points)


        with torch.no_grad():
            adv_points = output_points.data
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(adv_points.transpose(1, 2)))
            else:
                adv_logits = self.classifier(adv_points.transpose(1, 2))
            adv_target = adv_logits.argmax(-1)

        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1

        del normal_vec, grad, new_points, spin_axis_matrix, translation_matrix
        return adv_points, adv_target, (adv_logits.data.max(1)[1] != target).sum().item()
    
    def shape_invariant_query_attack(self, points, target):
        """Blaxk-box query-based attack based on point-cloud sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        normal_vec = points[:,:,-3:].data # N, [1, N, 3]
        normal_vec = normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True)) # N, [1, N, 3]
        points = points[:,:,:3].data # P, [1, N, 3]
        ori_points = points.data
        # initialization
        query_costs = 0
        with torch.no_grad():
            points = points.transpose(1, 2)
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.detach()))
            else:
                adv_logits = self.classifier(points)
            adv_target = adv_logits.max(1)[1]
            query_costs += 1
        # if categorized wrong
        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        if adv_target != target:
            return points.transpose(1, 2), adv_target, query_costs

        # P -> P', detach()
        points = points.transpose(1, 2)
        new_points, spin_axis_matrix, translation_matrix = get_transformed_point_cloud(points.detach(), normal_vec)
        new_points = new_points.detach()
        new_points.requires_grad = True

        # P' -> P
        inputs = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
        inputs = torch.min(torch.max(inputs, ori_points - self.eps), ori_points + self.eps)
        inputs = inputs.transpose(1, 2) # P, [1, 3, N]

        # get white-box gradients
        logits = self.wb_classifier(inputs)
        loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
        self.wb_classifier.zero_grad()
        loss.backward()

        grad = new_points.grad.data # g, [1, N, 3]
        grad[:,:,2] = 0.
        new_points.requires_grad = False
        rankings = torch.sqrt(grad[:,:,0] ** 2 + grad[:,:,1] ** 2) # \sqrt{g_{x'}^2+g_{y'}^2}, [1, N]
        directions = grad / (rankings.unsqueeze(-1)+1e-16) # (g_{x'}/r,g_{y'}/r,0), [1, N, 3]

        # rank the sensitivity map in the desending order
        point_list = []
        for i in range(points.size(1)):
            point_list.append((i, directions[:,i,:], rankings[:,i].item()))
        sorted_point_list = sorted(point_list, key=lambda c: c[2], reverse=True)

        # query loop
        i = 0
        best_loss = -999.
        while best_loss < 0 and i < len(sorted_point_list):
            # print(i, len(sorted_point_list))
            idx, direction, _ = sorted_point_list[i]
            for eps in {self.step_size, -self.step_size}:
                pert = torch.zeros_like(new_points).cuda()
                pert[:,idx,:] += eps * direction
                inputs = new_points + pert
                inputs = torch.matmul(spin_axis_matrix.transpose(-1, -2), inputs.unsqueeze(-1)) # U^T P', [1, N, 3, 1]
                inputs = inputs - translation_matrix.unsqueeze(-1) # P = U^T P' - (P \cdot N) N, [1, N, 3, 1]
                inputs = inputs.squeeze(-1).transpose(1, 2) # P, [1, 3, N]
                # inputs = torch.clamp(inputs, -1, 1)
                with torch.no_grad():
                    if not self.defense_method is None:
                        logits = self.classifier(self.pre_head(inputs.detach()))
                    else:
                        logits = self.classifier(inputs.detach()) # [1, num_class]
                    query_costs += 1
                    
                loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
                if loss.item() > best_loss:
                    # print(loss.item())
                    best_loss = loss.item()
                    new_points = new_points + pert
                    adv_target = logits.max(1)[1]
                    break
            i += 1

        adv_points = inputs.transpose(1, 2).data
        if self.top5_attack:
            target_top_5 = logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        del grad

        return adv_points, adv_target, query_costs
    
    def shape_invariant_ifgm_bp_query_attack(self, points, target):
        """Blaxk-box query-based attack based on point-cloud sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        normal_vec = points[:,:,-3:].data # N, [1, N, 3]
        normal_vec = normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True)) # N, [1, N, 3]
        points = points[:,:,:3].data # P, [1, N, 3]
        ori_points = points.data
        result_points = points.clone()
        # initialization
        query_costs = 0
        bp_optims = []
        if self.l2_weight != 0.0:
            bp_optims.append("l2")

        if self.hd_weight != 0.0:
            bp_optims.append("hd")

        if self.curv_weight != 0.0:
            bp_optims.append("curv")
    
        if self.cd_weight != 0.0:
            bp_optims.append("cd")
        bp = boundary_projection_query_si(self.args, bp_optims)
        
        with torch.no_grad():
            points = points.transpose(1, 2)
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.detach()))
            else:
                adv_logits = self.classifier(points)
            adv_target = adv_logits.max(1)[1]
            query_costs += 1
        # if categorized wrong
        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        if adv_target != target:
            return points.transpose(1, 2), adv_target, query_costs

        # P -> P', detach()
        points = points.transpose(1, 2)
        new_points, spin_axis_matrix, translation_matrix = get_transformed_point_cloud(points.detach(), normal_vec)
        new_points = new_points.detach()
        new_points.requires_grad = True

        # P' -> P
        inputs = get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
        inputs = torch.min(torch.max(inputs, ori_points - self.eps), ori_points + self.eps)
        inputs = inputs.transpose(1, 2) # P, [1, 3, N]

        # get white-box gradients
        logits = self.wb_classifier(inputs)
        loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
        self.wb_classifier.zero_grad()
        loss.backward()

        grad = new_points.grad.data # g, [1, N, 3]
        grad[:,:,2] = 0.
        
        rankings = torch.sqrt(grad[:,:,0] ** 2 + grad[:,:,1] ** 2) # \sqrt{g_{x'}^2+g_{y'}^2}, [1, N]
        directions = grad / (rankings.unsqueeze(-1)+1e-16) # (g_{x'}/r,g_{y'}/r,0), [1, N, 3]
        alpha_hat = bp.gradient_map_project(new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec, directions)

        new_points.requires_grad = False
        # rank the sensitivity map in the desending order
        point_list = []
        for i in range(points.size(1)):
            point_list.append((i, directions[:,i,:], rankings[:,i].item()))
        sorted_point_list = sorted(point_list, key=lambda c: c[2], reverse=True)

        # query loop
        i = 0
        best_loss = -999.
        while best_loss < 0 and i < len(sorted_point_list):
            idx, direction, _ = sorted_point_list[i]
            for eps in {self.step_size, -self.step_size}:
                pert = torch.zeros_like(new_points).cuda()
                pert[:,idx,:] += eps * direction
                inputs = new_points + pert
                inputs = torch.matmul(spin_axis_matrix.transpose(-1, -2), inputs.unsqueeze(-1)) # U^T P', [1, N, 3, 1]
                inputs = inputs - translation_matrix.unsqueeze(-1) # P = U^T P' - (P \cdot N) N, [1, N, 3, 1]
                inputs = inputs.squeeze(-1).transpose(1, 2) # P, [1, 3, N]
                # inputs = torch.clamp(inputs, -1, 1)
                with torch.no_grad():
                    if not self.defense_method is None:
                        logits = self.classifier(self.pre_head(inputs.detach()))
                    else:
                        logits = self.classifier(inputs.detach()) # [1, num_class]
                    query_costs += 1
                loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
                if loss.item() > best_loss:
                    alpha_hat = bp.gradient_map_project(new_points, spin_axis_matrix, translation_matrix, ori_points, normal_vec, directions)
                    pert[:,idx,:] -= 0.16 * alpha_hat[:,idx,:]
                    best_loss = loss.item()
                    new_points = new_points + pert
                    adv_target = logits.max(1)[1]
                    result_points = inputs.detach().clone()
                    break
                
            i += 1
        
        adv_points = inputs.transpose(1, 2).data
        if self.top5_attack:
            target_top_5 = logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        del grad

        return adv_points, adv_target, query_costs
    

    def simba_attack(self, points, target):
        """Blaxk-box query-based SimBA attack.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        points = points[:,:,:3].data # P, [1, N, 3]
        # initialization
        query_costs = 0
        with torch.no_grad():
            points = points.transpose(1, 2)
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.detach()))
            else:
                adv_logits = self.classifier(points)
            adv_target = adv_logits.max(1)[1]
            query_costs += 1
        # if categorized wrong
        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        if adv_target != target:
            return points.transpose(1, 2), adv_target, query_costs

        # constructing random list
        basis_list = []
        for j in range(points.shape[2]):
            for i in range(3):
                basis_list.append((i, j))
        basis_list = np.array(basis_list)
        np.random.shuffle(basis_list)

        # query loop
        i = 0
        best_loss = -999.
        while best_loss < 0 and i < len(basis_list):
            channel, idx = basis_list[i]
            for eps in {self.step_size, -self.step_size}:
                pert = torch.zeros_like(points).cuda() # \delta, [1, 3, N]
                pert[:,channel,idx] += eps
                inputs = points + pert
                with torch.no_grad():
                    if not self.defense_method is None:
                        logits = self.classifier(self.pre_head(inputs.detach()))
                    else:
                        logits = self.classifier(inputs.detach()) # [1, num_class]
                    query_costs += 1
                loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
                if loss.item() > best_loss:
                    # print(loss.item())
                    best_loss = loss.item()
                    points = points + pert
                    adv_target = logits.max(1)[1]
                    break
            i += 1


        adv_points = points.transpose(1, 2).data
        if self.top5_attack:
            target_top_5 = logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        # del grad

        return adv_points, adv_target, query_costs


    def simbapp_attack(self, points, target):
        """Blaxk-box query-based SimBA++ attack.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        points = points[:,:,:3].data # P, [1, N, 3]
        # initialization
        query_costs = 0
        with torch.no_grad():
            points = points.transpose(1, 2)
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.detach()))
            else:
                adv_logits = self.classifier(points)
            adv_target = adv_logits.max(1)[1]
            query_costs += 1
        # if categorized wrong
        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        if adv_target != target:
            return points.transpose(1, 2), adv_target, query_costs

        # get white-box gradients
        points = points.detach()
        points.requires_grad = True
        logits = self.wb_classifier(points)
        loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
        self.wb_classifier.zero_grad()
        loss.backward()
        grad = points.grad.data # g, [1, 3, N]
        grad = abs(grad).reshape(-1)

        # # rank 
        # basis_list = []
        # for j in range(points.shape[2]):
        #     for i in range(3):
        #         basis_list.append((i, j, grad[0][i][j]))
        # sorted_basis_list = sorted(basis_list, key=lambda c: c[2], reverse=True)

        # query loop
        i = 0
        best_loss = -999.
        while best_loss < 0 and i < grad.shape[0]:
            # channel, idx, _ = sorted_basis_list[i]
            m = Categorical(grad)
            choice = m.sample()
            channel = int(choice % 3)
            idx = int(choice // 3)
            for eps in {self.step_size, -self.step_size}:
                pert = torch.zeros_like(points).cuda() # \delta, [1, 3, N]
                pert[:,channel,idx] += (eps + 0.1*torch.randn(1).cuda())
                inputs = points + pert
                with torch.no_grad():
                    if not self.defense_method is None:
                        logits = self.classifier(self.pre_head(inputs.detach()))
                    else:
                        logits = self.classifier(inputs.detach()) # [1, num_class]
                    query_costs += 1
                loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
                if loss.item() > best_loss:
                    # print(loss.item())
                    best_loss = loss.item()
                    points = points + pert
                    adv_target = logits.max(1)[1]
                    break
            i += 1


        adv_points = points.transpose(1, 2).data
        if self.top5_attack:
            target_top_5 = logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        del grad, m

        return adv_points, adv_target, query_costs

