from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import sys
import time

import ipdb
import numpy as np
import open3d as o3d
from pytorch3d.ops import knn_points, knn_gather
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pytorch3d


# import torch_dct
from utils.utility import estimate_perpendicular, _compare, farthest_points_sample, pad_larger_tensor_with_index_batch, dct, idct
from utils.loss_utils import norm_l2_loss, chamfer_loss, pseudo_chamfer_loss, hausdorff_loss, curvature_loss, uniform_loss, _get_kappa_ori, _get_kappa_adv, spectral_loss

import modules.functional as F
from modules.voxelization import Voxelization

@torch.no_grad()
def eig_vector(data, K):
    b, n, _ = data.shape
    _, idx, _ = pytorch3d.ops.knn_points(data, data, K=K)  # idx (b,n,K)

    idx0 = torch.arange(0,b,device=data.device).reshape((b,1)).expand(-1,n*K).reshape((1,b*n*K))
    idx1 = torch.arange(0,n,device=data.device).reshape((1,n,1)).expand(b,n,K).reshape((1,b*n*K))
    idx = idx.reshape((1,b*n*K))
    idx = torch.cat([idx0, idx1, idx], dim=0) # (3, b*n*K)
    # print(b, n, K, idx.shape)
    ones = torch.ones(idx.shape[1], dtype=bool, device=data.device)
    A = torch.sparse_coo_tensor(idx, ones, (b, n, n))
    A = A.to(torch.uint8).to_dense().to(torch.bool) # (b,n,n)
    A = A | A.transpose(1, 2)
    A = A.float()
    deg = torch.diag_embed(torch.sum(A, dim=2))
    laplacian = deg - A
    u = torch.zeros((laplacian.size(0), laplacian.size(1)), device=data.device)
    v = torch.zeros((laplacian.size(0), laplacian.size(1), laplacian.size(1)), device=data.device)
    for i in range(0, laplacian.size(0)):
        u_, v_ = torch.eig(laplacian[i], eigenvectors = True) # (b,n,n)
        u[i] = u_[:, 0]
        v[i] = v_
    return v, laplacian, u

def GFT(pc_ori, K, factor):
    x = pc_ori.transpose(0,1) #(b,n,3)
    b, n, _ = x.shape
    v = eig_vector(x, K)
    x_ = torch.einsum('bij,bjk->bik',v.transpose(1,2), x) # (b,n,3)
    x_ = torch.einsum('bij,bi->bij', x_, factor)
    x = torch.einsum('bij,bjk->bik',v, x_)
    return x

def resample_reconstruct_from_pc(cfg, output_file_name, pc, normal=None, reconstruct_type='PRS'):
    assert pc.size() == 2
    assert pc.size(2) == 3
    assert normal.size() == pc.size()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if normal:
        pcd.normals = o3d.utility.Vector3dVector(normal)

    if reconstruct_type == 'BPA':
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist

        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

        output_mesh = bpa_mesh.simplify_quadric_decimation(100000)
        output_mesh.remove_degenerate_triangles()
        output_mesh.remove_duplicated_triangles()
        output_mesh.remove_duplicated_vertices()
        output_mesh.remove_non_manifold_edges()
    elif reconstruct_type == 'PRS':
        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
        bbox = pcd.get_axis_aligned_bounding_box()
        output_mesh = poisson_mesh.crop(bbox)

    o3d.io.write_triangle_mesh(os.path.join(cfg.output_path, output_file_name+"ply"), output_mesh)

    return o3d.geometry.TriangleMesh.sample_points_uniformly(output_mesh, number_of_points=cfg.npoint)

def offset_proj(offset, ori_pc, ori_normal, project='dir'):
    # offset: shape [b, 3, n], perturbation offset of each point
    # normal: shape [b, 3, n], normal vector of the object

    condition_inner = torch.zeros(offset.shape).cuda().byte()

    intra_KNN = knn_points(offset.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    normal = knn_gather(ori_normal.permute(0,2,1), intra_KNN.idx).permute(0,3,1,2).squeeze(3).contiguous() # [b, 3, n]

    normal_len = (normal**2).sum(1, keepdim=True).sqrt()
    normal_len_expand = normal_len.expand_as(offset) #[b, 3, n]

    # add 1e-6 to avoid dividing by zero
    offset_projected = (offset * normal / (normal_len_expand + 1e-6)).sum(1,keepdim=True) * normal / (normal_len_expand + 1e-6)

    # let perturb be the projected ones
    offset = torch.where(condition_inner, offset, offset_projected)

    return offset

def find_offset(ori_pc, adv_pc):
    intra_KNN = knn_points(adv_pc.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    knn_pc = knn_gather(ori_pc.permute(0,2,1), intra_KNN.idx).permute(0,3,1,2).squeeze(3).contiguous() # [b, 3, n]

    real_offset =  adv_pc - knn_pc

    return real_offset


def lp_clip(offset, cc_linf):
    lengths = (offset**2).sum(1, keepdim=True).sqrt() #[b, 1, n]
    lengths_expand = lengths.expand_as(offset) # [b, 3, n]

    condition = lengths > 1e-6
    offset_scaled = torch.where(condition, offset / lengths_expand * cc_linf, torch.zeros_like(offset))

    condition = lengths < cc_linf
    offset = torch.where(condition, offset, offset_scaled)

    return offset

def _forward_step(net, defense_head, pc_ori, input_curr_iter, normal_ori, ori_kappa, target, scale_const, cfg, targeted, v):
    #needed cfg:[arch, classes, cls_loss_type, confidence, dis_loss_type, is_cd_single_side, dis_loss_weight, hd_loss_weight, curv_loss_weight, curv_loss_knn]
    b,_,n=input_curr_iter.size()
    if not defense_head is None:
        output_curr_iter = net(defense_head(input_curr_iter))
    else:
        output_curr_iter = net(input_curr_iter)
        

    if cfg.cls_loss_type == 'Margin':
        target_onehot = torch.zeros(target.size() + (cfg.classes,)).cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)

        fake = (target_onehot * output_curr_iter).sum(1)
        other = ((1. - target_onehot) * output_curr_iter - target_onehot * 10000.).max(1)[0]

        if targeted:
            # if targeted, optimize for making the other class most likely
            cls_loss = torch.clamp(other - fake + cfg.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            cls_loss = torch.clamp(fake - other + cfg.confidence, min=0.)  # equiv to max(..., 0.)

    elif cfg.cls_loss_type == 'CE':
        if targeted:
            cls_loss = nn.CrossEntropyLoss(reduction='none').cuda()(output_curr_iter, Variable(target, requires_grad=False))
        else:
            cls_loss = - nn.CrossEntropyLoss(reduction='none').cuda()(output_curr_iter, Variable(target, requires_grad=False))
    elif cfg.cls_loss_type == 'None':
        cls_loss = torch.FloatTensor(b).zero_().cuda()
    else:
        assert False, 'Not support such clssification loss'

    info = 'cls_loss: {0:6.4f}\t'.format(cls_loss.mean().item())

    if cfg.dis_loss_type == 'CD':
        if cfg.is_cd_single_side:
            dis_loss = pseudo_chamfer_loss(input_curr_iter, pc_ori)
        else:
            dis_loss = chamfer_loss(input_curr_iter, pc_ori)

        constrain_loss = cfg.dis_loss_weight * dis_loss
        info = info + 'cd_loss: {0:6.4f}\t'.format(dis_loss.mean().item())
    elif cfg.dis_loss_type == 'L2':
        assert cfg.hd_loss_weight ==0
        dis_loss = norm_l2_loss(input_curr_iter, pc_ori)
        constrain_loss = cfg.dis_loss_weight * dis_loss
        info = info + 'l2_loss: {0:6.4f}\t'.format(dis_loss.mean().item())
    elif cfg.dis_loss_type == 'None':
        dis_loss = 0
        constrain_loss = 0
    elif cfg.dis_loss_type == 'Spectral':
        dis_loss = spectral_loss(input_curr_iter, pc_ori, v)
        constrain_loss = cfg.dis_loss_weight * dis_loss
        info = info + 'spectral_loss: {0:6.4f}\t'.format(dis_loss.mean().item())
    else:
        assert False, 'Not support such distance loss'

    # hd_loss
    if cfg.hd_loss_weight !=0:
        hd_loss = hausdorff_loss(input_curr_iter, pc_ori)
        constrain_loss = constrain_loss + cfg.hd_loss_weight * hd_loss
        info = info+'hd_loss : {0:6.4f}\t'.format(hd_loss.mean().item())
    else:
        hd_loss = 0

    # nor loss
    if cfg.curv_loss_weight !=0:
        adv_kappa, normal_curr_iter = _get_kappa_adv(input_curr_iter, pc_ori, normal_ori, cfg.curv_loss_knn)
        curv_loss = curvature_loss(input_curr_iter, pc_ori, adv_kappa, ori_kappa)
        constrain_loss = constrain_loss + cfg.curv_loss_weight * curv_loss
        info = info+'curv_loss : {0:6.4f}\t'.format(curv_loss.mean().item())
    else:
        normal_curr_iter = torch.zeros(b, 3, n).cuda()
        curv_loss = 0

    # uniform loss
    if cfg.uniform_loss_weight !=0:
        uniform = uniform_loss(input_curr_iter)
        constrain_loss = constrain_loss + cfg.uniform_loss_weight * uniform
        info = info+'uniform : {0:6.4f}\t'.format(uniform.mean().item())
    else:
        uniform = 0



    scale_const = scale_const.float().cuda()
    loss_n = cls_loss + scale_const * constrain_loss
    loss = loss_n.mean()

    return output_curr_iter, normal_curr_iter, loss, loss_n, cls_loss, dis_loss, hd_loss, curv_loss, constrain_loss, info

def _forward_step_bp(net, defense_head, pc_ori, input_curr_iter, normal_ori, ori_kappa, target, scale_const, cfg, targeted, v):
    #needed cfg:[arch, classes, cls_loss_type, confidence, dis_loss_type, is_cd_single_side, dis_loss_weight, hd_loss_weight, curv_loss_weight, curv_loss_knn]
    b,_,n=input_curr_iter.size()
    if not defense_head is None:
        output_curr_iter = net(defense_head(input_curr_iter))
    else:
        output_curr_iter = net(input_curr_iter)

    if cfg.cls_loss_type == 'Margin':
        target_onehot = torch.zeros(target.size() + (cfg.classes,)).cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)

        fake = (target_onehot * output_curr_iter).sum(1)
        other = ((1. - target_onehot) * output_curr_iter - target_onehot * 10000.).max(1)[0]

        if targeted:
            # if targeted, optimize for making the other class most likely
            cls_loss = torch.clamp(other - fake + cfg.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            cls_loss = torch.clamp(fake - other + cfg.confidence, min=0.)  # equiv to max(..., 0.)

    elif cfg.cls_loss_type == 'CE':
        if targeted:
            cls_loss = nn.CrossEntropyLoss(reduction='none').cuda()(output_curr_iter, Variable(target, requires_grad=False))
        else:
            cls_loss = - nn.CrossEntropyLoss(reduction='none').cuda()(output_curr_iter, Variable(target, requires_grad=False))
    elif cfg.cls_loss_type == 'None':
        cls_loss = torch.FloatTensor(b).zero_().cuda()
    else:
        assert False, 'Not support such clssification loss'

    info = 'cls_loss: {0:6.4f}\t'.format(cls_loss.mean().item())

    loss_n = cls_loss
    loss = loss_n.mean()

    return output_curr_iter, None, loss, loss_n, cls_loss, None, None, None, None, info


def attack(net, defense_head, input_data, cfg):
    #needed cfg:[arch, classes, attack_label, initial_const, lr, optim, binary_max_steps, iter_max_steps, metric,
    #  cls_loss_type, confidence, dis_loss_type, is_cd_single_side, dis_loss_weight, hd_loss_weight, curv_loss_weight, curv_loss_knn,
    #  is_pre_jitter_input, calculate_project_jitter_noise_iter, jitter_k, jitter_sigma, jitter_clip,
    #  is_save_normal,
    #  ]

    if cfg.attack_label == 'Untarget':
        targeted = False
    else:
        targeted = True


    pc = input_data[0]
    normal = input_data[1]
    gt_labels = input_data[2]

    if pc.size(3) == 3:
        pc = pc.permute(0,1,3,2)
    if normal.size(3) == 3:
        normal = normal.permute(0,1,3,2)

    bs, l, _, n = pc.size()
    b = bs*l

    pc_ori = pc.view(b, 3, n).cuda()
    normal_ori = normal.view(b, 3, n).cuda()
    gt_target = gt_labels.view(-1)

    if cfg.attack_label == 'Untarget':
        target = gt_target.cuda()
    else:
        target = input_data[3].view(-1).cuda()

    if cfg.curv_loss_weight !=0:
        kappa_ori = _get_kappa_ori(pc_ori, normal_ori, cfg.curv_loss_knn)
    else:
        kappa_ori = None

    lower_bound = torch.ones(b) * 0
    scale_const = torch.ones(b) * cfg.initial_const
    upper_bound = torch.ones(b) * 1e10

    best_loss = [1e10] * b
    best_attack = torch.ones(b, 3, n).cuda()
    best_x_ = torch.zeros(b,n,3).cuda()
    best_gft = torch.zeros(b,n,3).cuda()
    best_factor_ = torch.zeros(b,n,3).cuda()
    best_attack_step = [-1] * b
    best_attack_BS_idx = [-1] * b
    all_loss_list = [[-1] * b] * cfg.iter_max_steps
    voxelization = Voxelization(128, normalize=True, eps=0)
    for search_step in range(cfg.binary_max_steps):
        iter_best_loss = [1e10] * b
        iter_best_score = [-1] * b
        constrain_loss = torch.ones(b) * 1e10
        attack_success = torch.zeros(b).cuda()


        input_all = None

        for step in range(cfg.iter_max_steps):
            if step == 0:
                    offset = torch.zeros(b, 3, n).cuda()
                    nn.init.normal_(offset, mean=0, std=1e-3)
                    offset.requires_grad_()
                    factor = torch.zeros(b,n,3).cuda()
                    response = torch.zeros(b,5,3).cuda()
                    response1 = torch.zeros(b,n,3).cuda()
                    response[:,0,:] = 1
                    mask = torch.ones(b,n,3).cuda()
                    mask[:,:cfg.band_frequency[0],:] = 0
                    mask[:, cfg.band_frequency[1]:, :] = 0
                    if cfg.spectral_attack:
                        nn.init.normal_(factor, mean=0, std=1e-3)
                        nn.init.normal_(response, mean=0, std=1e-3)
                        factor.requires_grad_()
                        response.requires_grad_()
                        response1.requires_grad_()
                        if cfg.optim == 'adam':
                            optimizer = optim.Adam([factor, response, response1], lr=cfg.lr)
                        elif cfg.optim == 'sgd':
                            optimizer = optim.SGD([factor, response, response1], lr=cfg.lr)
                        else:
                            assert False, 'Not support such optimizer.'
                        x = pc_ori.clone().transpose(1, 2) #(b,n,3)

                        K = cfg.KNN

                        v, laplacian, u = eig_vector(x, K)

                        u = u.unsqueeze(-1)
                        u_ = torch.cat((torch.ones_like(u).to(u.device), u, u*u, u*u*u, u*u*u*u), dim=-1) # (b, n, 5)
                        x_ = torch.einsum('bij,bjk->bik',v.transpose (1,2), x) # (b,n,3)
                    else:
                        if cfg.optim == 'adam':
                            optimizer = optim.Adam([offset], lr=cfg.lr)
                        elif cfg.optim == 'sgd':
                            optimizer = optim.SGD([offset], lr=cfg.lr)
                        else:
                            assert False, 'Not support such optimizer.'

                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9990, last_epoch=-1)

                    periodical_pc = pc_ori.clone()

            if cfg.spectral_attack:
                gft = x_.clone()
                if cfg.spectral_offset:
                    if cfg.spectral_restrict != 0:
                        factor_relative = torch.clamp(factor/x_, min = -cfg.spectral_restrict, max=cfg.spectral_restrict)
                        factor_ = x_.mul(factor_relative)
                    else:
                        factor_ = factor
                    gft += factor_ * mask
                GFT_pc = torch.einsum('bij,bjk->bik', v, gft).transpose(1,2)
                input_all = GFT_pc
            else:
                input_all = periodical_pc + offset

            if (input_all.size(2) > cfg.npoint) and (not cfg.is_partial_var) and cfg.is_subsample_opt:
                input_curr_iter = farthest_points_sample(input_all, cfg.npoint)
            else:
                input_curr_iter = input_all


            with torch.no_grad():
                for k in range(b):
                    if input_curr_iter.size(2) < input_all.size(2):
                        batch_k_pc = farthest_points_sample(torch.cat([input_all[k].unsqueeze(0)]*cfg.eval_num), cfg.npoint)
                        if not defense_head is None:
                            batch_k_adv_output = net(defense_head(batch_k_pc))
                        else:
                            batch_k_adv_output = net(batch_k_pc)
                        attack_success[k] = _compare(torch.max(batch_k_adv_output,1)[1].data, target[k], gt_target[k], targeted).sum() > 0.5 * cfg.eval_num
                        output_label = torch.max(batch_k_adv_output,1)[1].mode().values.item()
                    else:
                        if not defense_head is None:
                            adv_output = net(defense_head(input_curr_iter[k].unsqueeze(0)))
                        else:
                            adv_output = net(input_curr_iter[k].unsqueeze(0))
                        output_label = torch.argmax(adv_output).item()
                        attack_success[k] = _compare(output_label, target[k], gt_target[k].cuda(), targeted).item()

                    metric = constrain_loss[k].item()

                    if attack_success[k] and (metric <best_loss[k]):

                        best_loss[k] = metric
                        best_attack[k] = input_all.data[k].clone()
                        best_attack_BS_idx[k] = search_step
                        best_attack_step[k] = step
                        if cfg.spectral_attack:
                            best_gft[k] = gft[k].clone()
                            best_x_[k] = x_[k].clone()

                    if attack_success[k] and (metric <iter_best_loss[k]):
                        iter_best_loss[k] = metric
                        iter_best_score[k] = output_label

            _, normal_curr_iter, loss, loss_n, cls_loss, dis_loss, hd_loss, nor_loss, constrain_loss, info = _forward_step(net, defense_head, pc_ori, input_curr_iter, normal_ori, kappa_ori, target, scale_const, cfg, targeted, v)

            all_loss_list[step] = loss_n.detach().tolist()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            if cfg.is_use_lr_scheduler:
                lr_scheduler.step()


            if cfg.cc_linf != 0:
                with torch.no_grad():
                    proj_offset = lp_clip(offset, cfg.cc_linf)
                    offset.data = proj_offset.data

        # adjust the scale constants
        for k in range(b):
            if _compare(output_label, target[k], gt_target[k].cuda(), targeted).item() and iter_best_score[k] != -1:
                lower_bound[k] = max(lower_bound[k], scale_const[k])
                if upper_bound[k] < 1e9:
                    scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5
                else:
                    scale_const[k] *= 2
            else:
                upper_bound[k] = min(upper_bound[k], scale_const[k])
                if upper_bound[k] < 1e9:
                    scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5
                
    return best_attack, target, (np.array(best_loss)<1e10), best_attack_step, all_loss_list  #best_attack:[b, 3, n], target: [b], best_loss:[b], best_attack_step:[b], all_loss_list:[iter_max_steps, b]

def attack_bp(net, defense_head, input_data, cfg, attacker):
    #needed cfg:[arch, classes, attack_label, initial_const, lr, optim, binary_max_steps, iter_max_steps, metric,
    #  cls_loss_type, confidence, dis_loss_type, is_cd_single_side, dis_loss_weight, hd_loss_weight, curv_loss_weight, curv_loss_knn,
    #  is_pre_jitter_input, calculate_project_jitter_noise_iter, jitter_k, jitter_sigma, jitter_clip,
    #  is_save_normal,
    #  ]

    if cfg.attack_label == 'Untarget':
        targeted = False
    else:
        targeted = True


    pc = input_data[0]
    normal = input_data[1]
    gt_labels = input_data[2]

    if pc.size(3) == 3:
        pc = pc.permute(0,1,3,2)
    if normal.size(3) == 3:
        normal = normal.permute(0,1,3,2)

    bs, l, _, n = pc.size()
    b = bs*l

    pc_ori = pc.view(b, 3, n).cuda()
    normal_ori = normal.view(b, 3, n).cuda()
    gt_target = gt_labels.view(-1)

    if cfg.attack_label == 'Untarget':
        target = gt_target.cuda()
    else:
        target = input_data[3].view(-1).cuda()

    if cfg.curv_loss_weight !=0:
        kappa_ori = _get_kappa_ori(pc_ori, normal_ori, cfg.curv_loss_knn)
    else:
        kappa_ori = None

    scale_const = torch.ones(b) * cfg.initial_const


    best_loss = [1e10] * b
    best_attack_step = [-1] * b

    stage2 = False
    initial_lr = 0.10
    gamma = 0.01
    
    gamma_max = 1.0
    gamma_min = 0.05

    output_points = pc_ori.permute(0,2,1).detach()
    output_reg = torch.ones((pc_ori.size(0), 4)).cuda() * 1e10


    for step in range(cfg.iter_max_steps):
        if step == 0:
            offset = torch.zeros(b, 3, n).cuda()
            nn.init.normal_(offset, mean=0, std=1e-3)
            offset.requires_grad_()
            factor = torch.zeros(b,n,3).cuda()
            response = torch.zeros(b,5,3).cuda()
            response1 = torch.zeros(b,n,3).cuda()
            response[:,0,:] = 1
            mask = torch.ones(b,n,3).cuda()
            mask[:,:cfg.band_frequency[0],:] = 0
            mask[:, cfg.band_frequency[1]:, :] = 0
            if cfg.spectral_attack:
                nn.init.normal_(factor, mean=0, std=1e-3)
                nn.init.normal_(response, mean=0, std=1e-3)
                factor.requires_grad_()
                response.requires_grad_()
                response1.requires_grad_()
                if cfg.optim == 'adam':
                    optimizer = optim.Adam([factor, response, response1], lr=cfg.lr)
                elif cfg.optim == 'sgd':
                    optimizer = optim.SGD([factor, response, response1], lr=cfg.lr)
                else:
                    assert False, 'Not support such optimizer.'
                x = pc_ori.clone().transpose(1, 2) #(b,n,3)

                K = cfg.KNN

                v, laplacian, u = eig_vector(x, K)

                u = u.unsqueeze(-1)
                u_ = torch.cat((torch.ones_like(u).to(u.device), u, u*u, u*u*u, u*u*u*u), dim=-1) # (b, n, 5)
                x_ = torch.einsum('bij,bjk->bik',v.transpose (1,2), x) # (b,n,3)
            else:
                if cfg.optim == 'adam':
                    optimizer = optim.Adam([offset], lr=cfg.lr)
                elif cfg.optim == 'sgd':
                    optimizer = optim.SGD([offset], lr=cfg.lr)
                else:
                    assert False, 'Not support such optimizer.'

            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9990, last_epoch=-1)


            periodical_pc = pc_ori.clone()

        if cfg.spectral_attack:
            gft = x_.clone()
            if cfg.spectral_offset:
                if cfg.spectral_restrict != 0:
                    factor_relative = torch.clamp(factor/x_, min = -cfg.spectral_restrict, max=cfg.spectral_restrict)
                    factor_ = x_.mul(factor_relative)
                else:
                    factor_ = factor
                gft += factor_ * mask
            GFT_pc = torch.einsum('bij,bjk->bik', v, gft).transpose(1,2)
            input_all = GFT_pc
        else:
            input_all = periodical_pc + offset

        if (input_all.size(2) > cfg.npoint) and (not cfg.is_partial_var) and cfg.is_subsample_opt:
            input_curr_iter = farthest_points_sample(input_all, cfg.npoint)
        else:
            input_curr_iter = input_all

        if not stage2:
            _, normal_curr_iter, loss, loss_n, cls_loss, dis_loss, hd_loss, nor_loss, constrain_loss, info = _forward_step_bp(net, defense_head, pc_ori, input_curr_iter, normal_ori, kappa_ori, target, scale_const, cfg, targeted, v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if not defense_head is None:
                logits = net(defense_head(input_curr_iter))
            else:
                logits = net(input_curr_iter)

            if logits.argmax(-1).item() != target.item():
                stage2 = True
            if cfg.is_use_lr_scheduler:
                lr_scheduler.step()


            reg_loss = torch.zeros_like(output_reg).cuda()

            if cfg.l2_weight != 0.0:
                reg_loss_l2 = attacker.get_loss(input_curr_iter.permute(0,2,1), pc_ori.permute(0,2,1), None, "l2")
                reg_loss[:, 0] = reg_loss_l2
            
            if cfg.hd_weight != 0.0:
                reg_loss_hd = attacker.get_loss(input_curr_iter.permute(0,2,1), pc_ori.permute(0,2,1), None, "hd")
                reg_loss[:, 1] = reg_loss_hd

            if cfg.cd_weight != 0.0:
                reg_loss_cd = attacker.get_loss(input_curr_iter.permute(0,2,1), pc_ori.permute(0,2,1), None, "cd")
                reg_loss[:, 2] = reg_loss_cd
            
            if cfg.curv_weight != 0.0:
                reg_loss_curv = attacker.get_loss(input_curr_iter.permute(0,2,1), pc_ori.permute(0,2,1), normal_ori.permute(0,2,1), "curv")
                reg_loss[:, 3] = reg_loss_curv

            if logits.argmax(1).item() != target.item() and (reg_loss <= output_reg).all():
                output_points = input_curr_iter.permute(0,2,1).detach()
                output_reg = reg_loss
            
        else:
            f, output_points, output_reg, initial_lr = attacker.gsda_boundary_projection_2(pc_ori.permute(0,2,1), x_, v, factor, mask, normal_ori.permute(0,2,1), target, output_points, output_reg, initial_lr, gamma, step)
            factor.data = f.data
            
        if cfg.cc_linf != 0:
            with torch.no_grad():
                proj_offset = lp_clip(offset, cfg.cc_linf)
                offset.data = proj_offset.data


    return output_points, target, (np.array(best_loss)<1e10), best_attack_step, output_reg  


