# -*- coding: utf-8 -*-

import os
import argparse
import time
import numpy as np
from tqdm import tqdm
import pickle as pkl

import torch
from utils.utils import set_seed

from attacks import PointCloudAttack
from utils.loss_utils import norm_l2_loss, chamfer_loss, pseudo_chamfer_loss, hausdorff_loss, curvature_loss, kNN_smoothing_loss, _get_kappa_ori, _get_kappa_adv

from torch.utils.tensorboard import SummaryWriter

def data_preprocess(data):
    """Preprocess the given data and label.
    """
    points, target = data

    points = points # [B, N, C]
    target = target[:, 0] # [B]

    points = points.cuda()
    target = target.cuda()

    return points, target


ten_label_indexes = {0:17, 2:9, 4:36, 5:20, 8:3, 22:16, 30:34, 33:38, 35:23, 37:15}

def main():

    num_class = 0
    if args.dataset == 'ModelNet40':
        num_class = 40
    elif args.dataset == 'ShapeNetPart':
        num_class = 16
    assert num_class != 0
    args.num_class = num_class

    # load model
    attack = PointCloudAttack(args)

    # start attack
    avg_time_cost = 0.

    result = []
    recall = []
    
    with open(args.data_path, "rb") as inputs:
        datas = pkl.load(inputs)

    total_count = len(datas)
    datas = datas[args.rank * args.rank_count:(args.rank + 1) * args.rank_count]


    hd_loss = []
    d_hd_loss = []
    cd_loss = []
    p_cd_loss = []
    cur_loss = []
    l2_loss = []
    smooth_loss = []
    asr = []

    avg_time_cost = []

    hd_loss = []
    result = []
    recall = []
    
    query_costs = []

    if args.time_verify or args.ss_exp:
        datas = datas[::10]
        
    for batch_id, data in tqdm(enumerate(datas), total=len(datas)):

        data = list(data)
        data[0] = torch.from_numpy(data[0][np.newaxis, :])
        data[1] = torch.from_numpy(data[1][np.newaxis, :])
        
        points, target = data_preprocess(data)
        target = target.long()

        if args.target_model == "PointNet" or args.target_model == "PointNetPP_ssg":
            for b in range(0, target.size(0)):
                target[b] = ten_label_indexes[target[b].item()]
        
        
        with torch.no_grad():
            recall.append(target.item() == attack.predict(points).item())
        
        # start attack
        t0 = time.time()
        adv_points, adv_target, query_cost = attack.run(points, target)
        if not args.query_attack_method is None :
            query_costs.append(query_cost)
        t1 = time.time()
        avg_time_cost.append(t1 - t0)
        
        adv_target = attack.predict(adv_points)

        asr.append((adv_target.item() != target.item()) and (not (adv_points == 1).all()))

        with torch.no_grad():
            pc_normal = points[:, :, -3:].data
            pc_ori = points[:, :, 0:3].data
            pc_adv = adv_points[:, :, :].data

            pc_normal = pc_normal.cuda().permute(0,2,1)
            pc_ori = pc_ori.cuda().permute(0,2,1)
            pc_adv = pc_adv.cuda().permute(0,2,1)

            if recall[-1] and asr[-1]:

                l2 = norm_l2_loss(pc_adv, pc_ori)
                l2_loss.append(l2.cpu().detach().numpy())

                hd_1 = hausdorff_loss(pc_adv, pc_ori)
                hd_loss.append(hd_1.cpu().detach().numpy())
                hd_2 = hausdorff_loss(pc_ori, pc_adv)
                d_hd_loss.append((hd_2.cpu().detach().numpy() + hd_1.cpu().detach().numpy()).mean())

                cd = chamfer_loss(pc_adv, pc_ori)
                cd_loss.append(cd.cpu().detach().numpy())

                p_cd = pseudo_chamfer_loss(pc_adv, pc_ori)
                p_cd_loss.append(p_cd.cpu().detach().numpy())

                ori_kappa = _get_kappa_ori(pc_ori, pc_normal, args.curv_loss_knn)
                adv_kappa, normal_curr_iter = _get_kappa_adv(pc_adv, pc_ori, pc_normal, args.curv_loss_knn)
                cur = curvature_loss(pc_adv, pc_ori, adv_kappa, ori_kappa).mean().item()
                cur_loss.append(cur)

                smooth = kNN_smoothing_loss(pc_adv, args.curv_loss_knn)
                smooth_loss.append(smooth.cpu().detach().numpy())


        points = points[:,:,:3].data # P, [1, N, 3]
        result.append((np.array(adv_points.cpu()), np.array(adv_target.cpu())))

    if not args.time_verify:

        data_path = os.path.join(os.path.dirname(args.data_path), args.task_name)

        suffix_adv_name = ".adv"

        suffix_acc_name = ".acc"

        if not os.path.exists(data_path + suffix_adv_name):
            temp_data = [None] * total_count
            temp_data[args.rank * args.rank_count:(args.rank + 1) * args.rank_count] = result
            
            with open(data_path + suffix_adv_name, "wb") as output:
                pkl.dump(temp_data, output)
        else :
            with open(data_path + suffix_adv_name, "rb") as input:
                temp_data = pkl.load(input)
            temp_data[args.rank * args.rank_count:(args.rank + 1) * args.rank_count] = result
            with open(data_path + suffix_adv_name, "wb") as output:
                pkl.dump(temp_data, output)

        if not os.path.exists(data_path + suffix_acc_name):
            temp_data = [None] * total_count
            temp_data[args.rank * args.rank_count:(args.rank + 1) * args.rank_count] = recall
            with open(data_path + suffix_acc_name, "wb") as output:
                pkl.dump(temp_data, output)
        else :
            with open(data_path + suffix_acc_name, "rb") as input:
                temp_data = pkl.load(input)
            temp_data[args.rank * args.rank_count:(args.rank + 1) * args.rank_count] = recall
            with open(data_path + suffix_acc_name, "wb") as output:
                pkl.dump(temp_data, output)

        recall = np.array(recall)

        log = "Result: \n"
        log += f"Recalled samples:{np.array(recall).sum()}\n"
        log += f"A.S.R:{np.array(asr)[recall].mean()}±{np.array(asr)[recall].std()}\n"
        log += f"Average L2:{np.array(l2_loss).mean()}±{np.array(l2_loss).std()}\n"
        log += f"Average HD(double):{np.array(d_hd_loss).mean()}±{np.array(d_hd_loss).std()}\n"
        log += f"Average HD:{np.array(hd_loss).mean()}±{np.array(hd_loss).std()}\n"
        log += f"Average Pseudo CD:{np.array(p_cd_loss).mean()}±{np.array(p_cd_loss).std()}\n"
        log += f"Average CD:{np.array(cd_loss).mean()}±{np.array(cd_loss).std()}\n"
        log += f"Average Curv:{np.array(cur_loss).mean()}±{np.array(cur_loss).std()}\n"
        log += f"Average Smooth Loss:{np.array(smooth_loss).mean()}±{np.array(smooth_loss).std()}\n"
        if not args.query_attack_method is None :
            log += f"Average Query Cost:{np.array(query_costs).mean()}±{np.array(query_costs).std()}\n"
        print(log)
        
    
    print(f'Average time cost: {np.array(avg_time_cost).mean()}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shape-invariant 3D Adversarial Point Clouds')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', 
                        help='input batch size for training (default: 1)')
    parser.add_argument('--input_point_nums', type=int, default=1024,
                        help='Point nums of each point cloud')
    parser.add_argument('--seed', type=int, default=2022, metavar='S',
                        help='random seed (default: 2022)')
    parser.add_argument('--dataset', type=str, default='ModelNet40',
                        choices=['ModelNet40'])
    parser.add_argument('--data_path', type=str, 
                        default='./data/modelNet40_batch1_1000batches_test.pkl.clean')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Worker nums of data loading.')
    parser.add_argument('--transfer_attack_method', type=str, default=None,
                        choices=['ifgm_si_adv', 'ifgm_bp',
                                 'ifgm_bp_ours', 'ifgm_si_bp', 
                                 'geoa3', 'gsda', 'gsda_bp'])
    parser.add_argument('--query_attack_method', type=str, default=None,
                        choices=['ifgm_si_adv_query', 'ifgm_bp_ours_query', 'simbapp', 'simba'])
    parser.add_argument('--surrogate_model', type=str, default='pointnet_cls',
                        choices=['pointnet_cls', "PointNetPP_ssg", 'DGCNN', 'curvenet', 'paconv','dgcnn', 'point_transformer'])
    parser.add_argument('--target_model', type=str, default='pointnet_cls',
                        choices=['pointnet_cls', "PointNetPP_ssg", 'DGCNN', 'curvenet', 'paconv','dgcnn', 'point_transformer'])
    parser.add_argument('--defense_method', type=str, default=None,
                        choices=['sor', 'srs', 'dupnet'])

    parser.add_argument('--bp_version', type=str, default='bp3',
                        choices=['bp1', 'bp1_si', 'bp2', 'bp2_si', 'bp3', 'bp3_var', 'bp3_no_GS', 'bp3_si_no_GS', 'bp3_deepfool', 'bp3_deepfool_var', 'bp3_si', 'bp4', 'bp4_si'])
    parser.add_argument('--top5_attack', action='store_true', default=False,
                        help='Whether to attack the top-5 prediction [default: False]')

    parser.add_argument('--initial_const', type=float, default=10, help='')
    parser.add_argument('--binary_max_steps', type=int, default=10, help='')
    parser.add_argument('--curv_loss_knn', type=int, default=16, help='')

    parser.add_argument('--max_steps', default=100, type=int,
                        help='max iterations for black-box attack')
    parser.add_argument('--eps', default=0.16, type=float,
                        help='epsilon of perturbation')
    parser.add_argument('--step_size', default=0.007, type=float,
                        help='step-size of perturbation')
    parser.add_argument('--device', default=0, type=int,
                        help='specific device')
    parser.add_argument('--task_name', default=None, type=str,
                        help='specific device')
    parser.add_argument('--rank', type=int, default=0, help='')
    parser.add_argument('--rank_count', type=int, default=1000, help='')

    parser.add_argument('--stage2_steps', type=float, default=0.030, help='step-size of stage 2')
    parser.add_argument('--exponential_step', action='store_true', default=False,
                        help='Whether to use exponential_step [default: False]')

    parser.add_argument('--l2_weight', type=float, default=1.0, help='')
    parser.add_argument('--cd_weight', type=float, default=1.0, help='')
    parser.add_argument('--hd_weight', type=float, default=1.0, help='')
    parser.add_argument('--curv_weight', type=float, default=1.0, help='')
    parser.add_argument('--time_verify', action='store_true', default=False,
                        help='Whether to launch time_verify [default: False]')
    parser.add_argument('--ss_exp', action='store_true', default=False,
                        help='Whether to launch a small scale experiment [default: False]')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Whether to launch the tensorboard [default: False]')



    # Arguments for geoa3
    parser.add_argument('--attack_label', default='Untarget', type=str, help='For GEOA3 [All; ...; Untarget]')
    parser.add_argument('--curv_loss_weight', type=float, default=1.0, help='For GEOA3 ')
    parser.add_argument('--iter_max_steps',  default=500, type=int, metavar='M', help='For GEOA3 max steps')
    parser.add_argument('--optim', default='adam', type=str, help='For GEOA3 adam| sgd')
    parser.add_argument('--lr', type=float, default=0.010, help='For GEOA3 ')
    parser.add_argument('--cls_loss_type', default='CE', type=str, help='For GEOA3 Margin | CE')
    parser.add_argument('--dis_loss_type', default='CD', type=str, help='For GEOA3 CD | L2 | None')
    parser.add_argument('--dis_loss_weight', type=float, default=1.0, help='For GEOA3 ')
    parser.add_argument('--hd_loss_weight', type=float, default=0.1, help='For GEOA3 ')
    parser.add_argument('--is_use_lr_scheduler', dest='is_use_lr_scheduler', action='store_true', default=False, help='')
    parser.add_argument('--cc_linf', type=float, default=0.0, help='For GEOA3 Coefficient for infinity norm')

    # Arguments for GSDA
    parser.add_argument('--band_frequency',  type=int, nargs='+', default=[0, 1024], help='For GSDA band frequency')
    parser.add_argument('--spectral_attack', action='store_true', default=True, help='For GSDA use spectral attack')
    parser.add_argument('--KNN', type=int, default=10, help='K of K-NN graph')
    parser.add_argument('--spectral_offset', action='store_true', default=True, help='use spectral offset attack')
    parser.add_argument('--spectral_restrict', type=float, default=0.0, help='spectral restrict')
    parser.add_argument('--npoint', default=1024, type=int, help='')
    parser.add_argument('--is_partial_var', dest='is_partial_var', action='store_true', default=False, help='')
    parser.add_argument('--is_cd_single_side', action='store_true', default=False, help='')
    parser.add_argument('--uniform_loss_weight', type=float, default=0.0, help='')
  
    args = parser.parse_args()

    if args.task_name is None :
        args.task_name = os.path.basename(args.data_path)

    print(os.path.join(os.path.dirname(args.data_path), args.task_name))

    # basic configuration
    set_seed(args.seed)
    torch.cuda.set_device(args.device)
    args.device = torch.device("cuda:%d" % args.device)

    # main loop
    writer = SummaryWriter(log_dir=f'./logs/{args.task_name}')
    args.writer = writer
    main()
    writer.close()
