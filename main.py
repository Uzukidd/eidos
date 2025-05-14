# -*- coding: utf-8 -*-

import line_profiler
import argparse
import os
import pickle as pkl
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from attacks import PointCloudAttack
from utils.loss_utils import (
    _get_kappa_adv,
    _get_kappa_ori,
    chamfer_loss,
    curvature_loss,
    hausdorff_loss,
    kNN_smoothing_loss,
    norm_l2_loss,
    pseudo_chamfer_loss,
)
from utils.metric_utils import *
from utils.modelnet40_utils import ModelNetDataset
from utils.common_utils import set_seed


def data_preprocess(data: list[torch.Tensor, torch.Tensor]):
    """Preprocess the given data and label."""
    points, target = data

    points = points  # [B, N, C]
    target = target.squeeze(1)  # [B]

    points = points.cuda()
    target = target.cuda()

    return points, target


ten_label_indexes = {
    0: 17,
    2: 9,
    4: 36,
    5: 20,
    8: 3,
    22: 16,
    30: 34,
    33: 38,
    35: 23,
    37: 15,
}


def load_partial_modelnet40_dataset(data_path: str):
    class partial_modelnet40_dataset(Dataset):
        def __init__(self, pkl_file):
            with open(pkl_file, "rb") as f:
                self.data = pkl.load(f)  # list of (points, label)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            points, label = self.data[idx]

            if not isinstance(points, torch.Tensor):
                points = torch.tensor(points, dtype=torch.float32)
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
            return points, label

    dataset = partial_modelnet40_dataset(data_path)
    # dataloader = DataLoader(dataset, batch_size=batch_size,
    #                         shuffle=False, num_workers=64)
    return dataset


def load_modelnet40_dataset(data_path: str):
    # TRAIN_DATASET = ModelNetDataset(
    #     root=data_path, npoint=8192, split="train", normal_channel=False
    # )

    # train_dataLoader = DataLoader(
    #     TRAIN_DATASET, batch_size=batch_size, shuffle=True, num_workers=64
    # )

    TEST_DATASET = ModelNetDataset(root=data_path, npoint=8192,
                                   split="test", normal_channel=True)

    # test_dataLoader = DataLoader(
    #     TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=64
    # )

    return TEST_DATASET


@line_profiler.profile
def main():
    num_class = 0
    if args.dataset == "ModelNet40" or args.dataset == "ModelNet40Full":
        num_class = 40
    elif args.dataset == "ShapeNetPart":
        num_class = 16
    assert num_class != 0
    args.num_class = num_class

    # load model
    attack = PointCloudAttack(args)

    # start attack
    avg_time_cost = 0.0

    result = []
    recall = []

    if args.dataset == "ModelNet40":
        dataset = load_partial_modelnet40_dataset(
            args.data_path)
    elif args.dataset == "ModelNet40Full":
        dataset = load_modelnet40_dataset(
            args.data_path)

    collector = metric_collector()
    collector.register(ASR_metric(attack.classifier))
    collector.register(L2_metric())
    collector.register(HD_metric())
    collector.register(DoubleHD_metric())
    collector.register(CD_metric())
    collector.register(PseudoCD_metric())
    collector.register(Curvature_metric(k=args.curv_loss_knn))
    collector.register(Smooth_metric(k=args.curv_loss_knn))

    avg_time_cost = []

    recall = []

    query_costs = []

    if args.time_verify or args.ss_exp:
        dataset = Subset(dataset, range(0, dataset.__len__(), 20))

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=64
    )

    for batch_id, data in tqdm(
        enumerate(dataloader), total=dataloader.__len__()
    ):

        points, target = data_preprocess(data)
        target = target.long()

        if args.target_model == "PointNet" or args.target_model == "PointNetPP_ssg":
            for b in range(0, target.size(0)):
                target[b] = ten_label_indexes[target[b].item()]

        with torch.no_grad():
            recall = target == attack.predict(points)
        if not recall.any():
            continue

        points, target = points[recall], target[recall]

        # start attack
        t0 = time.time()
        adv_points, adv_target, query_cost = attack.run(points, target)
        if not args.query_attack_method is None:
            query_costs.append(query_cost)
        t1 = time.time()
        avg_time_cost.append(t1 - t0)

        adv_target: torch.Tensor = attack.predict(adv_points)

        result.append((points.cpu().numpy(), adv_points.cpu().numpy(),
                      target.cpu().numpy(), adv_target.cpu().numpy()))

        pc_normal = points[:, :, -3:]
        pc_ori = points[:, :, 0:3]
        pc_adv = adv_points[:, :, :]
        collector.update(pc_adv, pc_ori, pc_normal)

    if args.save_to_file:
        num = 0
        index_list = ""
        if isinstance(dataset, Subset):
            idx_to_classes = dataset.dataset.idx_to_classes
        else:
            idx_to_classes = dataset.idx_to_classes

        save_to_path = f"./outputs/{args.task_name}"
        os.makedirs(save_to_path, exist_ok=True)
        for batch_pts, batch_adv_pts, batch_target, batch_adv_target in result:
            B = batch_pts.shape[0]
            for batch_mask in range(B):
                ori_name = f"{num}_ori_{idx_to_classes[batch_target[batch_mask]]}_no_color.npy"
                adv_name = f"{num}_adv_{idx_to_classes[batch_adv_target[batch_mask]]}_no_color.npy"
                filename = os.path.join(
                    save_to_path, ori_name)
                np.save(filename, batch_pts[batch_mask][:, [0, 2, 1]])
                filename = os.path.join(
                    save_to_path, adv_name)
                np.save(filename, batch_adv_pts[batch_mask][:, [0, 2, 1]])
                index_list += f"{ori_name}, {adv_name}, {idx_to_classes[batch_target[batch_mask]]}, {idx_to_classes[batch_adv_target[batch_mask]]}\n"
                num += 1

        filename = os.path.join(
            save_to_path, "modelnet40_index.txt")
        with open(filename, "w+") as outputs:
            outputs.write(index_list)

    print(collector.output_str())
    # if not args.query_attack_method is None:
    #     log += f"Average Query Cost:{np.array(query_costs).mean()}±{np.array(query_costs).std()}\n"
    # print(log)

    print(f"Average time cost: {np.array(avg_time_cost).mean()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eidos 3D Adversarial Point Clouds"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 1)",
    )
    parser.add_argument(
        "--input_point_nums",
        type=int,
        default=1024,
        help="Point nums of each point cloud",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2022,
        metavar="S",
        help="random seed (default: 2022)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ModelNet40",
        choices=["ModelNet40", "ModelNet40Full"],
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/modelNet40_batch1_1000batches_test.pkl.clean",
    )
    parser.add_argument(
        "--normal",
        action="store_true",
        default=False,
        help="Whether to use normal information [default: False]",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Worker nums of data loading."
    )
    parser.add_argument(
        "--transfer_attack_method",
        type=str,
        default=None,
        choices=[
            "ifgm_si_adv",
            "ifgm_bp",
            "ifgm_bp_ours",
            "ifgm_si_bp",
            "geoa3",
            "gsda",
            "gsda_bp",
        ],
    )
    parser.add_argument(
        "--query_attack_method",
        type=str,
        default=None,
        choices=["ifgm_si_adv_query",
                 "ifgm_bp_ours_query", "simbapp", "simba"],
    )
    parser.add_argument(
        "--surrogate_model",
        type=str,
        default="pointnet_cls",
        choices=[
            "pointnet_cls",
            "PointNetPP_ssg",
            "DGCNN",
            "curvenet",
            "paconv",
            "dgcnn",
            "point_transformer",
            "pointllm_bert",
        ],
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="pointnet_cls",
        choices=[
            "pointnet_cls",
            "PointNetPP_ssg",
            "DGCNN",
            "curvenet",
            "paconv",
            "dgcnn",
            "point_transformer",
            "pointllm_bert",
        ],
    )
    parser.add_argument(
        "--defense_method", type=str, default=None, choices=["sor", "srs", "dupnet"]
    )

    parser.add_argument(
        "--bp_version",
        type=str,
        default="bp3",
        choices=[
            # "bp1",
            # "bp1_si",
            # "bp2",
            # "bp2_si",
            "bp3",
            # "bp3_var",
            # "bp3_no_GS",
            # "bp3_si_no_GS",
            # "bp3_deepfool",
            # "bp3_deepfool_var",
            # "bp3_si",
            # "bp4",
            # "bp4_si",
        ],
    )
    parser.add_argument(
        "--top5_attack",
        action="store_true",
        default=False,
        help="Whether to attack the top-5 prediction [default: False]",
    )

    parser.add_argument("--initial_const", type=float, default=10, help="")
    parser.add_argument("--binary_max_steps", type=int, default=10, help="")
    parser.add_argument("--curv_loss_knn", type=int, default=16, help="")

    parser.add_argument(
        "--max_steps", default=100, type=int, help="max iterations for black-box attack"
    )
    parser.add_argument(
        "--eps", default=0.16, type=float, help="epsilon of perturbation"
    )
    parser.add_argument(
        "--step_size", default=0.007, type=float, help="step-size of perturbation"
    )
    parser.add_argument("--device", default=0, type=int,
                        help="specific device")
    parser.add_argument("--task_name", default=None,
                        type=str, help="specific device")
    # parser.add_argument("--rank", type=int, default=0, help="")
    # parser.add_argument("--rank_count", type=int, default=1000, help="")

    parser.add_argument(
        "--stage2_steps", type=float, default=0.030, help="step-size of stage 2"
    )
    parser.add_argument(
        "--exponential_step",
        action="store_true",
        help="Whether to use exponential_step [default: False]",
    )
    parser.add_argument(
        "--save_to_file",
        action="store_true",
        help="Whether to save point clouds as .npy [default: False]",
    )

    parser.add_argument("--l2_weight", type=float, default=1.0, help="")
    parser.add_argument("--cd_weight", type=float, default=1.0, help="")
    parser.add_argument("--hd_weight", type=float, default=1.0, help="")
    parser.add_argument("--curv_weight", type=float, default=1.0, help="")
    parser.add_argument(
        "--time_verify",
        action="store_true",
        default=False,
        help="Whether to launch time_verify [default: False]",
    )
    parser.add_argument(
        "--ss_exp",
        action="store_true",
        help="Whether to launch a small scale experiment [default: False]",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=False,
        help="Whether to launch the tensorboard [default: False]",
    )

    # Arguments for geoa3
    # parser.add_argument(
    #     "--attack_label",
    #     default="Untarget",
    #     type=str,
    #     help="For GEOA3 [All; ...; Untarget]",
    # )
    # parser.add_argument(
    #     "--curv_loss_weight", type=float, default=1.0, help="For GEOA3 "
    # )
    # parser.add_argument(
    #     "--iter_max_steps",
    #     default=500,
    #     type=int,
    #     metavar="M",
    #     help="For GEOA3 max steps",
    # )
    # parser.add_argument("--optim", default="adam",
    #                     type=str, help="For GEOA3 adam| sgd")
    # parser.add_argument("--lr", type=float, default=0.010, help="For GEOA3 ")
    # parser.add_argument(
    #     "--cls_loss_type", default="CE", type=str, help="For GEOA3 Margin | CE"
    # )
    # parser.add_argument(
    #     "--dis_loss_type", default="CD", type=str, help="For GEOA3 CD | L2 | None"
    # )
    # parser.add_argument("--dis_loss_weight", type=float,
    #                     default=1.0, help="For GEOA3 ")
    # parser.add_argument("--hd_loss_weight", type=float,
    #                     default=0.1, help="For GEOA3 ")
    # parser.add_argument(
    #     "--is_use_lr_scheduler",
    #     dest="is_use_lr_scheduler",
    #     action="store_true",
    #     default=False,
    #     help="",
    # )
    # parser.add_argument(
    #     "--cc_linf",
    #     type=float,
    #     default=0.0,
    #     help="For GEOA3 Coefficient for infinity norm",
    # )

    # # Arguments for GSDA
    # parser.add_argument(
    #     "--band_frequency",
    #     type=int,
    #     nargs="+",
    #     default=[0, 1024],
    #     help="For GSDA band frequency",
    # )
    # parser.add_argument(
    #     "--spectral_attack",
    #     action="store_true",
    #     default=True,
    #     help="For GSDA use spectral attack",
    # )
    # parser.add_argument("--KNN", type=int, default=10, help="K of K-NN graph")
    # parser.add_argument(
    #     "--spectral_offset",
    #     action="store_true",
    #     default=True,
    #     help="use spectral offset attack",
    # )
    # parser.add_argument(
    #     "--spectral_restrict", type=float, default=0.0, help="spectral restrict"
    # )
    # parser.add_argument("--npoint", default=1024, type=int, help="")
    # parser.add_argument(
    #     "--is_partial_var",
    #     dest="is_partial_var",
    #     action="store_true",
    #     default=False,
    #     help="",
    # )
    # parser.add_argument(
    #     "--is_cd_single_side", action="store_true", default=False, help=""
    # )
    # parser.add_argument("--uniform_loss_weight",
    #                     type=float, default=0.0, help="")

    args = parser.parse_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.data_path)

    print(os.path.join(os.path.dirname(args.data_path), args.task_name))

    # basic configuration
    set_seed(args.seed)
    torch.cuda.set_device(args.device)
    args.device = torch.device("cuda:%d" % args.device)

    # main loop
    # writer = SummaryWriter(log_dir=f'./logs/{args.task_name}')
    # args.writer = writer
    main()
    # writer.close()
