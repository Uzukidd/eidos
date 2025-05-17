import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils.geoa3_utils import _compare, farthest_points_sample, lp_clip
from utils.loss_utils import (
    chamfer_loss,
    hausdorff_loss,
    _get_kappa_adv,
    _get_kappa_ori,
    curvature_loss,
    norm_l2_loss,
)


class geoa3_attack(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        defense_head: nn.Module,
        attack_label: str,
        curv_loss_weight: float,
        curv_loss_knn: int,
        initial_const: float,
        iter_max_steps: int,
        cls_loss_type: str,
        dis_loss_type: str,
        hd_loss_weight: int,
        binary_max_steps: int,
        optim: str,
        dis_loss_weight: int,
        lr: float,
        is_use_lr_scheduler: bool,
        cc_linf: int,
        npoint: int,
        confidence: int,
        classes: int,
        eval_num: int,
    ):
        super().__init__()
        self.model = model
        self.defense_head = defense_head
        self.attack_label = attack_label
        self.curv_loss_weight = curv_loss_weight
        self.curv_loss_knn = curv_loss_knn
        self.initial_const = initial_const
        self.iter_max_steps = iter_max_steps
        self.cls_loss_type = cls_loss_type
        self.dis_loss_type = dis_loss_type
        self.lr = lr
        self.hd_loss_weight = hd_loss_weight
        self.binary_max_steps = binary_max_steps
        self.optim = optim
        self.dis_loss_weight = dis_loss_weight
        self.is_use_lr_scheduler = is_use_lr_scheduler
        self.cc_linf = cc_linf
        self.npoint = npoint
        self.confidence = confidence
        self.classes = classes
        self.eval_num = eval_num

    def _forward_step(
        self,
        pc_ori,
        input_curr_iter,
        normal_ori,
        ori_kappa,
        target,
        scale_const,
        targeted,
    ):
        # needed self:[arch, classes, cls_loss_type, confidence, dis_loss_type, is_cd_single_side, dis_loss_weight, hd_loss_weight, curv_loss_weight, curv_loss_knn]
        b, n, _ = input_curr_iter.size()
        output_curr_iter = self.model(
            self.defense_head(input_curr_iter.transpose(1, 2))
        )

        if self.cls_loss_type == "Margin":
            target_onehot = torch.zeros(target.size() + (self.classes,)).cuda()
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

            fake = (target_onehot * output_curr_iter).sum(1)
            other = (
                (1.0 - target_onehot) * output_curr_iter - target_onehot * 10000.0
            ).max(1)[0]

            if targeted:
                # if targeted, optimize for making the other class most likely
                # equiv to max(..., 0.)
                cls_loss = torch.clamp(other - fake + self.confidence, min=0.0)
            else:
                # if non-targeted, optimize for making this class least likely.
                # equiv to max(..., 0.)
                cls_loss = torch.clamp(fake - other + self.confidence, min=0.0)

        elif self.cls_loss_type == "CE":
            if targeted:
                cls_loss = nn.CrossEntropyLoss(reduction="none").cuda()(
                    output_curr_iter, Variable(target, requires_grad=False)
                )
            else:
                cls_loss = -nn.CrossEntropyLoss(reduction="none").cuda()(
                    output_curr_iter, Variable(target, requires_grad=False)
                )
        elif self.cls_loss_type == "None":
            cls_loss = torch.FloatTensor(b).zero_().cuda()
        else:
            assert False, "Not support such clssification loss"

        info = "cls_loss: {0:6.4f}\t".format(cls_loss.mean().item())

        if self.dis_loss_type == "CD":
            dis_loss = chamfer_loss(input_curr_iter, pc_ori)

            constrain_loss = self.dis_loss_weight * dis_loss
            info = info + "cd_loss: {0:6.4f}\t".format(dis_loss.mean().item())
        elif self.dis_loss_type == "L2":
            assert self.hd_loss_weight == 0
            dis_loss = norm_l2_loss(input_curr_iter, pc_ori)
            constrain_loss = self.dis_loss_weight * dis_loss
            info = info + "l2_loss: {0:6.4f}\t".format(dis_loss.mean().item())
        elif self.dis_loss_type == "None":
            dis_loss = 0
            constrain_loss = 0
        else:
            assert False, "Not support such distance loss"

        # hd_loss
        if self.hd_loss_weight != 0:
            hd_loss = hausdorff_loss(input_curr_iter, pc_ori)
            constrain_loss = constrain_loss + self.hd_loss_weight * hd_loss
            info = info + "hd_loss : {0:6.4f}\t".format(hd_loss.mean().item())
        else:
            hd_loss = 0

        # nor loss
        if self.curv_loss_weight != 0:
            adv_kappa, normal_curr_iter = _get_kappa_adv(
                input_curr_iter, pc_ori, normal_ori, self.curv_loss_knn
            )
            curv_loss = curvature_loss(input_curr_iter, pc_ori, adv_kappa, ori_kappa)
            constrain_loss = constrain_loss + self.curv_loss_weight * curv_loss
            info = info + "curv_loss : {0:6.4f}\t".format(curv_loss.mean().item())
        else:
            normal_curr_iter = torch.zeros(b, 3, n).cuda()
            curv_loss = 0

        scale_const = scale_const.float().cuda()
        loss_n = cls_loss + scale_const * constrain_loss
        loss = loss_n.mean()

        return (
            output_curr_iter,
            normal_curr_iter,
            loss,
            loss_n,
            cls_loss,
            dis_loss,
            hd_loss,
            curv_loss,
            constrain_loss,
            info,
        )

    def forward(self, points: torch.Tensor, target: torch.Tensor):
        # needed self:[arch, classes, attack_label, initial_const, lr, optim, binary_max_steps, iter_max_steps, metric,
        #  cls_loss_type, confidence, dis_loss_type, is_cd_single_side, dis_loss_weight, hd_loss_weight, curv_loss_weight, curv_loss_knn,
        #  is_pre_jitter_input, calculate_project_jitter_noise_iter, jitter_k, jitter_sigma, jitter_clip,
        #  is_save_normal,
        #  ]
        if self.attack_label == "Untarget":
            targeted = False
        else:
            targeted = True

        pc_ori = points[:, :, :3]
        normal_ori = points[:, :, 3:]
        gt_labels = target.view(-1)
        gt_target = gt_labels.view(-1)

        b, n, _ = pc_ori.size()

        if self.attack_label == "Untarget":
            target = gt_target
        else:
            target = target.view(-1)

        if self.curv_loss_weight != 0:
            kappa_ori = _get_kappa_ori(pc_ori, normal_ori, self.curv_loss_knn)
        else:
            kappa_ori = None

        lower_bound = torch.ones(b) * 0
        scale_const = torch.ones(b) * self.initial_const
        upper_bound = torch.ones(b) * 1e10

        best_loss = [1e10] * b
        best_attack = pc_ori.clone()
        best_attack_step = [-1] * b
        best_attack_BS_idx = [-1] * b
        all_loss_list = [[-1] * b] * self.iter_max_steps

        for search_step in range(self.binary_max_steps):
            iter_best_loss = [1e10] * b
            iter_best_score = [-1] * b
            constrain_loss = torch.ones(b) * 1e10
            attack_success = torch.zeros(b).cuda()

            input_all = None
            dis_loss = 0
            hd_loss = 0
            nor_loss = 0
            for step in range(self.iter_max_steps):
                if step == 0:
                    offset = torch.zeros(b, n, 3).cuda()
                    nn.init.normal_(offset, mean=0, std=1e-3)
                    offset.requires_grad_()

                    if self.optim == "adam":
                        optimizer = optim.Adam([offset], lr=self.lr)
                    elif self.optim == "sgd":
                        optimizer = optim.SGD([offset], lr=self.lr)
                    else:
                        assert False, "Not support such optimizer."
                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer, gamma=0.9990, last_epoch=-1
                    )

                    periodical_pc = pc_ori.clone()

                input_all = periodical_pc + offset

                input_curr_iter = input_all

                with torch.no_grad():
                    for k in range(b):
                        if input_curr_iter.size(1) < input_all.size(1):
                            # batch_k_pc = torch.cat([input_curr_iter[k].unsqueeze(0)]*self.eval_num)
                            batch_k_pc = farthest_points_sample(
                                torch.cat(
                                    [input_curr_iter[k].unsqueeze(0)] * self.eval_num
                                ).unsqueeze(0),
                                self.npoint,
                            )
                            batch_k_adv_output = self.model(
                                self.defense_head(batch_k_pc.transpose(1, 2))
                            )
                            attack_success[k] = (
                                _compare(
                                    torch.max(batch_k_adv_output, 1)[1].data,
                                    target[k],
                                    gt_target[k],
                                    targeted,
                                ).sum()
                                > 0.5 * self.eval_num
                            )
                            output_label = (
                                torch.max(batch_k_adv_output, 1)[1].mode().values.item()
                            )
                        else:

                            adv_output = self.model(
                                self.defense_head(
                                    input_curr_iter[k].unsqueeze(0).transpose(1, 2)
                                )
                            )

                            output_label = torch.argmax(adv_output).item()
                            attack_success[k] = _compare(
                                output_label, target[k], gt_target[k].cuda(), targeted
                            ).item()

                        metric = constrain_loss[k].item()

                        if attack_success[k] and (metric < best_loss[k]):
                            best_loss[k] = metric
                            best_attack[k] = input_all.data[k].clone()
                            best_attack_BS_idx[k] = search_step
                            best_attack_step[k] = step
                            # print(info)
                        if attack_success[k] and (metric < iter_best_loss[k]):
                            iter_best_loss[k] = metric
                            iter_best_score[k] = output_label

                (
                    _,
                    normal_curr_iter,
                    loss,
                    loss_n,
                    cls_loss,
                    dis_loss,
                    hd_loss,
                    nor_loss,
                    constrain_loss,
                    info,
                ) = self._forward_step(
                    pc_ori,
                    input_curr_iter,
                    normal_ori,
                    kappa_ori,
                    target,
                    scale_const,
                    targeted,
                )
                all_loss_list[step] = loss_n.detach().tolist()

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()
                if self.is_use_lr_scheduler:
                    lr_scheduler.step()

                if self.cc_linf != 0:
                    with torch.no_grad():
                        proj_offset = lp_clip(offset, self.cc_linf)
                        offset.data = proj_offset.detach().clone()

            # adjust the scale constants
            for k in range(b):
                if (
                    _compare(
                        output_label, target[k], gt_target[k].cuda(), targeted
                    ).item()
                    and iter_best_score[k] != -1
                ):
                    lower_bound[k] = max(lower_bound[k], scale_const[k])
                    if upper_bound[k] < 1e9:
                        scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5
                    else:
                        scale_const[k] *= 2
                else:
                    upper_bound[k] = min(upper_bound[k], scale_const[k])
                    if upper_bound[k] < 1e9:
                        scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5

        # best_attack:[b, 3, n], target: [b], best_loss:[b], best_attack_step:[b], all_loss_list:[iter_max_steps, b]
        return (best_attack, target, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEOA3 Point Cloud Attacking")
    # ------------Model-----------------------
    parser.add_argument(
        "--arch", default="Pointself.model", type=str, metavar="ARCH", help=""
    )
    # ------------Dataset-----------------------
    parser.add_argument(
        "--data_dir_file",
        default="../Data/modelself.model10_250instances_1024.mat",
        type=str,
        help="",
    )
    parser.add_argument("--dense_data_dir_file", default="", type=str, help="")
    parser.add_argument(
        "-c",
        "--classes",
        default=40,
        type=int,
        metavar="N",
        help="num of classes (default: 40)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=2,
        type=int,
        metavar="B",
        help="batch_size (default: 2)",
    )
    parser.add_argument("--npoint", default=1024, type=int, help="")
    # ------------Attack-----------------------
    parser.add_argument(
        "--attack_label", default="All", type=str, help="[All; ...; Untarget; Random]"
    )
    parser.add_argument("--initial_const", type=float, default=10, help="")
    parser.add_argument("--lr", type=float, default=0.01, help="")
    parser.add_argument("--optim", default="adam", type=str, help="adam| sgd")
    parser.add_argument("--binary_max_steps", type=int, default=10, help="")
    parser.add_argument(
        "--iter_max_steps", default=500, type=int, metavar="M", help="max steps"
    )
    # cls loss
    parser.add_argument("--cls_loss_type", default="CE", type=str, help="Margin | CE")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0,
        help="confidence for margin based attack method",
    )
    # distance loss
    parser.add_argument(
        "--dis_loss_type", default="CD", type=str, help="CD | L2 | None"
    )
    parser.add_argument("--dis_loss_weight", type=float, default=1.0, help="")
    parser.add_argument(
        "--is_cd_single_side", action="store_true", default=False, help=""
    )
    # hausdorff loss
    parser.add_argument("--hd_loss_weight", type=float, default=0.1, help="")
    # normal loss
    parser.add_argument("--curv_loss_weight", type=float, default=0.1, help="")
    parser.add_argument("--curv_loss_knn", type=int, default=16, help="")
    # eval metric
    parser.add_argument(
        "--metric", default="Loss", type=str, help="[Loss | CDDis | HDDis | CurDis]"
    )
    # Jitter
    parser.add_argument(
        "--is_pre_jitter_input", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--calculate_project_jitter_noise_iter", default=50, type=int, help=""
    )
    parser.add_argument("--jitter_k", type=int, default=16, help="")
    parser.add_argument("--jitter_sigma", type=float, default=0.01, help="")
    parser.add_argument("--jitter_clip", type=float, default=0.05, help="")
    # ------------OS-----------------------
    parser.add_argument(
        "-j",
        "--num_workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 8)",
    )
    parser.add_argument("--is_save_normal", action="store_true", default=False, help="")
    self = parser.parse_args()
