from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from utils.bp_utils import *
from utils.clip_utils import *
from utils.loss_utils import *
from utils.siadv_utils import *


class eidos_attack(nn.Module):
    def __init__(
        self,
        eps: float,
        step_size: float,
        max_steps: int,
        classifier: nn.Module,
        pre_head: Optional[nn.Module],
        num_class: int,
        top5_attack: bool,
        # boundary projection arguments
        bp: str,
        l2_weight: float,
        hd_weight: float,
        cd_weight: float,
        curv_weight: float,
        curv_loss_knn: int,
        stage2_steps: float,
        exponential_step: bool,
    ):
        super().__init__()
        self.eps = eps
        self.step_size = step_size
        self.max_steps = max_steps
        self.classifier = classifier
        self.pre_head = pre_head
        self.num_class = num_class
        self.top5_attack = top5_attack
        self.bp = bp
        self.si_grad_required = False

        self.boundary_projection = boundary_projection_3(
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

        # if self.bp_version == "bp4":
        #     self.boundary_projection = boundary_projection_4(self.args)
        # elif self.bp_version == "bp4_si":
        #     self.boundary_projection = boundary_projection_4_si(self.args)
        #     si_grad_required = True
        # elif self.bp_version == "bp3":
        #     bp_optims = []
        #     if self.l2_weight != 0.0:
        #         bp_optims.append("l2")

        #     if self.hd_weight != 0.0:
        #         bp_optims.append("hd")

        #     if self.curv_weight != 0.0:
        #         bp_optims.append("curv")

        #     if self.cd_weight != 0.0:
        #         bp_optims.append("cd")
        #     bp = boundary_projection_3(self.args, bp_optims)
        # elif self.bp_version == "bp3_si":
        #     bp_optims = []
        #     if self.l2_weight != 0.0:
        #         bp_optims.append("l2")

        #     if self.hd_weight != 0.0:
        #         bp_optims.append("hd")

        #     if self.curv_weight != 0.0:
        #         bp_optims.append("curv")

        #     if self.cd_weight != 0.0:
        #         bp_optims.append("cd")
        #     bp = boundary_projection_3_si(self.args, bp_optims)
        #     si_grad_required = True
        # elif self.bp_version == "bp2":
        #     bp_weights = []
        #     bp_optims = []
        #     if self.l2_weight != 0.0:
        #         bp_weights.append(self.l2_weight)
        #         bp_optims.append("l2")

        #     if self.hd_weight != 0.0:
        #         bp_weights.append(self.hd_weight)
        #         bp_optims.append("hd")

        #     if self.curv_weight != 0.0:
        #         bp_weights.append(self.curv_weight)
        #         bp_optims.append("curv")

        #     if self.cd_weight != 0.0:
        #         bp_weights.append(self.cd_weight)
        #         bp_optims.append("cd")

        #     bp = boundary_projection_2(self.args, weights=bp_weights, optim_seq=bp_optims)
        # elif self.bp_version == "bp2_si":
        #     bp_weights = []
        #     bp_optims = []
        #     if self.l2_weight != 0.0:
        #         bp_weights.append(self.l2_weight)
        #         bp_optims.append("l2")

        #     if self.hd_weight != 0.0:
        #         bp_weights.append(self.hd_weight)
        #         bp_optims.append("hd")

        #     if self.curv_weight != 0.0:
        #         bp_weights.append(self.curv_weight)
        #         bp_optims.append("curv")

        #     if self.cd_weight != 0.0:
        #         bp_weights.append(self.cd_weight)
        #         bp_optims.append("cd")

        #     bp = boundary_projection_2_si(
        #         self.args, weights=bp_weights, optim_seq=bp_optims
        #     )
        #     si_grad_required = True
        # elif self.bp_version == "bp1_si":
        #     bp = boundary_projection_1_si(self.args)
        #     si_grad_required = True

    def forward(self, points, target):
        """White-box I-FGSM with boundary projection based on shape-invariant sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [B, N, 6].
            target (torch.cuda.LongTensor): the label for points, [B].
        """
        normal_vec = points[:, :, -3:].detach()  # N, [1, N, 3]
        normal_vec = normal_vec / torch.sqrt(
            torch.sum(normal_vec**2, dim=-1, keepdim=True)
        )  # N, [1, N, 3]
        points = points[:, :, :3].detach()  # P, [1, N, 3]
        ori_points = points.detach()
        clip_func = ClipPointsLinf(budget=self.eps)  # * np.sqrt(3*1024))
        stage2 = False

        self.boundary_projection.reset(points.size(0))

        # output_reg = torch.ones((points.size(0), 4)).cuda() * 1e10

        for i in range(self.max_steps):
            # print(self.max_steps)
            # bp.epoch = i

            if not stage2:
                # P -> P', detach()
                new_points, spin_axis_matrix, translation_matrix = (
                    get_transformed_point_cloud(points, normal_vec)
                )
                new_points = new_points.detach()
                new_points.requires_grad = True
                # P' -> P
                points = get_original_point_cloud(
                    new_points, spin_axis_matrix, translation_matrix
                )
                points = points.transpose(1, 2)  # P, [1, 3, N]
                # get white-box gradients
                logits = self.classifier(self.pre_head(points))

                loss = CWLoss(
                    logits, target, kappa=0.0, tar=False, num_classes=self.num_class
                )
                self.classifier.zero_grad()
                loss.backward()
                grad = new_points.grad.data  # g, [1, N, 3]
                grad[:, :, 2] = 0.0
                # update P', P and N
                # # Linf
                # new_points = new_points - self.step_size * torch.sign(grad)

                # L2
                norm = torch.sum(grad**2, dim=[1, 2]) ** 0.5
                new_points = new_points - self.step_size * np.sqrt(3 * 1024) * grad / (
                    norm[:, None, None] + 1e-9
                )
                points = get_original_point_cloud(
                    # P, [1, N, 3]
                    new_points,
                    spin_axis_matrix,
                    translation_matrix,
                )
                points = clip_func(points, ori_points)

                points = points.detach()

                normal_vec = get_normal_vector(points)  # N, [1, N, 3]

                logits = self.classifier(self.pre_head(points.transpose(1, 2)))

                logits = logits.argmax(-1)
                stage2 = (logits != target).all()

            else:
                points = points.detach()
                points.requires_grad = True

                logits = self.classifier(self.pre_head(points.transpose(1, 2)))

                logits = F.log_softmax(logits, dim=-1)
                loss = logits.gather(dim=1, index=target.unsqueeze(1)).sum()
                self.classifier.zero_grad()
                loss.backward()

                g = points.grad.detach()

                g_norm = (g**2).sum((1, 2)).sqrt()
                g_norm.clamp_(min=1e-12)
                g_hat = g / g_norm[:, None, None]

                points = self.boundary_projection(
                    points, ori_points, normal_vec, g_hat, logits, target
                )

                normal_vec = get_normal_vector(points)

                # else:

                # new_points, spin_axis_matrix, translation_matrix = (
                #     get_transformed_point_cloud(points, normal_vec)
                # )
                # new_points = new_points.detach()
                # new_points.requires_grad = True

                # points = get_original_point_cloud(
                #     new_points, spin_axis_matrix, translation_matrix
                # )

                # logits = self.classifier(self.pre_head(points.transpose(1, 2)))

                # logits = F.log_softmax(logits, dim=-1)
                # loss = logits[:, target.item()]
                # self.classifier.zero_grad()
                # loss.backward()

                # g = new_points.grad.detach().clone()
                # g[:, :, 2] = 0.0

                # new_points.grad.zero_()

                # g_norm = (g**2).sum((1, 2)).sqrt()
                # g_norm[g_norm == 0] = 1e-12
                # g_hat = g / g_norm[:, None, None]

                # normal_vec = torch.zeros_like(normal_vec)
                # normal_vec[:, :, 2] = 1

                # points = self.boundary_projection(
                #     new_points,
                #     spin_axis_matrix,
                #     translation_matrix,
                #     ori_points,
                #     normal_vec,
                #     g_hat,
                #     logits,
                #     target,
                # )

                # normal_vec = get_normal_vector(points)

        with torch.no_grad():
            # if not bp.output_points is None:
            #     adv_points = bp.output_points.clone()
            # else:
            #     adv_points = points.clone()
            adv_points = self.boundary_projection.output_points.clone()
            adv_logits = self.classifier(
                self.pre_head(adv_points.transpose(1, 2)))
            adv_target = adv_logits.argmax(-1)

        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1

        del normal_vec, grad, new_points, spin_axis_matrix, translation_matrix
        return (
            adv_points,
            adv_target,
            (adv_logits.data.max(1)[1] != target).sum().item(),
        )
