from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from utils.clip_utils import *
from utils.loss_utils import *
from utils.siadv_utils import *


class siadv_attack(nn.Module):

    def __init__(
        self,
        eps: float,
        step_size: float,
        max_steps: int,
        classifier: nn.Module,
        pre_head: Optional[nn.Module],
        num_class: int,
        top5_attack: bool,
    ):
        super().__init__()
        self.eps = eps
        self.step_size = step_size
        self.max_steps = max_steps
        self.classifier = classifier
        self.pre_head = pre_head
        self.num_class = num_class
        self.top5_attack = top5_attack

    def forward(self, points, target):
        """White-box I-FGSM based on shape-invariant sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [B, N, 6].
            target (torch.cuda.LongTensor): the label for points, [B].
        """
        normal_vec = points[:, :, -3:].data  # N, [B, N, 3]
        normal_vec = normal_vec / torch.sqrt(
            torch.sum(normal_vec**2, dim=-1, keepdim=True)
        )  # N, [B, N, 3]
        points = points[:, :, :3].data  # P, [B, N, 3]
        ori_points = points.data
        clip_func = ClipPointsLinf(budget=self.eps)  # * np.sqrt(3*1024))

        for i in range(self.max_steps):
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
                logits,
                target,
                kappa=0.0,
                tar=False,
                num_classes=self.num_class,
                top5_attack=self.top5_attack,
            )
            self.classifier.zero_grad()
            loss.backward()
            # print(loss.item(), logits.max(1)[1], target)
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

            normal_vec = get_normal_vector(points)  # N, [1, N, 3]

        with torch.no_grad():
            adv_points = points.data
            adv_logits = self.classifier(self.pre_head(points.transpose(1, 2).detach()))
            adv_target = adv_logits.argmax(1)

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
            (adv_logits.argmax(1) != target).sum().item(),
        )
