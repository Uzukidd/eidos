from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from utils.loss_utils import (
    _get_kappa_adv,
    _get_kappa_ori,
    chamfer_loss,
    hausdorff_loss,
    kNN_smoothing_loss,
    local_curvature_loss,
    norm_l2_loss,
    pseudo_chamfer_loss,
)

from typing import Optional

class metric(ABC):
    def __init__(self, name: str):
        self.name = name
        self.batch_metric = []  # list[Nx1] x B

    @abstractmethod
    def update(
        self, pc_adv: torch.Tensor, pc_ori: torch.Tensor, pc_normal: torch.Tensor
    ):
        raise NotImplementedError

    def compute(self):
        if not self.batch_metric:
            return None
        return torch.mean(torch.cat(self.batch_metric)), torch.std(
            torch.cat(self.batch_metric)
        )


class ASR_metric(metric):

    def __init__(self, cls_model: nn.Module):
        super().__init__("ASR")
        self.cls_model = cls_model

    def update(
        self, pc_adv: torch.Tensor, pc_ori: torch.Tensor, pc_normal: torch.Tensor
    ):
        asr = self.cls_model(pc_adv.transpose(1, 2)).argmax(-1) != self.cls_model(
            pc_ori.transpose(1, 2)
        ).argmax(-1)
        self.batch_metric.append(asr.float())


class L2_metric(metric):

    def __init__(self):
        super().__init__("L2 norm")

    def update(
        self, pc_adv: torch.Tensor, pc_ori: torch.Tensor, pc_normal: torch.Tensor
    ):
        l2 = norm_l2_loss(pc_adv, pc_ori)
        self.batch_metric.append(l2)


class HD_metric(metric):
    def __init__(self):
        super().__init__("Hausdorff Distance")

    def update(
        self, pc_adv: torch.Tensor, pc_ori: torch.Tensor, pc_normal: torch.Tensor
    ):
        hd_1 = hausdorff_loss(pc_adv, pc_ori)
        self.batch_metric.append(hd_1)


class DoubleHD_metric(metric):
    def __init__(self):
        super().__init__("Double Hausdorff Distance")

    def update(
        self, pc_adv: torch.Tensor, pc_ori: torch.Tensor, pc_normal: torch.Tensor
    ):
        hd_1 = hausdorff_loss(pc_adv, pc_ori)
        hd_2 = hausdorff_loss(pc_ori, pc_adv)
        self.batch_metric.append(((hd_1 + hd_2) / 2))


class CD_metric(metric):
    def __init__(self):
        super().__init__("Chamfer Distance")

    def update(
        self, pc_adv: torch.Tensor, pc_ori: torch.Tensor, pc_normal: torch.Tensor
    ):
        cd = chamfer_loss(pc_adv, pc_ori)
        self.batch_metric.append(cd)


class PseudoCD_metric(metric):
    def __init__(self):
        super().__init__("Pseudo Chamfer Distance")

    def update(
        self, pc_adv: torch.Tensor, pc_ori: torch.Tensor, pc_normal: torch.Tensor
    ):
        pcd = pseudo_chamfer_loss(pc_adv, pc_ori)
        self.batch_metric.append(pcd)


class Curvature_metric(metric):
    def __init__(self, k: int):
        super().__init__("Curvature Loss")
        self.k = k

    def update(
        self, pc_adv: torch.Tensor, pc_ori: torch.Tensor, pc_normal: torch.Tensor
    ):
        cur = local_curvature_loss(pc_adv, pc_ori, pc_normal, self.k)
        self.batch_metric.append(cur)


class Smooth_metric(metric):
    def __init__(self, k: int):
        super().__init__("kNN Smoothing Loss")
        self.k = k

    def update(
        self, pc_adv: torch.Tensor, pc_ori: torch.Tensor, pc_normal: torch.Tensor
    ):
        smooth = kNN_smoothing_loss(pc_adv, self.k)
        self.batch_metric.append(smooth)


class metric_collector:
    def __init__(self):
        self.metrics: list[metric] = []

    def register(self, metric_obj):
        self.metrics.append(metric_obj)

    def update(
        self, pc_adv: torch.Tensor, pc_ori: torch.Tensor, pc_normal: torch.Tensor, recall_mask: Optional[torch.Tensor] = None
    ):
        with torch.no_grad():
            for m in self.metrics:
                m.update(pc_adv, pc_ori, pc_normal)

    # def reset(self):
    #     for m in self.metrics:
    #         m.reset()

    # def compute_all(self):
    #     return {m.name: m.compute() for m in self.metrics}

    def output_str(self):
        log = "Metric Summary:\n"
        for m in self.metrics:
            result = m.compute()
            if result is None:
                log += f"{m.name}: No data.\n"
            else:
                mean, std = result
                log += f"{m.name}: {mean:.6f} ± {std:.6f}\n"
        return log
