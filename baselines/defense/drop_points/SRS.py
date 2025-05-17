"""SRS defense proposed by https://arxiv.org/pdf/1902.10899.pdf"""
import numpy as np

import torch
import torch.nn as nn


class SRSDefense(nn.Module):
    """Random dropping points as defense.
    """

    def __init__(self, drop_num=500):
        """SRS defense method.

        Args:
            drop_num (int, optional): number of points to drop.
                                        Defaults to 500.
        """
        super(SRSDefense, self).__init__()

        self.drop_num = drop_num

    def random_drop(self, pc):
        """Random drop self.drop_num points in each pc.

        Args:
            pc (torch.FloatTensor): batch input pc, [B, K, 3]
        """
        B, K = pc.shape[:2]
        if self.drop_num == 0:
            return pc
        assert self.drop_num < K, "drop_num must be less than the number of points K"
        
        idx = torch.stack([torch.randperm(K, device=pc.device)[:K - self.drop_num] for _ in range(B)])
        
        # Gather points using expanded indices [B, K - drop_num, 3]
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, 3)
        pc = torch.gather(pc, 1, idx_expanded)
        # B, K = pc.shape[:2]
        # idx = [np.random.choice(K, K - self.drop_num, replace=False) for _ in range(B)]
        # pc = torch.stack([pc[i][torch.from_numpy(idx[i]).long().to(pc.device)] for i in range(B)])
        return pc

    def forward(self, x:torch.Tensor):
        # with torch.no_grad():
        x = x.transpose(1, 2)
        x = self.random_drop(x)
        x = x.transpose(1, 2)
        return x
