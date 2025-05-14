import torch
from pytorch3d.ops import sample_farthest_points

import numpy as np

def farthest_points_normal_sample(obj_points, obj_normal, num_points):
    assert obj_points.size(1) == 3
    assert obj_points.size(2) == obj_normal.size(2)
    b,_,n = obj_points.size()

    selected = torch.randint(obj_points.size(2), [obj_points.size(0),1]).cuda()
    dists = torch.full([obj_points.size(0), obj_points.size(2)], fill_value = np.inf).cuda()

    for _ in range(num_points - 1):
        dists = torch.min(dists, torch.norm(obj_points - torch.gather(obj_points, 2, selected[:,-1].unsqueeze(1).unsqueeze(2).expand(b,3,1)), dim = 1))
        selected = torch.cat([selected, torch.argmax(dists, dim=1, keepdim=True)], dim = 1)
    res_points = torch.gather(obj_points, 2, selected.unsqueeze(1).expand(b, 3, num_points))
    res_normal = torch.gather(obj_normal, 2, selected.unsqueeze(1).expand(b, 3, num_points))

    return res_points, res_normal

def _compare(output, target, gt, targeted):
    if targeted:
        return output == target
    else:
        return output != gt
    
def farthest_points_sample(obj_points, num_points):
    """
    Args: 
        obj_points[B, N, 3]
        num_points: int
    """
    sampled_points, _ = sample_farthest_points(
        obj_points, 
        K=num_points,
        random_start_point=True
    )
    return sampled_points

def lp_clip(self, offset, cc_linf):
    """
    Args: 
        offset[B, N, 3]
        cc_linf: int
    """
    lengths = (offset**2).sum(2, keepdim=True).sqrt()  # [b, n, 1]
    lengths_expand = lengths.expand_as(offset)  # [b, n, 3]

    condition = lengths > 1e-6
    offset_scaled = torch.where(
        condition, offset / lengths_expand * cc_linf, torch.zeros_like(offset)
    )

    condition = lengths < cc_linf
    offset = torch.where(condition, offset, offset_scaled)

    return offset