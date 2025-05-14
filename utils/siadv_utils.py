import torch
import numpy as np
import open3d as o3d

def get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix):
    """Calculate the spin-axis matrix.

    Args:
        new_points (torch.cuda.FloatTensor): the transformed point cloud with N points, [B, N, 3].
        spin_axis_matrix (torch.cuda.FloatTensor): the rotate matrix for transformation, [B, N, 3, 3].
        translation_matrix (torch.cuda.FloatTensor): the offset matrix for transformation, [B, N, 3, 1].
    """
    inputs = torch.matmul(
        spin_axis_matrix.transpose(-1, -2), new_points.unsqueeze(-1)
    )  # U^T P', [B, N, 3, 1]

    inputs = inputs - translation_matrix.unsqueeze(
        -1
    )  # P = U^T P' - (P \cdot N) N, [B, N, 3, 1]
    inputs = inputs.squeeze(-1)  # P, [1, N, 3]
    return inputs


# def get_spin_axis_matrix(normal_vec):
#     """Calculate the spin-axis matrix.

#     Args:
#         normal_vec (torch.cuda.FloatTensor): the normal vectors for all N points, [1, N, 3].
#     """
#     _, N, _ = normal_vec.shape
#     x = normal_vec[:, :, 0]  # [1, N]
#     y = normal_vec[:, :, 1]  # [1, N]
#     z = normal_vec[:, :, 2]  # [1, N]
#     assert abs(normal_vec).max() <= 1
#     u = torch.zeros(1, N, 3, 3).cuda()
#     denominator = torch.sqrt(1 - z**2)  # \sqrt{1-z^2}, [1, N]
#     u[:, :, 0, 0] = y / denominator
#     u[:, :, 0, 1] = -x / denominator
#     u[:, :, 0, 2] = 0.0
#     u[:, :, 1, 0] = x * z / denominator
#     u[:, :, 1, 1] = y * z / denominator
#     u[:, :, 1, 2] = -denominator
#     u[:, :, 2] = normal_vec

#     # revision for |z| = 1, boundary case.
#     pos = torch.where(abs(z**2 - 1) < 1e-4)[1]
#     u[:, pos, 0, 0] = 1 / np.sqrt(2)
#     u[:, pos, 0, 1] = -1 / np.sqrt(2)
#     u[:, pos, 0, 2] = 0.0
#     u[:, pos, 1, 0] = z[:, pos] / np.sqrt(2)
#     u[:, pos, 1, 1] = z[:, pos] / np.sqrt(2)
#     u[:, pos, 1, 2] = 0.0
#     u[:, pos, 2, 0] = 0.0
#     u[:, pos, 2, 1] = 0.0
#     u[:, pos, 2, 2] = z[:, pos]
#     return u.data

# def get_transformed_point_cloud(points, normal_vec):
#     """Calculate the spin-axis matrix.

#     Args:
#         points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 3].
#         normal_vec (torch.cuda.FloatTensor): the normal vectors for all N points, [1, N, 3].
#     """
#     intercept = torch.mul(points, normal_vec).sum(
#         -1, keepdim=True
#     )  # P \cdot N, [B, N, 1]
#     spin_axis_matrix = get_spin_axis_matrix(normal_vec)  # U, [1, N, 3, 3]
#     translation_matrix = torch.mul(
#         intercept, normal_vec
#     ).data  # (P \cdot N) N, [1, N, 3]
#     new_points = points + translation_matrix  #  P + (P \cdot N) N, [1, N, 3]
#     new_points = new_points.unsqueeze(-1)  # P + (P \cdot N) N, [1, N, 3, 1]
#     new_points = torch.matmul(
#         spin_axis_matrix, new_points
#     )  # P' = U (P + (P \cdot N) N), [1, N, 3, 1]
#     new_points = new_points.squeeze(-1).data  # P', [1, N, 3]
#     return new_points, spin_axis_matrix, translation_matrix

# def get_normal_vector(points):
#     """Calculate the normal vector.

#     Args:
#         points (torch.cuda.FloatTensor): the point cloud with N points, [B, N, 3].
#     """
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points.squeeze(0).detach().cpu().numpy())
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
#     normal_vec = torch.from_numpy(np.asarray(pcd.normals)).float().cuda().unsqueeze(0)
#     return normal_vec

def get_spin_axis_matrix(normal_vec):
    """
    Calculate the spin-axis matrix for each normal vector.
    
    Args:
        normal_vec (torch.Tensor): Unit normal vectors of shape [B, N, 3]
    
    Returns:
        torch.Tensor: Spin-axis matrix of shape [B, N, 3, 3]
    """
    B, N, _ = normal_vec.shape
    device = normal_vec.device

    x = normal_vec[:, :, 0]  # [B, N]
    y = normal_vec[:, :, 1]  # [B, N]
    z = normal_vec[:, :, 2]  # [B, N]

    denominator = torch.sqrt(torch.clamp(1 - z**2, min=1e-8))  # [B, N]

    u = torch.zeros(B, N, 3, 3, device=device)

    u[:, :, 0, 0] = y / denominator
    u[:, :, 0, 1] = -x / denominator
    u[:, :, 0, 2] = 0.0

    u[:, :, 1, 0] = x * z / denominator
    u[:, :, 1, 1] = y * z / denominator
    u[:, :, 1, 2] = -denominator

    u[:, :, 2] = normal_vec

    boundary_mask = (z**2 - 1).abs() < 1e-4  # [B, N]
    b_idx, n_idx = torch.where(boundary_mask)

    sqrt2_inv = 1 / torch.sqrt(torch.tensor(2.0, device=device))

    u[b_idx, n_idx, 0, 0] = sqrt2_inv
    u[b_idx, n_idx, 0, 1] = -sqrt2_inv
    u[b_idx, n_idx, 0, 2] = 0.0

    u[b_idx, n_idx, 1, 0] = z[b_idx, n_idx] * sqrt2_inv
    u[b_idx, n_idx, 1, 1] = z[b_idx, n_idx] * sqrt2_inv
    u[b_idx, n_idx, 1, 2] = 0.0

    u[b_idx, n_idx, 2, 0] = 0.0
    u[b_idx, n_idx, 2, 1] = 0.0
    u[b_idx, n_idx, 2, 2] = z[b_idx, n_idx]

    return u

def get_transformed_point_cloud(points, normal_vec):
    """
    Transform point cloud into local frame aligned with normal vectors.

    Args:
        points (torch.FloatTensor): [B, N, 3] Point cloud batch
        normal_vec (torch.FloatTensor): [B, N, 3] Normal vectors per point

    Returns:
        new_points: [B, N, 3] Points transformed to local frame
        spin_axis_matrix: [B, N, 3, 3] Local frame rotation matrices
        translation_matrix: [B, N, 3] Local translation vectors
    """
    B, N, _ = points.shape
    device = points.device

    # Inner product: (P · N) ∈ [B, N, 1]
    intercept = torch.sum(points * normal_vec, dim=-1, keepdim=True)

    # Translation vector: (P · N) * N ∈ [B, N, 3]
    translation_matrix = intercept * normal_vec

    # Add translation vector: P' = P + (P · N) * N
    new_points = points + translation_matrix  # [B, N, 3]

    # Get rotation matrix: [B, N, 3, 3]
    spin_axis_matrix = get_spin_axis_matrix(normal_vec) 

    # Apply rotation: P'' = U @ P'
    new_points = torch.matmul(
        spin_axis_matrix, new_points.unsqueeze(-1)
    ).squeeze(-1)  # [B, N, 3]

    return new_points, spin_axis_matrix, translation_matrix

def get_normal_vector(points):
    """
    Estimate normals for batched point clouds using Open3D.

    Args:
        points (torch.FloatTensor): [B, N, 3] point clouds

    Returns:
        normal_vecs (torch.FloatTensor): [B, N, 3] normal vectors
    """
    B, N, _ = points.shape
    normal_vecs = []

    for b in range(B):
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pts_np = points[b].detach().cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(pts_np)

        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        normals_np = np.asarray(pcd.normals)
        normal_vecs.append(normals_np)

    # Stack and convert back to torch
    normal_vecs = torch.from_numpy(np.stack(normal_vecs, axis=0)).float().to(points.device)
    return normal_vecs

