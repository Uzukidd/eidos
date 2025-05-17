import torch
import pytorch3d
    
@torch.no_grad()
def eig_vector(data, K):
    b, n, _ = data.shape
    _, idx, _ = pytorch3d.ops.knn_points(data, data, K=K)  # idx (b,n,K)

    idx0 = (
        torch.arange(0, b, device=data.device)
        .reshape((b, 1))
        .expand(-1, n * K)
        .reshape((1, b * n * K))
    )
    idx1 = (
        torch.arange(0, n, device=data.device)
        .reshape((1, n, 1))
        .expand(b, n, K)
        .reshape((1, b * n * K))
    )
    idx = idx.reshape((1, b * n * K))
    idx = torch.cat([idx0, idx1, idx], dim=0)  # (3, b*n*K)
    # print(b, n, K, idx.shape)
    ones = torch.ones(idx.shape[1], dtype=bool, device=data.device)
    A = torch.sparse_coo_tensor(idx, ones, (b, n, n))
    A = A.to(torch.uint8).to_dense().to(torch.bool)  # (b,n,n)
    A = A | A.transpose(1, 2)
    A = A.float()
    deg = torch.diag_embed(torch.sum(A, dim=2))
    laplacian = deg - A
    u = torch.zeros((laplacian.size(0), laplacian.size(1)), device=data.device)
    v = torch.zeros(
        (laplacian.size(0), laplacian.size(1), laplacian.size(1)),
        device=data.device,
    )
    for i in range(0, laplacian.size(0)):
        u_, v_ = torch.linalg.eig(laplacian[i])  # (b,n,n)
        u[i] = u_.real
        v[i] = v_.real
    return v, laplacian, u