import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def hypersphere_autoencoder_loss(x: torch.Tensor,
                               alpha: float = 0.1,
                               beta: float = 0.05,
                               target_radius: float = 1.0):
    """
    Numerically stable version with:
    - Clipped uniformity gradients
    - Radius control via soft constraints
    - Normalized loss scales
    """
    norms = x.norm(p=2, dim=-1, keepdim=True)
    norms_no_grad = norms.clone().detach()
    # 1. Radius control (softly encourages target radius)
    radius_loss = ((norms - norms_no_grad) ** 2).mean()

    # 2. Normalized directions (stop gradient for norm)
    x_normed = x / (norms_no_grad + 1e-7)

    # Pairwise squared Euclidean distances
    pairwise_dists = torch.cdist(x_normed, x_normed, p=2)  # [N, N]

    # Mask out self-interactions
    mask = ~torch.eye(len(x), dtype=torch.bool, device=x.device)
    valid_dists = pairwise_dists[mask]

    # Inverse-square repulsion (clipped for stability)
    repulsion = (1.0 / (valid_dists.pow(2) + 1e-8)).mean()

    return alpha * radius_loss, beta * repulsion

def hypersphere_autoencoder_loss_2(x: torch.Tensor, alpha: float = 1.0, beta: float = 1.0):
    """
    Improved version with:
    - Stable radius control via sigmoid activation
    - Stronger uniformity enforcement
    - Balanced gradient flow
    """
    norms = x.norm(p=2, dim=-1)
    mean_norm = norms.mean().detach()  # Detach to avoid pushing mean_norm to zero

    # 1. Radius stabilization (sigmoid-shaped loss)
    radius_loss = F.mse_loss(norms, mean_norm * torch.ones_like(norms))

    # 2. Normalize for angular loss (stop gradient for norm to avoid conflict)
    x_normed = F.normalize(x, p=2, dim=-1)

    # 3. Uniformity loss (maximize mean minimal angle)
    cos_sim = x_normed @ x_normed.T
    cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)

    # Exclude self-similarities
    mask = ~torch.eye(len(x), dtype=torch.bool, device=x.device)
    uniformity_loss = torch.exp(5 * cos_sim[mask]).mean()  # Strong repulsion

    return alpha * radius_loss, beta * uniformity_loss


# torch_pi = torch.tensor(math.pi)

# # --- Custom Bessel Function with Autograd ---
# from torch.autograd import Function

# class BesselIVFunction(Function):
#     @staticmethod
#     def forward(ctx, v, x):
#         v_cpu = v.detach().cpu().numpy()
#         x_cpu = x.detach().cpu().numpy()

#         iv = scipy.special.iv(v_cpu, x_cpu)
#         ivm1 = scipy.special.iv(v_cpu - 1, x_cpu)
#         ivp1 = scipy.special.iv(v_cpu + 1, x_cpu)
#         ivp_v = scipy.special.ivp(v_cpu, x_cpu, n=1)

#         ctx.save_for_backward(v, x)
#         ctx.ivm1 = torch.tensor(ivm1, dtype=x.dtype, device='cpu')
#         ctx.ivp1 = torch.tensor(ivp1, dtype=x.dtype, device='cpu')
#         ctx.ivp_v = torch.tensor(ivp_v, dtype=x.dtype, device='cpu')

#         return torch.tensor(iv, dtype=x.dtype, device='cpu')

#     @staticmethod
#     def backward(ctx, grad_output):
#         v, x = ctx.saved_tensors
#         ivm1 = ctx.ivm1.to(grad_output.device)
#         ivp1 = ctx.ivp1.to(grad_output.device)
#         ivp_v = ctx.ivp_v.to(grad_output.device)

#         grad_x = 0.5 * (ivm1 + ivp1)
#         grad_v = ivp_v

#         return grad_output * grad_v, grad_output * grad_x


# def bessel_iv(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#     v_cpu = v.detach().to('cpu')
#     x_cpu = x.detach().to('cpu')
#     if v_cpu.ndim == 0 and x_cpu.ndim > 0:
#         v_cpu = v_cpu.expand_as(x_cpu)
#     result_cpu = BesselIVFunction.apply(v_cpu, x_cpu)
#     return result_cpu.to(x.device)


# class LogBessel(nn.Module):
#     def __init__(self, dim: int, eps=1e-10):
#         super().__init__()
#         self.dim = dim
#         self.nu = dim / 2 - 1
#         self.eps = eps

#     def forward(self, kappa: torch.Tensor) -> torch.Tensor:
#         kappa_safe = kappa.clamp(min=self.eps)
#         device, dtype = kappa.device, kappa.dtype
#         nu_tensor = torch.tensor(self.nu, dtype=dtype, device=device)
#         return torch.log(bessel_iv(nu_tensor, kappa_safe) + self.eps)

# def hypersphere_uniformity_loss(x: torch.Tensor) -> torch.Tensor:
#     """
#     Encourages uniform distribution on a hypersphere *without* penalizing antipodal points.
#     Minimizing this pushes points away from their nearest neighbors (not just orthogonal).
#     """
#     x = F.normalize(x, p=2, dim=-1)  # Work on unit sphere
#     cos_sim = x @ x.T  # [N, N] cosine similarities

#     # Mask out self-similarities and avoid numerical instability
#     mask = ~torch.eye(len(x), dtype=torch.bool, device=x.device)
#     cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)

#     # Penalize *only* nearby points (cos_sim close to 1 or -1)
#     # Using log(1 - cos_sim^2) to treat antipodal points equivalently
#     uniformity_loss = -torch.log(1.0 - cos_sim[mask].pow(2) + 1e-6).mean()
#     return uniformity_loss

# def hypersphere_autoencoder_loss(x: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
#     """
#     Loss for learning a stable, uniform hypersphere without explicit normalization.
#     Properties:
#     - Lets the network learn the natural radius.
#     - Prevents collapse/expansion via repulsion and radius stability.
#     - Encourages uniformity via angular dispersion.

#     Args:
#         x: Latent vectors (N, D), unnormalized.
#         alpha: Weight for uniformity term (default: 0.1).
#     Returns:
#         Scalar loss (minimize this).
#     """
#     norms = x.norm(p=2, dim=-1)  # [N]

#     # 1. Radius stability term (prevents collapse/expansion)
#     # Encourages stable but non-zero radius (log avoids hard constraints)
#     radius_loss = torch.log1p((norms - norms.mean()).abs()).mean()

#     # 2. Uniformity term (encourages points to spread out angularly)
#     # Works on *normalized* vectors (angular part only)
#     x_normalized = F.normalize(x, p=2, dim=-1)
#     uniformity_loss = hypersphere_uniformity_loss(x_normalized)

#     return radius_loss, uniformity_loss


# def spherical_knn_entropy(x: torch.Tensor, k: int = 1, eps: float = 1e-8) -> torch.Tensor:
#     """
#     Estimates the negative entropy of points on a unit hypersphere using kNN distances.
#     Minimizing this value will encourage uniform distribution of points.

#     Based on Kozachenko-Leonenko entropy estimator adapted for spherical geometry.

#     Args:
#         x (Tensor): Normalized data points of shape (N, D), assumed to lie on a unit hypersphere.
#         k (int): Number of nearest neighbors to consider (default = 1).
#         eps (float): Small constant to avoid numerical issues.

#     Returns:
#         Tensor: Scalar value proportional to negative entropy (minimize this for uniform distribution).
#     """
#     N, D = x.shape
#     device = x.device

#     # Compute pairwise angular distances (more stable than chord distance for small angles)
#     cos_sim = x @ x.T
#     cos_sim = cos_sim.clamp(-1 + eps, 1 - eps)  # Ensure numerical stability
#     angles = torch.acos(cos_sim)  # [N, N] matrix of angles

#     # Mask self-distances
#     angles.fill_diagonal_(float('inf'))

#     # Get k-th smallest angle for each point
#     knn_dists, _ = torch.topk(angles, k=k, dim=1, largest=False)
#     eps_i = knn_dists[:, -1]  # [N] vector of k-th NN distances

#     # Spherical entropy estimator (negative because we typically want to minimize loss)
#     # The constant terms don't affect optimization but are included for completeness
#     sphere_surface_area = (D * torch.pi**(D/2)) / torch.tensor(math.gamma(D/2 + 1))
#     entropy = torch.log(eps_i).mean() + torch.log(sphere_surface_area * (N-1))

#     # Return negative entropy (so minimization encourages uniformity)
#     return -entropy


# class EntropyLoss(nn.Module):
#     def __init__(self, dim, learn_mu=True, learn_kappa=True,
#                  repulsion_weight=0.5, radius_reg_weight=0.5):
#         super().__init__()
#         self.dim = dim
#         self.repulsion_weight = repulsion_weight
#         self.radius_reg_weight = radius_reg_weight

#         self.log_bessel_fn = LogBessel(dim=dim)
#         self.repulsion_strength = nn.Parameter(torch.tensor(0.01))

#     def forward(self, x: torch.Tensor):

#         # penalize norms away from average
#         norms2 = x.norm(p=2, dim=-1)
#         radius_reg = torch.mean((norms2 - norms2.detach().mean())**2)
#         # entropy estimation
#         norms = x.norm(p=2, dim=-1, keepdim=True)
#         x_unit = x / norms.detach()
#         h = -spherical_knn_entropy(x_unit, k=8)

#         # adds electrostatic-style repulsion
#         diff = x_unit.unsqueeze(1) - x_unit.unsqueeze(0)
#         dist_sq = (diff ** 2).sum(-1) + 1e-6
#         batch_size = x.size(0)
#         mask = ~torch.eye(batch_size, dtype=torch.bool, device=x.device)
#         inv_dist = 0.01 / (dist_sq[mask] + 1e-6)
#         repulsion = (inv_dist).mean()

#         # total_loss = h + self.radius_reg_weight * radius_reg + self.repulsion_weight * repulsion

#         metrics = {
#             'sph_ent': h,
#             'sph_rad': radius_reg,
#             'sph_rep': repulsion
#         }

#         return h, radius_reg, repulsion, metrics
