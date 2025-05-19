import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.special

torch_pi = torch.tensor(math.pi)

# --- Custom Bessel Function with Autograd ---
from torch.autograd import Function

class BesselIVFunction(Function):
    @staticmethod
    def forward(ctx, v, x):
        v_cpu = v.detach().cpu().numpy()
        x_cpu = x.detach().cpu().numpy()

        iv = scipy.special.iv(v_cpu, x_cpu)
        ivm1 = scipy.special.iv(v_cpu - 1, x_cpu)
        ivp1 = scipy.special.iv(v_cpu + 1, x_cpu)
        ivp_v = scipy.special.ivp(v_cpu, x_cpu, n=1)

        ctx.save_for_backward(v, x)
        ctx.ivm1 = torch.tensor(ivm1, dtype=x.dtype, device='cpu')
        ctx.ivp1 = torch.tensor(ivp1, dtype=x.dtype, device='cpu')
        ctx.ivp_v = torch.tensor(ivp_v, dtype=x.dtype, device='cpu')

        return torch.tensor(iv, dtype=x.dtype, device='cpu')

    @staticmethod
    def backward(ctx, grad_output):
        v, x = ctx.saved_tensors
        ivm1 = ctx.ivm1.to(grad_output.device)
        ivp1 = ctx.ivp1.to(grad_output.device)
        ivp_v = ctx.ivp_v.to(grad_output.device)

        grad_x = 0.5 * (ivm1 + ivp1)
        grad_v = ivp_v

        return grad_output * grad_v, grad_output * grad_x


def bessel_iv(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    v_cpu = v.detach().to('cpu')
    x_cpu = x.detach().to('cpu')
    if v_cpu.ndim == 0 and x_cpu.ndim > 0:
        v_cpu = v_cpu.expand_as(x_cpu)
    result_cpu = BesselIVFunction.apply(v_cpu, x_cpu)
    return result_cpu.to(x.device)


class LogBessel(nn.Module):
    def __init__(self, dim: int, eps=1e-10):
        super().__init__()
        self.dim = dim
        self.nu = dim / 2 - 1
        self.eps = eps

    def forward(self, kappa: torch.Tensor) -> torch.Tensor:
        kappa_safe = kappa.clamp(min=self.eps)
        device, dtype = kappa.device, kappa.dtype
        nu_tensor = torch.tensor(self.nu, dtype=dtype, device=device)
        return torch.log(bessel_iv(nu_tensor, kappa_safe) + self.eps)


def spherical_knn_entropy(x: torch.Tensor, k: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Estimate entropy of points on a unit hypersphere using Kozachenko–Leonenko (KL) method.

    Args:
        x (Tensor): Normalized data points of shape (N, D), assumed to lie on a unit hypersphere.
        k (int): Number of nearest neighbors to consider (default = 1).
        eps (float): Small constant to avoid log(0).

    Returns:
        Tensor: Estimated differential entropy (scalar).
    """
    N, D = x.shape
    device = x.device

    # Compute cosine similarity
    cos_sim = x @ x.T
    cos_sim = cos_sim.clamp(-1 + eps, 1 - eps)

    # Convert to squared chord distance on sphere
    chord_sq = 2 - 2 * cos_sim

    # Mask diagonal by setting to large value without in-place op
    mask = torch.eye(N, device=device, dtype=torch.bool)
    chord_sq_masked = chord_sq.masked_fill(mask, float('inf'))

    # Find k-th nearest neighbor distance
    knn_dists, _ = torch.topk(chord_sq_masked, k=k, dim=1, largest=False)
    eps_i = knn_dists[:, -1] + eps

    # Entropy estimate
    entropy = ((D - 1) * torch.log(eps_i)).mean()
    return entropy


class VMFLoss(nn.Module):
    def __init__(self, dim, learn_mu=True, learn_kappa=True,
                 repulsion_weight=0.2, radius_reg_weight=0.8):
        super().__init__()
        self.dim = dim
        self.repulsion_weight = repulsion_weight
        self.radius_reg_weight = radius_reg_weight

        self.mu = nn.Parameter(torch.randn(dim)) if learn_mu else self.register_buffer('mu', torch.randn(dim))
        self.kappa = nn.Parameter(torch.tensor(1.0)) if learn_kappa else self.register_buffer('kappa', torch.tensor(1.0))

        self.log_bessel_fn = LogBessel(dim=dim)
        self.repulsion_strength = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor):

        # entropy estimation
        x_norm = F.normalize(x, p=2, dim=-1)
        h = -spherical_knn_entropy(x_norm, k=3)
        # penalize norms away from average
        norms = x.norm(p=2, dim=-1)
        radius_reg = torch.mean((norms - norms.detach().mean())**2)

        # adds electrostatic-style repulsion
        norms = x.norm(p=2, dim=-1, keepdim=True)
        x_unit = x / norms.detach()
        diff = x_unit.unsqueeze(1) - x_unit.unsqueeze(0)
        dist_sq = (diff ** 2).sum(-1) + 1e-6
        batch_size = x.size(0)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=x.device)
        inv_dist = 1.0 / (dist_sq[mask] + 1e-6)
        repulsion = F.sigmoid(self.repulsion_strength) * (inv_dist.mean())

        total_loss = h + radius_reg + self.repulsion_weight * repulsion

        metrics = {
            'sph_vmf': h,
            'sph_rad': radius_reg,
            'sph_rep': self.repulsion_weight * repulsion
        }

        return total_loss, metrics
