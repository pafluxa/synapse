import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.special
import numpy as np
from torch.autograd import Function

# --- Custom Bessel Function with Autograd ---
class BesselIVFunction(Function):
    @staticmethod
    def forward(ctx, v, x):
        v_cpu = v.detach().cpu().numpy()
        x_cpu = x.detach().cpu().numpy()

        iv = scipy.special.iv(v_cpu, x_cpu)
        ivm1 = scipy.special.iv(v_cpu - 1, x_cpu)
        ivp1 = scipy.special.iv(v_cpu + 1, x_cpu)
        ivp_v = scipy.special.ivp(v_cpu, x_cpu, n=1)

        # Save tensors for backward
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
    # Move inputs to CPU for scipy
    v_cpu = v.detach().to('cpu')
    x_cpu = x.detach().to('cpu')

    # If v is scalar and x is tensor, expand v
    if v_cpu.ndim == 0 and x_cpu.ndim > 0:
        v_cpu = v_cpu.expand_as(x_cpu)

    result_cpu = BesselIVFunction.apply(v_cpu, x_cpu)
    return result_cpu.to(x.device)


class LogBessel(nn.Module):
    def __init__(self, dim: int, eps=1e-10, threshold_low=1.0, threshold_high=10.0):
        super().__init__()
        self.dim = dim
        self.nu = dim / 2 - 1
        self.eps = eps

        self.threshold_low = threshold_low
        self.threshold_high = max(threshold_high, 5 * self.nu)

        # Precompute log gamma(nu + 1)
        nu_plus_1 = self.nu + 1
        self.register_buffer('gamma_nu', torch.lgamma(torch.tensor(nu_plus_1)))

    def forward(self, kappa: torch.Tensor) -> torch.Tensor:
        kappa_safe = kappa.clamp(min=self.eps)
        device = kappa.device
        dtype = kappa.dtype

        # Compute exact Bessel and log
        nu_tensor = torch.tensor(self.nu, dtype=dtype, device=device)
        log_iv_exact = torch.log(bessel_iv(nu_tensor, kappa_safe) + self.eps)

        return log_iv_exact

class VMFLoss(nn.Module):
    def __init__(self, dim, learn_mu=True, learn_kappa=True, repulsion_weight=0.2, radius_reg_weight=0.8):
        super().__init__()
        self.dim = dim
        self.repulsion_weight = repulsion_weight
        self.radius_reg_weight = radius_reg_weight

        self.mu = nn.Parameter(torch.randn(dim)) if learn_mu else self.register_buffer("mu", torch.ones(dim))
        self.kappa = nn.Parameter(torch.tensor(1.0)) if learn_kappa else self.register_buffer("kappa", torch.tensor(1.0))

        self.log_bessel_fn = LogBessel(dim=dim)

    def forward(self, x: torch.Tensor):
        x_normalized = F.normalize(x, dim=-1)
        kappa = torch.clamp(self.kappa, min=1e-3)
        mu = F.normalize(self.mu, p=2, dim=-1)

        log_C = ( (self.dim / 2 - 1) * torch.log(kappa)
                  - (self.dim / 2) * torch.log(torch.tensor(2 * math.pi, device=kappa.device))
                  - self.log_bessel_fn(kappa) )

        dot_product = torch.sum(mu * x_normalized, dim=-1)
        nll = -(log_C + kappa * dot_product)
        vmf = nll.mean()

        norms = x.norm(p=2, dim=-1)
        radius_reg = torch.mean((norms - norms.detach().mean()) ** 2)

        batch_size = x.size(0)
        if batch_size > 1:
            cos_sim = torch.matmul(x_normalized, x_normalized.T)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=x.device)
            cos_vals = cos_sim[mask]
            repulsion = torch.mean(F.relu(cos_vals) ** 2)
        else:
            repulsion = torch.tensor(0.0, device=x.device)

        total_loss = vmf + self.radius_reg_weight * radius_reg + self.repulsion_weight * repulsion

        metrics = {
            'sph_vmf': vmf,
            'sph_rad': self.radius_reg_weight * radius_reg,
            'sph_rep': self.repulsion_weight * repulsion
        }

        return total_loss, metrics
