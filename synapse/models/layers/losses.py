import torch
import torch.nn as nn
import torch.nn.functional as F
import math




class LogBesselApprox(nn.Module):
    def __init__(self, dim: int, eps=1e-10, threshold_low=1.0, threshold_high=10.0):
        super().__init__()
        self.dim = dim
        self.nu = dim / 2 - 1
        self.eps = eps
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.gamma_nu = math.lgamma(self.nu + 1)

    @staticmethod
    def log_bessel_approx(kappa: torch.Tensor, v: int,
        threshold_low: float = 1.0,
        threshold_high: float = 10.0,
        eps: float = 1e-8):
        """
        Differentiable approximation of log(I_v(kappa)) for any v.
        """

        # Safe kappa
        kappa_safe = torch.clamp(kappa, min=eps)

        # Large kappa approximation: log(I_v(k)) ≈ k - 0.5 * log(2πk)
        log_iv_large = kappa_safe - 0.5 * torch.log(2 * torch.pi * kappa_safe)

        # Small kappa approximation: log(I_v(k)) ≈ v*log(k/2) - logΓ(v+1)
        log_iv_small = v * torch.log(kappa_safe / 2 + eps) - torch.lgamma(torch.tensor(v + 1.0, device=kappa.device))

        # Smooth interpolation (sigmoid blending)
        w_low = torch.sigmoid(5 * (threshold_low - kappa_safe))
        w_high = torch.sigmoid(5 * (kappa_safe - threshold_high))
        w_mid = 1 - w_low - w_high

        # 0 in the middle = neutral estimate
        log_iv = w_low * log_iv_small + w_mid * 0 + w_high * log_iv_large

        return log_iv

    def forward(self, kappa: torch.Tensor) -> torch.Tensor:
        kappa_safe = kappa.clamp(min=self.eps)
        # iv = torch.special.iv(self.nu, kappa_safe)
        # log_iv = torch.log(iv + self.eps)
        log_iv = self.log_bessel_approx(kappa, self.dim)

        large_kappa_approx = kappa_safe - 0.5 * torch.log(2 * math.pi * kappa_safe + self.eps)
        small_kappa_approx = self.nu * torch.log(kappa_safe / 2 + self.eps) - self.gamma_nu

        w_low = torch.sigmoid(5 * (self.threshold_low - kappa_safe))
        w_high = torch.sigmoid(5 * (kappa_safe - self.threshold_high))
        w_exact = 1 - w_low - w_high

        return w_low * small_kappa_approx + w_exact * log_iv + w_high * large_kappa_approx


class VMFLoss(nn.Module):
    def __init__(self, dim,
            learn_mu=True,
            learn_kappa=True,
            repulsion_weight=0.2,
            radius_reg_weight=0.8):
        super().__init__()
        self.dim = dim
        self.repulsion_weight = repulsion_weight
        self.radius_reg_weight = radius_reg_weight

        # Mean direction μ
        if learn_mu:
            self.mu = nn.Parameter(torch.randn(dim))
        else:
            self.register_buffer("mu", torch.ones(dim))

        # Concentration κ
        if learn_kappa:
            self.kappa = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("kappa", torch.tensor(1.0))

        self.log_bessel_fn = LogBesselApprox(dim=dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): (batch, dim), unnormalized vectors
        Returns:
            loss (Tensor)
        """
        # Normalize to unit sphere, then scale to learnable radius
        x_normalized = F.normalize(x, dim=-1)

        # μ normalized
        kappa = torch.clamp(self.kappa, min=1e-3)

        # log normalizer
        log_C = (
            (self.dim / 2 - 1) * torch.log(kappa)
            - (self.dim / 2) * torch.log(torch.tensor(2 * math.pi))
            - self.log_bessel_fn(kappa)
        )

        # Dot product κ μᵀx
        dot_product = torch.sum(self.mu * x_normalized, dim=-1)

        # vMF negative log-likelihood
        nll = -(log_C + kappa * dot_product)

        # Radius regularization: encourage all vectors to lie on the same-radius sphere
        norms = x.norm(p=2, dim=-1)
        radius_reg = torch.mean((norms - norms.detach().mean()) ** 2)

        # Repulsion loss: discourage vector collapse (maximize spread)
        similarity_matrix = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
        identity = torch.eye(x.size(0), device=x.device)
        repulsion = ((similarity_matrix - identity) ** 2).mean()

        # Total loss
        vmf = nll.mean()
        total_loss = vmf \
                   + self.radius_reg_weight * radius_reg \
                   + self.repulsion_weight * repulsion

        metrics = {
            'sph_vmf': vmf,
            'sph_rad': self.radius_reg_weight * radius_reg,
            'sph_rep': self.repulsion_weight * repulsion
        }

        return total_loss, metrics
