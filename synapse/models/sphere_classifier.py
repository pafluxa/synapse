# ---------------------------------------------------------------------
#   SphereClassifier & rotation augmenters
# ---------------------------------------------------------------------
import os
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group


# ---------------- rotation helpers -----------------------------------
class SVDRotationAugmenter:
    """
    QR-based random rotation + SVD-controlled perturbation (heavier).
    """
    def __init__(
        self,
        dim: int,
        epsilon: float = 0.01,
        min_singular: float = 0.01,
        max_singular: float = 10.0,
    ):
        self.dim, self.epsilon = dim, epsilon
        self.min_singular, self.max_singular = min_singular, max_singular

    def _random_rotation(self, device):
        R = torch.from_numpy(special_ortho_group.rvs(self.dim)).float().to(device)
        return R

    def _perturb(self, R):
        U, S, Vh = torch.linalg.svd(R)
        S = torch.clamp(S * (1 + torch.randn_like(S) * self.epsilon),
                        self.min_singular, self.max_singular)
        return U @ torch.diag(S) @ Vh

    def generate_pair(self, device: str = "cuda"):
        R = self._random_rotation(device)
        P = self._perturb(R)
        return R, P


class _ResBlock(nn.Module):
    """Depth-wise 1-D residual block: Conv → BN → ReLU → Conv → BN + skip."""
    def __init__(self, ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(ch)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.leaky_relu(out + x)            # residual add


class RotationConvNet(nn.Module):
    """
    Deeper 1-D CNN backbone → global pooling → FC head.

    depth = 3 * n_blocks  (default 9 convs)
    """
    def __init__(self, in_dim: int, embed_dim: int = 32, width: int = 128, n_blocks: int = 4):
        super().__init__()

        # Stem: expand channels once
        self.stem = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=7, padding=3),
            nn.BatchNorm1d(width),
            nn.LeakyReLU(inplace=True),
        )

        # Stack residual blocks with increasing dilation
        blocks = []
        for i in range(n_blocks):
            k = 3 if i % 2 == 0 else 5               # alternate kernel sizes
            d = 2 ** (i % 3)                         # 1,2,4 dilations cycle
            blocks.append(_ResBlock(width, k, dilation=d))
        self.backbone = nn.Sequential(*blocks)

        # Global average over sequence length (D)
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Projection to embedding
        self.head = nn.Sequential(
            nn.Linear(width, width),
            nn.LeakyReLU(inplace=True),
            nn.Linear(width, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                 # [B, 1, D]
        h = self.stem(x)
        h = self.backbone(h)
        h = self.pool(h).squeeze(-1)       # [B, width]
        z = self.head(h)
        z = F.normalize(z, p=2, dim=1)     # unit sphere embedding
        return z


class _RotationConvNet(nn.Module):
    """
    1-D CNN → GlobalAvgPool → FC → 32-D embedding (default).
    * kernel sizes 3,5,7 capture angular structure
    * global pooling retains signal magnitude implicitly
    """

    def __init__(self, in_dim: int, embed_dim: int = 8, width: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, width, 3, padding='same'),
            nn.Conv1d(width, width, 3, padding=1),
            nn.BatchNorm1d(width),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(width, width, 5, padding=2),
            nn.Conv1d(width, width, 5, padding=2),
            nn.BatchNorm1d(width),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(width, width, 7, padding=3),
            nn.Conv1d(width, width, 7, padding=3),
            nn.BatchNorm1d(width),
            nn.LeakyReLU(inplace=True),
        )

        # global average over the length dimension (in_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(width, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),   # final embedding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, D]  →  [B, 1, D]
        x = x.unsqueeze(1)
        h = self.features(x)
        h = self.pool(h).squeeze(-1)           # [B, width]
        z = self.head(h)                       # [B, embed_dim]
        return z

    # -- helpers -------------------------------------------------------
    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward(x)
        return z

    # -- save / load ---------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)


    @classmethod
    def load(
        cls,
        path: str,
        codec_dim: int,
        device="cpu",
        hidden_dim: int = 128,
        strict: bool = True):
        """
        Restore weights.  Accepts either a *raw* state-dict or the trainer's
        checkpoint wrapper that has a ``"model"`` key.
        """
        model = cls(codec_dim).to(device)
        ckpt = torch.load(path, map_location=device)

        if isinstance(ckpt, dict) and "model" in ckpt:      # ← unwrap
            ckpt = ckpt["model"]

        model.load_state_dict(ckpt, strict=strict)
        model.eval()
        return model
