# ---------------------------------------------------------------------
#   SphereClassifier & rotation augmenters
# ---------------------------------------------------------------------
import os
from typing import Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionRadiusGatedMLP(nn.Module):
    """
    x ∈ ℝᴰ → split:
        u = x / ∥x∥₂          (direction, shape [B,D])
        r = log ∥x∥₂          (scalar, shape [B,1])

    A) Direction branch: residual MLP (depth ≥ 4).
    B) Radius branch   : small 2-layer MLP → gating vector g ∈ ℝ^{width}.
    C) Element-wise multiply: h_dir * g   (amplifies norm differences).
    D) FC head → *unnormalised* embedding ℝ^{E}.
    """
    def __init__(self, in_dim: int, embed_dim: int = 8,
                 width: int = 64, depth: int = 6):
        super().__init__()

        # ---- direction branch -------------------------------------
        layers = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth):
            layers += [
                nn.Linear(width, width),
                nn.LayerNorm(width),
                nn.GELU(),
            ]
        self.dir_mlp = nn.Sequential(*layers)

        # ---- radius branch ----------------------------------------
        self.rad_mlp = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.Sigmoid(),           # gating values (0,1)
        )

        # ---- head --------------------------------------------------
        self.head = nn.Linear(width, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=1, keepdim=True).clamp_(1e-8)   # [B,1]
        u    = x / norm                                   # direction
        r    = norm                            # scalar radius

        h_dir = self.dir_mlp(u)                           # [B,width]
        g     = self.rad_mlp(r)                           # [B,width]
        h     = h_dir * g                                 # gated

        z = self.head(h)                                  # [B,E]  (no L2 norm!)
        return z

    @classmethod
    def load(
        cls,
        path: str,
        in_dim: int,
        embed_dim: int = 8,
        width: int = 64,
        depth: int = 6,
        device: str | torch.device = "cpu",
        strict: bool = True,
    ):
        """
        Restore a saved MLP.

        Returns
        -------
        model      : the nn.Module (in eval mode, on `device`)
        meta       : dict with optional fields {"epoch", "val_loss"}
        """
        device = torch.device(device)
        ckpt = torch.load(path, map_location=device)

        # Unwrap if we have a trainer checkpoint
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model = cls(in_dim, embed_dim, width, depth).to(device)
        model.load_state_dict(state, strict=strict)
        model.eval()

        # Gather metadata if present
        meta = {}
        if isinstance(ckpt, dict):
            meta["epoch"] = ckpt.get("epoch")
            meta["val_loss"] = ckpt.get("val_loss")

        return model

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, to_numpy: bool = False):
        """
        Forward pass with **no gradient tracking**.

        Parameters
        ----------
        x         : tensor [B, D]  input vectors
        to_numpy  : bool           if True, returns np.ndarray on CPU

        Returns
        -------
        embedding : tensor [B, E]  (or ndarray) embedding produced by the net
        """
        self.eval()
        emb = self.forward(x.to(next(self.parameters()).device))
        if to_numpy:
            return emb.cpu().numpy()
        return emb

class DirectionRadiusMLP(nn.Module):
    """
    x ∈ ℝᴰ  →  split into:
        u = x / ∥x∥₂           (direction)
        r = log ∥x∥₂           (scalar radius)

    • Process u with a deep residual MLP.
    • Concatenate r after first block so the net *may* use the radius.
    • Output L2-normalised 32-D embedding.
    """

    def __init__(self, in_dim: int, embed_dim: int = 32, width: int = 256, depth: int = 4):
        super().__init__()

        def block():
            return nn.Sequential(
                nn.Linear(width, width),
                nn.LayerNorm(width),
                nn.GELU(),
            )

        # first linear expands from D -> width
        self.in_proj = nn.Linear(in_dim, width)

        # residual tower
        self.tower = nn.Sequential(*[block() for _ in range(depth)])

        # combine with radius scalar (1-d) after tower
        self.out_proj = nn.Linear(width + 1, embed_dim)

    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # split into direction + log-norm ------------------------
        norm = torch.norm(x, dim=1, keepdim=True).clamp(min=1e-8)
        u = x / norm
        r = norm

        # residual MLP on the direction -------------------------
        h = self.in_proj(u)                       # [B, width]
        h = F.gelu(h)
        for layer in self.tower:
            h = h + layer(h)                      # residual

        # concat radius and project -----------------------------
        h = torch.cat([h, r], dim=1)              # [B, width+1]
        z = self.out_proj(h)                      # [B, embed_dim]
        # z = F.normalize(z, p=2, dim=1)            # unit-norm
        return z



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
    def __init__(self, in_dim: int, embed_dim: int = 8, width: int = 8, n_blocks: int = 4):
        super().__init__()

        # Stem: expand channels once
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, width, kernel_size=7, padding=3),
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
        h = self.stem(x)
        h = self.backbone(h)
        h = self.pool(h).squeeze(-1)       # [B, width]
        z = self.head(h)
        z = F.normalize(z, p=2, dim=1)     # unit sphere embedding
        return z
