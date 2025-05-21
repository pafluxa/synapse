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
        epsilon: float = 0.001,
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


# ---------------- sphere-classifier model ----------------------------
class SphereClassifier(nn.Module):
    """
    Binary classifier that decides whether a *codec* lies on *some* sphere.

    Output
    ------
    logits : shape [B] – feed to `torch.sigmoid` (0 = on-sphere, 1 = off-sphere)
    h      : shape [B, hidden_dim] – embedding used for contrastive learning
    """
    def __init__(self, codec_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(codec_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    # -- forward -------------------------------------------------------
    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        logits = self.classifier(h).squeeze(-1)          # [B]
        return logits, h

    # -- helpers -------------------------------------------------------
    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent embedding without touching the classifier head."""
        return self.encoder(x)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
        """Return 0 / 1 decisions."""
        logits, _ = self.forward(x)
        return (logits > thresh).long()

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
        model = cls(codec_dim, hidden_dim).to(device)
        ckpt = torch.load(path, map_location=device)

        if isinstance(ckpt, dict) and "model" in ckpt:      # ← unwrap
            ckpt = ckpt["model"]

        model.load_state_dict(ckpt, strict=strict)
        model.eval()
        return model
