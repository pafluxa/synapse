# ---------------------------------------------------------------------
#  TabularBERT  –  now with:
#     • save(path)                     ← checkpoints at any moment
#     • @classmethod load(path, cfg)   ← restore weights
#     • encode(x_num, x_cat)           ← forward pass, no grads/decoder
# ---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import digamma
from einops import rearrange

from synapse.utils.config_parser import RunConfiguration
from synapse.layers.embeddings import CategoricalEmbedding
from synapse.layers.embeddings import NumericalEmbedding
from synapse.layers.feature_encoders import Zwei


def _spherical_cap_area(cos_phi: torch.Tensor,
                        p: int,
                        n_steps: int = 64) -> torch.Tensor:
    """
    Compute the area of a spherical cap of half-angle φ=arccos(cos_phi)
    on the unit (p-1)-sphere via numeric integration.
    Returns a tensor of shape (n,), giving S_phi > 0.
    """
    # φ in [0, π]
    phi = torch.acos(cos_phi)                   # (n,)

    # build a uniform grid u in [0,1], then θ = u*φ
    u = torch.linspace(0.0, 1.0, n_steps,
                       device=cos_phi.device).unsqueeze(0)   # (1, m)
    theta = u * phi.unsqueeze(1)                # (n, m)

    # integrand sin^(p-2)(θ)
    integrand = torch.sin(theta).pow(p - 2)     # (n, m)

    # trapezoidal approximate ∫₀^φ sin^(p-2)(θ) dθ
    integral = torch.trapz(integrand, theta, dim=1)  # (n,)

    # normalization constant C_p
    #   C_p = 2·π^(p/2) / (√π · Γ((p-1)/2))
    half = torch.tensor(0.5, device=cos_phi.device)
    C_p = (2 * torch.pi**(p/2)
        / (torch.sqrt(torch.tensor(torch.pi))
              * torch.exp(torch.lgamma((p - 1) * half))))

    return C_p * integral                         # (n,) all > 0

def knn_entropy(X: torch.Tensor,
                k: int,
                integral_steps: int = 64) -> torch.Tensor:
    """
    kNN entropy estimator (Eq. 9) on the unit (p-1)-sphere,
    using a positive-definite numeric integral for S_phi.
    """
    n, p = X.shape

    # 1) pairwise cosines, mask out self
    C = X @ X.t()
    idx = torch.arange(n, device=X.device)
    C[idx, idx] = -1.0

    # 2) k-th nearest cosine
    topk_vals, _ = C.topk(k, dim=1, largest=True, sorted=True)
    cos_phi = topk_vals[:, -1].clamp(-1 + 1e-7, 1 - 1e-7)  # (n,)

    # 3) compute S_phi via angle integral
    S_phi = _spherical_cap_area(cos_phi, p, n_steps=integral_steps)

    # 4) entropy H = E[ln(n·S_phi)] − ψ(k)
    t_k = torch.tensor(k, device=X.device)
    H = torch.log(n * S_phi).mean() - digamma(t_k)
    return H


# ---------- IMPORTANT: keep your other imports / helpers here --------
# from .something import ...

class TabularBERT(nn.Module):
    """
    Masked-reconstruction transformer for mixed numerical/categorical data.

    Added methods
    -------------
    save(path)                     – writes only model_state_dict
    load(path, cfg, device='cpu')  – class-method factory that restores weights
    encode(x_num, x_cat)           – returns latent code (no gradients, eval mode)
    """

    def __init__(self, config: RunConfiguration):
        super().__init__()

        self.d_model = config.embedding_dim
        self.nhead = config.num_heads
        self.num_layers = config.num_layers
        self.codec_dim = config.codec_dim

        # Tokenization setup
        self.depths = config.numerical_depths
        self.max_depth = max(self.depths)
        self.cat_dims = config.categorical_dims
        self.num_numerical = config.num_numerical
        self.num_features = self.max_depth * self.num_numerical + len(config.categorical_dims)

        # Embedding layers
        self.num_tokenizer = Zwei(config.numerical_ranges, self.depths)
        self.num_embedder = NumericalEmbedding(self.d_model, self.depths)
        self.cat_embedder = CategoricalEmbedding(self.d_model, self.cat_dims, min_emb_dim=self.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        # VAE bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(self.d_model * self.num_features, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LayerNorm(100),
            nn.Linear(100, self.codec_dim)
        )

        # Decoder
        self.decoder_expand = nn.Sequential(
            nn.Linear(self.codec_dim, 100),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, self.d_model * self.num_features)
        )
        self.log_R = torch.nn.Parameter(torch.zeros(1))  # Learn log(R)
        self.mse_threshold = 0.1  # Phase 1 -> Phase 2 trigger
        self.norm_var_threshold = 0.1  # Phase 2 -> Phase 3 trigger
        # phase counters
        self.phase_2_epochs = -20
        self.phase_3_epochs = -20

    def forward(self, x_num, x_cat):
        batch_size = x_num.size(0)

        # Embeddings
        cat_emb = self.cat_embedder(x_cat)
        num_tok = self.num_tokenizer(x_num)
        num_emb = self.num_embedder(num_tok)

        # Combine features
        cat_emb = rearrange(cat_emb, 's b e -> b s e')
        num_emb = rearrange(num_emb, 's b n e -> b (s n) e')
        combined_embd = torch.cat([num_emb, cat_emb], dim=1)

        # Encoding
        encoded = self.encoder(combined_embd)

        # bottleneck
        flattened = encoded.view(batch_size, -1)
        codec = self.bottleneck(flattened)
        codec += 0.01 * torch.randn_like(codec)

        # Decoding
        expanded = self.decoder_expand(codec)
        decoded_features = expanded.view(batch_size, self.num_features, self.d_model)

        return codec, decoded_features

    def loss(self, outputs, targets, mask, w1, w2, k: torch.Tensor = torch.tensor(4)):
        z, decoded = outputs
        x_num, x_cat = targets
        # Reconstruction loss
        num_tok = self.num_tokenizer(x_num)
        num_emb = self.num_embedder(num_tok)
        num_emb = rearrange(num_emb, 's b n e -> b (s n) e')
        cat_emb = self.cat_embedder(x_cat)
        cat_emb = rearrange(cat_emb, 's b e -> b s e')
        cmb_emb = torch.cat([num_emb, cat_emb], dim=1)

        metrics = {
            'loss': 0.0,
            'mse_loss': 0.0,
            'sph_uni': 0.0,
            'sph_rad': 0.0,
            'var_norm': 1e4,
            'mean_norm': 1e4
        }
        norms = torch.norm(z, dim=-1)
        var_norm = torch.var(norms)
        avg_norm = torch.mean(norms)
        z_unit = F.normalize(z, p=2, dim=1, eps=1e-8)

        # ------ Phase 1: Reconstruction only ------
        mse = torch.mean((decoded - cmb_emb)**2)

        # ------ Phase 2: Gradually enforce spherical constraint ------
        R = avg_norm.detach()
        spherical_loss = torch.mean((norms - R)**2)

        # ------ Phase 3: Enforce uniformity ------
        # cos_sim = z_unit @ z_unit.T
        # mask = ~torch.eye(len(z), dtype=bool, device=z.device)
        # ul = torch.mean(cos_sim[mask])
        uniformity_loss = torch.nn.functional.softplus(-knn_entropy(z_unit, k))

        total_loss = mse + w1 * spherical_loss + w2 * uniformity_loss
        metrics['loss'] = total_loss.item()
        metrics['mse_loss'] = mse.item()
        metrics['sph_rad'] = (w1 * spherical_loss).item()
        metrics['sph_uni'] = (w2 * uniformity_loss).item()
        metrics['var_norm'] = var_norm.item()
        metrics['mean_norm'] = avg_norm.item()

        return total_loss, metrics

    # ------------------- ADDED: encode ---------------------------------
    @torch.inference_mode()
    def encode(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Quickly obtain the latent code **without** gradient tracking
        and **without** running the decoder.

        Example
        -------
        >>> z = model.encode(x_num, x_cat)   # z shape: [B, codec_dim]
        """
        self.eval()
        z, _ = self.forward(x_num, x_cat)
        return z

    # ------------------- ADDED: save -----------------------------------
    def save(self, path: str) -> None:
        """
        Save just the model weights (state_dict).  Directory is created if needed.

        >>> model.save("checkpoints/my_run/epoch_12.pt")
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    # ------------------- ADDED: load -----------------------------------
    @classmethod
    def load(cls, path: str, cfg, device="cpu") -> "TabularBERT":
        model = cls(cfg).to(device)

        checkpoint = torch.load(path, map_location=device)

        # ⬇⬇ unwrap if necessary
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            checkpoint = checkpoint["model_state"]

        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        return model
