from pathlib import Path
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse.utils.config_parser import RunConfiguration
from synapse.layers.embeddings import MixedFeatureEmbedding


class MaskedTransformerAutoencoder(nn.Module):
    """Masked‑token **encoder–decoder** auto‑encoder for mixed numerical + categorical inputs.

    Why re‑introduce a decoder?
    ---------------------------
    When individual tokens (binary decisions, one‑hot categorical IDs) carry
    little mutual information, a *pure* encoder that only sees the **masked**
    sequence cannot possibly infer the hidden value — it will converge to the
    random‑guess floor and the loss plateaus.

    The remedy is a lightweight *sequence‑to‑sequence* setup:

    •  **Encoder** gets the **full, uncorrupted** sequence (so the information
       *is* somewhere in the model).
    •  **Decoder** sees the *masked* sequence and must copy the missing pieces
       from the encoder’s memory.
    •  We still compute the loss **only** on the masked positions, so training
       cannot cheat by copying its own input.

    Validation is deterministic by default so that metrics are stable, but the
    masking probability can be overridden at call‑time.
    """

    # ---------------------------------------------------------------------
    # Init
    # ---------------------------------------------------------------------
    def __init__(self, cfg: RunConfiguration):
        super().__init__()

        # ---------- feature encoders ----------
        # self.numerical_encoder = NumericalFeatureEncoder(cfg.numerical_ranges, cfg.numerical_depths)
        self.cat_cardinalities = cfg.categorical_dims
        self.num_cat = len(self.cat_cardinalities)
        self.mask_prob = float(cfg.mask_prob)

        # ---------- vocabulary ----------
        self.num_numeric_tokens: int = cfg.num_numerical  # = n_features * max_depth
        self.numeric_vocab_size: int = 2  #self.num_numeric_tokens * 2          # 0/1 choices

        self.cat_offset = self.numeric_vocab_size
        self.cat_vocab_size = sum(self.cat_cardinalities)
        self.vocab_size = self.numeric_vocab_size + self.cat_vocab_size

        # special tokens
        self.SEP_ID = self.vocab_size
        self.PAD_ID = self.vocab_size
        self.MASK_ID = self.vocab_size + 1
        self.total_vocab = self.vocab_size + 2

        # ---------- embeddings ----------
        # self.token_embedding = nn.Embedding(
        #     num_embeddings=self.total_vocab,
        #     embedding_dim=cfg.embedding_dim,
        #     padding_idx=self.PAD_ID,
        # )
        self.seq_len = self.num_numeric_tokens + self.num_cat
        self.pos_embedding = nn.Embedding(self.seq_len, cfg.embedding_dim)
        self.seq_embedding = MixedFeatureEmbedding(
            num_numerical = cfg.num_numerical,
            cat_cardinalities = self.cat_cardinalities,
            numerical_dim=cfg.embedding_dim
        )
        # ---------- transformer ----------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            batch_first=True,
            activation="gelu",
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_layers)

        # simple projection → separate prediction space from embedding space
        self.proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)

        # deterministic RNG for validation masks
        self._det_gen: torch.Generator | None = None

    # ------------------------------------------------------------------
    # Helper: token sequence ------------------------------------------------
    # ------------------------------------------------------------------
    def _build_token_sequence(self, num_raw: torch.Tensor, cat_raw: torch.Tensor) -> torch.Tensor:
        """Convert raw numerical + categorical features to token IDs."""
        # bin_tokens = self.numerical_encoder(num_raw)              # (B, n_feats, depth) with {0,1,-1}
        # B = bin_tokens.size(0)
        # device = bin_tokens.device
        # # positional encoding of numerical features
        # for i in range(bin_tokens.size(1)):
        #     bin_tokens[:, i, :] += 2 * i

        # flat = bin_tokens.view(B, self.num_numeric_tokens)        # (B, total_tokens)
        # # pos_ids = torch.arange(self.num_numeric_tokens, device=device).unsqueeze(0).expand(B, -1)
        # numeric_ids = flat.clamp(min=0)             # map 0/1 ➜ 2*idx + bit
        # numeric_ids[flat == 255] = self.PAD_ID                    # mark padded bits

        # cat_ids = torch.full((B, self.num_cat), self.PAD_ID, dtype=torch.long, device=device)
        # offset = self.cat_offset
        # for j, card in enumerate(self.cat_cardinalities):
        #     cat_ids[:, j] = cat_raw[:, j] + offset
        #     offset += card
        # return torch.cat([numeric_ids, cat_ids], dim=1)           # (B, L)
        emb = self.seq_embedding(num_raw, cat_raw)
        return emb

    # ------------------------------------------------------------------
    # Forward -----------------------------------------------------------
    # ------------------------------------------------------------------
    def forward(
        self,
        num_raw: torch.Tensor,
        cat_raw: torch.Tensor,
        mask_prob: float | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Run a forward pass with encoder–decoder masking.

        The encoder consumes the **original** sequence; the decoder gets the
        **masked** sequence.  We compute CE + 0.1×MSE over *masked* positions
        only.  ``mask_prob`` and ``deterministic`` behave as described above.
        """
        # ---------------- token ids ----------------
        # tokens = self._build_token_sequence(num_raw, cat_raw)  # (B, L)

        # embeddings
        emb = self.seq_embedding(num_raw, cat_raw)
        # emb = F.normalize(emb, dim=-1)
        B, N = emb.size(0), emb.size(1)
        # ---------------- mask ----------------
        p = self.mask_prob if mask_prob is None else float(mask_prob)
        if p == 0.0:
            mask = torch.zeros((B, N), dtype=torch.bool)
        else:
            if deterministic:
                if self._det_gen is None:
                    self._det_gen = torch.Generator(device=emb.device).manual_seed(0)
                rand = torch.rand((B, N), generator=self._det_gen, device=emb.device)
            else:
                rand = torch.rand((B, N), device=emb.device)
            mask = (rand < p)
        masked_emb = emb.clone()
        random_noise = torch.randn_like(emb, device=emb.device)
        # mask is random noise
        masked_emb[mask, :] = random_noise[mask, :]
        # recover padding
        masked_emb[emb == 0.0] = 0.0
        # ---------- transformer ----------
        memory = self.encoder(masked_emb)
        hidden = self.decoder(emb, memory)

        # ---------- projection & logits ----------
        recon = self.proj(hidden)

        # loss = ce_loss + 0.1 * mse_loss
        emb_avg_norm = torch.mean(torch.norm(emb, dim=-1, p=2)).detach()
        loss = F.mse_loss(recon, emb) + 0.01 * torch.mean((torch.norm(emb, dim=-1, p=2) - emb_avg_norm)**2)

        metrics = {
            "loss": float(loss.detach()),
            "mse": float(torch.norm(recon, p=2, dim=-1).mean().detach()),
            "ce": float(torch.norm(recon, p=2, dim=-1).var().detach()),
        }
        return recon, loss, metrics

    # ------------------------------------------------------------------
    # Utilities ---------------------------------------------------------
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, num_raw: torch.Tensor, cat_raw: torch.Tensor) -> torch.Tensor:
        """Return encoder memory for the *full* (unmasked) sequence."""
        self.eval()
        # tokens = self._build_token_sequence(num_raw, cat_raw)
        # pos_idx = torch.arange(self.seq_len, device=tokens.device).unsqueeze(0).expand(tokens.size(0), -1)
        # enc_x = self.token_embedding(tokens) + self.pos_embedding(pos_idx)
        return self.seq_embedding(num_raw, cat_raw)

    # -------------- I/O helpers ----------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        cfg: RunConfiguration,
        device: str | torch.device = "cpu",
    ) -> "MaskedTransformerAutoencoder":
        model = cls(cfg).to(device)
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            checkpoint = checkpoint["model_state"]
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        return model



class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the true class.

    Args:
        alpha (float): weight for the focal term (default=1.0)
        gamma (float): focusing parameter to adjust rate at which easy examples are down-weighted (default=2.0)
        reduction (str): 'none' | 'mean' | 'sum'
    """
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (batch_size, seq_len), targets: (batch_size, seq_len) in {0,1}
        # Compute element-wise binary cross-entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # p_t: probabilities of the true class
        p_t = torch.exp(-bce_loss)
        # Focal loss factor
        focal_factor = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create constant positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerAnomalyDetector(nn.Module):
    """
    Transformer-based anomaly detector with focal loss, padding mask, and positional encoding.
    """
    def __init__(
        self,
        cfg: RunConfiguration,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        d_model = cfg.embedding_dim

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers
        )

        # Dropout and classification head
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        # Initialize bias to favor negative class
        # nn.init.constant_(self.classifier.bias, -3.0)

        # Focal loss (no reduction)
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, emb_dim)
        Returns:
            logits: (batch_size, seq_len)
            pad_mask: (batch_size, seq_len) boolean mask indicating padding positions
        """
        x = x.swapaxes(0, 1)
        # Identify padding positions by zero-vectors
        pad_mask = x.abs().sum(dim=-1) == 0
        # Apply positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder with padding mask
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        x = self.dropout(x)

        # Classification head
        x = x.swapaxes(0, 1)
        logits = self.classifier(x).squeeze(-1)

        pad_mask = pad_mask.swapaxes(0, 1)
        return logits, pad_mask

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss, ignoring padded positions.
        """
        print(logits[targets == 1])
        print(targets[targets == 1])
        loss = self.criterion(logits, targets).mean()
        # Zero out loss on padding tokens
        # loss = loss.masked_fill(pad_mask, 0.0)
        # Average over non-padded tokens
        return loss # / (~pad_mask).sum()

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
    def load(cls, path: str, cfg: RunConfiguration, device: str | torch.device ="cpu") -> "TransformerAnomalyDetector":
        model = cls(cfg).to(device)

        checkpoint = torch.load(path, map_location=device)

        # ⬇⬇ unwrap if necessary
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            checkpoint = checkpoint["model_state"]

        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        return model


class __TransformerAnomalyDetector(nn.Module):
    """
    Transformer-based anomaly detector with integrated Focal Loss.
    """
    def __init__(
        self,
        cfg: RunConfiguration,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        # Positional encoding placeholder (replace with your implementation)
        self.pos_encoder = nn.Sequential()

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # Classification head
        self.classifier = nn.Linear(cfg.embedding_dim, 1)

        # Integrated Focal Loss
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, embedding_dim, embedding_dim)
        # x = self.pos_encoder(x)
        # x = x.transpose(0, 1)  # (embedding_dim, seq_len, embedding_dim)
        x = self.transformer_encoder(x)
        # x = x.transpose(0, 1)  # (batch_size, seq_len, embedding_dim)
        logits = self.classifier(x).squeeze(-1)  # (batch_size, seq_len)
        return logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss between logits and binary targets.
        """
        return self.criterion(logits, targets)

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
    def load(cls, path: str, cfg: RunConfiguration, device: str | torch.device ="cpu") -> "TransformerAnomalyDetector":
        model = cls(cfg).to(device)

        checkpoint = torch.load(path, map_location=device)

        # ⬇⬇ unwrap if necessary
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            checkpoint = checkpoint["model_state"]

        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        return model
