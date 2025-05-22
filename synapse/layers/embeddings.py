"""Embedding layers for categorical and numerical data.

This module provides embedding layers that can handle both categorical and numerical
data with proper padding and masking. The embeddings are designed to work with
variable-length sequences and batch processing.

Classes:
    CategoricalEmbedding: Embedding layer for categorical data.
    NumericalEmbedding: Embedding layer for numerical data.
"""

import math
from typing import List, Tuple, Optional

from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class CategoricalEmbedding(nn.Module):
    """Embedding layer for categorical data with padding and masking.

    This layer handles categorical variables of different cardinalities by creating
    separate embedding layers for each variable. The embedding dimension for each
    variable is determined by the square root of its cardinality.

    Attributes:
        padval_: Padding value used for masking (default: -1).
        min_emb_dim_: Minimum embedding dimension across all variables.
        max_emb_dim_: Maximum embedding dimension across all variables.
        embedding_layers_: List of embedding layers for each categorical variable.
    """

    padval_: int = -1

    def __init__(
        self,
        d_model: int,
        cardinalities: List[int],
        max_emb_dim: int = 0,
        min_emb_dim: int = 0,
    ) -> None:
        """Initialize the CategoricalEmbedding layer.

        Args:
            cardinalities: List of cardinalities for each categorical variable.
            max_emb_dim: Maximum embedding dimension to use (default: -1 for auto).
            min_emb_dim: Minimum embedding dimension to use (default: 1,000,000 for auto).
        """
        super().__init__()
        # Compute min/max embedding dimensions
        self.min_emb_dim_ = max([
            min_emb_dim,
            int(math.ceil(min(cardinalities)**0.5) + 1)
        ])
        self.max_emb_dim_ = max([
            max_emb_dim,
            int(math.ceil(max(cardinalities)**0.5) + 1)
        ])

        assert self.min_emb_dim_ > 0, "Minimum embedding dimension must be positive"

        self.embeddings_ = nn.ModuleList()
        for c in cardinalities:
            d = int(math.ceil(c**0.5) + 1)
            self.embeddings_.append(
                nn.Sequential(
                    nn.Embedding(c, d),
                )
            )
        self.projections_ = nn.ModuleList()
        for c in cardinalities:
            d = int(math.ceil(c**0.5) + 1)
            self.projections_.append(
                nn.Sequential(
                    nn.Linear(d, d_model),
                    nn.LayerNorm(d_model),
                )
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the embedding layer.

        Args:
            x: Categorical data tensor with shape (n_categories, batch_size).

        Returns:
            Tuple containing:
                - Padded embedding tensor with shape (seq_len, batch_size, emb_dim)
                - Padding mask tensor with shape (batch_size, seq_len)
        """
        embeddings = []
        for i, (emb, proj) in enumerate(zip(self.embeddings_, self.projections_)):
            x_emb = emb(x[:, i]) #, 'b d -> d b')
            x_prj = proj(x_emb)
            embeddings.append(x_prj)

        # padded_emb_tensor = pad_sequence(embeddings, padding_value=self.padval_)
        # padded_emb_tensor = rearrange(padded_emb_tensor, 'd s b -> b s d')
        # padding_mask = padded_emb_tensor == self.padval_

        return torch.stack(embeddings)


class NumericalEmbedding(nn.Module):
    """Embedding layer for numerical data with padding and masking.

    This layer handles numerical variables by first converting them to tokens and then
    applying embedding layers. The embedding dimension is determined by the maximum
    depth of the binary encoding.

    Attributes:
        padval_: Padding value used for masking (default: -1).
        min_emb_dim_: Minimum embedding dimension across all variables.
        max_emb_dim_: Maximum embedding dimension across all variables.
        embedding_layers_: List of embedding layers for each numerical variable.
    """

    padval_: int = -1

    def __init__(
        self,
        d_model: int,
        max_depths: List[int],
        max_emb_dim: int = -1,
        min_emb_dim: int = 1_000_000,
    ) -> None:
        """Initialize the NumericalEmbedding layer.

        Args:
            max_depths: List of maximum depths for binary encoding of each numerical variable.
            max_emb_dim: Maximum embedding dimension to use (default: -1 for auto).
            min_emb_dim: Minimum embedding dimension to use (default: 1,000,000 for auto).
        """
        super().__init__()
        # Compute min/max embedding dimensions
        self.min_emb_dim_ = min([
            min_emb_dim,
            int(math.ceil(min(max_depths)**0.5) + 1)
        ])
        self.max_emb_dim_ = max([
            max_emb_dim,
            int(math.ceil(max(max_depths)**0.5) + 1)
        ])

        assert self.min_emb_dim_ > 0, "Minimum embedding dimension must be positive"

        # Embedding layer for values {0, 1}
        self.embedding_layer = nn.Embedding(2, self.max_emb_dim_)

        self.embedding_layers_ = nn.ModuleList()
        for m in max_depths:
            d = int(math.ceil(m**0.5) + 1)
            self.embedding_layers_.append(
                nn.Sequential(
                    nn.Embedding(3, d),
                )
            )
        self.projections_ = nn.ModuleList()
        for m in max_depths:
            d = int(math.ceil(m**0.5) + 1)
            self.projections_.append(
                nn.Sequential(
                    nn.Linear(d, d_model),
                    nn.LayerNorm(d_model),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the embedding layer.

        Args:
            tokens: List of token tensors for each numerical variable.

        Returns:
            Tuple containing:
                - Padded embedding tensor with shape (seq_len, batch_size, emb_dim)
                - Padding mask tensor with shape (batch_size, seq_len)
        """
        embeddings = []
        for i, (emb, proj) in enumerate(zip(self.embedding_layers_, self.projections_)):
            xi = x[:, i, :]
            mask = xi < 255
            xi_clamped = torch.clamp(xi, max=1)
            emb = emb(xi_clamped)
            # mask invalid entries
            emb = emb * mask.unsqueeze(-1)
            prj = proj(emb)
            embeddings.append(prj)

        return torch.stack(embeddings)


class NumericalFeatureEmbedding(nn.Module):
    """Project each **scalar** numerical feature independently to a common
    embedding dimension using a small 1‑layer *MLP* (here just ``nn.Linear``).

    Parameters
    ----------
    num_features : int
        How many distinct numerical columns the table contains.
    emb_dim : int
        Size of the vector produced for **every** numerical feature.
    """

    def __init__(self, num_features: int, emb_dim: int):
        super().__init__()
        self.num_features = num_features
        self.emb_dim = emb_dim

        # A separate linear projection per column → shape (B, 1) → (B, D)
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, emb_dim, bias=False),
                    nn.LayerNorm(emb_dim)
                ) for _ in range(num_features)
            ]
        )

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:  # (B, N)
        if x_num.ndim != 2 or x_num.size(1) != self.num_features:
            raise ValueError(
                f"Expected x_num of shape (B, {self.num_features}); got {tuple(x_num.shape)}"
            )
        outs = []
        for j, linear in enumerate(self.proj):
            col = x_num[:, j].unsqueeze(-1)  # (B, 1)
            outs.append(linear(col))   # (B, D)
        return torch.stack(outs, dim=1)  # (B, N, D)


class CategoricalFeatureEmbedding(nn.Module):
    """Embed each categorical column with its **own** dimensionality
    ``d_j = ⌈0.5 · √C_j⌉`` then *left‑pad* with zeros so every returned vector
    has the same final size ``max(d_j)``.
    """

    def __init__(self, cardinalities: List[int]):
        super().__init__()
        if not cardinalities:
            raise ValueError("cardinalities list must not be empty")

        self.cardinalities = list(cardinalities)
        self.dims = [max(1, int(round(0.5 * math.sqrt(c)))) for c in self.cardinalities]
        self.max_dim = max(self.dims)

        self.embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(C, d),
                    nn.LayerNorm(d)
                ) for C, d in zip(self.cardinalities, self.dims)
            ]
        )

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:  # (B, M)
        if x_cat.ndim != 2 or x_cat.size(1) != len(self.cardinalities):
            raise ValueError(
                f"Expected x_cat of shape (B, {len(self.cardinalities)}); got {tuple(x_cat.shape)}"
            )
        outs = []
        for j, (emb, d_j) in enumerate(zip(self.embeddings, self.dims)):
            v = emb(x_cat[:, j])                 # (B, d_j)
            if d_j < self.max_dim:               # pad on the *right*
                pad = (0, self.max_dim - d_j)    # (left, right)
                v = F.pad(v, pad, value=0.0)
            outs.append(v)
        return torch.stack(outs, dim=1)  # (B, M, max_dim)


class MixedFeatureEmbedding(nn.Module):
    """Combine numerical + categorical embeddings into a single tensor of
    uniform width ``D`` ready for a Transformer or MLP‑Mixer.

    Parameters
    ----------
    num_numerical : int
        Number of scalar numerical columns.
    cat_cardinalities : list[int]
        Cardinalities *per* categorical column.
    numerical_dim : int | None, default ``None``
        If ``None`` we use the same *final* dimension as the categorical block
        (``max_dim``).  Otherwise we project every numerical feature to this
        size **and** pad/truncate categorical vectors to match.
    """

    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int],
        numerical_dim: int | None = None,
    ):
        super().__init__()

        self.cat_block = CategoricalFeatureEmbedding(cat_cardinalities)

        # Decide the final common dimension D
        self.emb_dim = max(self.cat_block.max_dim, numerical_dim or 0)
        if numerical_dim is not None and numerical_dim != self.emb_dim:
            # We will *pad* cat vectors up to numerical_dim
            self.emb_dim = numerical_dim

        self.num_block = NumericalFeatureEmbedding(num_numerical, self.emb_dim)

    # ------------------------------------------------------------------
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Return shape **(B, num_numerical + num_categorical, D)**."""
        num_vecs = self.num_block(x_num)               # (B, N, D)
        cat_vecs = self.cat_block(x_cat)               # (B, M, d*)

        # Pad categorical side if needed
        d_cat = cat_vecs.size(-1)
        if d_cat < self.emb_dim:
            pad = (0, self.emb_dim - d_cat)
            cat_vecs = F.pad(cat_vecs, pad, value=0.0)
        elif d_cat > self.emb_dim:
            cat_vecs = cat_vecs[..., : self.emb_dim]   # truncate (rare)

        return torch.cat([num_vecs, cat_vecs], dim=1)  # (B, N+M, D)
