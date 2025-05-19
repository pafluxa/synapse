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
