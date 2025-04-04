""" Embeddings module.
"""
import math

from typing import List, Tuple

from einops import rearrange

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

class CategoricalEmbedding(nn.Module):

    # default padding value
    padval_: int = -1

    def __init__(
        self,
        cardinalities: List[int],
        max_emb_dim: int = -1,
        min_emb_dim: int = 1000000,
    ):
        super(CategoricalEmbedding, self).__init__()
        # compute min/max embedding dimensions
        self.min_emb_dim_ = min([
            min_emb_dim,
            int(math.ceil(min(cardinalities)**0.5) + 1)])
        self.max_emb_dim_ = max([
            max_emb_dim,
            int(math.ceil(max(cardinalities)**0.5) + 1)])

        assert self.min_emb_dim_ > 0, "Minimum embedding dimension must be positive"

        self.embedding_layers_ = nn.ModuleList()
        for c in cardinalities:
            d = int(math.ceil(c**0.5) + 1)
            self.embedding_layers_.append(
                nn.Sequential(
                    nn.Embedding(c, d),
                    nn.LayerNorm(d),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: categorical data with shape (n_categories, batch_size)
        """
        embeddings = []
        for i, layer in enumerate(self.embedding_layers_):
            emb = rearrange(layer(x[i]), 'b d -> d b')
            embeddings.append(emb)
        padded_emb_tensor = pad_sequence(embeddings, padding_value=self.padval_)
        padded_emb_tensor = rearrange(padded_emb_tensor, 'd s b -> s b d')
        padding_mask = padded_emb_tensor == self.padval_
        padding_mask = rearrange(padding_mask, 's b d-> b s d')[:, :, 0]

        return padded_emb_tensor, padding_mask


class NumericalEmbedding(nn.Module):

    # default padding value
    padval_: int = -1

    def __init__(
        self,
        max_depths: List[int],
        max_emb_dim: int = -1,
        min_emb_dim: int = 1000000,
    ):
        super(NumericalEmbedding, self).__init__()
        # compute min/max embedding dimensions
        self.min_emb_dim_ = min([
            min_emb_dim,
            int(math.ceil(min(max_depths)**0.5) + 1)])
        self.max_emb_dim_ = max([
            max_emb_dim,
            int(math.ceil(max(max_depths)**0.5) + 1)])

        assert self.min_emb_dim_ > 0, "Minimum embedding dimension must be positive"

        self.embedding_layers_ = nn.ModuleList()
        for m in max_depths:
            d = int(math.ceil(m**0.5) + 1)
            self.embedding_layers_.append(
                nn.Sequential(
                    nn.Embedding(m, d),
                    nn.LayerNorm(d),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: categorical data with shape (n_categories, batch_size)
        """
        embeddings = []
        for i, layer in enumerate(self.embedding_layers_):
            emb = rearrange(layer(x[i]), 'b d -> d b')
            embeddings.append(emb)
        padded_emb_tensor = pad_sequence(embeddings, padding_value=self.padval_)
        padded_emb_tensor = rearrange(padded_emb_tensor, 'd s b -> s b d')
        padding_mask = padded_emb_tensor == self.padval_
        padding_mask = rearrange(padding_mask, 's b d-> b s d')[:, :, 0]

        return padded_emb_tensor, padding_mask
