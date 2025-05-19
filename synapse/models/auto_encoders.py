import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from synapse.models.layers.embeddings import CategoricalEmbedding, NumericalEmbedding
from synapse.models.layers.feature_encoders import Zwei
from synapse.models.layers.losses import VMFLoss


class TabularBERT(nn.Module):

    @staticmethod
    def smooth_growth(n, start, end, low_val=1e-5, high_val=1e-0):
        if n < start:
            return low_val
        elif n > end:
            return high_val
        else:
            # Normalize n to range [0, 1]
            x = (n - start) / (end - start)
            # Exponential interpolation
            factor = math.log10(high_val / low_val)
            return low_val * (10 ** (factor * x))

    def __init__(self, config):

        super().__init__()

        self.sph_loss_fn = VMFLoss(config.codec_dim)

        self.d_model = config.embedding_dim
        self.nhead = config.num_heads
        self.num_layers = config.num_layers
        self.codec_dim = config.codec_dim

        # perform tokenization of numerical (continuous) data
        depths = [8,] * config.num_numerical
        self.max_depth = max(depths)
        self.cat_dims = config.categorical_dims
        self.num_numerical = config.num_numerical
        self.num_features = self.max_depth * self.num_numerical + len(config.categorical_dims)

        self.num_tokenizer = Zwei(
            # ranges from 0 to 1
            [[0, 1],] * self.num_numerical,
            # max depth is transformer "size"
            depths,
        )
        # calculate numerical embeddings
        self.num_embedder = NumericalEmbedding(
            self.d_model,
            depths
        )
        # calculate categorical embeddings
        self.cat_embedder = CategoricalEmbedding(
            self.d_model,
            self.cat_dims,
            min_emb_dim=self.d_model
        )
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, self.nhead, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        # Global bottleneck (B, k)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.d_model * self.num_features, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.Linear(100, self.codec_dim, bias=False),
        )
        # Decoder expansion
        self.decoder_expand = nn.Sequential(
            nn.Linear(self.codec_dim, 100, bias=False),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.Linear(100, self.d_model * self.num_features)
        )

    def forward(self, x_num, x_cat):
        batch_size = x_num.size(0)
        # embeddings of categorical data
        cat_emb = self.cat_embedder(x_cat)
        # embeddings of numerical data
        num_tok = self.num_tokenizer(x_num)
        num_emb = self.num_embedder(num_tok)
        # combine embeddings
        cat_emb = rearrange(cat_emb, 's b e -> b s e')
        num_emb = rearrange(num_emb, 's b n e -> b (s n) e')
        combined_embd = torch.cat([num_emb, cat_emb], dim=1)
        # transform
        encoded = self.encoder(combined_embd)
        # compress to codec
        flattened = encoded.view(batch_size, -1)
        compressed = self.bottleneck(flattened)
        # expand back to embeddings
        expanded = self.decoder_expand(compressed)
        decoded_features = expanded.view(batch_size, self.num_features, self.d_model)

        return compressed, decoded_features

    def loss(self, outputs, targets, mask, epoch):
        codecs, decoded = outputs
        # process target
        x_num, x_cat = targets
        num_tok = self.num_tokenizer(x_num)
        num_emb = self.num_embedder(num_tok)
        num_emb = rearrange(num_emb, 's b n e -> b (s n) e')
        cat_emb = self.cat_embedder(x_cat)
        cat_emb = rearrange(cat_emb, 's b e -> b s e')
        cmb_emb = torch.cat([num_emb, cat_emb], dim=1)

        # Numerical loss (MSE)
        rec_loss = torch.mean((decoded - cmb_emb)**2)

        # spherical loss
        w1 = self.smooth_growth(epoch,   0,  50, low_val=1e-5, high_val=0.1)
        w2 = self.smooth_growth(epoch,  50, 100, low_val=1e-5, high_val=0.01)
        w3 = self.smooth_growth(epoch, 100, 150, low_val=1e-5, high_val=0.01)
        sph_ent, sph_rmu, sph_rep, sph_metrics = self.sph_loss_fn(codecs)

        total_loss = rec_loss + w1 * sph_rmu + w2 * sph_rep + w1 * sph_ent

        return total_loss, {
            'loss': total_loss.item(),
            'mse_loss': rec_loss.item(),
            'sph_vmf': w3 * sph_metrics['sph_vmf'].item(),
            'sph_rep': w2 * sph_metrics['sph_rep'].item(),
            'sph_rad': w1 * sph_metrics['sph_rad'].item(),
        }
