from posix import read
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse.models.transformers import BatchMaskedTransformer
from synapse.models.mixture_of_experts import BatchMoEDecoder
from synapse.models.layers.losses import MoEBalancingLoss
from synapse.models.convolutional import EmbeddingEncoder
from synapse.models.layers.losses import uniformity_loss

class CodecTransformerMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Transformer Encoder with Masking
        self.transformer = BatchMaskedTransformer(config)

        self.encoder = EmbeddingEncoder(
                    num_features=config.seq_len,
                    embedding_dim=config.embedding_dim,
                    codec_dim=config.codec_dim
                )
        # 3. MoE Decoder
        self.moe_decoder = BatchMoEDecoder(config)

        # 4. Loss Components
        self.moe_balancing_loss = MoEBalancingLoss(config.num_experts, config.seq_len)
        self.hypersphere_loss = HypersphereSurfaceLoss()

    def forward(self, x_num, x_cat):
        # 1. Transformer Encoding
        transformer_out = self.transformer(x_num, x_cat)
        encoded, mask_pos, mask_ratio = transformer_out.values()  # [B, S, D]
        codec = self.encoder(encoded)  # [B, S, D]

        # 3. MoE Decoding
        moe_out = self.moe_decoder(codec, mask_pos)

        return {
            'codec': codec,
            'mask_pos': moe_out['mask_pos'],
            'gates': moe_out['gates'],
            'expert_indices': moe_out['expert_indices'],
            'num_recon': moe_out['num_recon'],
            'cat_recon': moe_out['cat_recon'],
            'feature_indices': moe_out['feature_indices']
        }

    def compute_loss_and_metrics(self, x_num, x_cat, outputs):

        mask_pos = outputs['mask_pos']  # [B, S]

        # 1. Only compute loss on masked positions
        num_mask = mask_pos[:, :self.config.num_numerical]
        cat_mask = mask_pos[:, self.config.num_numerical:]

        # 2. Apply masks to reconstructed features
        num_loss = F.mse_loss(
            outputs['num_recon'] * num_mask,
            x_num * num_mask
        )

        n_hits = 0
        cat_loss = torch.tensor([0.0], device=x_num.device)
        for i, (logits, mask) in enumerate(zip(outputs['cat_recon'], cat_mask.T)):
            if logits[mask].size(0) < 1:
                continue
            cat_loss += F.cross_entropy(
                logits[mask],
                x_cat[:, i][mask]
            )
            n_hits += mask.sum().item()
        if n_hits > 0:
            cat_loss /= n_hits

        # 3. MoE Balancing Loss
        moe_loss, moe_metrics = self.moe_balancing_loss(
            outputs['gates'],
            outputs['expert_indices'],
            outputs['feature_indices']
        )

        # 4. Hypersphere Regularization
        codec_norms = torch.norm(outputs['codec'], dim=1)
        sphere_loss = self.hypersphere_loss(codec_norms)
        spread_loss = uniformity_loss(outputs['codec'])

        # 5. Total Loss
        total_loss = (
            1.0 * num_loss +
            1.0 * cat_loss +
            0.1 * moe_loss +
            0.01 * sphere_loss +
            0.01 * spread_loss
        )

        metrics = {
            'loss': total_loss.item(),
            'num_loss': num_loss.item(),
            'cat_loss': cat_loss.item(),
            'moe_loss': moe_loss.item(),
            'sphere_loss': sphere_loss.item(),
            'codec_norm': codec_norms.mean().item(),
            'mask_ratio': mask_pos.float().mean().item()
        }

        return total_loss, metrics

class HypersphereSurfaceLoss(nn.Module):
    """Encourages codecs to lie on hypersphere surface"""
    def forward(self, norms):
        return F.mse_loss(norms, torch.ones_like(norms))
