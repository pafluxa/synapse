import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse.models.transformers import BatchMaskedTransformer
from synapse.models.convolutional import EmbeddingEncoder
from synapse.models.mixture_of_experts import BatchMoEDecoder
from synapse.models.layers.losses import SphericalLoss

class CodecTransformer(nn.Module):
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


class TabularBERT(nn.Module):
    
    @staticmethod
    def smooth_growth(n, start, end, low_val=1e-5, high_val=1e-1):
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
        self.d_model = config.embedding_dim
        self.nhead = config.num_heads 
        self.num_layers = config.num_layers
        self.codec_dim = config.codec_dim
        self.num_numerical = config.num_numerical
        self.cat_dims = config.categorical_dims
        self.num_features = config.num_numerical + len(config.categorical_dims) 
        self.sph_loss_fn = SphericalLoss()
         
        # Numerical embeddings
        self.num_embedder = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU()
        )
        
        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(dim, self.d_model) for dim in self.cat_dims
        ])
        
        # Feature type embeddings
        self.feature_emb = nn.Embedding(self.num_features, self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, self.nhead, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        
        # Global bottleneck (B, k)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.d_model * self.num_features, self.codec_dim),
            nn.ReLU(),
            nn.Linear(self.codec_dim, self.codec_dim),
        )
        
        # Decoder expansion
        self.decoder_expand = nn.Sequential(
            nn.Linear(self.codec_dim, self.d_model * self.num_features),
            nn.GELU(),
            nn.Linear(self.d_model * self.num_features, self.d_model * self.num_features)
        )
        
        # Reconstruction heads
        self.num_recon = nn.Linear(self.d_model, 1)
        self.cat_recons = nn.ModuleList([
            nn.Sequential(
               nn.Linear(self.d_model, 2 * self.d_model), 
               nn.ReLU(),
               nn.Linear(2 * self.d_model, self.d_model), 
               nn.Linear(self.d_model, dim)
            )
            for dim in self.cat_dims
        ])

    def forward(self, x_num, x_cat):
        batch_size = x_num.size(0)
        
        # Numerical embeddings
        num_emb = self.num_embedder(x_num.unsqueeze(-1))
        num_ids = torch.arange(self.num_numerical, device=x_num.device)
        num_ids = num_ids.unsqueeze(0).expand(batch_size, -1)
        num_feat_emb = self.feature_emb(num_ids)
        num_combined = num_emb + num_feat_emb
        
        # Categorical embeddings
        cat_embs = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_embs.append(emb(x_cat[:, i]).unsqueeze(1))
        cat_emb = torch.cat(cat_embs, dim=1)
        cat_ids = torch.arange(
            self.num_numerical, self.num_features, device=x_cat.device
        ).unsqueeze(0).expand(batch_size, -1)
        cat_feat_emb = self.feature_emb(cat_ids)
        cat_combined = cat_emb + cat_feat_emb
        
        # Combine features
        combined = torch.cat([num_combined, cat_combined], dim=1)
        
        # Encode and flatten
        encoded = self.encoder(combined)
        flattened = encoded.view(batch_size, -1)
        
        # Bottleneck compression
        compressed = self.bottleneck(flattened)
        
        # Decoder expansion
        expanded = self.decoder_expand(compressed)
        decoded_features = expanded.view(batch_size, self.num_features, self.d_model)
        
        # Split and reconstruct
        num_recon = self.num_recon(decoded_features[:, :self.num_numerical]).squeeze(-1)
        cat_recons = [
            head(decoded_features[:, self.num_numerical+i]) 
            for i, head in enumerate(self.cat_recons)
        ]
        
        return compressed, num_recon, cat_recons
    
    def loss(self, outputs, targets, mask, epoch):
        codecs, num_recon, cat_recons = outputs
        x_num, x_cat = targets
        
        # Numerical loss (MSE)
        num_loss = torch.mean((num_recon - x_num)**2 * mask[:, :x_num.shape[1]].float())
        
        # Categorical loss (CrossEntropy)
        cat_loss = 0
        for i, (logits, dim) in enumerate(zip(cat_recons, self.cat_dims)):
            # print(dim, x_cat[0, i], s_logits[0])
            loss = nn.CrossEntropyLoss(reduction='none')(logits, x_cat[:, i])
            cat_loss += torch.mean(loss * mask[:, x_num.shape[1]+i].float())
        cat_loss /= len(self.cat_dims)
         
        # spherical loss
        w1 = self.smooth_growth(epoch, 1, 200)
        sph_loss, sph_metrics = self.sph_loss_fn(codecs)
        
        total_loss = num_loss + cat_loss + w1 * sph_loss
        
        return total_loss, {
            'loss': total_loss.item(),
            'num_loss': num_loss.item(),
            'cat_loss': cat_loss.item(),
            'sph_loss': w1 * sph_loss.item(),
            'uni_loss': sph_metrics['variance'].item(),
        }

