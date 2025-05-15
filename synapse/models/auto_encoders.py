import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse.models.layers.embeddings import CategoricalEmbedding, NumericalEmbedding
from synapse.models.layers.feature_encoders import Zwei
from synapse.models.layers.losses import SphericalLoss


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
        # self.num_embedder = nn.Sequential(
        #     nn.Linear(1, self.d_model),
        #     nn.GELU()
        # )
        self.num_encoder = Zwei([[0, 1],] * self.num_numerical, [self.d_model] * self.num_numerical)
        self.num_embedder = NumericalEmbedding([self.d_model] * self.num_numerical) 
        # Categorical embeddings
        # self.cat_embeddings = nn.ModuleList([
        #     nn.Embedding(dim, self.d_model) for dim in self.cat_dims
        # ])
        self.cat_embedder = CategoricalEmbedding(self.cat_dims, min_emb_dim=self.d_model)
        # Feature type embeddings
        # self.feature_emb = nn.Embedding(self.num_features, self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, self.nhead, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        
        # Global bottleneck (B, k)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.d_model * self.num_features, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.Linear(100, self.codec_dim, bias=False),
        )
        
        # Decoder expansion
        self.decoder_expand = nn.Sequential(
            nn.Linear(self.codec_dim, 100, bias=False),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 500),
            nn.LeakyReLU(),
            nn.Linear(500, self.d_model * self.num_features)
        )
        
        # Reconstruction heads
        # self.num_recons = nn.ModuleList([
        #     nn.Sequential(
        #        nn.Linear(self.d_model, 2 * self.d_model), 
        #        nn.ReLU(),
        #        nn.Linear(2 * self.d_model, self.d_model), 
        #        nn.Linear(self.d_model, self.d_model)
        #     )
        #     for _ in range(self.num_numerical)
        # ])
        # self.cat_recons = nn.ModuleList([
        #     nn.Sequential(
        #        nn.Linear(self.d_model, 2 * self.d_model), 
        #        nn.ReLU(),
        #        nn.Linear(2 * self.d_model, self.d_model), 
        #        nn.Linear(self.d_model, self.d_model)
        #     )
        #     for _ in self.cat_dims
        # ])

    def forward(self, x_num, x_cat):
        batch_size = x_num.size(0)
        
        # Numerical embeddings
        # num_emb = self.num_embedder(x_num.unsqueeze(-1))
        # num_ids = torch.arange(self.num_numerical, device=x_num.device)
        # num_ids = num_ids.unsqueeze(0).expand(batch_size, -1)
        # num_feat_emb = self.feature_emb(num_ids)
        # num_combined = num_emb + num_feat_emb
        
        # Categorical embeddings
        # cat_embs = []
        # for i, emb in enumerate(self.cat_embeddings):
        #     cat_embs.append(emb(x_cat[:, i]).unsqueeze(1))
        # cat_emb = self.cat_embeddings(x_cat) torch.cat(cat_embs, dim=1)
        # cat_emb, cat_pad_mask = self.cat_embeddings(x_cat)
        # cat_ids = torch.arange(
        #     self.num_numerical, self.num_features, device=x_cat.device
        # ).unsqueeze(0).expand(batch_size, -1)
        # cat_feat_emb = self.feature_emb(cat_ids)
        # cat_combined = cat_emb + cat_feat_emb
        
        num_feat, num_pad_mask = self.num_encoder(x_num)
        num_emb = self.num_embedder(num_feat)
        cat_emb, cat_pad_mask = self.cat_embedder(x_cat)
        print(num_feat.shape, num_emb.shape, num_pad_mask.shape)
        print(cat_emb.shape) 
        # Combine features
        combined_embd = torch.cat([num_emb, cat_emb], dim=1)
        combined_masks = torch.cat([num_pad_mask, cat_pad_mask], dim=1)[:, :, 0]
        # Encode and flatten
        encoded = self.encoder(combined_embd, src_key_padding_mask=combined_masks)
        flattened = encoded.view(batch_size, -1)
        
        # Bottleneck compression
        compressed = self.bottleneck(flattened)
        
        # Decoder expansion
        expanded = self.decoder_expand(compressed)
        decoded_features = expanded.view(batch_size, self.num_features, self.d_model)
        
        # Split and reconstruct
        num_features = decoded_features[:, :self.num_numerical, :]
        cat_features = decoded_features[:, self.num_numerical:, :]
        # num_recon = self.num_recon(decoded_features[:, :self.num_numerical]).squeeze(-1)
        # cat_recons = [
        #     head(decoded_features[:, self.num_numerical+i]) 
        #     for i, head in enumerate(self.cat_recons)
        # ]
        
        return compressed, num_features, cat_features
    
    def loss(self, outputs, targets, mask, epoch):
        codecs, num_recon, cat_recon = outputs
        x_num, x_cat = targets
        num_emb, num_pad_mask = self.num_encoder(x_num)
        cat_emb, cat_pad_mask = self.cat_embedder(x_cat)
        
        # Numerical loss (MSE)
        mask_num = mask[:, 0:self.num_numerical][:, :, None]
        mask_cat = mask[:, self.num_numerical::][:, :, None]
        # print(num_recon.shape, num_emb.shape, mask_num.shape)
        # print(cat_recon.shape, cat_emb.shape, mask_cat.shape)
        num_loss = torch.mean((num_recon - num_emb)**2 * mask_num.float())
        cat_loss = torch.mean((cat_recon - cat_emb)**2 * mask_cat.float())
        
        # Categorical loss (CrossEntropy)
        # cat_loss = 0
        # for i, (logits, dim) in enumerate(zip(cat_recons, self.cat_dims)):
        #     # print(dim, x_cat[0, i], logits[0])
        #     loss = nn.CrossEntropyLoss(reduction='none')(logits, x_cat[:, i])
        #     cat_loss += torch.mean(loss * mask[:, x_num.shape[1]+i].float())
        # cat_loss /= len(self.cat_dims)
         
        # spherical loss
        w1 = self.smooth_growth(epoch, 1, 200)
        sph_loss, sph_metrics = self.sph_loss_fn(codecs)
        
        total_loss = num_loss + 0.1 * cat_loss + w1 * sph_loss
        
        return total_loss, {
            'loss': total_loss.item(),
            'num_loss': num_loss.item(),
            'cat_loss': 0.1 * cat_loss.item(),
            'sph_loss': w1 * sph_loss.item(),
            'uni_loss': sph_metrics['variance'].item(),
        }

