import torch
import torch.nn as nn
import math

class BatchMaskedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = config.num_numerical + config.num_categorical
        self.embedding_dim = config.embedding_dim

        # 1. Input Embeddings
        self.num_projs = nn.ModuleList([
            nn.Linear(1, config.embedding_dim)
            for _ in range(config.num_numerical)
        ])

        self.cat_embeds = nn.ModuleList([
            nn.Embedding(dim, config.embedding_dim)
            for dim in config.categorical_dims
        ])

        # 2. Masking Components
        self.mask_token = nn.Parameter(torch.randn(1, config.embedding_dim))
        self.register_buffer('max_mask_ratio', torch.tensor(0.3))

        # Initialize positional encoding properly
        self.register_buffer('pos_enc', self._create_positional_encoding())

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

    def _create_positional_encoding(self):
        """Create non-learnable sinusoidal positional encoding"""
        position = torch.arange(self.seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2) *
            (-math.log(10000.0) / self.embedding_dim)
        )

        pe = torch.zeros(1, self.seq_len, self.embedding_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x_num, x_cat):
        batch_size = x_num.size(0)
        device = x_num.device

        # 1. Generate single mask for entire batch
        mask_ratio = torch.rand(1, device=device) * self.max_mask_ratio
        num_masked = max(1, int(self.seq_len * mask_ratio))
        rand_vals = torch.rand(self.seq_len, device=device)
        _, mask_indices = torch.topk(rand_vals, num_masked, largest=False)

        mask_pos = torch.zeros(self.seq_len, dtype=torch.bool, device=device)
        mask_pos[mask_indices] = True
        mask_pos = mask_pos.unsqueeze(0).expand(batch_size, -1)  # [B, S]

        # 2. Create feature embeddings
        num_emb = torch.stack([
            proj(x_num[:, i].unsqueeze(-1))
            for i, proj in enumerate(self.num_projs)
        ], dim=1)

        cat_emb = torch.stack([
            emb(x_cat[:, i])
            for i, emb in enumerate(self.cat_embeds)
        ], dim=1)

        features = torch.cat([num_emb, cat_emb], dim=1)  # [B, S, D]

        # 3. Apply masking
        mask_token = self.mask_token.expand(batch_size, self.seq_len, -1)
        features = torch.where(
            mask_pos.unsqueeze(-1),
            mask_token,
            features
        )

        # 4. Add positional encoding
        features += self.pos_enc

        # 5. Transformer encoding
        encoded = self.encoder(
            src=features,
            src_key_padding_mask=mask_pos
        )

        return {
            'encoded': encoded,
            'mask_pos': mask_pos,
            'mask_ratio': mask_ratio
        }

class FeatureSpecificEmbedding(nn.Module):
    """Legacy component (optional)"""
    def __init__(self, num_features, d_model):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.randn(num_features, d_model) * 0.02)

    def forward(self, x):
        return x + self.embedding.unsqueeze(0)
