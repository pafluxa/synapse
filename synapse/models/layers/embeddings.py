import torch
from torch import nn
import torch.nn.functional as F

class NumericalEmbedding(nn.Module):
    def __init__(self, num_numerical, embedding_dim):
        super().__init__()
        # Project numerical features to embedding space
        self.proj = nn.Linear(num_numerical, embedding_dim)

    def forward(self, x):
        # x: [batch_size, num_numerical]
        return self.proj(x)  # [batch_size, embedding_dim]

class CategoricalEmbedding(nn.Module):
    def __init__(self, categorical_dims, embedding_dim):
        super().__init__()
        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=embedding_dim)
            for dim in categorical_dims
        ])

    def forward(self, x_categorical):
        # x_categorical: [batch_size, num_categorical]
        embedded = [emb(x_categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embedded, dim=1).sum(dim=1)  # [batch_size, embedding_dim]


class FeatureSpecificEmbedding(nn.Module):
    def __init__(self, num_features, d_model):
        super().__init__()
        self.feature_pos = nn.Parameter(torch.randn(num_features, d_model))

    def forward(self, x_embeddings):
        # x_embeddings: [batch, num_features, d_model] (pre-combined numerical + categorical)
        return x_embeddings + self.feature_pos.unsqueeze(0)  # Add per-feature offset


class InverseEmbedding(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.embedding = embedding_layer  # The original embedding layer

    def forward(self, x):
        # x: shape [..., embedding_dim]
        x_norm = F.normalize(x, p=2, dim=-1)  # L2 normalize input
        emb_norm = F.normalize(self.embedding.weight, p=2, dim=-1)  # L2 normalize embeddings
        similarities = x_norm @ emb_norm.T

        return similarities
        # return F.softmax(similarities, dim=-1)
        # return torch.argmax(similarities, dim=-1)

class EnhancedInverseEmbedding(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.embedding = embedding_layer
        self.proj1 = nn.Linear(embedding_layer.embedding_dim,
                              embedding_layer.embedding_dim*2)
        self.proj2 = nn.Linear(embedding_layer.embedding_dim*2,
                              embedding_layer.embedding_dim)
        self.ln = nn.LayerNorm(embedding_layer.embedding_dim)

    def forward(self, x):
        residual = x
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        x = self.ln(x + residual)  # Skip connection
        return x @ self.embedding.weight.T
