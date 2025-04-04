import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        scale = self.fc(avg_out + max_out).view(b, c, 1)
        return x * scale

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(combined)
        return x * self.sigmoid(att)

class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention1D(channels, reduction)
        self.sa = SpatialAttention1D(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 expansion_ratio=6, attention=True):
        super().__init__()
        expanded = int(in_channels * expansion_ratio)
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            # Pointwise
            nn.Conv1d(in_channels, expanded, 1, bias=False),
            nn.BatchNorm1d(expanded),
            nn.SiLU(),
            # Depthwise
            nn.Conv1d(expanded, expanded, kernel_size, stride, padding,
                      groups=expanded, bias=False),
            nn.BatchNorm1d(expanded),
            # nn.SiLU(),
            # CBAM Attention
            CBAM1D(expanded) if attention else nn.Identity(),
            # Pointwise linear
            nn.Conv1d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class EmbeddingEncoder(nn.Module):
    def __init__(self,
                 num_features: int,  # Original number of features (columns)
                 embedding_dim: int,  # Size of each feature's embedding
                 codec_dim: int = 32,
                 width_mult: float = 1.0,
                 depth_mult: float = 1.0,
                 base_channels: int = 64):
        super().__init__()

        # Scale channels and depth
        channels = [max(round(c * width_mult), 1) for c in
                   [base_channels, base_channels*2, base_channels*4]]
        depths = [max(round(d * depth_mult), 1) for d in [2, 4, 2]]

        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(channels[0]),
            nn.SiLU()
        )

        # Feature dimensions: [B, 1, num_features, embedding_dim]
        # We process features as 1D signals along the feature dimension

        # Stage 1
        self.stage1 = self._make_layer(
            channels[0], channels[0], depths[0], kernel_size=3)

        # Stage 2
        self.stage2 = self._make_layer(
            channels[0], channels[1], depths[1], kernel_size=5)

        # Stage 3
        self.stage3 = self._make_layer(
            channels[1], channels[2], depths[2], kernel_size=7)

        # Final pooling and projection
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.projection = nn.Linear(channels[-1], codec_dim)

    def _make_layer(self, in_channels, out_channels, depth, kernel_size):
        layers = []
        layers.append(ConvBlock(in_channels, out_channels,
                              kernel_size=kernel_size, stride=2))
        for _ in range(1, depth):
            layers.append(ConvBlock(out_channels, out_channels,
                                  kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Input shape: [B, num_features, embedding_dim]"""
        # Add channel dimension: [B, 1, num_features, embedding_dim]
        x = x.unsqueeze(1)

        # Process each embedding dimension independently
        b, _, n_features, emb_dim = x.size()
        x = x.permute(0, 1, 3, 2)  # [B, 1, emb_dim, n_features]
        x = x.reshape(b, 1, -1)     # [B, 1, emb_dim * n_features]

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # Pool across spatial dimensions
        x = self.adaptive_pool(x)  # [B, C, 1]
        x = x.view(x.size(0), -1)  # [B, C]
        return self.projection(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
