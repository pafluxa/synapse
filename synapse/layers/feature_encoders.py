from typing import Iterable, Tuple

from einops import rearrange

import torch
import torch.nn as nn

class Zwei(nn.Module):

    def __init__(self,
                 feature_ranges: Iterable[Tuple[float, float]],
                 max_depths: Iterable[int]):
        """
        Vectorized binary encoder with padding to max depth

        Args:
            feature_ranges: List of (min, max) tuples for each feature
            max_depths: Either single int or list of ints per feature
        """
        super().__init__()

        self.min_emb_dim_ = max([0, min(max_depths)])
        self.max_emb_dim_ = max([0, max(max_depths)])

        # Convert to tensors
        self.feature_ranges = torch.tensor(feature_ranges, dtype=torch.float32)
        self.n_features = len(feature_ranges)

        self.max_depths = torch.tensor(max_depths, dtype=torch.long)

        # Determine padding
        self.max_depth = self.max_depths.max().item()
        self.total_tokens = self.n_features * self.max_depth

        # Register buffers
        self.register_buffer('depth_mask', self._create_depth_mask())

    def _create_depth_mask(self) -> torch.Tensor:
        """Create mask for valid tokens (1=real, 0=padding)"""
        mask = torch.zeros(self.n_features, self.max_depth, dtype=torch.bool)
        for i, depth in enumerate(self.max_depths):
            mask[i, :depth] = 1
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, n_features)

        Returns:
            Padded token tensor of shape (batch_size, n_features, max_depth)
            with -1 for padding positions
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize bounds [B, F]
        mins = self.feature_ranges[:, 0].expand(batch_size, -1).to(device)
        maxs = self.feature_ranges[:, 1].expand(batch_size, -1).to(device)
        lows = mins.clone()
        highs = maxs.clone()

        # Prepare output tensor with padding
        tokens = torch.full((batch_size, self.n_features, self.max_depth),
                          -1, dtype=torch.long, device=device)

        # Vectorized encoding
        for depth in range(self.max_depth):
            # Only process features that need this depth
            active = depth < self.max_depths.to(device)
            active_mask = active.unsqueeze(0)  # [1, F]

            # Compute decisions for active features
            mids = (lows + highs) / 2
            decisions = ((x >= mids) & active_mask).byte()  # [B, F]

            # Store tokens
            tokens[:, :, depth] = torch.where(
                active_mask,
                decisions,
                torch.tensor(-1, dtype=torch.long, device=device)
            )

            # Update bounds
            lows = torch.where(active_mask & (decisions == 1), mids, lows)
            highs = torch.where(active_mask & (decisions == 0), mids, highs)

        tokens = rearrange(tokens, 'b f d -> f b d')
        tokens = tokens.long()

        return tokens
