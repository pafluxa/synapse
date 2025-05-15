"""Feature encoders for numerical data.

This module provides a binary encoder for numerical features that converts continuous
values into discrete tokens using a binary search approach.

Classes:
    Zwei: Binary encoder for numerical features with padding.
"""

from typing import Iterable, Tuple, List

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class Zwei(nn.Module):
    """Vectorized binary encoder with padding to max depth.

    This encoder converts numerical features into binary tokens using a binary search
    approach, with proper padding for variable-length encodings.

    Attributes:
        min_emb_dim_: Minimum embedding dimension across all features.
        max_emb_dim_: Maximum embedding dimension across all features.
        feature_ranges: Tensor containing (min, max) ranges for each feature.
        n_features: Number of features being encoded.
        max_depths: Tensor containing maximum depths for each feature.
        max_depth: Maximum depth across all features.
        total_tokens: Total number of tokens (n_features * max_depth).
        pad_token: Padding token value (default: -1).
        depth_mask: Mask indicating valid tokens (1=real, 0=padding).
    """

    def __init__(
        self,
        feature_ranges: Iterable[Tuple[float, float]],
        max_depths: Iterable[int],
    ) -> None:
        """Initialize the Zwei encoder.

        Args:
            feature_ranges: List of (min, max) tuples for each feature.
            max_depths: Either single int or list of ints per feature.
        """
        super().__init__()

        self.min_emb_dim_ = max([0, min(max_depths)])
        self.max_emb_dim_ = max([0, max(max_depths)])

        # Convert to tensors
        self.feature_ranges = torch.tensor(feature_ranges, dtype=torch.float32)
        self.n_features = len(list(feature_ranges))
        self.max_depths = torch.tensor(max_depths, dtype=torch.long)

        # Determine padding
        self.max_depth = self.max_depths.max().item()
        self.total_tokens = self.n_features * self.max_depth
        self.padval_ = -100

        # Register buffers
        self.register_buffer('depth_mask', self._create_depth_mask())

    def _create_depth_mask(self) -> torch.Tensor:
        """Create mask for valid tokens (1=real, 0=padding).

        Returns:
            Boolean tensor of shape (n_features, max_depth) indicating valid tokens.
        """
        mask = torch.zeros(self.n_features, self.max_depth, dtype=torch.bool)
        for i, depth in enumerate(self.max_depths):
            mask[i, :depth] = 1
        return mask

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Convert numerical features to binary tokens with padding.

        Args:
            x: Input tensor of shape (batch_size, n_features).

        Returns:
            List of token tensors, one for each feature, with variable lengths.
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize bounds [B, F]
        mins = self.feature_ranges[:, 0].expand(batch_size, -1).to(device)
        maxs = self.feature_ranges[:, 1].expand(batch_size, -1).to(device)
        lows = mins.clone()
        highs = maxs.clone()

        # Prepare output tensor with padding
        tokens = torch.full(
            (batch_size, self.n_features, self.max_depth),
            -1,
            dtype=torch.long,
            device=device
        )

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

        tokens = tokens.long()
        mask = tokens == -1

        return tokens , mask
