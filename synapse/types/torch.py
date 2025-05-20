"""
Type aliases for PyTorch tensors and devices, for clearer type hints.
"""

from typing import TypeAlias
import torch

FloatTensor: TypeAlias = torch.FloatTensor
LongTensor: TypeAlias = torch.LongTensor
Tensor1D: TypeAlias = torch.Tensor  # shape: (N,)
Tensor2D: TypeAlias = torch.Tensor  # shape: (N, M)
Device: TypeAlias = torch.device
