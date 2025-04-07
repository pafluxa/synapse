"""Custom type definitions for the Synapse project.

This module defines type aliases and custom types for tensors, sequences, and other
common objects used throughout the project. These types help with static type checking
and provide better documentation of expected shapes and dtypes.
"""

from typing import (
    Any, Dict, List, Optional, Sequence, Tuple, Union,
    TypeVar, Generic
)

import torch
from torch import Tensor
import numpy as np

# Generic type variable for torch tensors
T = TypeVar('T', bound=torch.dtype)

# Basic tensor types with dtype specification
class FloatTensor(Generic[T], Tensor): ...
class LongTensor(Generic[T], Tensor): ...
class BoolTensor(Generic[T], Tensor): ...

# -----------------------------------------------------------------------------
# Tensor Types with Shape Annotations (using torch.jit.annotations)
# -----------------------------------------------------------------------------

# 1D tensors
class Tensor1D(FloatTensor):  # shape: [D]
    pass

class LongTensor1D(LongTensor):  # shape: [D]
    pass

# 2D tensors
class Tensor2D(FloatTensor):  # shape: [B, D]
    pass

class LongTensor2D(LongTensor):  # shape: [B, D]
    pass

# 3D tensors
class Tensor3D(FloatTensor):  # shape: [B, S, D]
    pass

class LongTensor3D(LongTensor):  # shape: [B, S, D]
    pass

# 4D tensors
class Tensor4D(FloatTensor):  # shape: [B, S1, S2, D]
    pass

# -----------------------------------------------------------------------------
# Specialized Tensor Types
# -----------------------------------------------------------------------------

# Embedding tensors
class EmbeddingTensor(Tensor3D):  # shape: [B, S, D]
    pass

class AttentionMask(BoolTensor):  # shape: [B, S] or [B, S, S]
    pass

class PaddingMask(BoolTensor):  # shape: [B, S]
    pass

# -----------------------------------------------------------------------------
# Collection Types
# -----------------------------------------------------------------------------

# For variable-length sequences
TensorSequence = Union[
    List[Tensor],
    Tuple[Tensor, ...]
]

# For numerical ranges
RangeTuple = Tuple[float, float]
RangeList = Sequence[RangeTuple]

# For device specification
Device = Union[str, torch.device]

# -----------------------------------------------------------------------------
# Function Annotations
# -----------------------------------------------------------------------------

# Type for forward pass return values
ForwardResult = Union[
    Tensor,
    Tuple[Tensor, ...],
    Dict[str, Tensor]
]

# Type for loss functions
LossFunction = Callable[[Tensor, Tensor], Tensor]

# -----------------------------------------------------------------------------
# Configuration Types
# -----------------------------------------------------------------------------

# For model configuration
class ModelConfig(TypedDict):
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float
    activation: str

# For training configuration
class TrainingConfig(TypedDict):
    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float

# -----------------------------------------------------------------------------
# Type Checking Utilities
# -----------------------------------------------------------------------------

def is_tensor_of_shape(x: Any, shape: Tuple[Optional[int], ...]) -> bool:
    """Check if a tensor matches the given shape pattern.

    Args:
        x: Object to check
        shape: Tuple where each element is either an integer (exact match)
               or None (any size allowed)

    Returns:
        bool: True if x is a tensor matching the shape pattern
    """
    if not isinstance(x, Tensor):
        return False
    if len(x.shape) != len(shape):
        return False
    for dim, pattern in zip(x.shape, shape):
        if pattern is not None and dim != pattern:
            return False
    return True
