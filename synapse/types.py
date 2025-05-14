"""Custom type definitions for the Synapse project.

This module defines type aliases and custom types for tensors, sequences, and other
common objects used throughout the project. These types help with static type checking
and provide better documentation of expected shapes and dtypes.
"""

from __future__ import annotations
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar,
    Union, Generic, Protocol, runtime_checkable, TypedDict
)
from typing_extensions import TypeAlias, Literal, ParamSpec

import torch
from torch import Tensor, nn
import numpy as np
import numpy.typing as npt

# -----------------------------------------------------------------------------
# Type Variables and Parameters
# -----------------------------------------------------------------------------
T = TypeVar('T', bound=torch.dtype)
P = ParamSpec('P')  # For parameter preservation in decorators
DType = TypeVar('DType', bound=torch.dtype)
DeviceType = Union[str, torch.device]

# -----------------------------------------------------------------------------
# Basic Tensor Types
# -----------------------------------------------------------------------------
class FloatTensor(Tensor, Generic[DType]): ...
class LongTensor(Tensor, Generic[DType]): ...
class BoolTensor(Tensor, Generic[DType]): ...

# -----------------------------------------------------------------------------
# Shape-Annotated Tensor Types (using NewType pattern)
# -----------------------------------------------------------------------------
# 1D Tensors
Tensor1D: TypeAlias = Tensor  # [D]
FloatTensor1D: TypeAlias = FloatTensor  # [D]
LongTensor1D: TypeAlias = LongTensor  # [D]

# 2D Tensors
Tensor2D: TypeAlias = Tensor  # [B, D]
FloatTensor2D: TypeAlias = FloatTensor  # [B, D]
LongTensor2D: TypeAlias = LongTensor  # [B, D]

# 3D Tensors
Tensor3D: TypeAlias = Tensor  # [B, S, D]
FloatTensor3D: TypeAlias = FloatTensor  # [B, S, D]
LongTensor3D: TypeAlias = LongTensor  # [B, S, D]

# 4D Tensors
Tensor4D: TypeAlias = Tensor  # [B, S1, S2, D]
FloatTensor4D: TypeAlias = FloatTensor  # [B, S1, S2, D]

# -----------------------------------------------------------------------------
# Specialized Tensor Types
# -----------------------------------------------------------------------------
# Embedding types
Embedding: TypeAlias = FloatTensor3D  # [B, S, D]
PositionalEncoding: TypeAlias = FloatTensor3D  # [B, S, D]

# Attention types
AttentionMask: TypeAlias = BoolTensor  # [B, S] or [B, S, S]
PaddingMask: TypeAlias = BoolTensor  # [B, S]

# Loss-related types
Logits: TypeAlias = FloatTensor  # [B, S, V] where V is vocab size
Loss: TypeAlias = FloatTensor1D  # [1]

# -----------------------------------------------------------------------------
# Collection Types
# -----------------------------------------------------------------------------
# Tensor collections
TensorSequence: TypeAlias = Union[Sequence[Tensor], Tensor]
TensorDict: TypeAlias = Dict[str, Tensor]
TensorTuple: TypeAlias = Tuple[Tensor, ...]

# Numerical ranges
Range: TypeAlias = Tuple[float, float]
RangeList: TypeAlias = Sequence[Range]

# Data structures
Batch: TypeAlias = Union[
    Tensor,
    Tuple[Tensor, ...],
    Dict[str, Tensor],
    Sequence[Tensor]
]

# Protocols for duck typing
@runtime_checkable
class ModuleLike(Protocol):
    def forward(self, *args, **kwargs) -> Tensor: ...
    def __call__(self, *args, **kwargs) -> Tensor: ...

@runtime_checkable
class DatasetLike(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Tuple[Tensor, ...]: ...

# -----------------------------------------------------------------------------
# Configuration Types
# -----------------------------------------------------------------------------
class ModelConfig(TypedDict, total=False):
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float
    activation: Literal['relu', 'gelu', 'silu']
    norm_first: bool

class TrainingConfig(TypedDict, total=False):
    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float
    gradient_clip: float
    device: DeviceType

# -----------------------------------------------------------------------------
# Device and Precision Types
# -----------------------------------------------------------------------------
PrecisionType: TypeAlias = Literal['fp32', 'fp16', 'bf16']
Device: TypeAlias = Union[str, torch.device]

# -----------------------------------------------------------------------------
# Numpy Compatibility Types
# -----------------------------------------------------------------------------
NDArray: TypeAlias = npt.NDArray[Any]
FloatNDArray: TypeAlias = npt.NDArray[np.float32]
LongNDArray: TypeAlias = npt.NDArray[np.int64]

# -----------------------------------------------------------------------------
# Type Checking Utilities
# -----------------------------------------------------------------------------
def is_tensor_of_shape(
    x: Any,
    shape: Tuple[Optional[int], ...],
    dtype: Optional[torch.dtype] = None
) -> bool:
    """Check if a tensor matches the given shape and dtype pattern.

    Args:
        x: Object to check
        shape: Tuple where each element is either an integer (exact match)
               or None (any size allowed)
        dtype: Expected dtype (optional)

    Returns:
        bool: True if x is a tensor matching the shape and dtype pattern
    """
    if not isinstance(x, Tensor):
        return False
    if len(x.shape) != len(shape):
        return False
    if dtype is not None and x.dtype != dtype:
        return False
    return all(
        (dim is None or dim == actual)
        for dim, actual in zip(shape, x.shape)
    )

def get_shape(tensor: Tensor) -> Tuple[int, ...]:
    """Get the shape of a tensor with type information for static checking."""
    return tuple(tensor.shape)  # type: ignore
