import pytest
import torch

from synapse.layers.feature_encoders import Zwei

def test_zwei_encoding():
    # Test Zwei with a sample input
    feature_ranges = [(0.0, 8.0), (0.0, 4.0)]
    max_depths = [3, 2]
    zwei = Zwei(feature_ranges, max_depths)
    x = torch.tensor([[5.0, 3.0], [6.0, 2.0]])  # Batch size 2, 2 features

    tokens = zwei(x)

    # Check output shape
    assert tokens.shape == (2, 2, 3)  # (n_features, batch_size, max_depth)

    # Test first feature encoding
    assert torch.all(tokens[0, 0, :] == torch.tensor([1, 0, 1]))  # 5.0 → [1,0,1]
    assert torch.all(tokens[0, 1, :] == torch.tensor([1, 1, 0]))  # 6.0 → [1,1,0]

    # Test second feature encoding and padding
    assert torch.all(tokens[1, 0, :2] == torch.tensor([1, 1]))    # 3.0 → [1,1]
    assert tokens[1, 0, 2] == -1  # Padding for second feature
    assert tokens[1, 1, 2] == -1  # Padding for second feature
