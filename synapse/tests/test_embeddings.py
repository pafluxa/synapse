import pytest

import torch

from einops import rearrange

# from synapse.layers.embeddings import CategoricalEmbedding
from synapse.layers.embeddings import NumericalEmbedding
from synapse.layers.feature_encoders import Zwei

# def test_numerical_embedding_shapes_and_padding():
#     max_depths = [2, 5]
#     num_emb = NumericalEmbedding(max_depths)
#     x = torch.tensor([[0, 1], [1, 4]]).T  # Shape (2, 2)

#     padded_emb, mask = num_emb(x)

#     # Check output shapes
#     assert padded_emb.shape == (2, 2, 4)  # (n_features, batch_size, max_emb_dim)
#     assert mask.shape == (2, 2)            # (batch_size, n_features)

#     # Verify padding in first feature's embeddings
#     assert torch.all(padded_emb[0, :, 3] == torch.tensor([-1, -1]))

#     # Mask should be False (padding only occurs beyond original embedding dim)
#     assert not mask.any()

# def test_categorical_embedding_shapes_and_padding():
#     cardinalities = [3, 5]
#     cat_emb = CategoricalEmbedding(cardinalities)
#     x = torch.tensor([[0, 1, 2], [0, 1, 4]]).T  # Shape (2, 3)

#     padded_emb, mask = cat_emb(x)

#     # Check output shapes
#     assert padded_emb.shape == (2, 3, 4)  # (n_features, batch_size, max_emb_dim)
#     assert mask.shape == (3, 2)            # (batch_size, n_features)

#     # Verify padding in first feature's embeddings
#     assert torch.all(padded_emb[0, :, 3] == torch.tensor([-1, -1, -1]))

#     # Mask should be False (no padding in first embedding dimension)
#     assert not mask.any()

# def test_embeddings_with_layer_norm():
#     # Verify LayerNorm is applied correctly
#     max_depths = [3]
#     num_emb = NumericalEmbedding(max_depths)
#     x = torch.tensor([[1]]).T  # Shape (1, 1)

#     padded_emb, _ = num_emb(x)
#     emb_layer = num_emb.embedding_layers_[0]

#     # Manually compute expected output
#     manual_emb = emb_layer(x[0])

#     assert torch.allclose(padded_emb[0, 0, :], manual_emb[0], atol=1e-6)

def test_zwei_with_numerical_embedding():
    # Test integration between Zwei and NumericalEmbedding
    feature_ranges = [(0.0, 8.0), (-1.0, 4.0)]
    zwei_max_depths = [16, 3]

    # Create components
    zwei = Zwei(feature_ranges, zwei_max_depths)
    numerical_emb = NumericalEmbedding(max_depths=zwei_max_depths)  # 2 possible values per binary token

    # Create test input
    x = torch.tensor([[5.0, 3.0], [-0.5, 3.5], [1.0, 4.0]])  # Batch size 2, 2 features

    # Get tokens from Zwei [n_features, batch_size, max_depth]
    print('x input shape = ', x.shape)
    tokens = zwei(x)
    for tok in tokens:
        print('tokens shape = ', tok.shape)
    # Reshape tokens for NumericalEmbedding: treat each depth as separate feature
    # n_features, batch_size, max_depth = tokens.shape
    # tokens = rearrange(tokens, 'n b s -> b (n s)')
    # Convert padding (-1) to valid indices (0) for embedding
    # valid_tokens = torch.where(reshaped_tokens == -1, 0, reshaped_tokens)

    # Get embeddings
    padded_emb, mask = numerical_emb(tokens)
    print(padded_emb.shape, mask.shape)

    # Verify shapes
    assert padded_emb.shape == (
        n_features,  # Total "features" (original features * max_depth)
        batch_size,
        numerical_emb.max_emb_dim_
    )

    # Verify padding mask (should mask original padding locations)
    assert mask.shape == (batch_size, n_features)

    # Check mask for padded positions (depth exceeding feature's max_depth)
    expected_mask = torch.tensor([
        # Feature 0 depths (3 real + 0 padded)
        [False, False, False,  # Feature 0, depths 0-2
         True, True, True],    # Padding from feature 1 depths 2 (original max_depth=2)
        [False, False, False,  # Feature 0, depths 0-2
         True, True, True]
    ])
    assert torch.all(mask == expected_mask)
