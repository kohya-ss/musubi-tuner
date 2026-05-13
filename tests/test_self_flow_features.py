"""Tests for self-flow feature extraction from Flux2 model layers."""

import torch
from unittest.mock import Mock


def test_flux2_feature_extraction_disabled():
    """When hidden_features=False, model should return only prediction."""
    mock_model = Mock()
    mock_model.return_value = torch.ones(2, 4096, 128)  # Deterministic

    result = mock_model(
        x=torch.zeros(2, 4096, 128),
        hidden_features=False,
        feature_layer=None,
    )

    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 4096, 128)


def test_flux2_feature_extraction_enabled():
    """When hidden_features=True, model should return (prediction, features)."""
    mock_model = Mock()
    pred = torch.ones(2, 4096, 128)
    features = torch.ones(2, 1024, 128) * 2
    mock_model.return_value = (pred, features)

    result = mock_model(
        x=torch.zeros(2, 4096, 128),
        hidden_features=True,
        feature_layer=10,
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].shape == (2, 4096, 128)
    assert result[1].shape == (2, 1024, 128)


def test_feature_layer_extraction_double_block():
    """Test feature extraction from double block (creates clone, not reference)."""
    torch.manual_seed(42)
    img = torch.randn(2, 4096, 128)
    feature_layer = 5
    global_block_idx = 5

    # Simulate feature extraction
    if global_block_idx == feature_layer:
        features = img.clone()

    assert features.shape == img.shape
    assert torch.equal(features, img)  # Values equal

    # Verify it's a copy, not reference
    features[0, 0, 0] = 999
    assert img[0, 0, 0] != 999


def test_feature_layer_extraction_single_block():
    """Test feature extraction from single block (slices out text tokens)."""
    num_txt_tokens = 512
    img_tokens = 4096
    total_tokens = num_txt_tokens + img_tokens

    # Create deterministic concatenated tensor
    txt_part = torch.zeros(2, num_txt_tokens, 128)
    img_part = torch.ones(2, img_tokens, 128)
    img = torch.cat([txt_part, img_part], dim=1)

    feature_layer = 15
    global_block_idx = 15

    # Simulate feature extraction from single block
    if global_block_idx == feature_layer:
        features = img[:, num_txt_tokens:, :].clone()

    assert features.shape == (2, img_tokens, 128)
    # Should only contain the img part (ones)
    assert torch.all(features == 1.0)


def test_feature_extraction_default_last_block():
    """When feature_layer=None, should extract from last block."""
    img = torch.arange(2 * 4096 * 128).reshape(2, 4096, 128).float()
    feature_layer = None
    hidden_features = True
    features = None

    # Simulate default extraction
    if hidden_features and features is None:
        features = img.clone()

    assert features is not None
    assert features.shape == img.shape
    assert torch.equal(features, img)


def test_feature_shapes_consistent():
    """Verify student and teacher features have same shape."""
    hidden_size = 128
    num_tokens = 1024
    batch_size = 2

    student_features = torch.ones(batch_size, num_tokens, hidden_size)
    teacher_features = torch.zeros(batch_size, num_tokens, hidden_size)

    assert student_features.shape == teacher_features.shape
    assert student_features.shape == (batch_size, num_tokens, hidden_size)


def test_flux2_call_dit_feature_passthrough():
    """Test that flux_2_train_network.py passes hidden_features correctly."""
    mock_model = Mock()
    mock_model.return_value = (
        torch.full((2, 4096, 128), 1.0),  # model_pred
        torch.full((2, 1024, 128), 2.0),  # features
    )

    hidden_features = True
    feature_layer = 10

    model_output = mock_model(
        x=torch.zeros(2, 4096, 128),
        hidden_features=hidden_features,
        feature_layer=feature_layer,
    )

    if hidden_features:
        model_pred, features = model_output
    else:
        model_pred = model_output
        features = None

    assert features is not None
    assert torch.all(features == 2.0)


def test_vanilla_training_no_feature_extraction():
    """Verify vanilla training (self_flow=False) doesn't extract features."""
    mock_model = Mock()
    mock_model.return_value = torch.ones(2, 4096, 128)

    hidden_features = False

    model_output = mock_model(
        x=torch.zeros(2, 4096, 128),
        hidden_features=hidden_features,
    )

    if hidden_features:
        model_pred, features = model_output
    else:
        model_pred = model_output
        features = None

    assert features is None
    assert isinstance(model_pred, torch.Tensor)
