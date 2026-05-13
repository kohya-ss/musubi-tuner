import pytest
import torch
from musubi_tuner.utils.dit_output import DitOutput


def test_fields_accessible():
    pred = torch.zeros(2, 4, 8)
    target = torch.ones(2, 4, 8)
    out = DitOutput(pred=pred, target=target)
    assert out.pred is pred
    assert out.target is target
    assert out.features is None


def test_features_field():
    feat = torch.randn(2, 4, 8)
    out = DitOutput(pred=torch.zeros(1), target=torch.zeros(1), features=feat)
    assert out.features is feat


def test_repr_contains_class_name():
    out = DitOutput(pred=torch.zeros(1), target=torch.zeros(1))
    assert "DitOutput" in repr(out)


def test_not_iterable_by_default():
    out = DitOutput(pred=torch.zeros(1), target=torch.zeros(1))
    with pytest.raises(TypeError):
        a, b = out
