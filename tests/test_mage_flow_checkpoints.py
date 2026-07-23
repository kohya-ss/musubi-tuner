from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.mage_flow import model as model_module
from musubi_tuner.mage_flow.model import MageFlow, load_mage_flow_transformer
from musubi_tuner.mage_flow.utils import ComponentValidationError, MageFlowConfig, inspect_component, normalize_dit_state_dict


def tiny_config(depth=2):
    return MageFlowConfig(
        in_channels=4,
        out_channels=4,
        context_in_dim=7,
        hidden_size=16,
        depth=depth,
        num_heads=2,
        axes_dim=(2, 2, 4),
        text_max_length=16,
    )


def write_checkpoint(path: Path, config=None, prefix=""):
    config = config or tiny_config()
    state = MageFlow(config).state_dict()
    save_file({prefix + key: value.contiguous() for key, value in state.items()}, str(path))
    return state


def test_loader_round_trips_a_strict_canonical_checkpoint(tmp_path):
    path = tmp_path / "dit.safetensors"
    expected = write_checkpoint(path)

    loaded = load_mage_flow_transformer(path, device="cpu", dtype=torch.float32, _config=tiny_config())

    assert loaded.training is False
    assert loaded.config == tiny_config()
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, expected[key])


@pytest.mark.parametrize("prefix", ["_orig_mod.", "transformer."])
def test_loader_accepts_only_documented_exact_prefix_layouts(tmp_path, prefix):
    path = tmp_path / "dit.safetensors"
    expected = write_checkpoint(path, prefix=prefix)

    inspected = inspect_component(path, "dit", config=tiny_config())
    loaded = load_mage_flow_transformer(path, device="cpu", dtype=torch.float32, _config=tiny_config())

    assert inspected.layout in {"official_compiled", "component_prefixed"}
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, expected[key])


def test_normalizer_rejects_mixed_or_unknown_layouts():
    with pytest.raises(ComponentValidationError, match="mixed"):
        normalize_dit_state_dict({"img_in.weight": torch.zeros(1), "transformer.img_in.bias": torch.zeros(1)})
    with pytest.raises(ComponentValidationError, match="unknown"):
        normalize_dit_state_dict({"model.diffusion_model.img_in.weight": torch.zeros(1)})


def test_bad_header_is_rejected_before_model_construction(tmp_path, monkeypatch):
    path = tmp_path / "bad.safetensors"
    state = write_checkpoint(path)
    del state
    tensors = MageFlow(tiny_config()).state_dict()
    tensors = {key: value for key, value in tensors.items() if not key.startswith("transformer_blocks.1.")}
    save_file(tensors, str(path))

    monkeypatch.setattr(model_module, "MageFlow", lambda *_args, **_kwargs: pytest.fail("allocated model before validation"))
    with pytest.raises(ComponentValidationError, match=r"transformer blocks.*expected \[0, 1\].*actual \[0\]"):
        load_mage_flow_transformer(path, device="cpu", _config=tiny_config())


def test_header_reports_shape_missing_and_unexpected_keys(tmp_path):
    path = tmp_path / "bad.safetensors"
    tensors = MageFlow(tiny_config()).state_dict()
    tensors["img_in.weight"] = torch.zeros(15, 4)
    del tensors["proj_out.bias"]
    tensors["mystery.weight"] = torch.zeros(1)
    save_file(tensors, str(path))

    with pytest.raises(ComponentValidationError) as error:
        inspect_component(path, "dit", config=tiny_config())

    message = str(error.value)
    assert "img_in.weight" in message
    assert "expected (16, 4), actual (15, 4)" in message
    assert "proj_out.bias" in message
    assert "mystery.weight" in message


@pytest.mark.parametrize(
    "path_factory,match",
    [
        (lambda root: root / "missing.safetensors", "does not exist"),
        (lambda root: root, "regular file"),
        (lambda root: root / "dit.bin", r"\.safetensors"),
    ],
)
def test_component_path_must_be_one_regular_safetensors_file(tmp_path, path_factory, match):
    path = path_factory(tmp_path)
    if path.suffix == ".bin":
        path.write_bytes(b"not safetensors")
    with pytest.raises(ComponentValidationError, match=match):
        inspect_component(path, "dit", config=tiny_config())


def test_released_loader_rejects_tiny_architecture_before_allocation(tmp_path, monkeypatch):
    path = tmp_path / "tiny.safetensors"
    write_checkpoint(path)
    monkeypatch.setattr(model_module, "MageFlow", lambda *_args, **_kwargs: pytest.fail("allocated incompatible released model"))

    with pytest.raises(ComponentValidationError, match="transformer blocks"):
        load_mage_flow_transformer(path, device="cpu")
