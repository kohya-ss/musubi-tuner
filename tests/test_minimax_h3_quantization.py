import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.minimax_h3.quantization import (
    H3Int8Embedding,
    H3Nvfp4Linear,
    inspect_comfy_quantized_layers,
    inspect_safetensors_tensors,
    load_comfy_quantized_model,
    swap_nvfp4_nibbles,
)
from musubi_tuner.minimax_h3 import model as h3_model
from musubi_tuner.networks.lora import LoRAModule


def _marker(config: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")), dtype=torch.uint8)


def _load_quantized_tiny(checkpoint: Path, factory):
    specs = inspect_safetensors_tensors([checkpoint])
    quantized = inspect_comfy_quantized_layers([checkpoint], specs)
    return load_comfy_quantized_model(
        factory,
        [checkpoint],
        quantized,
        device="cpu",
        output_dtype=torch.float32,
    )


def test_transformer_routes_non_convrot_artifacts_to_generic_loader(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "nvfp4-transformer.safetensors"
    save_file({"probe": torch.zeros(1)}, checkpoint)
    layer_specs = {"blocks.0.attn.out_proj": object()}
    config = object()
    loaded = SimpleNamespace()
    captured = {}

    monkeypatch.setattr(h3_model, "inspect_safetensors_tensors", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(h3_model, "inspect_comfy_quantized_layers", lambda *_args, **_kwargs: layer_specs)
    monkeypatch.setattr(h3_model, "uses_generic_comfy_loader", lambda value: value is layer_specs)
    monkeypatch.setattr(h3_model, "load_safetensors_metadata", lambda *_args, **_kwargs: {"config": "{}"})
    monkeypatch.setattr(h3_model, "parse_h3_transformer_config", lambda *_args, **_kwargs: config)

    def fake_load(factory, files, layers, **kwargs):
        captured.update(factory=factory, files=files, layers=layers, kwargs=kwargs)
        return loaded

    monkeypatch.setattr(h3_model, "load_comfy_quantized_model", fake_load)

    result = h3_model.load_h3_transformer(checkpoint, device="cpu", dtype=torch.bfloat16)

    assert result is loaded
    assert captured["layers"] is layer_specs
    assert captured["kwargs"]["output_dtype"] is torch.bfloat16


def test_prequantized_convrot_stays_linear_supports_lora_and_backward(tmp_path: Path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=False)

    checkpoint = tmp_path / "convrot.safetensors"
    save_file(
        {
            "linear.weight": torch.tensor(
                [[1, 2, 3, 4], [-4, -3, -2, -1], [1, -1, 1, -1], [2, 0, -2, 0]],
                dtype=torch.int8,
            ),
            "linear.weight_scale": torch.tensor([0.25, 0.5, 1.0, 2.0]),
            "linear.comfy_quant": _marker(
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}
            ),
        },
        checkpoint,
    )

    model = _load_quantized_tiny(checkpoint, Tiny)

    assert model.linear.__class__.__name__ == "Linear"
    assert model.linear.weight.dtype == torch.int8
    assert model.linear._h3_quant_format == "int8_convrot"
    scales_before_cast = model.linear.weight_scale.view(torch.float32).clone()
    model.to(dtype=torch.float16)
    assert model.linear.weight_scale.dtype == torch.uint8
    torch.testing.assert_close(model.linear.weight_scale.view(torch.float32), scales_before_cast)

    adapter = LoRAModule("tiny", model.linear, lora_dim=2, alpha=2)
    adapter.apply_to()
    with torch.no_grad():
        adapter.lora_up.weight.fill_(0.1)
    inputs = torch.randn(3, 4, requires_grad=True)
    adapter(inputs).square().mean().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert adapter.lora_down.weight.grad is not None
    assert adapter.lora_up.weight.grad is not None


def _to_blocked(scales: torch.Tensor) -> torch.Tensor:
    rows, cols = scales.shape
    row_blocks = (rows + 127) // 128
    col_blocks = (cols + 3) // 4
    padded = torch.zeros(row_blocks * 128, col_blocks * 4, dtype=scales.dtype)
    padded[:rows, :cols] = scales
    blocks = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def test_nvfp4_nibble_unswizzle_and_awq_forward(tmp_path: Path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 2, bias=False)

    # Low-nibble-first normalized storage: even columns are 0.5 (code 1),
    # odd columns are 1.0 (code 2). The checkpoint stores the pair reversed.
    normalized_packed = torch.full((2, 8), (2 << 4) | 1, dtype=torch.uint8)
    checkpoint_packed = swap_nvfp4_nibbles(normalized_packed)
    row_major_scales = torch.ones(2, 1, dtype=torch.float8_e4m3fn)
    checkpoint_scales = _to_blocked(row_major_scales)
    checkpoint = tmp_path / "nvfp4.safetensors"
    save_file(
        {
            "linear.weight": checkpoint_packed,
            "linear.weight_scale": checkpoint_scales,
            "linear.weight_scale_2": torch.tensor(1.0),
            "linear.pre_quant_scale": torch.full((16,), 2.0),
            "linear.comfy_quant": _marker({"format": "nvfp4"}),
        },
        checkpoint,
    )

    model = _load_quantized_tiny(checkpoint, Tiny)

    assert isinstance(model.linear, H3Nvfp4Linear)
    assert model.linear.weight.shape == (2, 8)
    assert model.linear.weight_scale.shape == (2, 1)
    model.to(dtype=torch.bfloat16)
    assert model.linear.weight_scale.dtype == torch.uint8
    assert model.linear.weight_scale_2.dtype == torch.uint8
    assert model.linear.pre_quant_scale.dtype == torch.uint8
    inputs = torch.ones(1, 16)
    output = model.linear(inputs)
    # sum([0.5, 1.0] * 8) * AWQ input scale 2
    torch.testing.assert_close(output, torch.full((1, 2), 24.0))


def test_int8_embedding_dequantizes_only_selected_rows_and_survives_dtype_cast(tmp_path: Path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(3, 4)

    checkpoint = tmp_path / "embedding.safetensors"
    save_file(
        {
            "embedding.weight": torch.tensor([[1, 2, 3, 4], [-1, -2, -3, -4], [2, 4, 6, 8]], dtype=torch.int8),
            "embedding.weight_scale": torch.tensor([0.5, 1.0, 0.25]),
            "embedding.comfy_quant": _marker({"format": "int8_tensorwise"}),
        },
        checkpoint,
    )

    model = _load_quantized_tiny(checkpoint, Tiny)
    assert isinstance(model.embedding, H3Int8Embedding)
    model.to(dtype=torch.bfloat16)
    actual = model.embedding(torch.tensor([[2, 0]]))
    expected = torch.tensor([[[0.5, 1.0, 1.5, 2.0], [0.5, 1.0, 1.5, 2.0]]])
    torch.testing.assert_close(actual, expected)
