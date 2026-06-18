import argparse
from dataclasses import dataclass
import importlib.machinery
from pathlib import Path
import sys
import types

import pytest
import torch
from safetensors.torch import load_file, safe_open, save_file


@dataclass
class CacheItem:
    caption: str
    text_encoder_output_cache_path: str


def _parser():
    if "musubi_tuner.lens.lens_text_encoder" not in sys.modules:
        lens_text_encoder = types.ModuleType("musubi_tuner.lens.lens_text_encoder")
        lens_text_encoder.LensTextEmbedder = object
        sys.modules["musubi_tuner.lens.lens_text_encoder"] = lens_text_encoder
    sys.modules.setdefault("musubi_tuner.cache_text_encoder_outputs", types.ModuleType("musubi_tuner.cache_text_encoder_outputs"))
    for name in ("cv2", "av"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            if name == "av":
                module.logging = types.SimpleNamespace(ERROR=40, set_level=lambda *_args, **_kwargs: None)
            sys.modules[name] = module
    from musubi_tuner.lens_cache_text_encoder_outputs import lens_setup_parser

    return lens_setup_parser(argparse.ArgumentParser())


def _item(tmp_path: Path) -> CacheItem:
    return CacheItem("caption", str(tmp_path / "sample_lens_te.safetensors"))


def _features(dtype=torch.bfloat16) -> list[torch.Tensor]:
    base = torch.linspace(-2.0, 2.0, steps=5 * 7, dtype=torch.float32).reshape(5, 7)
    return [base.to(dtype), (base * 0.5 + 0.25).to(dtype)]


def _four_features(dtype=torch.bfloat16) -> list[torch.Tensor]:
    base = _features(dtype)
    return [base[0], base[1], (base[0] * -0.25).to(dtype), (base[1] + 0.75).to(dtype)]


def test_lens_cache_precision_parser_accepts_new_values():
    parser = _parser()

    args = parser.parse_args(
        [
            "--text_encoder",
            "gpt_oss_20b_nvfp4.safetensors",
            "--text_encoder_dtype",
            "fp8",
            "--text_encoder_cache_precision",
            "fp8",
        ]
    )

    assert args.text_encoder_dtype == "fp8"
    assert args.text_encoder_cache_precision == "fp8"
    assert not hasattr(args, "fp8_text_encoder")


def test_lens_text_encoder_dtype_accepts_fp8_dtype():
    _parser()
    from musubi_tuner.lens_cache_text_encoder_outputs import get_lens_text_encoder_dtype

    args = argparse.Namespace(text_encoder_dtype="fp8")

    assert get_lens_text_encoder_dtype(args) == torch.float8_e4m3fn


def test_lens_text_encoder_dtype_preserves_existing_default_and_override():
    _parser()
    from musubi_tuner.lens_cache_text_encoder_outputs import get_lens_text_encoder_dtype

    assert get_lens_text_encoder_dtype(argparse.Namespace(text_encoder_dtype=None)) == torch.bfloat16
    assert get_lens_text_encoder_dtype(argparse.Namespace(text_encoder_dtype="float16")) == torch.float16


def test_lens_fp8_storage_keeps_forward_compute_dtype():
    from musubi_tuner.lens.lens_fp8 import apply_lens_fp8_storage

    class GptOssExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 4
            self.intermediate_size = 3
            self.num_experts = 2
            self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 4, 6, dtype=torch.bfloat16))
            self.gate_up_proj_bias = torch.nn.Parameter(torch.randn(2, 6, dtype=torch.bfloat16))
            self.down_proj = torch.nn.Parameter(torch.randn(2, 3, 4, dtype=torch.bfloat16))
            self.down_proj_bias = torch.nn.Parameter(torch.randn(2, 4, dtype=torch.bfloat16))
            self.alpha = 1.702
            self.limit = 7.0

        def forward(self, *_args, **_kwargs):
            raise AssertionError("apply_lens_fp8_storage should replace this forward")

    class TinyLens(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4).to(torch.bfloat16)
            self.experts = GptOssExperts()

    model = TinyLens()
    converted = apply_lens_fp8_storage(model, torch.float8_e4m3fn)

    assert converted == {"linear": 1, "experts": 1}
    assert model.linear.weight.dtype == torch.float8_e4m3fn
    assert model.experts.gate_up_proj.dtype == torch.float8_e4m3fn
    assert model.experts.down_proj.dtype == torch.float8_e4m3fn
    assert model.experts.gate_up_proj_bias.dtype == torch.bfloat16

    hidden = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    linear_out = model.linear(hidden)
    router_indices = torch.tensor([[0], [1], [0]])
    routing_weights = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.bfloat16)
    expert_out = model.experts(hidden, router_indices=router_indices, routing_weights=routing_weights)

    assert linear_out.dtype == torch.bfloat16
    assert expert_out.shape == hidden.shape
    assert expert_out.dtype == torch.bfloat16
    assert torch.isfinite(expert_out.float()).all()


def test_auto_cache_preserves_legacy_lens_keys_and_dtype(tmp_path):
    from musubi_tuner.lens.lens_text_cache import load_lens_text_cache, save_lens_text_cache

    item = _item(tmp_path)
    save_lens_text_cache(item, _features(torch.bfloat16), "auto")

    sd = load_file(item.text_encoder_output_cache_path)
    assert sorted(sd.keys()) == ["varlen_lens_ctx_0_bfloat16", "varlen_lens_ctx_1_bfloat16"]
    assert sd["varlen_lens_ctx_0_bfloat16"].dtype == torch.bfloat16

    loaded = load_lens_text_cache(item.text_encoder_output_cache_path)
    assert sorted(loaded.keys()) == ["varlen_lens_ctx_0_bfloat16", "varlen_lens_ctx_1_bfloat16"]
    assert torch.equal(loaded["varlen_lens_ctx_0_bfloat16"], _features(torch.bfloat16)[0])


def test_fp8_cache_uses_native_float8_and_remains_collatable(tmp_path):
    from musubi_tuner.lens import lens_utils
    from musubi_tuner.lens.lens_text_cache import load_lens_text_cache, save_lens_text_cache

    item = _item(tmp_path)
    save_lens_text_cache(item, _features(torch.bfloat16), "fp8")

    sd = load_file(item.text_encoder_output_cache_path)
    assert sorted(sd.keys()) == ["varlen_lens_ctx_0_float8_e4m3fn", "varlen_lens_ctx_1_float8_e4m3fn"]
    assert sd["varlen_lens_ctx_0_float8_e4m3fn"].dtype == torch.float8_e4m3fn

    loaded = load_lens_text_cache(item.text_encoder_output_cache_path)
    tensor, mask = lens_utils.pad_lens_text_features([loaded["varlen_lens_ctx_0_float8_e4m3fn"]])
    assert tensor.dtype == torch.float8_e4m3fn
    assert mask.tolist() == [[True, True, True, True, True]]
    assert tensor.to(torch.bfloat16).dtype == torch.bfloat16


def test_nvfp4_cache_round_trips_shape_and_finite_bf16(tmp_path):
    from musubi_tuner.lens.lens_text_cache import load_lens_text_cache, save_lens_text_cache

    item = _item(tmp_path)
    features = _features(torch.bfloat16)
    save_lens_text_cache(item, features, "nvfp4")

    with safe_open(item.text_encoder_output_cache_path, framework="pt") as f:
        metadata = f.metadata()
        keys = set(f.keys())

    assert metadata["lens_text_cache_precision"] == "nvfp4"
    assert "varlen_lens_ctx_0_nvfp4_packed" in keys
    assert "varlen_lens_ctx_0_nvfp4_block_scale" in keys
    assert "varlen_lens_ctx_0_nvfp4_global_scale" in keys

    loaded = load_lens_text_cache(item.text_encoder_output_cache_path)
    assert sorted(loaded.keys()) == ["varlen_lens_ctx_0_bfloat16", "varlen_lens_ctx_1_bfloat16"]
    assert loaded["varlen_lens_ctx_0_bfloat16"].shape == features[0].shape
    assert loaded["varlen_lens_ctx_0_bfloat16"].dtype == torch.bfloat16
    assert torch.isfinite(loaded["varlen_lens_ctx_0_bfloat16"].float()).all()


def test_nvfp4_cache_preserves_empty_varlen_feature(tmp_path):
    from musubi_tuner.lens.lens_text_cache import load_lens_text_cache, save_lens_text_cache

    item = _item(tmp_path)
    empty = torch.empty(0, 7, dtype=torch.bfloat16)
    save_lens_text_cache(item, [empty], "nvfp4")

    loaded = load_lens_text_cache(item.text_encoder_output_cache_path)

    assert loaded["varlen_lens_ctx_0_bfloat16"].shape == (0, 7)
    assert loaded["varlen_lens_ctx_0_bfloat16"].dtype == torch.bfloat16


def test_recache_overwrites_stale_lens_cache_tensors(tmp_path):
    from musubi_tuner.lens.lens_text_cache import save_lens_text_cache

    item = _item(tmp_path)
    save_lens_text_cache(item, _features(torch.bfloat16), "nvfp4")
    save_lens_text_cache(item, _features(torch.bfloat16), "fp8")

    with safe_open(item.text_encoder_output_cache_path, framework="pt") as f:
        keys = set(f.keys())
        metadata = f.metadata()

    assert metadata["lens_text_cache_precision"] == "fp8"
    assert keys == {"varlen_lens_ctx_0_float8_e4m3fn", "varlen_lens_ctx_1_float8_e4m3fn"}


@pytest.mark.parametrize(
    ("cache_precision", "expected_dtype"),
    [
        ("fp8", torch.float8_e4m3fn),
        ("nvfp4", torch.bfloat16),
    ],
)
def test_lens_bucket_loads_compressed_cache_as_training_keys(tmp_path, cache_precision, expected_dtype):
    from musubi_tuner.dataset.architectures import ARCHITECTURE_LENS
    from musubi_tuner.dataset.bucket import BucketBatchManager
    from musubi_tuner.lens import lens_utils
    from musubi_tuner.lens.lens_text_cache import save_lens_text_cache

    item = _item(tmp_path)
    item.latent_cache_path = str(tmp_path / "sample_lens.safetensors")
    save_file({"latents_4x4_bfloat16": torch.zeros(3, 4, 4, dtype=torch.bfloat16)}, item.latent_cache_path)
    save_lens_text_cache(item, _four_features(torch.bfloat16), cache_precision)

    manager = BucketBatchManager({(64, 64): [item]}, batch_size=1, architecture=ARCHITECTURE_LENS)
    batch = manager[0]

    assert "latents" in batch
    for i in range(4):
        assert f"lens_ctx_{i}" in batch
        tensor, mask = lens_utils.pad_lens_text_features(batch[f"lens_ctx_{i}"])
        assert tensor.dtype == expected_dtype
        assert tensor.to(torch.bfloat16).dtype == torch.bfloat16
        assert mask.tolist() == [[True, True, True, True, True]]
    assert "lens_ctx_0_nvfp4_packed" not in batch
    assert "lens_ctx_0_nvfp4_block_scale" not in batch
