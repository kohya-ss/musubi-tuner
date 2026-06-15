import torch

from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.utils.train_utils import resolve_save_dtype


def test_default_is_fp32():
    assert resolve_save_dtype(None, full_fp16=False, full_bf16=False) == torch.float32


def test_explicit_precision():
    assert resolve_save_dtype("fp32", full_fp16=False, full_bf16=False) == torch.float32
    assert resolve_save_dtype("float", full_fp16=False, full_bf16=False) == torch.float32
    assert resolve_save_dtype("fp16", full_fp16=False, full_bf16=False) == torch.float16
    assert resolve_save_dtype("bf16", full_fp16=False, full_bf16=False) == torch.bfloat16


def test_full_precision_flags_used_when_unset():
    assert resolve_save_dtype(None, full_fp16=True, full_bf16=False) == torch.float16
    assert resolve_save_dtype(None, full_fp16=False, full_bf16=True) == torch.bfloat16


def test_explicit_precision_overrides_full_flags():
    assert resolve_save_dtype("fp32", full_fp16=True, full_bf16=False) == torch.float32
    assert resolve_save_dtype("fp16", full_fp16=False, full_bf16=True) == torch.float16


def test_parser_has_save_precision_defaulting_to_none():
    parser = setup_parser_common()
    args = parser.parse_args(["--dataset_config", "x.toml", "--dit", "x.safetensors"])
    assert args.save_precision is None
