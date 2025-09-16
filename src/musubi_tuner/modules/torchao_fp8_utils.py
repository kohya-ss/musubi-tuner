# torchao_fp8_utils.py
from typing import Optional, Union, Iterable, Callable
import torch
import torch.nn as nn
import logging

# TorchAO
from torchao.quantization import quantize_, Float8WeightOnlyConfig  # , Int8WeightOnlyConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _make_filter_fn(
    target_layer_keys: Optional[Iterable[str]], exclude_layer_keys: Optional[Iterable[str]]
) -> Callable[[nn.Module, str], bool]:
    """
    quantize the layers whose FQN contains any of target_layer_keys (if not None)
    and does not contain any of exclude_layer_keys (if not None)
    """

    def _is_target(name: str) -> bool:
        if target_layer_keys is not None and not any(p in name for p in target_layer_keys):
            return False
        if exclude_layer_keys is not None and any(p in name for p in exclude_layer_keys):
            return False
        return True

    def _filter(mod: nn.Module, name: str) -> bool:
        return isinstance(mod, nn.Linear) and _is_target(name)

    return _filter


@torch.no_grad()
def quantize_model_weights_fp8(
    model: nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    target_layer_keys: Optional[Iterable[str]] = None,
    exclude_layer_keys: Optional[Iterable[str]] = None,
    weight_dtype: torch.dtype = torch.float8_e4m3fn,
) -> nn.Module:
    """
    Quantize model weights for Linear layers to FP8 (default E4M3).
    This only quantizes the weights, using per-channel (row-wise) symmetric quantization,
    with scale derived from amax (= max(abs)) of each row.
    No need to replace forward.

    Args:
        model: model to be quantized (inplace update)
        device: device to place the quantized model (e.g. "cuda"). If None, do not move.
        target_layer_keys / exclude_layer_keys: filtering by FQN.
        weight_dtype: torch.float8_e4m3fn or torch.float8_e5m2, etc.

    Returns:
        model: quantized model (same instance as input)
    """
    cfg = Float8WeightOnlyConfig(weight_dtype=weight_dtype)
    # cfg = Int8WeightOnlyConfig()
    filter_fn = _make_filter_fn(target_layer_keys, exclude_layer_keys)
    quantize_(model, cfg, filter_fn=filter_fn, device=device)  # inplace update
    return model
