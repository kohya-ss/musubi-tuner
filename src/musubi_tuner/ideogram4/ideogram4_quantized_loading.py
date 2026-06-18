from __future__ import annotations

import torch
import torch.nn as nn

bnb = None
FP8_WEIGHT_DTYPE = getattr(torch, "float8_e4m3fn", None)


def require_fp8_dtype() -> torch.dtype:
    if FP8_WEIGHT_DTYPE is None:
        raise RuntimeError("Ideogram 4 FP8 weights require torch.float8_e4m3fn support")
    return FP8_WEIGHT_DTYPE


def _require_bnb():
    global bnb
    if bnb is None:
        try:
            import bitsandbytes as bnb_module
        except ImportError as exc:
            raise ImportError("bitsandbytes is required for Ideogram 4 bnb 4-bit checkpoints") from exc
        bnb = bnb_module
    return bnb


_BNB_SIBLING_SUFFIXES = (
    ".absmax",
    ".quant_map",
    ".nested_absmax",
    ".nested_quant_map",
)

FP8_SCALE_SUFFIX = ".weight_scale"
COMFY_FP8_MARKER_SUFFIX = ".comfy_quant"
# Marker written into the text encoder's config.json so the loader knows to take
# the custom weight-only FP8 path instead of transformers' from_pretrained.
FP8_TEXT_ENCODER_CONFIG_FLAG = "ideogram_fp8_weight_only"


def is_bnb4bit_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """True if any key looks like a bnb 4-bit quant_state sibling."""
    return any(".quant_state.bitsandbytes__" in k for k in state_dict)


def swap_linears_to_bnb4bit(
    module: nn.Module,
    compute_dtype: torch.dtype,
    *,
    quant_type: str = "nf4",
    compress_statistics: bool = False,
) -> None:
    bnb = _require_bnb()
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new_linear = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                compress_statistics=compress_statistics,
                quant_type=quant_type,
            )
            setattr(module, name, new_linear)
        else:
            swap_linears_to_bnb4bit(
                child,
                compute_dtype,
                quant_type=quant_type,
                compress_statistics=compress_statistics,
            )


def load_bnb4bit_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    bnb = _require_bnb()
    consumed: set[str] = set()
    for full_name, tensor in state_dict.items():
        if ".quant_state." in full_name or full_name.endswith(_BNB_SIBLING_SUFFIXES):
            continue
        parent_path, _, param_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        current = parent._parameters.get(param_name)
        if not isinstance(current, bnb.nn.Params4bit):
            continue
        prefix = full_name + "."
        quantized_stats = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
        # bnb's from_prequantized pops keys it consumes from the dict, so snapshot
        # the names first.
        consumed.add(full_name)
        consumed.update(quantized_stats.keys())
        parent._parameters[param_name] = bnb.nn.Params4bit.from_prequantized(
            data=tensor,
            quantized_stats=quantized_stats,
            requires_grad=False,
            device=device,
        )

    remaining = {k: v for k, v in state_dict.items() if k not in consumed}
    for k in list(remaining):
        if remaining[k].is_floating_point():
            remaining[k] = remaining[k].to(device=device, dtype=dtype)
        else:
            remaining[k] = remaining[k].to(device=device)

    missing, unexpected = model.load_state_dict(remaining, strict=False)
    # Quantized weights are loaded via from_prequantized above, so they appear in
    # `missing` from load_state_dict's perspective — filter those out.
    real_missing = [m for m in missing if m not in consumed]
    if real_missing:
        raise RuntimeError(f"missing keys after quantized load: {real_missing[:10]}")
    if unexpected:
        raise RuntimeError(f"unexpected keys after quantized load: {unexpected[:10]}")

    for p in model.parameters():
        if isinstance(p, bnb.nn.Params4bit):
            continue
        if p.is_floating_point() and p.dtype != dtype:
            p.data = p.data.to(dtype=dtype)
        if p.device != device:
            p.data = p.data.to(device=device)
    for name, b in list(model.named_buffers()):
        if b.is_floating_point() and b.dtype != dtype:
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            parent.register_buffer(
                leaf,
                b.to(device=device, dtype=dtype),
                persistent=leaf not in parent._non_persistent_buffers_set,
            )
        elif b.device != device:
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            parent.register_buffer(
                leaf,
                b.to(device=device),
                persistent=leaf not in parent._non_persistent_buffers_set,
            )


# ---------------------------------------------------------------------------
# Weight-only FP8 (e4m3)
#
# Activations stay in the compute dtype (e.g. bfloat16); only Linear weights are
# stored as float8 with a per-output-channel (per-row) float32 scale. At forward
# time the weight is dequantized back to the compute dtype and a normal bf16
# matmul runs, so this needs no FP8 tensor-core hardware and works on any device
# that can store float8 (CPU included). The win is ~2x smaller Linear weights.
# ---------------------------------------------------------------------------


def is_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """True if the checkpoint carries weight-only FP8 Linear weights."""
    return any(k.endswith(FP8_SCALE_SUFFIX) for k in state_dict) or (
        FP8_WEIGHT_DTYPE is not None and any(v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values())
    )
