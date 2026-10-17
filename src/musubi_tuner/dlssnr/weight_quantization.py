"""Export-aligned weight publication. Activation publication has different rounding."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch

from musubi_tuner.dlssnr.profiles import PROFILE_ID, build_records, get_profile

STORAGE_DTYPES = {
    "e4": torch.float8_e4m3fn,
    "f16": torch.float16,
    "f16frag": torch.float16,
    "prior": torch.float16,
    "f32": torch.float32,
}


def native_storage_kinds(profile: str = PROFILE_ID) -> dict[str, str]:
    return {view.name: region.kind for record in build_records(profile) for region in record.regions for view in region.views}


class _NativeWeightRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, kind, validation):
        if weight.dtype != torch.float32:
            raise ValueError("native weight publication requires FP32 effective weights")
        dtype = STORAGE_DTYPES[kind]
        validated = validation is not None and validation.matches(weight, kind)
        if not validated and not (torch.isfinite(weight) & (weight.abs() <= torch.finfo(dtype).max)).all():
            raise ValueError(f"native {kind} weights must be finite and within the storage range")
        return weight.to(dtype).float()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None, None


def native_weight_ste(weight: torch.Tensor, kind: str, *, validation=None) -> torch.Tensor:
    """Direct FP32 -> storage -> FP32, with identity gradients and no silent clipping."""
    if kind not in STORAGE_DTYPES:
        raise ValueError(f"unsupported native weight storage {kind}")
    return _NativeWeightRound.apply(weight, kind, validation)


@dataclass
class _ValidatedWeight:
    weight: torch.Tensor
    kind: str
    version: int
    pointer: int
    active: bool = False

    def matches(self, weight, kind):
        return (
            self.active
            and weight is self.weight
            and kind == self.kind
            and weight._version == self.version
            and weight.data_ptr() == self.pointer
        )


@contextmanager
def _validation_scope(bindings):
    missing = object()
    previous = []
    try:
        for module, attribute, ticket in bindings:
            previous.append((module, attribute, getattr(module, attribute, missing), ticket))
            ticket.active = True
            setattr(module, attribute, ticket)
        yield
    finally:
        for module, attribute, value, ticket in reversed(previous):
            ticket.active = False
            if value is missing:
                delattr(module, attribute)
            else:
                setattr(module, attribute, value)


@torch.no_grad()
def native_weight_validation(model):
    """Prevalidate direct full-training weights; scope must end before an update or weight swap.

    Validation happens before returning the scope so DDP callers can coordinate failures
    before entering a model collective. Tickets live on modules, including during worker-thread
    checkpoint replay, and never cache quantized tensors or survive optimizer attempts.
    """
    from musubi_tuner.dlssnr.model import ChannelLinear

    if not getattr(model, "native_weight_qat", False):
        return nullcontext()
    modules = list(model.modules())
    if any(hasattr(module, "_dlssnr_lora_target") for module in modules):
        return nullcontext()
    bindings = []
    for module in modules:
        if isinstance(module, ChannelLinear) and module.native_weight_kind is not None:
            if module.materialized_weight() is not module.weight:
                return nullcontext()
            bindings.append((module, "_native_weight_validation", module.weight, module.native_weight_kind))
    head = model.blocks[str(get_profile(getattr(model, "profile", PROFILE_ID)).head_block)]
    bindings.append((model, "_native_blend_validation", head.blend_scale, "f16"))
    if len({weight.device for _, _, weight, _ in bindings}) != 1:
        return nullcontext()
    checks = []
    for _, _, weight, kind in bindings:
        if kind not in STORAGE_DTYPES:
            raise ValueError(f"unsupported native weight storage {kind}")
        if weight.dtype != torch.float32:
            raise ValueError("native weight publication requires FP32 effective weights")
        checks.append((torch.isfinite(weight) & (weight.abs() <= torch.finfo(STORAGE_DTYPES[kind]).max)).all())
    valid = torch.stack(checks)
    if not bool(valid.all()):
        for (_, _, _, kind), finite in zip(bindings, valid.tolist()):
            if not finite:
                raise ValueError(f"native {kind} weights must be finite and within the storage range")
    return _validation_scope(
        [
            (module, attribute, _ValidatedWeight(weight, kind, weight._version, weight.data_ptr()))
            for module, attribute, weight, kind in bindings
        ]
    )


def _effective_weights(model, network=None):
    from musubi_tuner.dlssnr.fp8 import iter_canonical_tensors

    adapters = {adapter.target: adapter for adapter in network.adapters} if network is not None else {}
    attached = {module._dlssnr_lora_target for module in model.modules() if hasattr(module, "_dlssnr_lora_target")}
    if attached != set(adapters):
        raise ValueError("native diagnostics require the model's complete LoRA network")
    for name, value in iter_canonical_tensors(model):
        yield name, value + adapters[name].delta_weight() if name in adapters else value


@torch.no_grad()
def capture_native_reference(model, network=None) -> dict[str, torch.Tensor]:
    """Keep a detached CPU snapshot in native storage, not another FP32 model."""
    kinds = native_storage_kinds(getattr(model, "profile", PROFILE_ID))
    return {
        name: native_weight_ste(value.detach().cpu(), kinds[name]).to(STORAGE_DTYPES[kinds[name]]).clone()
        for name, value in _effective_weights(model, network)
    }


def _summarize(rows):
    fields = (
        "values",
        "trained_changed_values",
        "blended_changed_values",
        "exported_changed_values",
        "rounded_values",
        "lost_update_values",
    )
    total = {field: sum(row[field] for row in rows) for field in fields}
    count = total["values"]
    total.update(
        flip_fraction=total["exported_changed_values"] / count,
        max_abs_quantization_error=max(row["max_abs_quantization_error"] for row in rows),
        mean_abs_quantization_error=sum(row["mean_abs_quantization_error"] * row["values"] for row in rows) / count,
        rmse=(sum(row["rmse"] ** 2 * row["values"] for row in rows) / count) ** 0.5,
    )
    return total


@torch.no_grad()
def native_quantization_report(model, reference, network=None) -> dict:
    from musubi_tuner.dlssnr.native import quantization_statistics, quantize_tensor

    kinds = native_storage_kinds(getattr(model, "profile", PROFILE_ID))
    rows = []
    for name, value in _effective_weights(model, network):
        value = value.detach().cpu().contiguous().numpy()
        source = reference[name].float().numpy()
        if value.shape != source.shape:
            raise ValueError(f"{name}: native reference shape mismatch")
        quantized = quantize_tensor(value, kinds[name])
        rows.append(quantization_statistics(name, kinds[name], source, value, value, quantized))
    if not rows or {row["name"] for row in rows} != set(reference):
        raise ValueError("native reference parameter map does not match the model")
    return {
        "schema": "dlssnr_native_quantization_v1",
        "reference": "initial_native_weights",
        "mix": 1.0,
        "strength": 1.0,
        "lora_multiplier": 1.0,
        "native_equivalent": False,
        "totals": _summarize(rows),
        "by_storage": {
            kind: _summarize([row for row in rows if row["storage"] == kind]) for kind in sorted({row["storage"] for row in rows})
        },
        "tensors": rows,
    }
