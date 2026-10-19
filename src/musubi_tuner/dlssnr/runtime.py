"""Explicit model-local numerical policy; default execution remains FP32 NR."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from musubi_tuner.dlssnr.model import ChannelLinear, GlobalAttn, WindowAttn
from musubi_tuner.dlssnr.numerics import SURROGATE_FLAGS, fp32_execution
from musubi_tuner.dlssnr.profiles import PROFILE_ID
from musubi_tuner.dlssnr.weight_quantization import native_storage_kinds

RUNTIME_SCHEMA = "dlssnr_runtime_v3"
COMPUTE_DTYPES = {"no": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def default_runtime_policy() -> dict:
    return {
        "schema": RUNTIME_SCHEMA,
        "numerics_profile": "train_surrogate",
        "mixed_precision": "no",
        "gradient_checkpointing": False,
        "gradient_checkpointing_min_pixels": 0,
        "attention_backend": "native",
        "attention_scope": "all",
        "fp8_base": False,
        "fp8_scaled": False,
        "native_weight_qat": False,
    }


def runtime_policy(config: dict) -> dict:
    policy = {
        "schema": RUNTIME_SCHEMA,
        "numerics_profile": config["model"]["numerics_profile"],
        "mixed_precision": config["precision"]["mixed_precision"],
        "gradient_checkpointing": config["training"]["gradient_checkpointing"],
        "gradient_checkpointing_min_pixels": config["training"].get("gradient_checkpointing_min_pixels", 0),
        "attention_backend": config["model"].get("attention_backend", "native"),
        "attention_scope": config["model"].get("attention_scope", "all"),
        "fp8_base": config["precision"].get("fp8_base", False),
        "fp8_scaled": config["precision"].get("fp8_scaled", False),
    }
    return with_native_weight_qat(policy, config["model"].get("native_weight_qat", False))


def with_native_weight_qat(policy: dict, enabled: bool) -> dict:
    policy = dict(policy)
    if type(enabled) is not bool:
        raise ValueError("native_weight_qat must be a boolean")
    policy["native_weight_qat"] = enabled
    return policy


def validate_runtime_policy(policy: dict, *, training=False) -> None:
    if not isinstance(policy, dict):
        raise ValueError("NR runtime policy must be an object")
    if policy.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("unsupported NR runtime policy schema")
    if set(policy) != set(default_runtime_policy()):
        raise ValueError("NR runtime policy has missing or unsupported fields")
    for field in ("gradient_checkpointing", "fp8_base", "fp8_scaled", "native_weight_qat"):
        if type(policy[field]) is not bool:
            raise ValueError(f"runtime policy {field} must be a boolean")
    minimum_pixels = policy["gradient_checkpointing_min_pixels"]
    if type(minimum_pixels) is not int or minimum_pixels < 0:
        raise ValueError("gradient_checkpointing_min_pixels must be a nonnegative integer")
    if minimum_pixels and not policy["gradient_checkpointing"]:
        raise ValueError("gradient_checkpointing_min_pixels requires gradient_checkpointing")
    if policy.get("numerics_profile") not in ("train_surrogate", "train_experimental"):
        raise ValueError("unsupported NR runtime numerics_profile")
    if policy.get("mixed_precision") not in COMPUTE_DTYPES:
        raise ValueError("unsupported NR mixed_precision")
    if policy["mixed_precision"] != "no" and policy["numerics_profile"] != "train_experimental":
        raise ValueError("mixed precision requires --numerics_profile train_experimental")
    if policy.get("fp8_scaled") and not policy.get("fp8_base"):
        raise ValueError("fp8_scaled requires fp8_base")
    if policy.get("fp8_base") and policy["numerics_profile"] != "train_experimental":
        raise ValueError("FP8 storage requires train_experimental")
    backend = policy.get("attention_backend")
    if backend not in ("native", "sdpa", "flash_attn", "xformers", "sage_attn") or policy.get("attention_scope") not in (
        "all",
        "global",
    ):
        raise ValueError("unsupported NR attention backend/scope")
    if backend != "native" and policy["numerics_profile"] != "train_experimental":
        raise ValueError("alternative attention requires train_experimental")
    if backend in ("flash_attn", "sage_attn"):
        if policy["attention_scope"] != "global":
            raise ValueError(f"{backend} requires explicit --attention_scope global; NR window priors cannot be dropped")
        if policy["mixed_precision"] not in ("fp16", "bf16"):
            raise ValueError(f"{backend} requires fp16 or bf16 compute")
    if training and backend == "sage_attn":
        raise ValueError("SageAttention is inference-only for NR; no verified training backward is available")


def validate_runtime_device(policy: dict, device: torch.device, *, training=False) -> None:
    from musubi_tuner.dlssnr.attention import probe_backend

    validate_runtime_policy(policy, training=training)
    if policy["mixed_precision"] != "no":
        if device.type != "cuda":
            raise ValueError("NR mixed precision currently requires CUDA")
        if policy["mixed_precision"] == "bf16":
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("BF16 was requested but is unsupported by this CUDA device")
    probe_backend(policy, device, training=training)


def configure_model_runtime(model, policy: dict, *, training=False) -> None:
    validate_runtime_policy(policy, training=training)
    device = next(model.parameters()).device
    # Training may configure a CPU model before Accelerator moves it to CUDA.
    if not training:
        validate_runtime_device(policy, device)
    dtype = COMPUTE_DTYPES[policy["mixed_precision"]]
    kinds = native_storage_kinds(getattr(model, "profile", PROFILE_ID)) if policy.get("native_weight_qat", False) else {}
    for name, module in model.named_modules():
        if isinstance(module, (ChannelLinear, WindowAttn, GlobalAttn)):
            module.compute_dtype = dtype
        if isinstance(module, ChannelLinear):
            if kinds and f"{name}.weight" not in kinds:
                raise ValueError(f"unknown native projection {name}.weight")
            module.native_weight_kind = kinds.get(f"{name}.weight")
        if isinstance(module, GlobalAttn):
            module.attention_backend = policy["attention_backend"]
        elif isinstance(module, WindowAttn):
            module.attention_backend = policy["attention_backend"] if policy["attention_scope"] == "all" else "native"
    if training and policy["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    elif hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = False
    model.gradient_checkpointing_min_pixels = policy["gradient_checkpointing_min_pixels"] if training else 0
    model.runtime_policy = dict(policy)
    model.native_weight_qat = policy.get("native_weight_qat", False)


def numerics_metadata(policy: dict) -> dict:
    flags = (
        dict(SURROGATE_FLAGS)
        if policy["numerics_profile"] == "train_surrogate"
        else {
            **SURROGATE_FLAGS,
            "profile": "train_experimental",
            "compute_dtype": {"no": "float32", "fp16": "mixed_float16_products", "bf16": "mixed_bfloat16_products"}[
                policy["mixed_precision"]
            ],
            "matmul_accumulator": "autocast_products"
            if policy["mixed_precision"] != "no"
            else SURROGATE_FLAGS["matmul_accumulator"],
            "publication_dtype": "float32",
            "native_equivalent": False,
            "weight_publication": "fp8_storage_materialized_fp32_plus_lora"
            if policy["fp8_base"]
            else SURROGATE_FLAGS["weight_publication"],
            "attention_backend": policy["attention_backend"],
            "attention_scope": policy["attention_scope"],
            "softmax": "standard_softmax_scale_1.0" if policy["attention_backend"] != "native" else SURROGATE_FLAGS["softmax"],
            "vit_padding": "excluded_from_softmax" if policy["attention_backend"] != "native" else SURROGATE_FLAGS["vit_padding"],
        }
    )
    if policy.get("native_weight_qat", False):
        flags.update(weight_publication="native_storage_direct_fp32_rne_ste", native_equivalent=False)
    return flags


@contextmanager
def native_weight_runtime(model):
    """Evaluate exportable weights with native attention, without changing masters or optimizer state."""
    missing = object()
    saved = [
        (model, name, getattr(model, name, missing))
        for name in ("runtime_policy", "gradient_checkpointing", "gradient_checkpointing_min_pixels", "native_weight_qat")
    ]
    for module in model.modules():
        for name in ("compute_dtype", "attention_backend", "native_weight_kind"):
            if hasattr(module, name):
                saved.append((module, name, getattr(module, name)))
    try:
        configure_model_runtime(model, with_native_weight_qat(default_runtime_policy(), True), training=True)
        with fp32_execution():
            yield
    finally:
        for module, name, value in saved:
            if value is missing:
                if hasattr(module, name):
                    delattr(module, name)
            else:
                setattr(module, name, value)
