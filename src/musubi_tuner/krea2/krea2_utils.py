"""Shared loaders / helpers for the Krea 2 (K2) integration."""

import logging
from typing import Optional, Union

import torch

from musubi_tuner.krea2.krea2_encoder import (
    QWEN3_VL_4B_INSTRUCT_REPO_ID,
    Qwen3VLConditioner,
    TextEncoderConfig,
    load_qwen3_vl_conditioner,
)
from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer, apply_convrot_int8_monkey_patch
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.modules.nvfp4_utils import NvFp4Quantizer, apply_nvfp4_monkey_patch
from musubi_tuner.modules.quantization_utils import validate_nvfp4_requirements, validate_quantization_scheme
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.safetensors_utils import load_safetensors

logger = logging.getLogger(__name__)


# Dynamic fp8 quantization scope for the DiT: the per-block (SingleStreamBlock) attention
# and SwiGLU Linear weights — the heavy, repeated compute, matching the LoRA target. The
# modulation (`mod.lin`) is a raw nn.Parameter and the RMSNorm scales must stay in compute
# dtype, so both are excluded (cf. Z-Image's split). `txtfusion` (the text-fusion transformer,
# whose submodule is also named `layerwise_blocks` and so matches "blocks.") is small and
# delicate, so it is kept in compute dtype too.
KREA2_FP8_OPTIMIZATION_TARGET_KEYS = ["blocks."]
KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS = ["mod.", "norm", "txtfusion"]


# The single config shipped with the OSS checkpoints (single_mmdit_large_wide).
single_mmdit_large_wide = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)


def load_krea2_dit(
    dit_path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    config: SingleMMDiTConfig = single_mmdit_large_wide,
    fp8_scaled: bool = False,
    loading_device: Optional[Union[str, torch.device]] = None,
    attn_mode: str = "torch",
    split_attn: bool = False,
    lora_weights: Optional[list] = None,
    lora_multipliers: Optional[list] = None,
    convrot_int8: bool = False,
    convrot_int8_bwd: str = "bf16",
    nvfp4: bool = False,
    nvfp4_columnwise_chunk_rows: int = 1024,
    training: bool = True,
) -> SingleStreamDiT:
    """Build the K2 single-stream MMDiT on meta and load weights (assign=True).

    When ``fp8_scaled`` is True, the per-block Linear weights are dynamically quantized to
    scaled fp8 at load time and the matching Linear forwards are monkey-patched to
    dequantize on the fly (cf. Z-Image / qwen_image). ``dtype`` is then ignored — non-target
    weights (norms, modulation, embedders, heads) keep their checkpoint dtype.

    ``convrot_int8`` follows the same scheme with ConvRot int8 (Hadamard-rotated int8 weights,
    fused Triton forward, custom backward for LoRA training) instead of fp8; the same
    target/exclude scope applies. Mutually exclusive with ``fp8_scaled``.

    ``nvfp4`` loads a ComfyUI pre-quantized NVFP4 DiT checkpoint (per-block Linears already
    stored as packed FP4 + block/tensor scales) and, when ``training=True`` (the default),
    trains against it with true FP4x4 tensor-core forward/backward (``NvFp4LinearFn``) --
    no dynamic quantization, the file dictates which layers are NVFP4. Mutually exclusive
    with ``fp8_scaled``/``convrot_int8``. Cannot be combined with ``lora_weights``
    (pre-quantized NVFP4 cannot be merged at load time). ``dtype`` is ignored for
    NVFP4-quantized layers, same as ``fp8_scaled``/``convrot_int8`` -- non-target weights
    keep their checkpoint dtype.

    ``training`` (only meaningful when ``nvfp4`` is set) selects between the autograd
    training forward (default, ``True`` -- also builds the extra columnwise backward
    buffers ``nvfp4_weight_t``/``nvfp4_block_scale_t``/``nvfp4_scale_t``) and the
    non-autograd inference forward (``False`` -- no backward buffers built, pure memory
    savings since there is no backward pass to feed). Generic name (not NVFP4-specific):
    ``fp8_scaled``/``convrot_int8`` don't have this distinction today, but a future scheme
    that does can reuse this same parameter instead of adding another one-off flag.

    ``nvfp4_columnwise_chunk_rows`` (only used when ``nvfp4`` is set) is forwarded to
    ``apply_nvfp4_monkey_patch``'s ``columnwise_chunk_rows`` -- see there for what it controls.

    ``lora_weights`` (a list of loaded LoRA state dicts, with optional ``lora_multipliers``)
    are merged into the base weights at load time. This is the only correct route under fp8
    (fp8-quantized weights cannot be post-hoc merged), and it also keeps loading uniform for
    block swap: the merged/quantized state dict is produced before the model is placed, so the
    offloader can stream blocks afterward without an external weight mutation.

    For block swap, pass ``loading_device="cpu"``: the weights stay on CPU (``move_to_device``
    is then False) and the caller's ``enable_block_swap`` / ``move_to_device_except_swap_blocks``
    places the resident blocks on ``device`` and keeps the swap blocks on CPU.
    """
    assert sum([fp8_scaled, convrot_int8, nvfp4]) <= 1, "fp8_scaled, convrot_int8, and nvfp4 are mutually exclusive"
    assert not (nvfp4 and lora_weights), (
        "nvfp4 cannot be combined with lora_weights: pre-quantized NVFP4 weights cannot be "
        "merged at load time. Requantizing after a merge is possible in principle but is not "
        "implemented; use the original BF16 weights with LoRA instead."
    )
    device = torch.device(device)
    loading_device = device if loading_device is None else torch.device(loading_device)
    has_lora = lora_weights is not None and len(lora_weights) > 0

    logger.info(
        f"Loading Krea 2 DiT weights from {dit_path}"
        + (" (fp8 scaled)" if fp8_scaled else "")
        + (" (convrot int8)" if convrot_int8 else "")
        + (" (nvfp4)" if nvfp4 else "")
        + (f" (+{len(lora_weights)} LoRA merged)" if has_lora else "")
    )
    with torch.device("meta"):
        dit = SingleStreamDiT(config, attn_mode=attn_mode, split_attn=split_attn)

    quantized = fp8_scaled or convrot_int8 or nvfp4
    if quantized or has_lora:
        # Single load path that merges LoRA (if any) into the base weights and optionally
        # quantizes the per-block Linears (scaled fp8 or ConvRot int8). Targets/excludes only
        # apply when quantizing; without quantization the weights are merged and cast to
        # ``dtype`` as-is.
        quantizer = None
        if convrot_int8:
            quantizer = ConvRotInt8Quantizer(KREA2_FP8_OPTIMIZATION_TARGET_KEYS, KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS)
        elif nvfp4:
            quantizer = NvFp4Quantizer()
        sd = load_safetensors_with_lora_and_fp8(
            model_files=dit_path,
            lora_weights_list=lora_weights,
            lora_multipliers=lora_multipliers,
            fp8_optimization=fp8_scaled,
            calc_device=device,
            move_to_device=(loading_device == device),
            dit_weight_dtype=None if quantized else dtype,
            target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS if fp8_scaled else None,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS if fp8_scaled else None,
            quantizer=quantizer,
        )
        if fp8_scaled:
            apply_fp8_monkey_patch(dit, sd, use_scaled_mm=False)
        elif convrot_int8:
            apply_convrot_int8_monkey_patch(dit, sd, bwd_mode=convrot_int8_bwd, groupsize_map=quantizer.module_groupsizes)
            # int8 tensors cannot be wrapped as Parameters with requires_grad=True (only
            # floating dtypes can require grad), and load_state_dict(assign=True) re-wraps
            # incoming tensors with the meta params' requires_grad (default True). The base
            # is frozen right after load anyway, so drop requires_grad first.
            dit.requires_grad_(False)
        elif nvfp4:
            apply_nvfp4_monkey_patch(
                dit,
                sd,
                quantizer.nvfp4_module_shapes,
                quantizer.int8_embedding_modules,
                use_scaled_mm=True,
                training=training,
                calc_device=device,
                columnwise_chunk_rows=nvfp4_columnwise_chunk_rows,
            )
            # Same requires_grad concern as ConvRot above: NVFP4 weights are uint8 (packed
            # nibbles), not a floating dtype.
            dit.requires_grad_(False)
        if loading_device.type != "cpu":
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)
        dit.load_state_dict(sd, strict=True, assign=True)
    else:
        # Load without mmap (disable_mmap=True) to avoid the official load_file's transient ~2x
        # RAM (mmap page cache + materialized tensor), file locking, and lazy disk reads. Load
        # directly to the target device+dtype (assign=True) so the loaded tensors become the params.
        sd = load_safetensors(dit_path, device=loading_device, disable_mmap=True, dtype=dtype)
        dit.load_state_dict(sd, strict=True, assign=True)

    return dit


def validate_krea2_quantization_args(
    fp8_scaled: bool,
    convrot_int8: bool,
    convrot_int8_bwd: str,
    nvfp4: bool,
    nvfp4_columnwise_chunk_rows: int,
    turbo_dit: Optional[str],
    scaled_mm_available: bool,
    cuda_available: bool,
    device_capability: Optional[tuple],
    blocks_to_swap: int = 0,
    block_swap_h2d_only: bool = False,
    require_block_swap_h2d_only_with_nvfp4: bool = True,
) -> None:
    """Validate Krea2's quantization-scheme CLI args; shared by the trainer and standalone
    inference so there is exactly one copy of this logic.

    Composes the generic checks in ``modules.quantization_utils`` (mutual exclusivity,
    NVFP4 runtime requirements) with the Krea2-specific ones (``turbo_dit`` incompatibility,
    ``convrot_int8_bwd`` requiring ``convrot_int8``, the chunk-rows multiple-of-128 rule).

    ``require_block_swap_h2d_only_with_nvfp4`` defaults to True (the trainer's requirement:
    the default block-swap offloader doesn't know about NVFP4's training-only columnwise
    backward buffers). Standalone inference passes False -- under ``training=False`` those
    buffers are never built (see ``apply_nvfp4_monkey_patch``), so the default offloader has
    nothing to be unaware of.

    Callers (not this function) compute ``scaled_mm_available``/``cuda_available``/
    ``device_capability`` via ``nvfp4_scaled_mm_available()``/``torch.cuda.*`` themselves --
    keeps this function pure and keeps existing tests' monkeypatching of the *caller's*
    module-level references working.
    """
    validate_quantization_scheme(fp8_scaled, convrot_int8, nvfp4)
    if convrot_int8 and turbo_dit:
        raise ValueError("--convrot_int8 is not supported together with --turbo_dit yet; omit one of them.")
    if convrot_int8_bwd == "int8" and not convrot_int8:
        raise ValueError("--convrot_int8_bwd int8 requires --convrot_int8.")
    if nvfp4 and turbo_dit:
        raise ValueError("--nvfp4 is not supported together with --turbo_dit yet; omit one of them.")
    validate_nvfp4_requirements(nvfp4, scaled_mm_available, cuda_available, device_capability)
    if nvfp4 and blocks_to_swap and require_block_swap_h2d_only_with_nvfp4 and not block_swap_h2d_only:
        raise ValueError(
            "--nvfp4 with --blocks_to_swap requires --block_swap_h2d_only. The default block-swap"
            " offloader (ModelOffloader) does not know about NVFP4's extra columnwise backward buffers"
            " (nvfp4_weight_t/nvfp4_block_scale_t/nvfp4_scale_t) and would leave them GPU-resident for"
            " every block, defeating most of block swap's memory savings. Pass --block_swap_h2d_only,"
            " or omit --blocks_to_swap if the model fits without it."
        )
    if nvfp4 and (nvfp4_columnwise_chunk_rows <= 0 or nvfp4_columnwise_chunk_rows % 128 != 0):
        raise ValueError(
            f"--nvfp4_columnwise_chunk_rows must be a positive multiple of 128 (cuBLAS block-scale tile"
            f" height), got {nvfp4_columnwise_chunk_rows}"
        )


def load_krea2_dit_state_dict(
    dit_path: str,
    fp8_scaled: bool = False,
    calc_device: Union[str, torch.device] = "cpu",
    result_device: Union[str, torch.device] = "cpu",
    config: SingleMMDiTConfig = single_mmdit_large_wide,
) -> dict:
    """Produce a Krea 2 DiT state dict matching a model loaded via ``load_krea2_dit``.

    Unlike ``load_krea2_dit`` this builds no ``nn.Module`` — it returns only the weights,
    for swapping the base weights of an already-built model in place (e.g. RAW-train /
    Turbo-sample). When ``fp8_scaled`` is True the per-block Linears are dynamically
    quantized exactly as in ``load_krea2_dit`` (quantization runs on ``calc_device``), so
    the returned keys include the matching ``.scale_weight`` entries and line up 1:1 with
    the live model's ``named_parameters()`` + ``named_buffers()``. The result is moved to
    ``result_device``.

    When ``result_device`` equals ``calc_device`` (e.g. both the GPU, used by the M2 turbo/raw
    swap), the dict is built straight on that device with no full intermediate CPU dict — the
    CPU peak stays at ~1 tensor. When ``result_device`` is CPU (e.g. the M1 resident stash),
    the fp8 path quantizes on ``calc_device`` and then lands the dict on CPU.
    """
    calc_dev = torch.device(calc_device)
    rd = torch.device(result_device)
    # Keep the fp8-quantized tensors on calc_device when that is also the result device, so the
    # dict never round-trips through a full CPU copy (the M2 GPU-direct swap path).
    move_to_device = calc_dev == rd

    if fp8_scaled:
        sd = load_safetensors_with_lora_and_fp8(
            model_files=dit_path,
            lora_weights_list=None,
            lora_multipliers=None,
            fp8_optimization=True,
            calc_device=calc_dev,
            move_to_device=move_to_device,
            dit_weight_dtype=None,
            target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS,
        )
    else:
        # Load without mmap (disable_mmap=True) to avoid the official load_file's transient ~2x
        # RAM, file locking, and lazy disk reads. Load directly to result_device in bf16.
        sd = load_safetensors(dit_path, device=result_device, disable_mmap=True, dtype=torch.bfloat16)

    sd = {k: v.to(rd) for k, v in sd.items()}
    return sd


def load_krea2_text_encoder(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cpu",
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple = TextEncoderConfig.select_layers,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
) -> Qwen3VLConditioner:
    """Load the Qwen3-VL-4B conditioner used by K2: weights from ``path`` (local safetensors,
    ComfyUI or official key layout), tokenizer from ``tokenizer_repo`` (Hub id or local dir)."""
    return load_qwen3_vl_conditioner(
        path,
        dtype=dtype,
        device=device,
        max_length=max_length,
        select_layers=select_layers,
        tokenizer_repo=tokenizer_repo,
    )


@torch.no_grad()
def get_krea2_prompt_embeds(encoder: Qwen3VLConditioner, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hiddens, mask).

    hiddens: (B, seq, num_select_layers, hidden) stacked selected hidden states.
    mask:    (B, seq) bool attention mask (valid tokens incl. suffix, padding=False).
    """
    hiddens, mask = encoder(prompts)
    return hiddens, mask.to(dtype=torch.bool)
