"""Shared, architecture-agnostic quantization-scheme validation.

fp8_scaled/convrot_int8/nvfp4 live in modules/ (not any one architecture's subpackage)
because the schemes themselves are architecture-agnostic: the checks here have no
architecture-specific knowledge, so any architecture wiring these schemes can call them
directly instead of re-deriving the same mutual-exclusivity / runtime-requirement logic.

Both functions are pure: they take already-computed primitives (booleans, a capability
tuple) rather than calling torch.cuda or nvfp4_scaled_mm_available themselves. Callers
own that side-effecting lookup, which keeps these functions trivially testable and keeps
existing monkeypatch-based tests (which patch the *caller's* module-level references)
working unchanged.
"""

from __future__ import annotations


def validate_quantization_scheme(fp8_scaled: bool, convrot_int8: bool, nvfp4: bool) -> None:
    """Raise ValueError if the selected schemes are incompatible.

    ``fp8_scaled`` is exclusive of the other two. ``convrot_int8`` and ``nvfp4`` may be
    combined -- together they select mixed-format prequantized loading (each Linear's
    format is declared per-module in the checkpoint via its ``.comfy_quant`` spec; see
    ``MixedQuantizer``).
    """
    if fp8_scaled and (convrot_int8 or nvfp4):
        raise ValueError(
            "--fp8_scaled is exclusive of --convrot_int8 and --nvfp4: choose --fp8_scaled alone, or"
            " --convrot_int8 and/or --nvfp4 (both together loads a mixed-format prequantized checkpoint)."
        )


def validate_nvfp4_requirements(
    nvfp4: bool,
    scaled_mm_available: bool,
    cuda_available: bool,
    device_capability: tuple[int, int] | None,
) -> None:
    """Raise ValueError if --nvfp4 is selected but the runtime can't support it.

    ``device_capability`` should be ``torch.cuda.get_device_capability()`` when
    ``cuda_available`` is True, else ``None`` (CLI validation can run before the process
    has been placed on a GPU, e.g. early in a multi-process accelerate launch).
    """
    if not nvfp4:
        return
    if not scaled_mm_available:
        raise ValueError("--nvfp4 requires PyTorch 2.10+ with torch.float4_e2m1fn_x2/torch.nn.functional.scaled_mm support.")
    if cuda_available and device_capability is not None and device_capability[0] < 10:
        major, minor = device_capability
        raise ValueError(
            f"--nvfp4 requires a Blackwell GPU (compute capability 10.0+) for real FP4x4"
            f" tensor-core scaled_mm; detected compute capability {major}.{minor}. Use"
            f" --convrot_int8 or --fp8_scaled instead on this GPU."
        )


def validate_quantization_scheme_args(
    fp8_scaled: bool,
    convrot_int8: bool,
    convrot_int8_bwd: str,
    nvfp4: bool,
    nvfp4_columnwise_chunk_rows: int,
    scaled_mm_available: bool,
    cuda_available: bool,
    device_capability: tuple[int, int] | None,
    blocks_to_swap: int = 0,
    block_swap_h2d_only: bool = False,
    require_block_swap_h2d_only_with_nvfp4: bool = True,
) -> None:
    """Validate the full quantization-scheme CLI arg set shared by every architecture wiring
    fp8_scaled/convrot_int8/nvfp4 (Krea2, Flux.2, ...).

    Composes ``validate_quantization_scheme`` (mutual exclusivity) and
    ``validate_nvfp4_requirements`` (NVFP4 runtime requirements) with the two checks that are
    common to every architecture but don't live in either of those: ``convrot_int8_bwd``
    requiring ``convrot_int8``, and the NVFP4 + block-swap + chunk-rows rules. Architecture-specific
    checks (e.g. Krea2's ``--turbo_dit`` incompatibility) are the caller's responsibility --
    see ``krea2_utils.validate_krea2_quantization_args`` for an example wrapper.

    ``require_block_swap_h2d_only_with_nvfp4`` defaults to True (the trainer's requirement:
    --block_swap_h2d_only + LoRA/LoHa/LoKr's ring-buffer streaming is the only combination
    validated end-to-end for NVFP4 training's columnwise backward buffers). Standalone inference
    should pass False -- under ``training=False`` those buffers are never built, so there is
    nothing block-swap-specific left to validate.
    """
    validate_quantization_scheme(fp8_scaled, convrot_int8, nvfp4)
    if convrot_int8_bwd == "int8" and not convrot_int8:
        raise ValueError("--convrot_int8_bwd int8 requires --convrot_int8.")
    validate_nvfp4_requirements(nvfp4, scaled_mm_available, cuda_available, device_capability)
    if nvfp4 and blocks_to_swap and require_block_swap_h2d_only_with_nvfp4 and not block_swap_h2d_only:
        raise ValueError(
            "--nvfp4 with --blocks_to_swap requires --block_swap_h2d_only. ModelOffloader (the"
            " non-h2d_only path) now tracks NVFP4's extra columnwise backward buffers"
            " (nvfp4_weight_t/nvfp4_block_scale_t/nvfp4_scale_t) via its swap_tensor_selector, but"
            " --block_swap_h2d_only + LoRA/LoHa/LoKr's ring-buffer streaming is the only combination"
            " validated end-to-end for NVFP4 training memory savings. Pass --block_swap_h2d_only,"
            " or omit --blocks_to_swap if the model fits without it."
        )
    if nvfp4 and (nvfp4_columnwise_chunk_rows <= 0 or nvfp4_columnwise_chunk_rows % 128 != 0):
        raise ValueError(
            f"--nvfp4_columnwise_chunk_rows must be a positive multiple of 128 (cuBLAS block-scale tile"
            f" height), got {nvfp4_columnwise_chunk_rows}"
        )
