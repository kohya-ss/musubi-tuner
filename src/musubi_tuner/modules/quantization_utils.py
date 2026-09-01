"""Shared, architecture-agnostic quantization-scheme validation.

fp8_scaled/convrot_int8/nvfp4 live in modules/ (not any one architecture's subpackage)
because the schemes themselves are architecture-agnostic. Only Krea2 has adopted
convrot_int8/nvfp4 so far, but the checks here have no Krea2-specific knowledge, so any
future architecture wiring these schemes can call them directly instead of re-deriving
the same mutual-exclusivity / runtime-requirement logic.

Both functions are pure: they take already-computed primitives (booleans, a capability
tuple) rather than calling torch.cuda or nvfp4_scaled_mm_available themselves. Callers
own that side-effecting lookup, which keeps these functions trivially testable and keeps
existing monkeypatch-based tests (which patch the *caller's* module-level references)
working unchanged.
"""

from __future__ import annotations


def validate_quantization_scheme(fp8_scaled: bool, convrot_int8: bool, nvfp4: bool) -> None:
    """Raise ValueError if more than one quantization scheme is selected."""
    if sum([fp8_scaled, convrot_int8, nvfp4]) > 1:
        raise ValueError("--fp8_scaled, --convrot_int8, and --nvfp4 are mutually exclusive: choose at most one.")


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
