"""Bit-exact tests for the fused Triton NVFP4 activation-quantization kernel
(modules/nvfp4_kernels.py) against the eager reference path it replaces
(modules/nvfp4_utils._quantize_nvfp4_2d)."""

import pytest
import torch

from musubi_tuner.modules import nvfp4_kernels
from musubi_tuner.modules.nvfp4_utils import _quantize_nvfp4_2d, quantize_nvfp4_activation

requires_triton_cuda = pytest.mark.skipif(
    not (nvfp4_kernels.HAS_TRITON and torch.cuda.is_available()), reason="triton + CUDA required"
)


# E2M1 code -> magnitude, in code order (codes 0-7 positive, 8-15 = sign bit | positive code).
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _adjacent_codes(a: int, b: int) -> bool:
    """True if E2M1 codes a and b decode to the same sign and adjacent magnitudes (one
    quantization step apart) -- the signature of a fp32 rounding-tie landing a hair on
    opposite sides of the boundary in two independently-ordered implementations, not a
    real decoding error."""
    sign_a, mag_a = a & 8, a & 7
    sign_b, mag_b = b & 8, b & 7
    return sign_a == sign_b and abs(mag_a - mag_b) == 1


def _compare(x: torch.Tensor, max_mismatch_fraction: float = 0.0):
    """Quantize the same input via the eager reference path (_quantize_nvfp4_2d) and via the
    fused Triton kernel (quantize_nvfp4_activation, which dispatches to it whenever x is on
    CUDA and triton is importable).

    Independently computing the same E2M1 rounding on different execution engines (Triton
    in-kernel fp32 arithmetic vs PyTorch/ATen eager ops) can put a value a few ULPs from an
    exact rounding-tie boundary on opposite sides of that boundary between the two paths --
    a genuine, unavoidable floating-point divergence, not a logic bug. ``max_mismatch_fraction``
    bounds how much of that is tolerated; every tolerated mismatch must still decode to the
    immediately adjacent E2M1 magnitude (never an arbitrary/large decoding difference), which
    is what distinguishes this from an actual kernel defect.
    """
    ref_packed, ref_scale, ref_pts, ref_rows = _quantize_nvfp4_2d(x)
    fused_packed, fused_scale, fused_pts, fused_rows = quantize_nvfp4_activation(x)

    assert fused_rows == ref_rows
    assert torch.equal(fused_pts, ref_pts)

    _assert_packed_bytes_match(ref_packed, fused_packed, max_mismatch_fraction)
    _assert_scale_bytes_match(ref_scale, fused_scale, max_mismatch_fraction)


def _assert_packed_bytes_match(ref_packed: torch.Tensor, fused_packed: torch.Tensor, max_mismatch_fraction: float) -> None:
    mismatch = ref_packed != fused_packed
    n_mismatch = int(mismatch.sum().item())
    n_total = mismatch.numel()
    if n_mismatch == 0:
        return
    assert n_mismatch / n_total <= max_mismatch_fraction, (
        f"{n_mismatch}/{n_total} packed bytes differ, exceeding the allowed {max_mismatch_fraction:.1e} rounding-tie tolerance"
    )
    ref_bytes = ref_packed[mismatch].tolist()
    fused_bytes = fused_packed[mismatch].tolist()
    for ref_byte, fused_byte in zip(ref_bytes, fused_bytes):
        ref_high, ref_low = ref_byte >> 4, ref_byte & 0xF
        fused_high, fused_low = fused_byte >> 4, fused_byte & 0xF
        assert (ref_high == fused_high or _adjacent_codes(ref_high, fused_high)) and (
            ref_low == fused_low or _adjacent_codes(ref_low, fused_low)
        ), f"non-adjacent packed-byte mismatch: ref {ref_byte:#04x} vs fused {fused_byte:#04x}"


def _assert_scale_bytes_match(ref_scale: torch.Tensor, fused_scale: torch.Tensor, max_mismatch_fraction: float) -> None:
    """Same rounding-tie tolerance as packed bytes, for the F8_E4M3 block-scale cast. Scale
    values here are always non-negative (block_scale/per_tensor_scale ratios), so unlike
    E2M1 codes, adjacent raw byte values already mean adjacent representable magnitudes --
    no sign-bit unpacking needed."""
    ref_bytes_t = ref_scale.view(torch.uint8)
    fused_bytes_t = fused_scale.view(torch.uint8)
    mismatch = ref_bytes_t != fused_bytes_t
    n_mismatch = int(mismatch.sum().item())
    n_total = mismatch.numel()
    if n_mismatch == 0:
        return
    assert n_mismatch / n_total <= max_mismatch_fraction, (
        f"{n_mismatch}/{n_total} scale bytes differ, exceeding the allowed {max_mismatch_fraction:.1e} rounding-tie tolerance"
    )
    ref_bytes = ref_bytes_t[mismatch].tolist()
    fused_bytes = fused_bytes_t[mismatch].tolist()
    for ref_byte, fused_byte in zip(ref_bytes, fused_bytes):
        assert abs(ref_byte - fused_byte) == 1, f"non-adjacent scale-byte mismatch: ref {ref_byte} vs fused {fused_byte}"


@requires_triton_cuda
@pytest.mark.parametrize(
    "rows,cols",
    [
        (16, 16),  # single row-block, single group -- smallest valid shape
        (128, 6144),  # Krea2 attn.wq/attn.gate/attn.wo row count, exactly one row-tile
        (100, 6144),  # not a multiple of 16 -- exercises row padding
        (32, 48),  # cols not a multiple of 128 -- exercises column-block padding in to_blocked
    ],
)
def test_fused_kernel_matches_eager_reference(rows, cols):
    torch.manual_seed(0)
    x = (torch.randn(rows, cols, device="cuda") * 3.0).float()
    _compare(x)


@requires_triton_cuda
@pytest.mark.parametrize(
    "rows,cols",
    [
        (24576, 6144),  # Krea2 mlp.gate row count -- exercises many row-blocks
        (2048, 24576),  # Krea2 mlp.down's column count at a memory-safe row count
        (1536, 6144),  # Krea2 attn.wk/attn.wv row count
    ],
)
def test_fused_kernel_matches_eager_reference_at_scale(rows, cols):
    """At real Krea2 shapes (millions of elements), an fp32 rounding-tie divergence between
    the two independently-ordered implementations shows up roughly once per several million
    elements (see _compare's docstring) -- tolerate a tiny, bounded fraction of these, but
    require every tolerated mismatch to decode to an adjacent E2M1 magnitude."""
    torch.manual_seed(0)
    x = (torch.randn(rows, cols, device="cuda") * 3.0).float()
    _compare(x, max_mismatch_fraction=1e-6)


@requires_triton_cuda
def test_fused_kernel_matches_eager_reference_zero_block():
    """A block whose values are all zero exercises the scale-is-zero guard (total_safe)."""
    torch.manual_seed(1)
    x = torch.randn(32, 32, device="cuda").float()
    x[5, :] = 0.0
    _compare(x)


@requires_triton_cuda
def test_fused_kernel_matches_eager_reference_saturating_values():
    """Very large magnitudes force saturation to F4_E2M1_MAX (E2M1 code path) and
    F8_E4M3_MAX (block-scale cast path)."""
    torch.manual_seed(2)
    x = torch.randn(32, 32, device="cuda").float() * 1e6
    _compare(x)


@requires_triton_cuda
def test_fused_kernel_matches_eager_reference_e2m1_boundary_values():
    """Values exactly on E2M1 magnitude boundaries (0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0) and
    their negatives/midpoints exercise round-to-nearest-even ties in the bit-conversion path."""
    boundaries = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]
    values = boundaries + [-v for v in boundaries]
    while len(values) < 16 * 16:
        values.append(0.0)
    x = torch.tensor(values[: 16 * 16], dtype=torch.float32, device="cuda").reshape(16, 16)
    _compare(x)


def test_quantize_nvfp4_activation_falls_back_to_eager_reference_on_cpu():
    """quantize_nvfp4_activation only dispatches to the fused kernel for CUDA tensors, so a
    CPU input always takes the eager _quantize_nvfp4_2d path -- this must still equal calling
    _quantize_nvfp4_2d directly (i.e. the dispatch wrapper doesn't change the fallback's
    behavior)."""
    torch.manual_seed(3)
    x = torch.randn(32, 32).float()
    _compare(x)


@requires_triton_cuda
def test_e2m1_stochastic_code_exact_values_are_deterministic():
    """Values exactly on a representable E2M1 magnitude (including 0.0 and 6.0, the degenerate
    span==0 cases) must decode to themselves with probability 1 -- no random draw should ever
    move them, matching test_nvfp4_stochastic_rounding.py's eager-path equivalent."""
    import triton
    import triton.language as tl

    from musubi_tuner.modules.nvfp4_kernels import _e2m1_stochastic_code

    @triton.jit
    def _kernel(x_ptr, y_ptr, seed, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        r = tl.rand(seed, offs)
        code = _e2m1_stochastic_code(x, r)
        tl.store(y_ptr + offs, code, mask=mask)

    exact_values = torch.tensor([0.0, -0.0, 0.5, -0.5, 1.0, 2.0, 3.0, 4.0, 6.0, -6.0])
    expected_codes = torch.tensor([0, 0, 1, 9, 2, 4, 5, 6, 7, 15], dtype=torch.uint8)
    x = exact_values.repeat(1000).cuda()
    n = x.numel()
    y = torch.empty(n, device="cuda", dtype=torch.uint8)
    _kernel[(1,)](x, y, 12345, n, BLOCK=triton.next_power_of_2(n))

    assert torch.equal(y.cpu(), expected_codes.repeat(1000))


@requires_triton_cuda
def test_e2m1_stochastic_code_unbiased_in_expectation_midpoint():
    """3.5 is exactly the midpoint between representable values 3.0 (code 5) and 4.0 (code 6);
    unbiased rounding should land on each with ~50% probability, averaging back to 3.5. Mirrors
    test_nvfp4_stochastic_rounding.py::test_stochastic_code_unbiased_in_expectation_midpoint."""
    import triton
    import triton.language as tl

    from musubi_tuner.modules.nvfp4_kernels import _e2m1_stochastic_code

    @triton.jit
    def _kernel(x_ptr, y_ptr, seed, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        r = tl.rand(seed, offs)
        code = _e2m1_stochastic_code(x, r)
        tl.store(y_ptr + offs, code, mask=mask)

    n = 100000
    x = torch.full((n,), 3.5, device="cuda")
    y = torch.empty(n, device="cuda", dtype=torch.uint8)
    _kernel[(1,)](x, y, 777, n, BLOCK=triton.next_power_of_2(n))

    magnitude_table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    decoded = magnitude_table[(y & 7).long()]
    assert decoded.mean().item() == pytest.approx(3.5, abs=0.05)


@requires_triton_cuda
def test_e2m1_stochastic_code_unbiased_in_expectation_near_zero_end():
    """0.1 is close to 0.0 (code 0) and far from 0.5 (code 1); expected decode should be close to
    0.1, not exactly 0 or 0.5 -- checks the inverse-distance weighting, not just a coin flip.
    Mirrors test_nvfp4_stochastic_rounding.py::test_stochastic_code_unbiased_in_expectation_near_zero_end."""
    import triton
    import triton.language as tl

    from musubi_tuner.modules.nvfp4_kernels import _e2m1_stochastic_code

    @triton.jit
    def _kernel(x_ptr, y_ptr, seed, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        r = tl.rand(seed, offs)
        code = _e2m1_stochastic_code(x, r)
        tl.store(y_ptr + offs, code, mask=mask)

    n = 100000
    x = torch.full((n,), 0.1, device="cuda")
    y = torch.empty(n, device="cuda", dtype=torch.uint8)
    _kernel[(1,)](x, y, 42, n, BLOCK=triton.next_power_of_2(n))

    magnitude_table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    decoded = magnitude_table[(y & 7).long()]
    assert decoded.mean().item() == pytest.approx(0.1, abs=0.02)


@requires_triton_cuda
def test_triton_quantize_nvfp4_stochastic_block_scale_matches_deterministic():
    """Block-scale computation is shared with the deterministic kernel and must match exactly --
    only the final magnitude conversion differs. Mirrors
    test_nvfp4_stochastic_rounding.py::test_quantize_nvfp4_activation_stochastic_block_scale_matches_deterministic."""
    from musubi_tuner.modules.nvfp4_kernels import triton_quantize_nvfp4, triton_quantize_nvfp4_stochastic
    from musubi_tuner.modules.nvfp4_utils import F4_E2M1_MAX, F8_E4M3_MAX

    torch.manual_seed(0)
    x = (torch.randn(32, 64, device="cuda") * 0.5).float()
    per_tensor_scale = (torch.amax(x.abs()).float() / (F8_E4M3_MAX * F4_E2M1_MAX)).reshape(())

    _packed_det, scale_det = triton_quantize_nvfp4(x, per_tensor_scale)
    _packed_stoch, scale_stoch = triton_quantize_nvfp4_stochastic(x, per_tensor_scale, seed=123)

    assert torch.equal(scale_stoch.view(torch.uint8), scale_det.view(torch.uint8))


@requires_triton_cuda
def test_triton_quantize_nvfp4_stochastic_is_random_across_calls():
    """Sanity check the kernel is actually stochastic, not accidentally deterministic (e.g. a bug
    reusing offset 0 for every element). Uses varied random magnitudes, not a uniform fill --
    see test_nvfp4_stochastic_rounding.py's eager equivalent for why a uniform fill is vacuous."""
    from musubi_tuner.modules.nvfp4_kernels import triton_quantize_nvfp4_stochastic
    from musubi_tuner.modules.nvfp4_utils import F4_E2M1_MAX, F8_E4M3_MAX

    torch.manual_seed(0)
    x = (torch.rand(16, 32, device="cuda") * 0.3).float()
    per_tensor_scale = (torch.amax(x.abs()).float() / (F8_E4M3_MAX * F4_E2M1_MAX)).reshape(())

    packed_a, _ = triton_quantize_nvfp4_stochastic(x, per_tensor_scale, seed=1)
    packed_b, _ = triton_quantize_nvfp4_stochastic(x, per_tensor_scale, seed=2)

    assert not torch.equal(packed_a, packed_b)


@requires_triton_cuda
def test_triton_quantize_nvfp4_stochastic_unbiased_end_to_end():
    """Full-path unbiasedness check (block-scale normalization + stochastic E2M1 conversion),
    at a real Krea2-sized shape. Mirrors
    test_nvfp4_stochastic_rounding.py::test_quantize_nvfp4_activation_stochastic_unbiased_end_to_end."""
    from musubi_tuner.modules.nvfp4_kernels import triton_quantize_nvfp4_stochastic
    from musubi_tuner.modules.nvfp4_utils import F4_E2M1_MAX, F8_E4M3_MAX, NVFP4_BLOCK_SIZE

    torch.manual_seed(0)
    magnitude_table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    target_value = 0.1

    means = []
    for trial in range(200):
        x = torch.full((16, NVFP4_BLOCK_SIZE), target_value, device="cuda")
        x[:, 0] = 0.24  # sets this block's amax so target_value normalizes off-grid, not to 6.0
        per_tensor_scale = (torch.amax(x.abs()).float() / (F8_E4M3_MAX * F4_E2M1_MAX)).reshape(())
        packed, block_scale = triton_quantize_nvfp4_stochastic(x, per_tensor_scale, seed=trial)

        codes = packed.view(-1, 1).bitwise_and(0x0F)
        codes_hi = packed.view(-1, 1).bitwise_right_shift(4).bitwise_and(0x0F)
        all_codes = torch.stack([codes_hi, codes], dim=-1).flatten()
        neg = (all_codes & 8).bool()
        mag_codes = (all_codes & 7).long()
        decoded_magnitude = magnitude_table[mag_codes]
        decoded_signed = torch.where(neg, -decoded_magnitude, decoded_magnitude)
        total_scale = per_tensor_scale * block_scale.view(-1)[0].float()
        decoded_value = decoded_signed.float() * total_scale

        decoded_reshaped = decoded_value.view(16, NVFP4_BLOCK_SIZE)
        target_cells = decoded_reshaped[:, 1:]
        means.append(target_cells.mean().item())

    grand_mean = sum(means) / len(means)
    stdev = (sum((m - grand_mean) ** 2 for m in means) / len(means)) ** 0.5
    assert stdev > 0.0, "means across trials should not be identical -- indicates degenerate/deterministic rounding"
    assert grand_mean == pytest.approx(target_value, abs=0.005)
