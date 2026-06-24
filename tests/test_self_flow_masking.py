import torch

from musubi_tuner.flux_2_train_network_self_flow import apply_per_token_mask


def test_mask_ratio_zero_returns_student_unchanged():
    student = torch.randn(2, 4, 8, 8)
    teacher = torch.randn(2, 4, 8, 8)
    out, mask = apply_per_token_mask(student, teacher, 0.0, torch.device("cpu"))
    assert torch.equal(out, student)
    assert mask.shape == (2, 64)
    assert not mask.any()


def test_mask_ratio_one_returns_teacher():
    # ratio 1.0 is invalid for training (paper caps at 0.5) but pins the boundary
    student = torch.zeros(2, 4, 8, 8)
    teacher = torch.ones(2, 4, 8, 8)
    out, mask = apply_per_token_mask(student, teacher, 1.0, torch.device("cpu"))
    assert torch.equal(out, teacher)
    assert mask.all()


def test_masked_positions_take_teacher_values():
    torch.manual_seed(0)
    student = torch.zeros(2, 4, 8, 8)
    teacher = torch.ones(2, 4, 8, 8)
    out, mask = apply_per_token_mask(student, teacher, 0.25, torch.device("cpu"))
    mask_spatial = mask.view(2, 1, 8, 8).expand_as(student)
    assert torch.equal(out[mask_spatial], teacher[mask_spatial])
    assert torch.equal(out[~mask_spatial], student[~mask_spatial])


def test_mask_applies_across_all_channels():
    torch.manual_seed(0)
    student = torch.zeros(1, 4, 8, 8)
    teacher = torch.ones(1, 4, 8, 8)
    out, _ = apply_per_token_mask(student, teacher, 0.5, torch.device("cpu"))
    # every (h, w) position is either all-teacher or all-student across channels
    per_pos = out.sum(dim=1)  # (1, 8, 8)
    assert ((per_pos == 0) | (per_pos == 4)).all()


def test_5d_independent_draw_shape():
    torch.manual_seed(0)
    student = torch.zeros(1, 4, 2, 8, 8)
    teacher = torch.ones(1, 4, 2, 8, 8)
    out, mask = apply_per_token_mask(student, teacher, 0.25, torch.device("cpu"))
    assert mask.shape == (1, 2 * 8 * 8)
    assert out.shape == student.shape


def test_approximate_ratio():
    torch.manual_seed(0)
    student = torch.zeros(8, 4, 32, 32)
    teacher = torch.ones(8, 4, 32, 32)
    _, mask = apply_per_token_mask(student, teacher, 0.25, torch.device("cpu"))
    assert abs(mask.float().mean().item() - 0.25) < 0.03


# --- Fix 1: tensor mask_ratio support ---


def test_tensor_ratio_per_sample_zero_one():
    """Tensor mask_ratio (B,) = [0.0, 1.0]: sample 0 all-student, sample 1 all-teacher."""
    torch.manual_seed(42)
    B = 2
    student = torch.zeros(B, 4, 8, 8)
    teacher = torch.ones(B, 4, 8, 8)
    ratio = torch.tensor([0.0, 1.0])
    out, mask = apply_per_token_mask(student, teacher, ratio, torch.device("cpu"))
    # Sample 0: ratio=0.0 => all student (zeros)
    assert not mask[0].any(), "sample 0 with ratio=0.0 must have no masked tokens"
    assert torch.equal(out[0], student[0]), "sample 0 output must equal student"
    # Sample 1: ratio=1.0 => all teacher (ones)
    assert mask[1].all(), "sample 1 with ratio=1.0 must have all tokens masked"
    assert torch.equal(out[1], teacher[1]), "sample 1 output must equal teacher"


def test_tensor_ratio_scalar_same_as_float():
    """0-dim tensor ratio should behave identically to float ratio."""
    torch.manual_seed(7)
    student = torch.zeros(4, 4, 8, 8)
    teacher = torch.ones(4, 4, 8, 8)
    ratio_float = 0.3
    ratio_tensor = torch.tensor(0.3)

    torch.manual_seed(7)
    _, mask_float = apply_per_token_mask(student, teacher, ratio_float, torch.device("cpu"))
    torch.manual_seed(7)
    _, mask_tensor = apply_per_token_mask(student, teacher, ratio_tensor, torch.device("cpu"))
    assert torch.equal(mask_float, mask_tensor), "0-dim tensor ratio must give same result as float"


def test_statistical_marginal_property():
    """Fix 1 marginal property: with R_M=0.25 and coin-flip, the overall fraction
    of tokens assigned to the cleaner (teacher) level must be ~0.5 over many samples.

    This is THE invariant paper Eq. 4 preserves: each token's marginal timestep
    distribution equals p(t), independent of R_M.

    We test the coin-flip helper directly using per-sample tensor ratios.
    """
    torch.manual_seed(12345)
    R_M = 0.25
    B = 512
    N_h, N_w = 16, 16  # 256 tokens per sample (4D latents)
    student = torch.zeros(B, 4, N_h, N_w)
    teacher = torch.ones(B, 4, N_h, N_w)
    device = torch.device("cpu")

    # Simulate coin-flip: half of samples get R_M, half get 1 - R_M
    coin = torch.rand(B) < 0.5
    effective_ratio = torch.where(coin, torch.full((B,), R_M), torch.full((B,), 1.0 - R_M))
    _, mask = apply_per_token_mask(student, teacher, effective_ratio, device)

    overall_fraction = mask.float().mean().item()
    assert abs(overall_fraction - 0.5) < 0.03, (
        f"Marginal fraction of masked (teacher) tokens should be ~0.5, got {overall_fraction:.4f}. "
        "Coin-flip Eq. 4 equivalence is broken."
    )


def test_per_sample_bimodal_fractions():
    """Per-sample masked fractions must be bimodal: near R_M or near 1-R_M, NOT near 0.5.

    This confirms the per-sample coin-flip structure (each sample has a SINGLE fraction
    of R_M or 1-R_M), not per-token averaging. Uses large N for tight per-sample
    estimates (N=4096 tokens => within ±0.1 of true fraction with high probability).
    """
    torch.manual_seed(99)
    R_M = 0.25
    B = 64
    N_h, N_w = 64, 64  # 4096 tokens
    student = torch.zeros(B, 4, N_h, N_w)
    teacher = torch.ones(B, 4, N_h, N_w)
    device = torch.device("cpu")

    coin = torch.rand(B) < 0.5
    effective_ratio = torch.where(coin, torch.full((B,), R_M), torch.full((B,), 1.0 - R_M))
    _, mask = apply_per_token_mask(student, teacher, effective_ratio, device)

    per_sample_frac = mask.float().mean(dim=1)  # (B,)
    tol = 0.10
    near_rm = (per_sample_frac - R_M).abs() < tol
    near_inv = (per_sample_frac - (1.0 - R_M)).abs() < tol
    all_bimodal = (near_rm | near_inv).all().item()
    assert all_bimodal, (
        f"Per-sample fractions must be near {R_M} or {1 - R_M} (±{tol}); "
        f"offending values: {per_sample_frac[(~near_rm & ~near_inv)].tolist()}"
    )
