import pytest
import torch

from musubi_tuner.dlssnr import numerics
from musubi_tuner.dlssnr.model import DenseFFN, ExpertFFN, Split512FFN


def test_cubic_activation_has_the_native_gain_and_unbounded_positive_tail():
    x = torch.tensor([-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0])
    expected = torch.tensor([-0.0, -0.0, -0.447265625, -0.503173828125, 0.0, 1.285888671875, 3.130859375, 7.15625, 14.3125])
    torch.testing.assert_close(numerics.mp_cubic_silu(x), expected, atol=0, rtol=0)


def test_cubic_activation_is_differentiable_without_detaching_training():
    x = torch.tensor([-5.0, -2.3, -0.2, 0.0, 0.7, 2.4, 5.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(numerics.mp_cubic_silu, (x,))
    derivative = torch.autograd.grad(numerics.mp_cubic_silu(x).sum(), x)[0]
    assert derivative[3].item() == 0.89453125
    assert derivative[-1].item() == 1.7890625


@pytest.mark.parametrize("family", ["dense", "expert", "split512"])
def test_every_ffn_uses_cubic_instead_of_ordinary_silu(family):
    if family == "dense":
        ffn = DenseFFN(32, 128)
        projections = [ffn.fc1, ffn.fc2]
        channels = 32
    elif family == "expert":
        ffn = ExpertFFN(64)
        projections = [ffn.experts[0]["fc1"], ffn.experts[0]["fc2"], ffn.fc3]
        channels = 64
    else:
        ffn = Split512FFN()
        branch = ffn.branches[0]
        projections = [branch["in_proj"], branch["fc1"], branch["fc2"], ffn.contract]
        channels = 512
    with torch.no_grad():
        for parameter in ffn.parameters():
            parameter.zero_()
        for projection in projections:
            projection.weight[0, 0] = 1
    x = torch.ones(1, channels, 1, 1, requires_grad=True)
    result = ffn(x)
    expected = torch.zeros_like(x)
    # Native FFNs publish the cubic hidden activation on the E4 grid.
    expected[:, 0] = 1.25
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    result.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0, 0, 0, 0] > 1


def test_publication_rounds_half_before_e4_and_keeps_a_float_gradient():
    from musubi_tuner.dlssnr import arithmetic

    x = torch.tensor([-0.0, 1.0626, 1.063, 500.0, -500.0], requires_grad=True)
    published = arithmetic.e4m3_ste(x)
    torch.testing.assert_close(published, torch.tensor([-0.0, 1.0, 1.125, 448.0, -448.0]), atol=0, rtol=0)
    assert torch.signbit(published[0])
    (published * 0.123456).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.123456, 0.123456, 0.123456, 0.0, 0.0]), atol=0, rtol=0)


@pytest.mark.parametrize("vit", [False, True])
def test_exponential_matches_independent_opendlss_scalar_reference(vit):
    from musubi_tuner.dlssnr.arithmetic import exp_weight

    # OpenDLSS-NR browser CPU reference, commit 9d08f418 (not the implementation under test).
    scores = torch.tensor([-10.0, -6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 10.0], requires_grad=True)
    window = [0.00006103515625, 0.00006103515625, 0.00128173828125, 0.00927734375, 0.025390625, 0.06640625, 0.484375, 9.75, 9.75]
    global_values = [
        0.0040283203125,
        0.0040283203125,
        0.00408935546875,
        0.02978515625,
        0.083984375,
        0.22265625,
        1.640625,
        1.640625,
        1.640625,
    ]
    output = exp_weight(scores, vit=vit)
    torch.testing.assert_close(output, torch.tensor(global_values if vit else window), atol=0, rtol=0)
    output.sum().backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[4] > 0
    assert scores.grad[0] == scores.grad[-1] == 0


def test_half_cosine_zero_rows_have_finite_backward_but_nan_is_not_hidden():
    from musubi_tuner.dlssnr.arithmetic import cosine_half

    x = torch.stack([torch.zeros(32), torch.ones(32)]).requires_grad_()
    actual = cosine_half(x)
    expected = torch.stack([torch.zeros(32), torch.full((32,), 0.1767578125)])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isnan(cosine_half(torch.full((1, 32), float("nan")))).all()


def test_native_cubic_half_publications_preserve_training_gradients():
    from musubi_tuner.dlssnr.arithmetic import cubic_half

    x = torch.tensor([-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0], requires_grad=True)
    expected = torch.tensor([-0.0, -0.167724609375, -0.5029296875, 0.0, 1.2861328125, 5.19921875, 17.890625])
    actual = cubic_half(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[3].item() == 0.89453125


@pytest.mark.parametrize("global_mode", [False, True])
def test_attention_publishes_values_on_e4_grid_and_has_finite_gradients(global_mode):
    x = torch.full((1, 32, 5, 5) if global_mode else (1, 32, 8, 8), 0.2, requires_grad=True)
    qkv = torch.zeros(96, 32)
    qkv[64:] = torch.eye(32)
    if global_mode:
        result = numerics.global_attention(x, qkv, torch.eye(32), torch.ones(1), torch.zeros(32))
    else:
        result = numerics.window_attention(x, qkv, torch.eye(32), torch.zeros(1, 64, 64), torch.ones(1), torch.zeros(32), 0)
    torch.testing.assert_close(result, torch.full_like(x, 0.203125), atol=0, rtol=0)
    result.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0


def test_half_sum64_and_zero_norm_reject_wrong_head_geometry():
    from musubi_tuner.dlssnr.arithmetic import cosine_half, sum64_half

    assert sum64_half(torch.ones(3, 64)).tolist() == [[64.0]] * 3
    with pytest.raises(ValueError, match="64 keys"):
        sum64_half(torch.ones(3, 63))
    with pytest.raises(ValueError, match="32 channels"):
        cosine_half(torch.ones(3, 16))
