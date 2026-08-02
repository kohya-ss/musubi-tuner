import pytest
import torch

from musubi_tuner.mage_flow.mage_vae import decode_mage_vae_latents
from musubi_tuner.mage_flow.sampling import euler_step


@pytest.mark.parametrize(
    ("dtype", "latent_value", "velocity_value", "next_sigma", "expected"),
    [
        (torch.float16, 0.1, 0.3, 0.6666667, 0.0),
        (torch.bfloat16, 0.1, 0.1, 0.3, 0.0302734375),
    ],
)
def test_euler_step_accumulates_in_float32_before_restoring_model_dtype(
    dtype,
    latent_value,
    velocity_value,
    next_sigma,
    expected,
):
    latent = torch.tensor([latent_value], dtype=dtype)
    velocity = torch.tensor([velocity_value], dtype=dtype)

    stepped = euler_step(latent, velocity, sigma=1.0, next_sigma=next_sigma)

    assert stepped.dtype == dtype
    torch.testing.assert_close(stepped.float(), torch.tensor([expected]), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("vae_dtype", "expected_autocast_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.float32, None),
    ],
)
def test_decode_uses_requested_vae_compute_dtype(vae_dtype, expected_autocast_dtype):
    captured = {}

    class FakeVAE:
        device = torch.device("cpu")
        dtype = vae_dtype

        def decode(self, latents):
            captured["latent_dtype"] = latents.dtype
            captured["autocast_enabled"] = torch.is_autocast_enabled("cpu")
            if captured["autocast_enabled"]:
                captured["autocast_dtype"] = torch.get_autocast_dtype("cpu")
            return latents

    result = decode_mage_vae_latents(FakeVAE(), torch.zeros(1, 4, 2, 2, dtype=vae_dtype))

    assert result.shape == (1, 4, 2, 2)
    assert captured["latent_dtype"] == torch.float32
    assert captured["autocast_enabled"] is (expected_autocast_dtype is not None)
    if expected_autocast_dtype is not None:
        assert captured["autocast_dtype"] == expected_autocast_dtype
