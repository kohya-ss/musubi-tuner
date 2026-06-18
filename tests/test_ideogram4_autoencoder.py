import torch
from torch import nn

from musubi_tuner.ideogram4 import ideogram4_utils
from musubi_tuner.ideogram4.ideogram4_autoencoder import AutoEncoder, AutoEncoderParams


class FakeEncoder(nn.Module):
    def __init__(self, moments: torch.Tensor):
        super().__init__()
        self.register_buffer("moments", moments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moments.to(device=x.device, dtype=x.dtype)


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.seen = z.detach().clone()
        return torch.zeros(z.shape[0], 3, z.shape[2] * 8, z.shape[3] * 8, device=z.device, dtype=z.dtype)


def make_autoencoder(z_channels: int = 32) -> AutoEncoder:
    params = AutoEncoderParams(resolution=16, ch=32, ch_mult=[1], num_res_blocks=0, z_channels=z_channels)
    return AutoEncoder(params)


def poison_batch_norm(autoencoder: AutoEncoder) -> None:
    channels = autoencoder.bn.running_mean.numel()
    autoencoder.bn.running_mean.copy_(torch.arange(channels, dtype=torch.float32) + 10.0)
    autoencoder.bn.running_var.copy_(torch.full((channels,), 4.0))


def test_encode_returns_patchified_raw_encoder_mean_without_batch_norm():
    autoencoder = make_autoencoder()
    poison_batch_norm(autoencoder)
    mean = torch.arange(1 * 32 * 4 * 6, dtype=torch.float32).reshape(1, 32, 4, 6)
    moments = torch.cat([mean, torch.zeros_like(mean)], dim=1)
    autoencoder.encoder = FakeEncoder(moments)

    actual = autoencoder.encode(torch.zeros(1, 3, 16, 16))

    assert torch.equal(actual, ideogram4_utils.patchify_vae_latents(mean))
    assert torch.equal(ideogram4_utils.unpatchify_vae_latents(actual, 2, 3), mean)


def test_decode_unpatchifies_tokens_without_batch_norm_inverse():
    autoencoder = make_autoencoder()
    poison_batch_norm(autoencoder)
    decoder = FakeDecoder()
    autoencoder.decoder = decoder
    latents = torch.arange(1 * 32 * 4 * 6, dtype=torch.float32).reshape(1, 32, 4, 6)
    tokens = ideogram4_utils.patchify_vae_latents(latents)

    autoencoder.decode(tokens)

    assert torch.equal(decoder.seen, latents)


def test_is_diffusers_vae_state_dict_distinguishes_layouts():
    native = {
        "encoder.down.0.block.0.norm1.weight": torch.zeros(1),
        "encoder.norm_out.weight": torch.zeros(1),
        "bn.running_mean": torch.zeros(1),
    }
    diffusers = {
        "encoder.down_blocks.0.resnets.0.norm1.weight": torch.zeros(1),
        "encoder.conv_norm_out.weight": torch.zeros(1),
    }
    assert not ideogram4_utils._is_diffusers_vae_state_dict(native)
    assert ideogram4_utils._is_diffusers_vae_state_dict(diffusers)
