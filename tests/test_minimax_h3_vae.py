from pathlib import Path
import json
import sys

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import musubi_tuner.minimax_h3.video_vae as video_vae
from musubi_tuner.minimax_h3.audio_vae import MiniMaxH3AudioVAE, encode_audio_mode
from musubi_tuner.minimax_h3.video_vae import (
    CausalConv3d,
    MiniMaxH3VideoVAE,
    ViT3DDecoder,
    encode_video_condition,
    encode_video_target,
)
from musubi_tuner.minimax_h3.checkpoint import load_safetensors_module, resolve_safetensors_files


class _ExplodingProjection(nn.Module):
    def forward(self, inputs):
        raise AssertionError("logs_proj must not be evaluated for MiniMax-H3 audio caching")


def test_video_vae_dtype_policy_uses_fp32_encoding_and_fp16_decoding():
    assert getattr(video_vae, "VIDEO_VAE_ENCODE_DTYPE", None) is torch.float32
    assert getattr(video_vae, "VIDEO_VAE_DECODE_DTYPE", None) is torch.float16


def test_video_vae_keeps_the_published_checkpoint_structure_after_provenance_rewrite():
    with torch.device("meta"):
        vae = MiniMaxH3VideoVAE()
    state = vae.state_dict()

    assert len(state) == 562
    assert state["encoder.down.1.block.0.nin_shortcut.weight"].shape == (256, 128, 1, 1, 1)
    assert state["decoder.mask_token"].shape == (1, 1, 2048)
    assert state["decoder.transformer_blocks.0.attn.to_qkv.weight"].shape == (6144, 2048)
    assert state["decoder.transformer_blocks.35.ff.w2.weight"].shape == (2048, 8192)


def test_audio_vae_encode_uses_mean_projection_without_logs_projection():
    vae = MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE)
    nn.Module.__init__(vae)
    vae.hop_length = 4
    vae.encoder = nn.Conv1d(1, 4, kernel_size=4, stride=4, bias=False)
    vae.pre_block = nn.Identity()
    vae.mean_proj = nn.Conv1d(4, 4, kernel_size=1, bias=False)
    vae.logs_proj = _ExplodingProjection()
    vae.register_buffer("latents_mean", torch.zeros(4))
    vae.register_buffer("latents_std", torch.ones(4))
    nn.init.constant_(vae.encoder.weight, 0.25)
    nn.init.eye_(vae.mean_proj.weight[:, :, 0])

    waveform = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8) / 16
    latents = vae.encode(waveform)

    assert latents.shape == (1, 4, 2, 2)
    torch.testing.assert_close(latents[:, :, 0], latents[:, :, 0].clone())


def test_encode_audio_mode_preserves_published_feature_stereo_time_layout():
    expected = torch.randn(1, 32, 2, 8)

    class FakeAudioVAE:
        def encode(self, waveform):
            assert waveform.shape == (1, 2, 6400)
            return expected

    actual = encode_audio_mode(FakeAudioVAE(), torch.zeros(1, 2, 6400))

    assert actual is expected
    assert actual.shape == (1, 32, 2, 8)


def test_video_target_posterior_is_stable_per_cache_seed_and_item_key():
    class FakeVideoVAE:
        latents_mean = torch.zeros(24)
        latents_std = torch.ones(24)

        def encode_moments(self, pixels):
            shape = (pixels.shape[0], 24, 2, 1, 1)
            return torch.cat([torch.zeros(shape), torch.zeros(shape)], dim=1)

    pixels = torch.zeros(1, 3, 5, 16, 16)
    vae = FakeVideoVAE()

    first = encode_video_target(vae, pixels, cache_seed=123, canonical_item_key="C:/data/clip.mp4")
    repeated = encode_video_target(vae, pixels, cache_seed=123, canonical_item_key="C:/data/clip.mp4")
    other_item = encode_video_target(vae, pixels, cache_seed=123, canonical_item_key="C:/data/other.mp4")

    torch.testing.assert_close(first, repeated)
    assert not torch.equal(first, other_item)


def test_video_condition_uses_seed_42_and_fp16_round_trip_before_normalization():
    class FakeVideoVAE:
        latents_mean = torch.linspace(-0.2, 0.2, 24)
        latents_std = torch.linspace(0.8, 1.2, 24)

        def encode_moments(self, pixels):
            shape = (pixels.shape[0], 24, 1, 1, 1)
            mean = torch.full(shape, 0.1234567)
            logvar = torch.full(shape, -2.0)
            return torch.cat([mean, logvar], dim=1)

    vae = FakeVideoVAE()
    pixels = torch.zeros(1, 3, 1, 16, 16)
    generator = torch.Generator(device="cpu").manual_seed(42)
    noise = torch.randn((1, 24, 1, 1, 1), generator=generator)
    raw = torch.full_like(noise, 0.1234567) + torch.exp(torch.tensor(-1.0)) * noise
    rounded = raw.to(torch.float16).to(torch.float32)
    expected = (rounded - vae.latents_mean.view(1, -1, 1, 1, 1)) / vae.latents_std.view(1, -1, 1, 1, 1)

    actual = encode_video_condition(vae, pixels)

    torch.testing.assert_close(actual, expected)


def test_causal_conv_single_frame_truncates_temporal_kernel():
    conv = CausalConv3d(1, 1, kernel_size=3, padding=1)
    nn.init.ones_(conv.weight)
    nn.init.zeros_(conv.bias)
    inputs = torch.ones(1, 1, 1, 3, 3)

    output = conv(inputs)

    assert output.shape == inputs.shape
    assert output[0, 0, 0, 1, 1].item() == pytest.approx(9.0)


def test_tiny_vit3d_decoder_runs_without_comfy_runtime():
    decoder = ViT3DDecoder(
        patch_size=2,
        patch_size_t=1,
        in_channels=2,
        out_channels=3,
        num_layers=1,
        heads=2,
        dim_head=8,
        rope_dim_ratio=0.75,
        num_register_tokens=1,
    )
    for parameter in decoder.parameters():
        nn.init.uniform_(parameter, -0.1, 0.1)

    output = decoder(torch.randn(1, 2, 1, 2, 2))

    assert output.shape == (1, 3, 1, 4, 4)
    assert torch.isfinite(output).all()


class _TinyCheckpointModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 3)
        self.second = nn.Linear(3, 1)


def test_checkpoint_loader_streams_sharded_directory_and_loads_strictly(tmp_path: Path):
    expected = _TinyCheckpointModule().state_dict()
    for index, tensor in enumerate(expected.values(), start=1):
        tensor.fill_(index)
    first_keys = ["first.weight", "first.bias"]
    second_keys = ["second.weight", "second.bias"]
    save_file({key: expected[key] for key in first_keys}, tmp_path / "model-00001-of-00002.safetensors")
    save_file({key: expected[key] for key in second_keys}, tmp_path / "model-00002-of-00002.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in expected.values())},
                "weight_map": {
                    **{key: "model-00001-of-00002.safetensors" for key in first_keys},
                    **{key: "model-00002-of-00002.safetensors" for key in second_keys},
                },
            }
        ),
        encoding="utf-8",
    )

    files = resolve_safetensors_files(tmp_path)
    actual = load_safetensors_module(lambda: _TinyCheckpointModule(), files, device="cpu", dtype=torch.float32)

    assert [path.name for path in files] == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for key, tensor in actual.state_dict().items():
        torch.testing.assert_close(tensor, expected[key])


def test_checkpoint_loader_supports_explicit_key_transforms(tmp_path: Path):
    expected = _TinyCheckpointModule().state_dict()
    checkpoint = tmp_path / "prefixed.safetensors"
    save_file({f"source.{key}": tensor for key, tensor in expected.items()}, checkpoint)

    actual = load_safetensors_module(
        lambda: _TinyCheckpointModule(),
        [checkpoint],
        device="cpu",
        dtype=torch.float32,
        key_transform=lambda key: key.removeprefix("source."),
    )

    for key, tensor in actual.state_dict().items():
        torch.testing.assert_close(tensor, expected[key])


@pytest.mark.parametrize(
    "state_dict",
    [
        {"first.weight": torch.zeros(3, 2)},
        {
            "first.weight": torch.zeros(3, 2),
            "first.bias": torch.zeros(3),
            "second.weight": torch.zeros(1, 3),
            "second.bias": torch.zeros(1),
            "unexpected": torch.zeros(1),
        },
    ],
)
def test_checkpoint_loader_rejects_missing_or_unexpected_keys(tmp_path: Path, state_dict: dict[str, torch.Tensor]):
    checkpoint = tmp_path / "model.safetensors"
    save_file(state_dict, checkpoint)

    with pytest.raises(ValueError, match="checkpoint key mismatch"):
        load_safetensors_module(lambda: _TinyCheckpointModule(), [checkpoint], device="cpu", dtype=torch.float32)
