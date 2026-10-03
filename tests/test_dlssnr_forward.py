import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from musubi_tuner.dlssnr.config import build_train_config
from musubi_tuner.dlssnr.dataset import load_single_frame_manifest
from musubi_tuner.dlssnr.geometry import resolve_geometry
from musubi_tuner.dlssnr.model import DenseFFN, NRModel, WindowAttn
from musubi_tuner.dlssnr.numerics import PHASE_SHIFTS, _scatter_windows, _window_qkv, window_attention
from musubi_tuner.dlssnr.preprocess import build_features, center_proxy
from musubi_tuner.dlssnr.profiles import canonical_names
from musubi_tuner.training.dlssnr_parser import setup_parser
from musubi_tuner.training.dlssnr_trainer import (
    build_optimizer,
    single_frame_update,
)


def test_geometry_matches_the_audited_fields():
    field = resolve_geometry(512, 512)
    assert (field.full_width, field.full_height) == (576, 512)
    assert field.levels == ((288, 256), (144, 128), (72, 64), (36, 32), (20, 16), (12, 8))
    assert (resolve_geometry(768, 768).full_width, resolve_geometry(768, 768).full_height) == (832, 768)
    assert (resolve_geometry(644, 768).full_width, resolve_geometry(644, 768).full_height) == (768, 768)
    assert (resolve_geometry(1920, 1080).full_width, resolve_geometry(1920, 1080).full_height) == (1920, 1152)
    assert (resolve_geometry(3840, 2160).full_width, resolve_geometry(3840, 2160).full_height) == (3840, 2176)
    with pytest.raises(ValueError):
        resolve_geometry(32, 32)


@pytest.mark.parametrize("phase", range(4))
@pytest.mark.parametrize("shape", [(16, 24), (12, 20)])
def test_window_tokens_preserve_coordinates_and_zero_only_outside_the_field(phase, shape):
    height, width = shape
    sx, sy = PHASE_SHIFTS[phase]
    wy, wx = (height + sy + 7) // 8, (width + sx + 7) // 8
    pixels = (100 * torch.arange(height)[:, None] + torch.arange(width)[None, :] + 1).float()
    qkv = pixels[None, None].expand(1, 96, height, width)
    query, key, value = _window_qkv(qkv, height, width, wy, wx, sx, sy, 1)
    expected = torch.zeros(wy * wx, 64)
    for window_y in range(wy):
        for window_x in range(wx):
            for token in range(64):
                y, x = window_y * 8 + token // 8 - sy, window_x * 8 + token % 8 - sx
                if 0 <= y < height and 0 <= x < width:
                    expected[window_y * wx + window_x, token] = 100 * y + x + 1
    for tensor in (query, key, value):
        torch.testing.assert_close(tensor[0, :, 0, :, 0], expected, rtol=0, atol=0)
    tokens = value.permute(0, 1, 3, 2, 4).reshape(1, wy * wx, 64, 32)
    restored = _scatter_windows(tokens, height, width, wy, wx, sx, sy)
    torch.testing.assert_close(restored[0, 0], pixels, rtol=0, atol=0)


def test_center_and_feature_lanes():
    proxy = torch.zeros(1, 3, 48, 48)
    proxy[:, :, 10, 12] = torch.tensor([0.0, 0.5, 1.0])
    centered = center_proxy(proxy)
    assert centered.dtype == torch.float32
    assert centered[0, 1, 10, 12].item() == 0.0
    controls = torch.zeros(1, 5, 48, 48)
    controls[:, 0] = 2 / 128
    controls[:, 1] = 0.5
    features = build_features(proxy, controls, resolve_geometry(48, 48), frame_seed=7)
    assert features.shape[1] == 16
    assert torch.all(features[:, 3] == 1)
    assert torch.all(features[:, 15] == 0)
    assert torch.allclose(features[:, 10], torch.full_like(features[:, 10], 2 / 128))
    again = build_features(proxy, controls, resolve_geometry(48, 48), frame_seed=7)
    assert torch.equal(features[:, 0:3], again[:, 0:3])


def test_window_block_residual_and_constant_attention():
    torch.manual_seed(0)
    ffn = DenseFFN(32, 128)
    attn = WindowAttn(32, 0)
    for module in (ffn.fc1, ffn.fc2, attn.qkv, attn.proj):
        module.weight.data.zero_()
    ffn.skip_scale.data.fill_(1)
    attn.skip_scale.data.fill_(1)
    source = torch.randn(1, 32, 8, 8)
    published = source.half().to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(attn(ffn(source)), published, rtol=0, atol=0)

    y = torch.ones(1, 32, 8, 8)
    qkv = torch.full((96, 32), 1.0 / 32)
    projection = torch.eye(32)
    prior = torch.zeros(1, 64, 64)
    temperature = torch.ones(1)
    output = window_attention(y, qkv, projection, prior, temperature, torch.zeros(32), 0)
    assert torch.allclose(output, torch.ones_like(y), atol=1e-5)
    shifted = window_attention(y, qkv, projection, prior, temperature, torch.zeros(32), 1)
    assert torch.isfinite(shifted).all()


def test_parameter_names_match_the_canonical_checkpoint():
    model = NRModel()
    assert set(model.state_dict()) == set(canonical_names())
    assert torch.count_nonzero(model.blocks["0"].input_adapter.weight[:, 15]) == 0


def test_single_frame_step_updates_image_weights_and_freezes_temporal_head():
    torch.manual_seed(1)
    model = NRModel()
    optimizer = build_optimizer(model, learning_rate=1e-3, multipliers={"priors": 0.1, "scales": 0.1}, weight_decay=0.0)
    source = torch.full((1, 3, 48, 48), 0.5)
    target = torch.full((1, 3, 48, 48), 0.2)
    controls = torch.zeros(1, 5, 48, 48)
    before_logit = model.blocks["70"].head.logit.weight.detach().clone()
    before_blend = model.blocks["70"].blend_scale.detach().clone()
    before_matrix = model.blocks["1"].ffn.fc1.weight.detach().clone()
    metrics = single_frame_update(
        model, optimizer, source, target, controls, frame_seed=3, loss_weights={"pre": 1, "out": 1, "edge": 0.05}
    )
    assert metrics["blend_max"] == 0.0
    assert np.isfinite(metrics["loss"])
    assert torch.equal(model.blocks["70"].head.logit.weight, before_logit)
    assert torch.equal(model.blocks["70"].blend_scale, before_blend)
    assert torch.count_nonzero(model.blocks["0"].input_adapter.weight[:, 15]) == 0
    assert not torch.allclose(model.blocks["1"].ffn.fc1.weight, before_matrix)


def test_manifest_and_config_reject_bad_single_frame_inputs(tmp_path: Path):
    image = tmp_path / "frame.png"
    Image.fromarray(np.full((48, 48, 3), 128, dtype=np.uint8)).save(image)
    controls = tmp_path / "controls.npy"
    np.save(controls, np.zeros((5, 48, 48), dtype=np.float32))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema": "dlssnr_pairs_v1",
                "sample_id": "one",
                "sequence_id": "scene",
                "source_encoding": "srgb_proxy",
                "target_encoding": "srgb_proxy",
                "controls_encoding": "dlssnr_lanes_10_14_v1",
                "frames": [
                    {
                        "frame_index": 0,
                        "input_path": "frame.png",
                        "target_path": "frame.png",
                        "controls_path": "controls.npy",
                        "reset": True,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = load_single_frame_manifest(manifest, 48, 48)
    assert samples[0]["source"].shape == (3, 48, 48)
    config = tmp_path / "train.toml"
    config.write_text(
        """
[general]
resolution = [48, 48]
[[datasets]]
train_manifest = "train.jsonl"
""".strip(),
        encoding="utf-8",
    )
    args = setup_parser().parse_args(
        [
            "--dataset_config",
            str(config),
            "--output_dir",
            str(tmp_path / "out"),
            "--output_name",
            "run",
            "--development_smoke",
            "--training_mode",
            "temporal",
        ]
    )
    with pytest.raises(ValueError, match="sequence_length"):
        build_train_config(args)
