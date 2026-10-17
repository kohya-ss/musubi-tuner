import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from musubi_tuner.dlssnr.dataset import load_temporal_manifest
from musubi_tuner.dlssnr.model import NRModel
from musubi_tuner.dlssnr.pipeline import forward_frame, rollout_sequence
from musubi_tuner.dlssnr.temporal import stable_frame_seed, warp_bilinear
from musubi_tuner.training.dlssnr_trainer import build_optimizer, temporal_clip_update


class TinyHistory(nn.Module):
    """A stand-in whose RGB head reads the history lanes, so the clip graph is cheap to test."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))
        block = nn.Module()
        block.blend_scale = nn.Parameter(torch.tensor([0.5]))
        self.blocks = nn.ModuleDict({"70": block})

    def forward(self, features: torch.Tensor, geometry) -> torch.Tensor:
        height, width = geometry.valid_height, geometry.valid_width
        output = features.new_zeros(features.shape[0], 4, features.shape[-2], features.shape[-1])
        output[:, 0:3, :height, :width] = features[:, 7:10, :height, :width] * self.gain
        output[:, 3, :height, :width] = 2.0
        return output


def test_bilinear_warp_follows_current_to_previous_pixels():
    image = torch.zeros(1, 1, 4, 4)
    image[0, 0, 1, 2] = 1
    motion = torch.zeros(1, 2, 4, 4)
    motion[0, 0] = 1
    sampled, inside = warp_bilinear(image, motion)
    assert inside[0, 0, 1, 1]
    assert sampled[0, 0, 1, 1].item() == torch.tensor(1.0).item() or abs(sampled[0, 0, 1, 1].item() - 1) < 1e-5
    motion = torch.zeros(1, 2, 4, 4)
    motion[0, 0] = -1
    sampled, inside = warp_bilinear(image, motion)
    assert inside[0, 0, 1, 3]
    assert abs(sampled[0, 0, 1, 3].item() - 1) < 1e-5
    motion = torch.full((1, 2, 4, 4), 100.0)
    sampled, inside = warp_bilinear(image, motion)
    assert not bool(inside.any())
    assert torch.count_nonzero(sampled) == 0


def test_frame_seed_is_stable_and_depends_on_the_sample_id():
    assert stable_frame_seed(5, 0, "clip", 1) == stable_frame_seed(5, 0, "clip", 1)
    assert stable_frame_seed(5, 0, "clip", 1) != stable_frame_seed(5, 0, "other", 1)
    assert stable_frame_seed(5, 0, "clip", 1) != stable_frame_seed(5, 0, "clip", 2)


def test_later_train_frame_reaches_the_earlier_frame_but_not_burn_in():
    model = TinyHistory()
    source = torch.full((1, 3, 48, 48), 0.4)
    controls = torch.zeros(1, 5, 48, 48)
    motion = torch.zeros(1, 2, 48, 48)
    valid = torch.ones(1, 1, 48, 48)
    with torch.no_grad():
        burned = forward_frame(model, source, controls, 1, history=None, reset=torch.tensor([True]))
    assert not burned["next_history"].requires_grad
    first = forward_frame(
        model, source, controls, 2, history=burned["next_history"], motion=motion, history_valid=valid, reset=torch.tensor([False])
    )
    first["next_history"].retain_grad()
    second = forward_frame(
        model, source, controls, 3, history=first["next_history"], motion=motion, history_valid=valid, reset=torch.tensor([False])
    )
    second["rendered_proxy"].sum().backward()
    assert first["next_history"].grad is not None
    assert float(first["next_history"].grad.abs().sum()) > 0
    reset = forward_frame(
        model, source, controls, 4, history=burned["next_history"], motion=motion, history_valid=valid, reset=torch.tensor([True])
    )
    assert float(reset["blend_weight"].detach().max()) == 0.0


def test_rollout_detaches_every_frame():
    model = TinyHistory()
    frames = 64
    source = torch.full((frames, 3, 48, 48), 0.4)
    controls = torch.zeros(frames, 5, 48, 48)
    motion = torch.zeros(frames, 2, 48, 48)
    history_valid = torch.ones(frames, 1, 48, 48)
    reset = torch.zeros(frames, dtype=torch.bool)
    reset[0] = True
    rendered = rollout_sequence(model, source, controls, motion, reset, history_valid, list(range(frames)))
    assert rendered.shape == (frames, 3, 48, 48)
    assert torch.isfinite(rendered).all()
    assert not rendered.requires_grad


def test_temporal_manifest_requires_motion_layout(tmp_path: Path):
    image = tmp_path / "frame.png"
    from PIL import Image

    Image.fromarray(np.full((48, 48, 3), 100, dtype=np.uint8)).save(image)
    np.save(tmp_path / "controls.npy", np.zeros((5, 48, 48), np.float32))
    np.save(tmp_path / "motion.npy", np.zeros((2, 48, 48), np.float32))
    np.save(tmp_path / "valid.npy", np.ones((1, 48, 48), np.float32))
    manifest = tmp_path / "clip.jsonl"
    frame = {
        "frame_index": 0,
        "input_path": "frame.png",
        "target_path": "frame.png",
        "controls_path": "controls.npy",
        "reset": True,
    }
    second = {
        "frame_index": 1,
        "input_path": "frame.png",
        "target_path": "frame.png",
        "controls_path": "controls.npy",
        "motion_path": "motion.npy",
        "history_valid_path": "valid.npy",
        "temporal_valid_path": "valid.npy",
        "reset": False,
    }
    manifest.write_text(
        json.dumps(
            {
                "schema": "dlssnr_pairs_v1",
                "sample_id": "clip",
                "sequence_id": "scene",
                "source_encoding": "srgb_proxy",
                "target_encoding": "srgb_proxy",
                "controls_encoding": "dlssnr_lanes_10_14_v1",
                "motion_layout": "chw",
                "frames": [frame, second],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    clips = load_temporal_manifest(manifest, 48, 48, 2)
    assert clips[0]["frames"][1]["motion"].shape == (2, 48, 48)
    assert bool(clips[0]["frames"][0]["reset"])


def test_real_model_temporal_step_trains_blend_and_logit():
    torch.manual_seed(2)
    model = NRModel()
    optimizer = build_optimizer(model, 1e-2, {"priors": 0.1, "scales": 0.1, "temporal_blend": 1.0}, 0.0, include_temporal=True)
    source = torch.full((1, 2, 3, 48, 48), 0.55)
    target = torch.full((1, 2, 3, 48, 48), 0.15)
    controls = torch.zeros(1, 2, 5, 48, 48)
    motion = torch.zeros(1, 2, 2, 48, 48)
    history_valid = torch.ones(1, 2, 1, 48, 48)
    history_valid[:, 0] = 0
    batch = {
        "source": source,
        "target": target,
        "controls": controls,
        "motion": motion,
        "history_valid": history_valid,
        "temporal_valid": history_valid.clone(),
        "reset": torch.tensor([[True, False]]),
    }
    before_blend = model.blocks["70"].blend_scale.detach().clone()
    before_logit = model.blocks["70"].head.logit.weight.detach().clone()
    metrics = temporal_clip_update(
        model, optimizer, batch, [[11, 12]], {"pre": 1.0, "out": 1.0, "edge": 0.05, "temporal": 0.1}, burn_in=1
    )
    assert np.isfinite(metrics["loss"])
    assert metrics["blend_max"] > 0
    assert not torch.equal(model.blocks["70"].blend_scale, before_blend)
    assert not torch.equal(model.blocks["70"].head.logit.weight, before_logit)
    assert torch.count_nonzero(model.blocks["0"].input_adapter.weight[:, 15]) == 0
