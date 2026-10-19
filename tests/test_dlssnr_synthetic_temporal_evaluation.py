"""Synthetic temporal evidence uses label support without changing real-clip metrics."""

from copy import deepcopy
import json

import pytest
import toml
import torch

from musubi_tuner.dlssnr import evaluation
from musubi_tuner.dlssnr.config import build_train_config
from musubi_tuner.dlssnr.runtime import configure_model_runtime, default_runtime_policy, native_weight_runtime
from musubi_tuner.dlssnr.synthetic_temporal import synthesize_clip
from musubi_tuner.training.dlssnr_trainer import _datasets
from test_dlssnr_synthetic_temporal import bilinear_oracle, still_sample
from test_dlssnr_synthetic_temporal_training import synthetic_args
from test_dlssnr_training import SmallNR


def test_marked_validation_is_fixed_epoch_zero_but_sequence_manifest_stays_real(tmp_path):
    args = synthetic_args(tmp_path, randomize=True, evaluate=True)
    row = json.loads((tmp_path / "validation.jsonl").read_text())
    row.update(sample_id="real_clip", sequence_id="real_scene")
    row["frames"] = [{**row["frames"][0], "frame_index": 20 + index, "reset": index == 0} for index in range(3)]
    (tmp_path / "real.jsonl").write_text(json.dumps(row), encoding="utf-8")
    data = toml.load(args.dataset_config)
    data["datasets"][0]["sequence_manifest"] = "real.jsonl"
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    args.min_sequence_frames = 3
    train, datasets = _datasets(build_train_config(args))
    model = SmallNR().eval()
    configure_model_runtime(model, default_runtime_policy())
    first = evaluation.evaluate(model, datasets, 4, torch.device("cpu"), compare_native=True)
    assert first["validation"][0]["frames"] == 3
    assert first["validation"][0]["synthetic_temporal"] == datasets["validation"].synthetic_protocol
    assert first["validation"][0]["native"]["synthetic_temporal"] == datasets["validation"].synthetic_protocol
    assert "synthetic_temporal" not in first["sequences"][0]
    assert "synthetic_temporal" not in first["sequences"][0]["native"]
    assert [frame["frame_index"] for frame in datasets["sequences"][0]["frames"]] == [20, 21, 22]
    assert datasets["sequences"][0]["frames"][1]["history_valid"].all()
    trajectory = datasets["validation"][0]["frames"][1]["motion"].clone()
    train.get_sample(0, epoch=31)
    datasets["validation"].get_sample(0, epoch=17)
    assert torch.equal(trajectory, datasets["validation"][0]["frames"][1]["motion"])
    assert evaluation.evaluate(model, datasets, 4, torch.device("cpu"), compare_native=True) == first
    assert datasets["validation"][0]["frames"][0]["controls"][1:].eq(1).all()


class FrameCase:
    def __init__(self, frames, *, synthetic=True):
        self.frames = frames
        self.rows = [{"sample_id": "case", "crop_id": 0, "frames": [{"frame_index": frame["frame_index"]} for frame in frames]}]
        if synthetic:
            self.synthetic_protocol = {
                "schema": "dlssnr_synthetic_temporal_v1",
                "temporal_metric": "joint_loss_mask_normalized_residual",
            }

    def iter_frames(self, index):
        yield from self.frames


def masked_frames():
    sample = still_sample()
    sample["loss_mask"] *= 0.7
    sample["loss_mask"][..., 8:16, 10:23] = 0
    frames = synthesize_clip(sample, torch.tensor([[0, 0], [0.5, 0], [0, 0], [0.5, 0]]))["frames"]
    frames[2]["reset"] = True
    frames[2]["motion"] = torch.zeros_like(frames[2]["motion"])
    frames[2]["loss_mask"] = frames[2]["loss_mask"].clone()
    frames[2]["loss_mask"][..., 32:] *= 0.25
    return frames


def affine_forward(model, source, controls, seed, **kwargs):
    output = source * 0.8 + 0.03
    return {
        "raw_head": torch.cat((output, torch.zeros_like(output[:, :1])), dim=1),
        "rendered_proxy": output,
        "neural_preclamp": output,
        "next_history": output,
        "blend_weight": torch.zeros_like(output[:, :1]),
    }


def direct_temporal_metric(frames, *, joint):
    numerator = denominator = 0.0
    previous = None
    for frame in frames:
        current = frame["source"] * 0.8 + 0.03
        if not frame["reset"] and previous is not None:
            output, target, mask = previous
            offset = frame["motion"][:, 0, 0]
            if joint:
                coverage = bilinear_oracle(mask, offset, border=False)
                warped = bilinear_oracle((output - target) * mask, offset, border=False)
                error = current - frame["target"] - warped / torch.where(coverage > 0, coverage, 1)
                valid = frame["temporal_valid"] * frame["loss_mask"] * coverage
            else:
                error = (current - bilinear_oracle(output, offset, border=False)) - (
                    frame["target"] - bilinear_oracle(target, offset, border=False)
                )
                valid = frame["temporal_valid"]
            numerator += float((error.abs() * valid).sum())
            denominator += float(valid.sum()) * 3
        previous = current, frame["target"], frame["loss_mask"]
    return numerator / denominator


def test_synthetic_temporal_metric_ignores_neutral_fill_and_resets(monkeypatch):
    frames = masked_frames()
    monkeypatch.setattr(evaluation, "forward_frame", affine_forward)
    report = evaluation.evaluate(None, {"synthetic": FrameCase(frames)}, 4, torch.device("cpu"))["synthetic"][0]
    assert report["temporal_mae"] == pytest.approx(direct_temporal_metric(frames, joint=True), rel=1e-5, abs=1e-7)
    changed = deepcopy(frames)
    for frame in changed:
        frame["target"].masked_fill_(frame["loss_mask"].expand_as(frame["target"]) == 0, 500)
    actual = evaluation.evaluate(None, {"synthetic": FrameCase(changed)}, 4, torch.device("cpu"))["synthetic"][0]
    assert actual == report
    reset_pair = evaluation.evaluate(None, {"synthetic": FrameCase(frames[2:])}, 4, torch.device("cpu"))["synthetic"][0]
    assert reset_pair["temporal_mae"] == pytest.approx(direct_temporal_metric(frames[2:], joint=True), rel=1e-5, abs=1e-7)


def test_real_temporal_metric_keeps_legacy_unmasked_residual(monkeypatch):
    frames = masked_frames()
    monkeypatch.setattr(evaluation, "forward_frame", affine_forward)
    report = evaluation.evaluate(None, {"real": FrameCase(frames, synthetic=False)}, 4, torch.device("cpu"))["real"][0]
    assert "synthetic_temporal" not in report
    assert report["temporal_mae"] == pytest.approx(direct_temporal_metric(frames, joint=False), rel=1e-5, abs=1e-7)
    assert report["temporal_mae"] != pytest.approx(direct_temporal_metric(frames, joint=True), rel=1e-3)


def test_native_comparison_uses_previous_frame_mask_for_both_modes():
    datasets = {"synthetic": FrameCase(masked_frames())}
    model = SmallNR().eval()
    configure_model_runtime(model, default_runtime_policy())
    combined = evaluation.evaluate(model, datasets, 4, torch.device("cpu"), compare_native=True)["synthetic"][0]
    with native_weight_runtime(model):
        standalone = evaluation.evaluate(model, datasets, 4, torch.device("cpu"))["synthetic"][0]
    assert combined["native"] == standalone
    assert combined["native"]["synthetic_temporal"] == datasets["synthetic"].synthetic_protocol


def test_synthetic_validation_rejects_multiframe_manifest_and_source_overlap(tmp_path):
    args = synthetic_args(tmp_path, evaluate=True)
    manifest = tmp_path / "validation.jsonl"
    row = json.loads(manifest.read_text())
    row["frames"].append({**row["frames"][0], "frame_index": 11, "reset": False})
    manifest.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="sequence_length|single.frame"):
        _datasets(build_train_config(args))
    row["frames"] = row["frames"][:1]
    row["sequence_id"] = "train"
    manifest.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        _datasets(build_train_config(args))
