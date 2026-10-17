"""Synthetic dataset configuration, optimizer integration and combined acceptance."""

import json

import pytest
import toml
import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file

from musubi_tuner.dlssnr.config import build_train_config, config_sha256, load_dataset_config
from musubi_tuner.dlssnr.dataset import NRBatchPlan
from musubi_tuner.training import dlssnr_trainer as trainer
from test_dlssnr_config import make_args as config_args
from test_dlssnr_control_randomization_training import assert_nested_equal, fixed_args
from test_dlssnr_directory_dataset import paired_directories
from test_dlssnr_ema import assert_ema_state
from test_dlssnr_training import SmallNR, small_math  # noqa: F401


def synthetic_args(tmp_path, *, lora=False, randomize=False, evaluate=False):
    args = fixed_args(tmp_path, lora=lora, evaluate=evaluate)
    args.training_mode, args.sequence_length, args.burn_in, args.tbptt_length, args.loss_temporal = "temporal", 3, 1, 2, 0.1
    args.control_randomization = randomize
    data = toml.load(args.dataset_config)
    data["datasets"][0]["synthetic_temporal"] = True
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    return args


def test_synthetic_configuration_is_opt_in_and_does_not_enable_other_losses(tmp_path):
    args = fixed_args(tmp_path)
    plain = build_train_config(args)
    assert "synthetic_temporal" not in plain["data"]
    args = synthetic_args(tmp_path)
    args.loss_temporal = 0
    enabled = build_train_config(args)
    assert enabled["data"]["synthetic_temporal"] == {"schema": "dlssnr_synthetic_temporal_v1", "max_shift_px": 0.5}
    assert enabled["loss"]["temporal"] == 0 and "control_randomization" not in enabled
    data = toml.load(args.dataset_config)
    data["datasets"][0]["synthetic_max_shift_px"] = 0.7
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    assert config_sha256(build_train_config(args)) != config_sha256(enabled)


def test_synthetic_inheritance_respects_dataset_disable(tmp_path):
    data = {
        "general": {"synthetic_temporal": True, "synthetic_max_shift_px": 0.75},
        "datasets": [{"train_manifest": "one.jsonl"}, {"train_manifest": "two.jsonl", "synthetic_temporal": False}],
    }
    args = config_args(tmp_path, dataset=data)
    resolved = load_dataset_config(args.dataset_config)["datasets"]
    assert resolved[0]["synthetic_temporal"]["max_shift_px"] == 0.75
    assert "synthetic_temporal" not in resolved[1]
    data["datasets"][1]["synthetic_max_shift_px"] = 0.2
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic"):
        load_dataset_config(args.dataset_config)
    data["datasets"] = [{"train_manifest": "one.jsonl", "synthetic_temporal": False}]
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic"):
        load_dataset_config(args.dataset_config)


@pytest.mark.parametrize(
    "entry",
    [
        {"synthetic_temporal": "true"},
        {"synthetic_temporal": 1},
        {"synthetic_temporal": True, "synthetic_max_shift_px": -0.1},
        {"synthetic_temporal": True, "synthetic_max_shift_px": 1.1},
        {"synthetic_temporal": True, "synthetic_max_shift_px": True},
        {"synthetic_temporal": True, "synthetic_max_shift_px": float("nan")},
        {"synthetic_max_shift_px": 0.5},
    ],
)
def test_invalid_synthetic_dataset_settings_are_rejected(tmp_path, entry):
    args = config_args(tmp_path, dataset={"datasets": [{"train_manifest": "one.jsonl", **entry}]})
    with pytest.raises(ValueError, match="synthetic"):
        load_dataset_config(args.dataset_config)


def test_synthetic_flag_requires_temporal_mode_and_explicit_lengths(tmp_path):
    args = synthetic_args(tmp_path)
    args.training_mode, args.sequence_length, args.burn_in, args.tbptt_length, args.loss_temporal = "single_frame", 1, 0, 1, 0
    with pytest.raises(ValueError, match="temporal"):
        build_train_config(args)
    args.training_mode, args.sequence_length = "temporal", None
    with pytest.raises(ValueError, match="sequence_length"):
        build_train_config(args)


def test_marked_directories_load_generated_clips_but_unmarked_temporal_dirs_fail(tmp_path):
    args = synthetic_args(tmp_path)
    entry = {**paired_directories(tmp_path), "synthetic_temporal": True}
    data = {"general": {"resolution": [64, 48]}, "datasets": [entry]}
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    dataset, _ = trainer._datasets(build_train_config(args))
    assert len(dataset[0]["frames"]) == 3 and dataset[0]["fixed_controls"]["nr_tone"] == 1
    assert not list((tmp_path / "pair").rglob("*.npy"))
    assert not list((tmp_path / "pair").rglob("*.jsonl"))
    del entry["synthetic_temporal"]
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        build_train_config(args)


def test_marked_real_manifest_is_not_silently_flattened_and_real_validation_stays_strict(tmp_path):
    args = fixed_args(tmp_path, mode="temporal")
    data = toml.load(args.dataset_config)
    data["datasets"][0]["synthetic_temporal"] = True
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="sequence_length|single.frame"):
        trainer._datasets(build_train_config(args))
    del data["datasets"][0]["synthetic_temporal"]
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    rows = [json.loads(line) for line in (tmp_path / "data.jsonl").read_text().splitlines()]
    del rows[0]["frames"][1]["motion_path"]
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="motion_path"):
        trainer._datasets(build_train_config(args))


def test_trainer_microbatches_load_their_epoch_before_sampling_motion(tmp_path):
    args = synthetic_args(tmp_path)
    data = toml.load(args.dataset_config)
    data["datasets"][0]["num_repeats"] = 2
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    config = build_train_config(args)
    dataset, _ = trainer._datasets(config)
    plan = NRBatchPlan(dataset, 1)
    first, _ = trainer._microbatch(dataset, 0, config, plan)
    later, _ = trainer._microbatch(dataset, len(plan), config, plan)
    expected = dataset.get_sample(0, epoch=1)
    assert torch.equal(later["motion"][0], torch.stack([frame["motion"] for frame in expected["frames"]]))
    assert not torch.equal(later["motion"], first["motion"])
    again, _ = trainer._microbatch(dataset, len(plan), config, plan)
    assert_nested_equal(again, later)


@pytest.mark.usefixtures("small_math")
def test_synthetic_only_training_updates_temporal_head_without_allocating_reference(tmp_path, monkeypatch):
    models = []

    def create(profile="dlss_nr_310_8_0"):
        model = SmallNR(profile=profile)
        models.append(model)
        return model

    def unexpected(*args, **kwargs):
        pytest.fail("Synthetic clips alone must not allocate a frozen NR reference")

    monkeypatch.setattr(trainer, "NRModel", create)
    monkeypatch.setattr(trainer, "NRBaseAnchor", unexpected)
    args = synthetic_args(tmp_path)
    trainer.train_from_args(args)
    assert len(models) == 1
    for parameter in (models[0].blocks["70"].head.logit.weight, models[0].blocks["70"].blend_scale):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    folder = args.output_dir / args.output_name
    metadata = json.loads((folder / "run_config.json").read_text())
    assert "control_randomization" not in metadata and "base_anchor" not in metadata
    assert all(bucket["frames"] == 3 for bucket in metadata["bucket_plan"]["buckets"])
    assert metadata["synthetic_temporal"] == metadata["identity"]["synthetic_temporal"]
    assert metadata["synthetic_temporal"][0]["protocol"]["schema"] == "dlssnr_synthetic_temporal_v1"
    assert "synthetic_temporal.py" in metadata["identity"]["implementation"]
    rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    assert [row["update"] for row in rows] == [1, 2]
    assert all(row["loss/temporal"] >= 0 for row in rows)


def textured_pairs(root):
    y, x = np.indices((48, 48))
    source = np.stack((30 + 4 * x, 20 + 4 * y, 50 + ((x + 2 * y) % 16) * 10), axis=-1).astype(np.uint8)
    target = (np.roll(source, 1, axis=1).astype(np.float32) * 0.8 + 20).astype(np.uint8)
    Image.fromarray(source).save(root / "source.png")
    Image.fromarray(target).save(root / "target.png")


@pytest.mark.usefixtures("small_math")
@pytest.mark.parametrize(
    "lora,randomize,frequency", [(False, False, False), (False, True, True), (True, False, True), (True, True, False)]
)
def test_synthetic_runtime_matrix_and_exact_resume_preserve_trajectories(tmp_path, monkeypatch, lora, randomize, frequency):
    from test_dlssnr_dino import tiny_dino_loss

    monkeypatch.setattr(trainer, "create_dino_loss", tiny_dino_loss)
    if lora:
        from musubi_tuner.networks import lora_dlssnr
        from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject

        monkeypatch.setattr(trainer, "NRModel", TinyFP8NR)
        monkeypatch.setattr(lora_dlssnr, "inject", tiny_fp8_inject)
    args = synthetic_args(tmp_path, lora=lora, randomize=randomize, evaluate=True)
    textured_pairs(tmp_path)
    args.ema_decay, args.dino_loss_weight = 0.5, 0.1
    args.native_weight_qat = args.eval_native = True
    args.base_anchor_weight = 0.2 if randomize or lora else 0
    args.loss_edge = 0.2
    if frequency:
        args.loss_profile, args.loss_lowpass_sigma = "frequency_split", 4
    if lora:
        args.network_dropout, args.fp8_base, args.fp8_scaled, args.numerics_profile = 0, True, True, "train_experimental"
    original = trainer.NRTrainModule.forward
    seen = []

    def observed(self, batch, seeds, *args, **kwargs):
        assert batch["temporal_support"] == "joint_loss_mask"
        if randomize:
            assert torch.equal(batch["controls"], batch["controls"][:, :1].expand_as(batch["controls"]))
        seen.append(
            {
                "sample_id": list(batch["sample_id"]),
                "seeds": seeds,
                **{name: batch[name].clone() for name in ("controls", "motion", "target", "loss_mask")},
            }
        )
        return original(self, batch, seeds, *args, **kwargs)

    monkeypatch.setattr(trainer.NRTrainModule, "forward", observed)
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    folder = args.output_dir / args.output_name
    filename = "adapter.safetensors" if lora else "model.safetensors"
    raw, averaged = load_file(folder / "final" / filename), load_file(folder / "final/ema" / filename)
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    report = json.loads((folder / "evaluation/step000002.json").read_text())
    assert state["identity"]["synthetic_temporal"][0]["protocol"]["sequence_length"] == 3
    assert not any("reference" in key or "base_anchor" in key for key in (*raw.keys(), *state["ema"]["shadow"].keys()))
    assert len(seen) == 4
    expected = seen[2:]
    assert not torch.equal(seen[0]["motion"], seen[2]["motion"])
    seen.clear()
    args.resume = folder / "state-step000001"
    train(args)
    assert_nested_equal(seen, expected)
    torch.testing.assert_close(load_file(folder / "final" / filename), raw, rtol=0, atol=0)
    torch.testing.assert_close(load_file(folder / "final/ema" / filename), averaged, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(resumed["ema"], state["ema"])
    assert_nested_equal(resumed["optimizer"], state["optimizer"])
    assert_nested_equal(resumed["rank_states"], state["rank_states"])
    assert json.loads((folder / "evaluation/step000002.json").read_text()) == report
    for candidate in (report["candidate"], report["ema_candidate"]):
        case = candidate["validation"][0]
        assert case["synthetic_temporal"]["schema"] == "dlssnr_synthetic_temporal_v1"
        assert case["native"]["synthetic_temporal"] == case["synthetic_temporal"]


@pytest.mark.usefixtures("small_math")
def test_synthetic_resume_rejects_changed_shift_seed_clip_sampler_and_files(tmp_path, monkeypatch):
    args = synthetic_args(tmp_path, randomize=True)
    trainer.train_from_args(args)
    args.resume = args.output_dir / args.output_name / "state-step000001"
    data = toml.load(args.dataset_config)
    data["datasets"][0]["synthetic_max_shift_px"] = 0.75
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        trainer.train_from_args(args)
    del data["datasets"][0]["synthetic_max_shift_px"]
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    args.seed = 5
    with pytest.raises(ValueError, match="identity"):
        trainer.train_from_args(args)
    args.seed, args.sequence_length, args.tbptt_length = 4, 4, 3
    with pytest.raises(ValueError, match="identity"):
        trainer.train_from_args(args)
    args.sequence_length, args.tbptt_length = 3, 2
    original = trainer.NRSyntheticTemporalDataset

    def changed_sampler(*args, **kwargs):
        dataset = original(*args, **kwargs)
        dataset.synthetic_protocol["seed_policy"] = "changed"
        return dataset

    with monkeypatch.context() as patch:
        patch.setattr(trainer, "NRSyntheticTemporalDataset", changed_sampler)
        with pytest.raises(ValueError, match="identity"):
            trainer.train_from_args(args)
    textured_pairs(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        trainer.train_from_args(args)


@pytest.mark.usefixtures("small_math")
def test_synthetic_control_retry_reuses_motion_points_and_dropout_rng(tmp_path, monkeypatch):
    args = synthetic_args(tmp_path, lora=True, randomize=True)
    textured_pairs(tmp_path)
    args.ema_decay = 0.5
    trainer.train_lora_from_args(args)
    folder = args.output_dir / args.output_name
    raw = load_file(folder / "final/adapter.safetensors")
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    update, forward = trainer.optimizer_update, trainer.NRTrainModule.forward
    calls, batches = [], []

    def retry_once(*args, **kwargs):
        calls.append(1)
        return False if len(calls) == 1 else update(*args, **kwargs)

    def capture(self, batch, seeds, *args, **kwargs):
        batches.append(
            {"seeds": seeds, **{name: batch[name].clone() for name in ("source", "motion", "controls", "control_ratios")}}
        )
        return forward(self, batch, seeds, *args, **kwargs)

    monkeypatch.setattr(trainer, "optimizer_update", retry_once)
    monkeypatch.setattr(trainer.NRTrainModule, "forward", capture)
    args.output_name = "retry"
    trainer.train_lora_from_args(args)
    actual_folder = args.output_dir / args.output_name
    actual = torch.load(actual_folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert len(calls) == 3 and len(batches) == 6
    assert_nested_equal(batches[:2], batches[2:4])
    torch.testing.assert_close(load_file(actual_folder / "final/adapter.safetensors"), raw, rtol=0, atol=0)
    assert_nested_equal(actual["optimizer"], state["optimizer"])
    assert_nested_equal(actual["rank_states"], state["rank_states"])
    assert_ema_state(actual["ema"], state["ema"])
    assert actual["global_update"] == actual["ema"]["num_updates"] == 2
