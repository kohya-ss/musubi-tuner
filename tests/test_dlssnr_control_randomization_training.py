"""Opt-in control supervision through the actual NR data, optimizer and persistence paths."""

import json
import random

import numpy as np
import pytest
import toml
import torch
from safetensors.torch import load_file

from musubi_tuner.dlssnr import control_randomization as controls
from musubi_tuner.dlssnr import training_step
from musubi_tuner.dlssnr.base_anchor import NRBaseAnchor
from musubi_tuner.dlssnr.config import build_train_config, config_sha256
from musubi_tuner.dlssnr.dataset import NRBatchPlan
from musubi_tuner.dlssnr.temporal import AUGMENTATION_SEED_POLICY, stable_frame_seed
from musubi_tuner.training import dlssnr_trainer as trainer
from musubi_tuner.training.dlssnr_parser import setup_parser
from musubi_tuner.training.dlssnr_services import capture_rng
from test_dlssnr_base_anchor import batch_and_seeds
from test_dlssnr_control_randomization import SETTINGS
from test_dlssnr_ema import assert_ema_state
from test_dlssnr_frequency_loss import PROFILE, clip_batch, still_batch
from test_dlssnr_training import SmallNR, make_args, small_inject, small_math  # noqa: F401


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            assert_nested_equal(actual[name], expected[name])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested_equal(left, right)
    else:
        assert actual == expected


def fixed_args(tmp_path, *, lora=False, mode="single_frame", evaluate=False):
    args = make_args(tmp_path, lora=lora, mode=mode, evaluate=evaluate)
    data = toml.load(args.dataset_config)
    for entry in data["datasets"]:
        entry.update(nr_controls_mode="fixed", nr_tone=1, nr_structure=1)
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    return args


@pytest.mark.parametrize("lora", [False, True])
def test_control_settings_are_opt_in_and_bound_to_cli_config(tmp_path, lora):
    args = fixed_args(tmp_path, lora=lora)
    disabled = build_train_config(args, lora=lora)
    assert "control_randomization" not in disabled
    args.control_randomization = True
    enabled = build_train_config(args, lora=lora)
    assert enabled["control_randomization"] == {
        "schema": "dlssnr_control_randomization_v1",
        "seed_policy": AUGMENTATION_SEED_POLICY,
        **SETTINGS,
    }
    assert config_sha256(disabled) != config_sha256(enabled)
    for name, value in (("residual_sigma", 4), ("anchor_probability", 0.5), ("corner_probability", 0.5)):
        setattr(args, f"control_{name}", value)
        changed = build_train_config(args, lora=lora)
        assert changed["control_randomization"][name] == value
        assert config_sha256(changed) != config_sha256(enabled)
        setattr(args, f"control_{name}", None)
    parsed = setup_parser(lora=lora).parse_args(
        [
            "--dataset_config",
            str(args.dataset_config),
            "--output_dir",
            str(tmp_path / "cli"),
            "--output_name",
            "cli",
            "--development_smoke",
            "--control_randomization",
            "--control_residual_sigma",
            "6",
            "--control_anchor_probability",
            "0.25",
            "--control_corner_probability",
            "0.25",
        ]
    )
    assert build_train_config(parsed, lora=lora)["control_randomization"] == enabled["control_randomization"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("residual_sigma", 0),
        ("residual_sigma", 33),
        ("residual_sigma", True),
        ("anchor_probability", -0.1),
        ("corner_probability", float("inf")),
        ("corner_probability", float("nan")),
        ("anchor_probability", 0.9),
    ],
)
def test_invalid_control_options_are_rejected(tmp_path, field, value):
    args = fixed_args(tmp_path)
    args.control_randomization = True
    setattr(args, f"control_{field}", value)
    with pytest.raises(ValueError, match="control"):
        build_train_config(args)


@pytest.mark.parametrize("field", ["residual_sigma", "anchor_probability", "corner_probability"])
def test_disabled_randomization_rejects_explicit_unused_options(tmp_path, field):
    args = fixed_args(tmp_path)
    setattr(args, f"control_{field}", SETTINGS[field])
    with pytest.raises(ValueError, match="control_randomization"):
        build_train_config(args)


@pytest.mark.parametrize("invalid", ["files", "tone_zero", "structure_fp16_zero", "mixed_files"])
def test_invalid_reference_is_rejected_before_model_or_teacher_allocation(tmp_path, monkeypatch, invalid):
    args = fixed_args(tmp_path)
    args.control_randomization = True
    data = toml.load(args.dataset_config)
    entry = data["datasets"][0]
    if invalid == "files":
        data["datasets"] = [{"train_manifest": "data.jsonl"}]
    elif invalid == "mixed_files":
        data["datasets"].append({"train_manifest": "data.jsonl"})
    else:
        entry["nr_tone" if invalid == "tone_zero" else "nr_structure"] = 0 if invalid == "tone_zero" else 1e-12
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid fixed references must fail before any model allocation")

    monkeypatch.setattr(trainer, "NRModel", unexpected)
    monkeypatch.setattr(trainer, "NRBaseAnchor", unexpected)
    with pytest.raises(ValueError, match="fixed|reference"):
        trainer.train_from_args(args)


@pytest.mark.parametrize("clip", [False, True])
def test_attachment_is_per_clip_and_target_restoration_is_time_major(clip):
    source = torch.full((2, 3, 3, 5, 7) if clip else (2, 3, 5, 7), 0.2)
    target = torch.linspace(0.2, 0.8, source.numel()).reshape_as(source)
    original = clip_batch(source, target) if clip else still_batch(source, target)
    saved_controls = original["controls"].clone()
    draws = [controls.encode_control_point({}, (0, 0)), controls.encode_control_point({}, (1, 1))]
    batch = controls.attach_control_batch(original, draws)
    assert "control_ratios" not in original and torch.equal(original["controls"], saved_controls)
    assert batch["temporal_support"] == "joint_loss_mask"
    for index, draw in enumerate(draws):
        expected = draw["sampled_lanes"][:, None, None].expand(5, 5, 7)
        if clip:
            expected = expected.expand(3, 5, 5, 7)
        assert torch.equal(batch["controls"][index], expected)
        assert torch.equal(batch["control_reference_lanes"][index], draw["reference_lanes"])
    count = 4 if clip else 2
    rendered = torch.arange(1, count + 1).float()[:, None, None, None].expand(count, 3, 5, 7) * 0.1
    snapshot = {"rendered_proxy": rendered, "neural_preclamp": rendered + 1}
    actual, _ = controls.apply_control_targets(
        batch, snapshot, snapshot, burn_in=int(clip), settings=SETTINGS, loss_profile=PROFILE
    )
    assert torch.equal(original["target"], target) and torch.equal(batch["target"], target)
    if clip:
        for key in ("target", "preclamp_target", "edge_target"):
            assert torch.equal(actual[key][:, 0], target[:, 0])
        assert torch.equal(actual["target"][0, 1], rendered[0])
        assert torch.equal(actual["target"][0, 2], rendered[2])
        assert torch.equal(actual["preclamp_target"][0, 2], rendered[2] + 1)
        assert torch.equal(actual["target"][1, 1:], target[1, 1:])
        assert torch.equal(actual["edge_target"][1, 1:], source[1, 1:])
    else:
        assert torch.equal(actual["target"][0], rendered[0])
        assert torch.equal(actual["preclamp_target"][0], rendered[0] + 1)
        assert torch.equal(actual["target"][1], target[1])
        assert torch.equal(actual["edge_target"][1], source[1])


def test_control_metrics_use_global_rgb_mass():
    source = torch.full((2, 3, 2, 3), 0.8)
    mask = torch.tensor([[[[0.5, 0.5, 0], [0, 0, 0]]], [[[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]]]])
    original = still_batch(source, torch.full_like(source, 0.9), mask)
    draws = [controls.encode_control_point({}, (0.5, 0.5)), controls.encode_control_point({}, (0, 0))]
    batch = controls.attach_control_batch(original, draws)
    reference = {"rendered_proxy": torch.zeros_like(source), "neural_preclamp": torch.full_like(source, 1.2)}
    sampled = {**reference, "rendered_proxy": torch.full_like(source, 0.8)}
    actual, raw = controls.apply_control_targets(batch, reference, sampled, burn_in=0, settings=SETTINGS, loss_profile=PROFILE)
    assert raw == pytest.approx(
        {
            "control/_mass": 7.5,
            "control/_tone": 1.5,
            "control/_structure": 1.5,
            "control/_reference": 0,
            "control/_zero": 4.5,
            "control/_rgb_clipped": 3,
            "control/_edge_clipped": 3,
        }
    )
    assert actual["target"].max() == 1 and actual["edge_target"].max() == 1
    assert actual["preclamp_target"].min() > 1
    partial = []
    for i in range(2):
        part = {key: value[i : i + 1] if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        _, stats = controls.apply_control_targets(
            part,
            {key: value[i : i + 1] for key, value in reference.items()},
            {key: value[i : i + 1] for key, value in sampled.items()},
            burn_in=0,
            settings=SETTINGS,
            loss_profile=PROFILE,
        )
        partial.append(stats)
    reduced = {key: sum(part[key] for part in partial) for key in raw}
    assert reduced == raw
    metrics = controls.finalize_control_metrics({**reduced, "loss": 0.2})
    assert metrics == pytest.approx(
        {
            "loss": 0.2,
            "control/tone_mean": 0.2,
            "control/structure_mean": 0.2,
            "control/reference_fraction": 0,
            "control/zero_fraction": 0.6,
            "control/rgb_clipped_fraction": 0.4,
            "control/edge_clipped_fraction": 0.4,
        }
    )
    assert all(isinstance(value, float) for value in raw.values())


def test_empty_local_shard_keeps_all_raw_metric_keys():
    source = torch.ones(1, 3, 2, 3) * 0.5
    batch = controls.attach_control_batch(
        still_batch(source, source, torch.zeros_like(source[:, :1])), [controls.encode_control_point({}, (1, 1))]
    )
    snapshot = {"rendered_proxy": source, "neural_preclamp": source + 1}
    _, raw = controls.apply_control_targets(batch, snapshot, snapshot, burn_in=0, settings=SETTINGS, loss_profile=None)
    assert set(raw) == {
        f"control/_{name}" for name in ("mass", "tone", "structure", "reference", "zero", "rgb_clipped", "edge_clipped")
    }
    assert all(value == 0.0 for value in raw.values())


@pytest.mark.parametrize("mode", ["single_frame", "temporal"])
def test_microbatch_draws_use_final_repeat_ids_epoch_and_unchanged_frame_noise(tmp_path, mode):
    args = fixed_args(tmp_path, mode=mode)
    args.control_randomization, args.control_anchor_probability, args.control_corner_probability = True, 0, 0
    data = toml.load(args.dataset_config)
    data["datasets"][0]["num_repeats"] = 2
    args.dataset_config.write_text(toml.dumps(data), encoding="utf-8")
    config = build_train_config(args)
    dataset, _ = trainer._datasets(config)
    sample = dataset[0]
    assert sample["fixed_controls"]["nr_tone"] == 1
    sample["fixed_controls"]["nr_tone"] = 0
    assert dataset[0]["fixed_controls"]["nr_tone"] == 1
    plan = NRBatchPlan(dataset, 1)
    previous = []
    for micro_index in (0, 2, len(plan)):
        batch, seeds = trainer._microbatch(dataset, micro_index, config, plan)
        item = dataset[plan.indices(micro_index)[0]]
        epoch = micro_index // len(plan)
        expected = controls.sample_control_point(
            item["fixed_controls"],
            config["control_randomization"],
            seed=4,
            epoch=epoch,
            sample_id=item["sample_id"],
            crop_id=item["crop_id"],
        )
        assert torch.equal(batch["control_ratios"][0], expected["ratios"])
        frames = [item] if mode == "single_frame" else item["frames"]
        expected_seeds = [stable_frame_seed(4, epoch, item["sample_id"], frame["frame_index"], item["crop_id"]) for frame in frames]
        assert seeds == [expected_seeds[0]] if mode == "single_frame" else seeds == [expected_seeds]
        previous.append(batch["control_ratios"])
        assert batch["temporal_support"] == "joint_loss_mask"
    assert not torch.equal(previous[0], previous[1]) and not torch.equal(previous[0], previous[2])


@pytest.mark.usefixtures("small_math")
@pytest.mark.parametrize("lora", [False, True])
def test_randomization_teacher_does_not_enable_base_anchor(tmp_path, monkeypatch, lora):
    created, denominators = [], []
    original_denominators = trainer.loss_denominators

    def create(model, network):
        reference = NRBaseAnchor(model, network)
        created.append(reference)
        return reference

    def measure(*args, **kwargs):
        values = original_denominators(*args, **kwargs)
        denominators.append(values)
        return values

    monkeypatch.setattr(trainer, "NRBaseAnchor", create)
    monkeypatch.setattr(trainer, "loss_denominators", measure)
    args = fixed_args(tmp_path, lora=lora)
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    filename = "adapter.safetensors" if lora else "model.safetensors"
    ordinary_keys = set(load_file(args.output_dir / args.output_name / "final" / filename))
    assert not created
    args.control_randomization, args.output_name = True, "randomized"
    train(args)
    folder = args.output_dir / args.output_name
    assert len(created) == 1
    assert all(not value.requires_grad and value.grad is None for value in created[0].parameters())
    assert all("base_anchor" not in value for value in denominators)
    metadata = json.loads((folder / "run_config.json").read_text())
    assert "base_anchor" not in metadata and "base_anchor" not in metadata["identity"]
    assert metadata["control_randomization"] == metadata["identity"]["control_randomization"]
    assert metadata["control_randomization"]["reference"] == created[0].reference_identity
    assert metadata["control_randomization"]["reference_controls"][0]["reference_lanes"] == [0, 1, 1, 1, 1]
    assert {"base_anchor.py", "control_randomization.py"} <= metadata["identity"]["implementation"].keys()
    assert not any("base_anchor" in name or "reference" in name for name in metadata["trainable_parameters"])
    assert set(load_file(folder / "final" / filename)) == ordinary_keys
    rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert not any("base_anchor" in key or key.startswith("control/_") for key in row)
        assert 0 <= row["control/tone_mean"] <= 1
        assert 0 <= row["control/reference_fraction"] <= 1
        assert row["loss"] == pytest.approx(args.loss_pre * row["loss/pre"] + args.loss_out * row["loss/out"])


@pytest.mark.usefixtures("small_math")
@pytest.mark.parametrize(
    "lora,mode,profile",
    [(False, "single_frame", None), (False, "temporal", PROFILE), (True, "single_frame", PROFILE), (True, "temporal", None)],
)
def test_reference_only_controls_preserve_ordinary_updates_and_share_rollouts(tmp_path, monkeypatch, lora, mode, profile):
    args = fixed_args(tmp_path, lora=lora, mode=mode)
    for index in range(2):
        np.save(tmp_path / f"loss{index}.npy", np.ones((1, 48, 48), np.float32))
    if profile:
        args.loss_profile, args.loss_edge = "frequency_split", 0.2
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    filename = "adapter.safetensors" if lora else "model.safetensors"
    expected = load_file(args.output_dir / args.output_name / "final" / filename)
    calls = []
    original = NRBaseAnchor.predict

    def observe(self, model, network, batch, seeds, burn_in):
        calls.append(batch["control_ratios"].clone())
        return original(self, model, network, batch, seeds, burn_in)

    monkeypatch.setattr(NRBaseAnchor, "predict", observe)
    args.control_randomization, args.control_anchor_probability, args.control_corner_probability = True, 1, 0
    args.output_name = "reference_point"
    train(args)
    torch.testing.assert_close(load_file(args.output_dir / args.output_name / "final" / filename), expected, rtol=1e-5, atol=2e-7)
    assert len(calls) == 4 and all(value.eq(1).all() for value in calls)


@pytest.mark.usefixtures("small_math")
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("mode", ["single_frame", "temporal"])
def test_randomized_qat_fp8_dino_ema_training_resumes_exactly(tmp_path, monkeypatch, lora, mode):
    from test_dlssnr_dino import tiny_dino_loss

    monkeypatch.setattr(trainer, "create_dino_loss", tiny_dino_loss)
    if lora:
        from musubi_tuner.networks import lora_dlssnr
        from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject

        monkeypatch.setattr(trainer, "NRModel", TinyFP8NR)
        monkeypatch.setattr(lora_dlssnr, "inject", tiny_fp8_inject)
    args = fixed_args(tmp_path, lora=lora, mode=mode, evaluate=True)
    args.control_randomization = args.native_weight_qat = args.eval_native = True
    args.loss_profile, args.loss_lowpass_sigma, args.loss_edge = "frequency_split", 4, 0.2
    args.ema_decay, args.dino_loss_weight = 0.5, 0.1
    args.base_anchor_weight = 0.2 if mode == "temporal" else 0
    if lora:
        args.network_dropout, args.fp8_base, args.fp8_scaled, args.numerics_profile = 0, True, True, "train_experimental"
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    folder = args.output_dir / args.output_name
    filename = "adapter.safetensors" if lora else "model.safetensors"
    raw, averaged = load_file(folder / "final" / filename), load_file(folder / "final/ema" / filename)
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert not any("reference" in key or "base_anchor" in key for key in (*raw.keys(), *state["ema"]["shadow"].keys()))
    evaluations = json.loads((folder / "evaluation/step000002.json").read_text())
    args.resume = folder / "state-step000001"
    train(args)
    torch.testing.assert_close(load_file(folder / "final" / filename), raw, rtol=0, atol=0)
    torch.testing.assert_close(load_file(folder / "final/ema" / filename), averaged, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(resumed["ema"], state["ema"])
    assert_nested_equal(resumed["optimizer"], state["optimizer"])
    assert_nested_equal(resumed["rank_states"], state["rank_states"])
    assert resumed["identity"]["control_randomization"] == state["identity"]["control_randomization"]
    assert json.loads((folder / "evaluation/step000002.json").read_text()) == evaluations
    for field, value in (("residual_sigma", 5), ("anchor_probability", 0.5), ("corner_probability", 0.5)):
        setattr(args, f"control_{field}", value)
        with pytest.raises(ValueError, match="identity"):
            train(args)
        setattr(args, f"control_{field}", None)

    class ChangedReference(NRBaseAnchor):
        def __init__(self, model, network):
            super().__init__(model, network)
            self.reference_identity["base_parameters_sha256"] = "changed"

    monkeypatch.setattr(trainer, "NRBaseAnchor", ChangedReference)
    with pytest.raises(ValueError, match="identity"):
        train(args)


def test_control_reference_failure_restores_adapters_and_rng(monkeypatch):
    model = SmallNR()
    network = small_inject(model, {"dropout": 0.3})
    module = trainer.NRTrainModule(
        model, {"out": 1}, network=network, base_anchor=NRBaseAnchor(model, network), control_randomization=SETTINGS
    )
    batch, seeds = batch_and_seeds()
    batch = controls.attach_control_batch(batch, [controls.encode_control_point({}, (0.2, 0.7))] * 2)
    model.blocks["0"].eval()
    modes = [child.training for child in model.modules()]
    before = capture_rng()
    original = training_step.forward_frame
    calls = []

    def fail_on_reference_point(*args, **kwargs):
        assert not network.enabled and not torch.is_grad_enabled()
        calls.append(1)
        if len(calls) == 2:
            torch.rand(1)
            np.random.rand()
            random.random()
            raise RuntimeError("control reference failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(training_step, "forward_frame", fail_on_reference_point)
    with pytest.raises(RuntimeError, match="control reference failed"):
        module(batch, seeds)
    assert len(calls) == 2 and network.enabled and network.training
    assert [child.training for child in model.modules()] == modes
    assert_nested_equal(capture_rng(), before)


def test_checkpoint_replay_after_control_references_keeps_adapters_enabled():
    from musubi_tuner.dlssnr.model import _run_window_stage
    from musubi_tuner.networks.lora_dlssnr import DLSSNRLoRA
    from test_dlssnr_checkpointing import _blocks

    class CheckpointNR(SmallNR):
        def __init__(self, profile="dlss_nr_310_8_0"):
            super().__init__(profile=profile)
            self.stages = _blocks()
            self.checkpointing = False

        def forward(self, features, geometry):
            hidden = self.blocks["0"].input_adapter(features)
            hidden = _run_window_stage(self.stages, hidden, range(2), checkpointing=self.checkpointing)
            return torch.cat((self.blocks["70"].head.rgb(hidden), self.blocks["70"].head.logit(hidden)), dim=1)

    model = CheckpointNR()
    network = DLSSNRLoRA()
    for index in range(2):
        network.add(f"stages.{index}.ffn.fc1.weight", model.stages[str(index)].ffn.fc1, 2, 2, 0.3)
    for adapter in network.adapters:
        torch.nn.init.normal_(adapter.lora_up, std=0.02)
    model.requires_grad_(False)
    module = trainer.NRTrainModule(
        model, {"out": 1, "pre": 1}, network=network, base_anchor=NRBaseAnchor(model, network), control_randomization=SETTINGS
    )
    batch, seeds = batch_and_seeds()
    batch = controls.attach_control_batch(batch, [controls.encode_control_point({}, (0.3, 0.6))] * 2)
    visits, results = [], []

    def observe(*_):
        visits.append((network.enabled, torch.is_grad_enabled()))

    handle = model.stages["0"].ffn.register_forward_pre_hook(observe)
    try:
        for enabled in (False, True):
            network.zero_grad(set_to_none=True)
            model.checkpointing = enabled
            torch.manual_seed(71)
            visits.clear()
            loss, _ = module(batch, seeds)
            loss.backward()
            gradients = {name: parameter.grad.clone() for name, parameter in network.named_parameters()}
            assert any(value.abs().sum() > 0 for value in gradients.values())
            assert visits.count((False, False)) == 2
            assert visits.count((True, True)) == (2 if enabled else 1)
            assert network.enabled and (False, True) not in visits
            results.append((loss.detach(), gradients, capture_rng()))
    finally:
        handle.remove()
    assert_nested_equal(results[0], results[1])


@pytest.mark.usefixtures("small_math")
def test_control_overflow_retry_reuses_draws_and_restores_dropout_ema_optimizer_rng(tmp_path, monkeypatch):
    args = fixed_args(tmp_path, lora=True, mode="temporal")
    args.control_randomization, args.ema_decay, args.base_anchor_weight = True, 0.5, 0.2
    trainer.train_lora_from_args(args)
    folder = args.output_dir / args.output_name
    weights = load_file(folder / "final/adapter.safetensors")
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    original_update, original_forward, original_sample = (
        trainer.optimizer_update,
        trainer.NRTrainModule.forward,
        trainer.sample_control_point,
    )
    attempts, batches, draws = [], [], []

    def overflow_once(*args, **kwargs):
        attempts.append(1)
        return False if len(attempts) == 1 else original_update(*args, **kwargs)

    def record_batch(self, batch, seeds, *args, **kwargs):
        batches.append((batch["control_ratios"].clone(), seeds))
        return original_forward(self, batch, seeds, *args, **kwargs)

    def record_draw(*args, **kwargs):
        draw = original_sample(*args, **kwargs)
        draws.append(draw)
        return draw

    monkeypatch.setattr(trainer, "optimizer_update", overflow_once)
    monkeypatch.setattr(trainer.NRTrainModule, "forward", record_batch)
    monkeypatch.setattr(trainer, "sample_control_point", record_draw)
    args.output_name = "retry"
    trainer.train_lora_from_args(args)
    actual_folder = args.output_dir / args.output_name
    actual = torch.load(actual_folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert len(attempts) == 3 and len(batches) == 6 and len(draws) == 4
    torch.testing.assert_close(batches[:2], batches[2:4], rtol=0, atol=0)
    torch.testing.assert_close(load_file(actual_folder / "final/adapter.safetensors"), weights, rtol=0, atol=0)
    assert_ema_state(actual["ema"], state["ema"])
    assert_nested_equal(actual["optimizer"], state["optimizer"])
    assert_nested_equal(actual["rank_states"], state["rank_states"])
    assert actual["global_update"] == actual["ema"]["num_updates"] == 2
