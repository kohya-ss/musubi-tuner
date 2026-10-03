import json

import numpy as np
import pytest
import toml
import torch
from PIL import Image
from safetensors.torch import load_file

from musubi_tuner.dlssnr.model import ChannelLinear, NRModel
from musubi_tuner.networks.lora_dlssnr import DLSSNRLoRA
from musubi_tuner.training import dlssnr_trainer as trainer
from musubi_tuner.training.dlssnr_parser import setup_parser


class SmallNR(torch.nn.Module):
    """Replace expensive NR math, not the real optimizer, state, loader or runner."""

    def __init__(self):
        super().__init__()
        first, last, head = torch.nn.Module(), torch.nn.Module(), torch.nn.Module()
        first.input_adapter = ChannelLinear(32, 16)
        head.rgb, head.logit = ChannelLinear(3, 32), ChannelLinear(1, 32)
        last.head = head
        last.blend_scale = torch.nn.Parameter(torch.tensor([0.5]))
        self.blocks = torch.nn.ModuleDict({"0": first, "70": last})
        for name, parameter in self.named_parameters():
            if name.endswith("weight"):
                torch.nn.init.normal_(parameter, std=0.05)
        self.enforce_lane15()

    enforce_lane15 = NRModel.enforce_lane15
    freeze_single_frame = NRModel.freeze_single_frame
    load_canonical = NRModel.load_canonical

    def forward(self, features, geometry):
        features = self.blocks["0"].input_adapter(features)
        return torch.cat((self.blocks["70"].head.rgb(features), self.blocks["70"].head.logit(features)), dim=1)


def small_inject(model, table):
    network = DLSSNRLoRA()
    network.profile = "manual"
    target = "blocks.70.head.rgb.weight"
    network.add(target, model.blocks["70"].head.rgb, 2, 2, table.get("dropout", 0.0))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    network.report = {"profile": "manual", "targets": [{"name": target, "rank": 2, "alpha": 2, "in": 32, "out": 3}]}
    return network


@pytest.fixture
def small_math(monkeypatch):
    from accelerate.state import AcceleratorState
    from musubi_tuner.networks import lora_dlssnr

    AcceleratorState._reset_state(reset_partial_state=True)
    monkeypatch.setattr(trainer, "NRModel", SmallNR)
    monkeypatch.setattr(lora_dlssnr, "inject", small_inject)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


def make_args(tmp_path, *, lora=False, accum=2, batch=1, mode="single_frame", evaluate=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((48, 48, 3), 140, np.uint8)).save(tmp_path / "source.png")
    Image.fromarray(np.full((48, 48, 3), 60, np.uint8)).save(tmp_path / "target.png")
    np.save(tmp_path / "controls.npy", np.ones((5, 48, 48), np.float32))
    np.save(tmp_path / "motion.npy", np.zeros((2, 48, 48), np.float32))
    np.save(tmp_path / "valid.npy", np.ones((1, 48, 48), np.float32))
    rows = []
    length = 1 if mode == "single_frame" else 3
    for sample in range(2):
        loss_mask = np.ones((1, 48, 48), np.float32)
        if sample:
            loss_mask[:, :32] = 0
        np.save(tmp_path / f"loss{sample}.npy", loss_mask)
        frames = []
        for index in range(length):
            frames.append(
                {
                    "frame_index": index + 10,
                    "input_path": "source.png",
                    "target_path": "target.png",
                    "controls_path": "controls.npy",
                    "reset": index == 0,
                    "motion_path": "motion.npy",
                    "history_valid_path": "valid.npy",
                    "temporal_valid_path": "valid.npy",
                    "loss_mask_path": f"loss{sample}.npy",
                }
            )
        rows.append(
            {
                "schema": "dlssnr_pairs_v1",
                "sample_id": f"sample{sample}",
                "sequence_id": "train",
                "source_encoding": "srgb_proxy",
                "target_encoding": "srgb_proxy",
                "controls_encoding": "dlssnr_lanes_10_14_v1",
                "motion_layout": "chw",
                "frames": frames,
            }
        )
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    dataset = {"train_manifest": "data.jsonl"}
    if evaluate:
        rows[0]["sequence_id"] = "validation"
        (tmp_path / "validation.jsonl").write_text(json.dumps(rows[0]), encoding="utf-8")
        dataset["validation_manifest"] = "validation.jsonl"
    path = tmp_path / "dataset.toml"
    path.write_text(toml.dumps({"general": {"resolution": [48, 48], "batch_size": batch}, "datasets": [dataset]}), encoding="utf-8")
    argv = [
        "--dataset_config",
        str(path),
        "--development_smoke",
        "--device",
        "cpu",
        "--seed",
        "4",
        "--training_mode",
        mode,
        "--sequence_length",
        str(length),
        "--burn_in",
        "0" if length == 1 else "1",
        "--tbptt_length",
        "1" if length == 1 else "2",
        "--gradient_accumulation_steps",
        str(accum),
        "--max_train_steps",
        "2",
        "--learning_rate",
        "0.001",
        "--loss_edge",
        "0",
        "--loss_temporal",
        "0" if length == 1 else "0.1",
        "--output_dir",
        str(tmp_path / "output"),
        "--output_name",
        "dlssnr",
        "--save_every_n_steps",
        "1",
        "--save_state",
    ]
    if lora:
        argv.extend(["--network_dim", "16", "--network_alpha", "16", "--network_dropout", "0.2"])
    if evaluate:
        argv.extend(["--sample_every_n_steps", "1"])
    return setup_parser(lora=lora).parse_args(argv)


@pytest.mark.parametrize("lora", [False, True])
def test_linear_scheduler_counts_updates_and_resumes_exactly(tmp_path, small_math, lora):
    args = make_args(tmp_path, lora=lora, accum=2)
    args.lr_scheduler = "linear"
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    output = tmp_path / "output/dlssnr"
    first = torch.load(output / "state-step000001/trainer_state.pt", weights_only=True)
    second = torch.load(output / "state-step000002/trainer_state.pt", weights_only=True)
    assert first["scheduler"]["last_epoch"] == 1
    assert second["scheduler"]["last_epoch"] == 2
    assert first["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.0005)
    assert second["optimizer"]["param_groups"][0]["lr"] == 0
    filename = "adapter.safetensors" if lora else "model.safetensors"
    expected = {name: tensor.clone() for name, tensor in load_file(output / "final" / filename).items()}
    args.resume = output / "state-step000001"
    train(args)
    for name, tensor in load_file(output / "final" / filename).items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


def test_scheduler_does_not_advance_on_retried_update(tmp_path, small_math, monkeypatch):
    args = make_args(tmp_path)
    args.lr_scheduler = "linear"
    update = trainer.optimizer_update
    attempts = []

    def overflow_once(accelerator, module, optimizer, max_grad_norm):
        attempts.append(optimizer.param_groups[0]["lr"])
        return False if len(attempts) == 1 else update(accelerator, module, optimizer, max_grad_norm)

    monkeypatch.setattr(trainer, "optimizer_update", overflow_once)
    trainer.train_from_args(args)
    assert attempts == pytest.approx([0.001, 0.001, 0.0005])
    state = torch.load(tmp_path / "output/dlssnr/state-step000002/trainer_state.pt", weights_only=True)
    assert state["scheduler"]["last_epoch"] == 2


@pytest.mark.parametrize("lora", [False, True])
def test_directory_pairs_train_with_fixed_conditions_and_scheduler(tmp_path, small_math, lora):
    args = make_args(tmp_path, lora=lora)
    inputs, targets = tmp_path / "inputs", tmp_path / "targets"
    inputs.mkdir()
    targets.mkdir()
    with Image.open(tmp_path / "source.png") as source:
        source.save(inputs / "pair.png")
    with Image.open(tmp_path / "target.png") as target:
        target.save(targets / "pair.png")
    args.dataset_config.write_text(
        toml.dumps(
            {
                "general": {"resolution": [48, 48]},
                "datasets": [{"image_directory": "targets", "control_directory": "inputs", "nr_auto_mask": False}],
            }
        ),
        encoding="utf-8",
    )
    args.lr_scheduler = "cosine"
    (trainer.train_lora_from_args if lora else trainer.train_from_args)(args)
    output = tmp_path / "output/dlssnr"
    metadata = json.loads((output / "run_config.json").read_text())
    assert metadata["config"]["data"]["fixed_controls"]["nr_auto_mask"] is False
    assert metadata["bucket_plan"]["samples_per_epoch"] == 1
    state = torch.load(output / "state-step000002/trainer_state.pt", weights_only=True)
    assert state["scheduler"]["last_epoch"] == 2
    assert state["optimizer"]["param_groups"][0]["lr"] == 0


@pytest.mark.parametrize("lora", [False, True])
def test_output_names_isolate_weights_state_logs_and_evaluation(tmp_path, small_math, lora):
    args = make_args(tmp_path, lora=lora, evaluate=True)
    args.output_name = "experiment_a"
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    output = tmp_path / "output"
    first_files = {file: file.read_bytes() for file in output.rglob("*") if file.is_file()}
    assert first_files
    args.output_name, args.seed = "experiment_b", 5
    train(args)
    for file, contents in first_files.items():
        assert file.read_bytes() == contents, f"the second run overwrote {file.relative_to(output)}"
    filename = "adapter.safetensors" if lora else "model.safetensors"
    for name in ("experiment_a", "experiment_b"):
        run = output / name
        for relative in (
            f"final/{filename}",
            f"state-step000001/{filename}",
            "state-step000001/trainer_state.pt",
            "metrics.jsonl",
            "run_config.json",
            "evaluation/step000002.json",
        ):
            assert (run / relative).is_file()
        assert json.loads((run / "run_config.json").read_text())["config"]["output"]["output_name"] == name


def test_new_training_rejects_an_existing_run_directory(tmp_path, small_math):
    args = make_args(tmp_path)
    run = tmp_path / "output" / "dlssnr"
    run.mkdir(parents=True)
    previous = run / "existing.safetensors"
    previous.write_bytes(b"previous run must remain untouched")
    with pytest.raises(FileExistsError, match="output_name|resume"):
        trainer.train_from_args(args)
    assert previous.read_bytes() == b"previous run must remain untouched"
    assert list(run.iterdir()) == [previous]


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("mode", ["single_frame", "temporal"])
def test_full_and_lora_resume_with_accumulation_and_comments(tmp_path, small_math, lora, mode):
    args = make_args(tmp_path, lora=lora, mode=mode)
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    output = tmp_path / "output" / "dlssnr"
    filename = "adapter.safetensors" if lora else "model.safetensors"
    expected = {key: tensor.clone() for key, tensor in load_file(output / "final" / filename).items()}
    saved = output / "state-step000001"
    assert (saved / "trainer_state.pt").is_file()
    assert (output / "final" / "training_metadata.json").is_file()
    args.dataset_config.write_text(args.dataset_config.read_text() + "\n# comments do not change a run\n", encoding="utf-8")
    args.resume = saved
    train(args)
    actual = load_file(output / "final" / filename)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    state = torch.load(output / "state-step000002" / "trainer_state.pt", weights_only=True)
    assert state["global_update"] == 2
    assert state["consumed_samples"] == 4
    Image.fromarray(np.full((48, 48, 3), 61, np.uint8)).save(tmp_path / "target.png")
    with pytest.raises(ValueError, match="identity|data"):
        train(args)


@pytest.mark.parametrize("lora", [False, True])
def test_normal_training_without_evidence_preserves_unvalidated_metadata_and_resume(tmp_path, small_math, lora):
    from test_dlssnr_artifacts import make_canonical

    args = make_args(tmp_path, lora=lora)
    args.development_smoke = False
    args.model_dir = make_canonical(tmp_path / "base", SmallNR())
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    output = args.output_dir / "dlssnr"
    run = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert run["config"]["training"]["development_smoke"] is False
    assert run["source_forward_validated"] is False
    assert run["experimental_surrogate"] is True
    assert "forward_validation_report" not in run["identity"]["source"]
    metadata = json.loads((output / "final/training_metadata.json").read_text(encoding="utf-8"))
    for name in ("float_validated", "temporal_validated", "native_export_validated"):
        assert metadata[name] is False
    filename = "adapter.safetensors" if lora else "model.safetensors"
    expected = {key: tensor.clone() for key, tensor in load_file(output / "final" / filename).items()}
    (args.model_dir / "forward_validation_report.json").write_text("unrequested sidecar", encoding="utf-8")
    args.resume = output / "state-step000001"
    train(args)
    actual = load_file(output / "final" / filename)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("lora", [False, True])
def test_explicit_source_evidence_never_validates_trained_outputs(tmp_path, small_math, lora):
    from test_dlssnr_artifacts import make_canonical, make_validation_report
    from musubi_tuner.dlssnr.identity import file_sha256

    args = make_args(tmp_path, lora=lora)
    args.development_smoke = False
    args.model_dir = make_canonical(tmp_path / "base", SmallNR())
    args.forward_validation_report = make_validation_report(args.model_dir)
    args.max_train_steps = 1
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    output = args.output_dir / "dlssnr"
    metadata = json.loads((output / "final/training_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_forward_validated"] is True
    assert metadata["identity"]["source"]["forward_validation_report"] == file_sha256(args.forward_validation_report)
    for name in ("float_validated", "temporal_validated", "native_export_validated"):
        assert metadata[name] is False
    args.resume = output / "state-step000001"
    report = json.loads(args.forward_validation_report.read_text(encoding="utf-8"))
    report["reference_identity"] = {"test_fixture": "different synthetic evidence"}
    args.forward_validation_report.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        train(args)


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("development_smoke", [False, True])
def test_training_rejects_invalid_explicit_source_evidence(tmp_path, small_math, lora, development_smoke):
    from test_dlssnr_artifacts import make_canonical

    args = make_args(tmp_path, lora=lora)
    args.development_smoke = development_smoke
    args.model_dir = make_canonical(tmp_path / "base", SmallNR())
    args.forward_validation_report = args.model_dir / "invalid.json"
    args.forward_validation_report.write_text("{}", encoding="utf-8")
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    with pytest.raises(ValueError, match="forward validation"):
        train(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("field,value", [("learning_rate", 0.002), ("optimizer_type", "SGD"), ("network_dim", 8)])
def test_resume_rejects_changed_cli_hyperparameters(tmp_path, small_math, field, value):
    args = make_args(tmp_path, lora=True)
    trainer.train_lora_from_args(args)
    args.resume = tmp_path / "output/dlssnr/state-step000001"
    setattr(args, field, value)
    with pytest.raises(ValueError, match="identity"):
        trainer.train_lora_from_args(args)


@pytest.mark.parametrize("lora", [False, True])
def test_accumulation_matches_a_larger_batch(tmp_path, small_math, lora):
    accumulated = make_args(tmp_path / "accum", lora=lora, accum=2, batch=1)
    batched = make_args(tmp_path / "batch", lora=lora, accum=1, batch=2)
    if lora:
        accumulated.network_dropout = batched.network_dropout = 0.0
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(accumulated)
    train(batched)
    filename = "adapter.safetensors" if lora else "model.safetensors"
    left = load_file(accumulated.output_dir / "dlssnr" / "final" / filename)
    right = load_file(batched.output_dir / "dlssnr" / "final" / filename)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=1e-5, atol=1e-7)


def test_evaluation_runs_without_changing_training_rng_or_dropout(tmp_path, small_math):
    plain = make_args(tmp_path / "plain", lora=True)
    evaluated = make_args(tmp_path / "evaluated", lora=True, evaluate=True)
    trainer.train_lora_from_args(plain)
    trainer.train_lora_from_args(evaluated)
    left = load_file(plain.output_dir / "dlssnr/final/adapter.safetensors")
    right = load_file(evaluated.output_dir / "dlssnr/final/adapter.safetensors")
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    report = json.loads((evaluated.output_dir / "dlssnr/evaluation/step000002.json").read_text())
    assert report["baseline"] and report["candidate"]
    assert report["candidate"]["validation"][0]["rgb_mae"] >= 0


def test_train_validation_overlap_is_rejected(tmp_path, small_math):
    args = make_args(tmp_path, evaluate=True)
    manifest = tmp_path / "validation.jsonl"
    row = json.loads(manifest.read_text())
    row["sequence_id"] = "train"
    manifest.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="sequence"):
        trainer.train_from_args(args)


def test_max_grad_norm_is_applied_to_optimizer_updates(tmp_path, small_math, monkeypatch):
    models, originals = [], []

    def capture_model():
        model = SmallNR()
        models.append(model)
        originals.append({name: value.detach().clone() for name, value in model.named_parameters()})
        return model

    monkeypatch.setattr(trainer, "NRModel", capture_model)
    args = make_args(tmp_path, accum=1)
    args.optimizer_type = "SGD"
    args.max_train_steps = 1
    for name, maximum in (("unclipped", 0.0), ("clipped", 1e-8)):
        args.output_name, args.max_grad_norm = name, maximum
        trainer.train_from_args(args)
    changes = [
        sum((value.detach() - before[name]).square().sum() for name, value in model.named_parameters()).sqrt()
        for model, before in zip(models, originals)
    ]
    assert changes[0] > 0
    assert changes[1] < changes[0] * 0.01


def test_nonfinite_gradients_never_reach_optimizer_or_checkpoint(tmp_path, small_math, monkeypatch):
    def corrupt_model():
        model = SmallNR()
        model.blocks["70"].head.rgb.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
        return model

    monkeypatch.setattr(trainer, "NRModel", corrupt_model)
    with pytest.raises(RuntimeError, match="non-finite gradient"):
        trainer.train_from_args(make_args(tmp_path))
    assert not (tmp_path / "output/dlssnr/state-step000001").exists()


def test_save_state_is_written_at_the_final_update_without_a_periodic_save(tmp_path, small_math):
    args = make_args(tmp_path)
    args.save_every_n_steps = 0
    trainer.train_from_args(args)
    assert (tmp_path / "output/dlssnr/state-step000002/trainer_state.pt").is_file()


def test_disabling_state_still_saves_periodic_adapter_weights(tmp_path, small_math):
    args = make_args(tmp_path, lora=True)
    args.save_state = False
    trainer.train_lora_from_args(args)
    assert (tmp_path / "output/dlssnr/step000001/adapter.safetensors").is_file()
    assert not list((tmp_path / "output").rglob("trainer_state.pt"))


def test_checkpoint_checksum_rejects_modified_weights(tmp_path, small_math):
    args = make_args(tmp_path, lora=True)
    trainer.train_lora_from_args(args)
    folder = tmp_path / "output/dlssnr/state-step000001"
    from safetensors.torch import save_file

    weights = {key: tensor.clone() for key, tensor in load_file(folder / "adapter.safetensors").items()}
    weights[next(iter(weights))].add_(1)
    save_file(weights, str(folder / "adapter.safetensors"))
    args.resume = folder
    with pytest.raises(ValueError, match="modified"):
        trainer.train_lora_from_args(args)


def test_fp32_training_disables_tf32_without_leaking_global_settings(tmp_path, small_math, monkeypatch):
    class CheckPrecision(SmallNR):
        def forward(self, features, geometry):
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            return super().forward(features, geometry)

    monkeypatch.setattr(trainer, "NRModel", CheckPrecision)
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        trainer.train_from_args(make_args(tmp_path))
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
