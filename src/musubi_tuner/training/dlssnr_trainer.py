"""Shared optimizer-boundary runner for NR full and LoRA supervised training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from accelerate.utils import set_seed

from musubi_tuner.dlssnr.artifacts import inspect_canonical, save_canonical, write_json
from musubi_tuner.dlssnr.config import build_train_config, config_sha256
from musubi_tuner.dlssnr.dataset import (
    NRBatchPlan,
    NRDatasetCollection,
    collate_clips,
    collate_single_frames,
    load_single_frame_manifest,
    load_temporal_manifest,
    load_directory_pairs,
)
from musubi_tuner.dlssnr.evaluation import evaluate
from musubi_tuner.dlssnr.identity import file_sha256, implementation_identity
from musubi_tuner.dlssnr.model import NRModel
from musubi_tuner.dlssnr.numerics import fp32_execution
from musubi_tuner.dlssnr.runtime import configure_model_runtime, numerics_metadata, runtime_policy, validate_runtime_device
from musubi_tuner.dlssnr.profiles import PROFILE_ID
from musubi_tuner.dlssnr.temporal import SEED_POLICY, stable_frame_seed
from musubi_tuner.dlssnr.training_step import loss_denominators, training_loss
from musubi_tuner.training.dlssnr_services import (
    assert_finite_gradients,
    assert_finite_parameters,
    capture_rng,
    coordinated_call,
    create_accelerator,
    create_nr_optimizer,
    create_nr_lr_scheduler,
    evaluation_mode,
    gather_rank_values,
    move_batch,
    optimizer_update,
    reject_unsupported_runtime,
    reduce_values,
    restore_rng,
)
from musubi_tuner.training.dlssnr_state import read_state, restore_state, save_state

logger = logging.getLogger(__name__)


def parameter_group(name: str) -> str:
    if "head.logit" in name:
        return "temporal_head"
    if name.endswith("blend_scale"):
        return "temporal_blend"
    if name.endswith("prior"):
        return "priors"
    if name.endswith(("skip_scale", "temperature", "up_scale", "adapter_scale")):
        return "scales"
    if "input_adapter" in name or "head.rgb" in name:
        return "input_rgb_head"
    return "matrices"


def build_optimizer(model, learning_rate, multipliers, weight_decay, *, include_temporal=False, optimizer_config=None):
    if include_temporal:
        model.blocks["70"].head.logit.weight.requires_grad_(True)
        model.blocks["70"].blend_scale.requires_grad_(True)
    else:
        model.freeze_single_frame()
    grouped = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            grouped.setdefault(parameter_group(name), []).append(parameter)
    groups = [
        {"params": parameters, "lr": learning_rate * multipliers.get(name, 1.0), "name": name}
        for name, parameters in grouped.items()
    ]
    if optimizer_config is None:
        optimizer_config = {
            "type": "AdamW",
            "args": [f"weight_decay={weight_decay!r}"],
            "learning_rate": learning_rate,
            "lr_scheduler": "constant",
            "max_grad_norm": 0.0,
        }
    optimizer = create_nr_optimizer(groups, optimizer_config)
    _check_optimizer(model, optimizer)
    return optimizer


def _check_optimizer(module, optimizer):
    opted = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    trainable = {id(parameter) for parameter in module.parameters() if parameter.requires_grad}
    if len(opted) != len(set(opted)) or set(opted) != trainable:
        raise RuntimeError("optimizer parameters must exactly match the unique trainable parameters")


def clear_lane15_state(model, optimizer):
    model.enforce_lane15()
    weight = model.blocks["0"].input_adapter.weight
    state = optimizer.state.get(weight, {})
    for value in state.values():
        # Integer-coded quantized states cannot be cleared as floating-point moments.
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.shape == weight.shape:
            value[:, 15] = 0


class NRTrainModule(torch.nn.Module):
    """The only prepared model; adapters have exactly one registered owner."""

    def __init__(self, model, loss_weights, burn_in=0, network=None):
        super().__init__()
        self.model = model
        self.network = network
        self.loss_weights = loss_weights
        self.burn_in = burn_in

    def forward(self, batch, seeds, normalizers=None):
        # Accelerator owns scaling; NR owns exactly which products use autocast.
        with torch.autocast(batch["source"].device.type, enabled=False):
            return training_loss(self.model, batch, seeds, self.loss_weights, self.burn_in, normalizers)


def _one_update(model, optimizer, batch, seeds, weights, burn_in):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = training_loss(model, batch, seeds, {"pre": 1.0, "out": 1.0, "edge": 0.05, **weights}, burn_in)
    loss.backward()
    model.enforce_lane15()
    assert_finite_gradients([parameter for group in optimizer.param_groups for parameter in group["params"]])
    optimizer.step()
    clear_lane15_state(model, optimizer)
    assert_finite_parameters(model)
    return metrics


def single_frame_update(model, optimizer, source, target, controls, frame_seed, loss_weights):
    return _one_update(model, optimizer, {"source": source, "target": target, "controls": controls}, frame_seed, loss_weights, 0)


def temporal_clip_update(model, optimizer, batch, seeds, loss_weights, burn_in):
    return _one_update(model, optimizer, batch, seeds, loss_weights, burn_in)


def _datasets(config):
    data, training = config["data"], config["training"]
    entries = data.get(
        "datasets",
        [
            {
                **data,
                "batch_size": training["batch_size"],
                **{key: config["evaluation"][key] for key in ("sequence_manifest",) if key in config["evaluation"]},
            }
        ],
    )
    sources, validation = [], {}
    for index, entry in enumerate(entries):
        train, evaluations = _dataset_entry(entry, training, config["loss"], config["evaluation"])
        sources.append(train)
        validation.update({(f"dataset{index}_{name}" if len(entries) > 1 else name): value for name, value in evaluations.items()})
    train = NRDatasetCollection(sources, entries) if len(entries) > 1 or entries[0].get("num_repeats", 1) != 1 else sources[0]
    for name, dataset in validation.items():
        if overlap := train.sequence_ids & dataset.sequence_ids:
            raise ValueError(f"train/{name} sequence overlap: {sorted(overlap)}")
    for dataset in (train, *validation.values()):
        dataset.validate()
    return train, validation


def _dataset_entry(data, training, loss, evaluation):
    width, height = data["bucket_size"]
    buckets = {name: data[name] for name in ("enable_bucket", "bucket_no_upscale")}
    buckets["fixed_controls"] = data.get("fixed_controls")
    if "image_directory" in data:
        if training["mode"] != "single_frame":
            raise ValueError("temporal training requires a manifest with explicit frames and motion")
        train = load_directory_pairs(data)
    elif training["mode"] == "single_frame":
        train = load_single_frame_manifest(data["train_manifest"], width, height, **buckets)
    else:
        train = load_temporal_manifest(
            data["train_manifest"],
            width,
            height,
            training["sequence_length"],
            require_temporal_mask=bool(loss["temporal"]),
            **buckets,
        )
    validation = {}
    if data.get("validation_manifest"):
        validation["validation"] = load_temporal_manifest(
            data["validation_manifest"], width, height, None, require_temporal_mask=False, **buckets
        )
    if data.get("sequence_manifest"):
        sequences = load_temporal_manifest(data["sequence_manifest"], width, height, None, require_temporal_mask=False, **buckets)
        if any(len(row["frames"]) < evaluation["min_sequence_frames"] for row in sequences.rows):
            raise ValueError("evaluation sequence is shorter than min_sequence_frames")
        validation["sequences"] = sequences
    return train, validation


def _microbatch(dataset, microbatch_index, config, plan):
    training = config["training"]
    samples = [dataset[index] for index in plan.indices(microbatch_index)]
    epoch = microbatch_index // len(plan)
    seeds = []
    for sample in samples:
        frames = [sample] if training["mode"] == "single_frame" else sample["frames"]
        seeds.append(
            [
                stable_frame_seed(training["seed"], epoch, sample["sample_id"], frame["frame_index"], sample["crop_id"])
                for frame in frames
            ]
        )
    if training["mode"] == "single_frame":
        return collate_single_frames(samples), [item[0] for item in seeds]
    return collate_clips(samples), seeds


class NRSupervisedTrainer:
    def __init__(self, config, *, lora=False):
        self.config = config
        self.lora = lora

    @fp32_execution()
    def train(self, resume=None):
        accelerator = create_accelerator(self.config["training"], self.config["precision"])
        try:
            self._train(accelerator, resume)
        finally:
            accelerator.end_training()
            accelerator.free_memory()

    def _train(self, accelerator, resume):
        from musubi_tuner.networks.lora_dlssnr import load_adapter
        from musubi_tuner.dlssnr.convert import _git_commit

        config = self.config
        policy = runtime_policy(config)
        training, output = config["training"], config["output"]
        coordinated_call(accelerator, lambda: validate_runtime_device(policy, accelerator.device, training=True))
        train_data, validation = coordinated_call(accelerator, lambda: _datasets(config))
        batch_plan = NRBatchPlan(train_data, training["batch_size"])
        if accelerator.is_main_process:
            batch_plan.show_bucket_info()
        source_dir = config["model"].get("model_dir")
        source_identity = coordinated_call(
            accelerator,
            lambda: (
                inspect_canonical(
                    source_dir,
                    development_smoke=training["development_smoke"],
                    validation_report=config["model"].get("forward_validation_report"),
                )
                if source_dir
                else {}
            ),
        )
        source_forward_validated = "forward_validation_report" in source_identity
        model, network, optimizer, base_identity = coordinated_call(accelerator, lambda: self._initialize_model(policy))
        optimizer_cfg = config["optimizer"]
        scheduler = coordinated_call(
            accelerator, lambda: create_nr_lr_scheduler(optimizer, optimizer_cfg, training["max_train_steps"])
        )
        train_module = NRTrainModule(model, config["loss"], training["burn_in"], network)
        _check_optimizer(train_module, optimizer)
        parameter_map = {
            name: {"shape": list(parameter.shape), "elements": parameter.numel()}
            for name, parameter in train_module.named_parameters()
            if parameter.requires_grad
        }
        optimizer_groups = [
            {
                "name": group.get("name", "lora"),
                "lr": group["lr"],
                "weight_decay": group.get("weight_decay", 0.0),
                "elements": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in optimizer.param_groups
        ]
        data_identity = coordinated_call(
            accelerator, lambda: {name: dataset.fingerprint() for name, dataset in {"train": train_data, **validation}.items()}
        )
        devices = gather_rank_values(
            accelerator,
            {
                "rank": accelerator.process_index,
                "type": accelerator.device.type,
                "cuda_device": torch.cuda.get_device_name(accelerator.device) if accelerator.device.type == "cuda" else None,
            },
        )
        from musubi_tuner.dlssnr.attention import backend_identity

        identity = {
            "config_sha256": config_sha256(config),
            "source": source_identity,
            "base_parameters_sha256": base_identity,
            "base_quantization": getattr(model, "base_quantization", None),
            "data": data_identity,
            "implementation": implementation_identity(),
            "lora": self.lora,
            "runtime_policy": policy,
            "attention_implementation": backend_identity(policy["attention_backend"]),
            "world_size": accelerator.num_processes,
            "device_type": accelerator.device.type,
            "torch_version": str(torch.__version__),
            "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "seed_policy": SEED_POLICY,
            "bucket_plan": batch_plan.report(),
            "cuda_version": torch.version.cuda if accelerator.device.type == "cuda" else None,
            "devices": devices,
        }
        for name in (
            "config.py",
            "controls.py",
            "dataset.py",
            "filenames.py",
            "training_step.py",
            "losses.py",
            "runtime.py",
            "fp8.py",
            "attention.py",
        ):
            identity["implementation"][name] = file_sha256(Path(__file__).parents[1] / "dlssnr" / name)
        for name in ("dlssnr_trainer.py", "dlssnr_services.py", "dlssnr_state.py", "optimizer_setup.py", "lr_scheduler.py"):
            identity["implementation"][name] = file_sha256(Path(__file__).parent / name)
        for name in ("bucket.py", "architectures.py"):
            identity["implementation"][f"dataset/{name}"] = file_sha256(Path(__file__).parents[1] / "dataset" / name)
        identity["implementation"]["lora_dlssnr.py"] = file_sha256(Path(__file__).parents[1] / "networks/lora_dlssnr.py")
        if policy["fp8_base"]:
            identity["implementation"]["fp8_optimization_utils.py"] = file_sha256(
                Path(__file__).parents[1] / "modules/fp8_optimization_utils.py"
            )
        if any(value != identity for value in gather_rank_values(accelerator, identity)):
            raise ValueError("DDP ranks disagree on config, data, base or runtime identity")
        restored = coordinated_call(accelerator, lambda: read_state(resume, identity)) if resume else None
        completed = restored["global_update"] if restored else 0
        cursor = restored["consumed_samples"] if restored else 0
        steps, accum = training["max_train_steps"], training["gradient_accumulation_steps"]
        world, rank = accelerator.num_processes, accelerator.process_index
        if completed > steps or cursor != batch_plan.sample_count(completed * accum * world):
            raise ValueError("resume counters are inconsistent with max_train_steps or the consumed batch plan")
        output_dir = Path(output["output_dir"]) / output["output_name"]

        def create_run_dir():
            try:
                output_dir.mkdir(parents=True, exist_ok=restored is not None)
            except FileExistsError:
                raise FileExistsError(
                    f"run directory already exists: {output_dir}; choose a different output_name or use --resume"
                ) from None

        coordinated_call(accelerator, create_run_dir, main_only=True)
        wrapped, optimizer = accelerator.prepare(train_module, optimizer)
        if world > 1:
            set_seed(training["seed"] + rank)
        logger.info(
            "NR device=%s trainable=%d mode=%s lora=%s",
            accelerator.device,
            sum(item["elements"] for item in parameter_map.values()),
            training["mode"],
            self.lora,
        )
        metadata = {
            "config": config,
            "identity": identity,
            "profile": PROFILE_ID,
            "numerics": numerics_metadata(policy),
            "runtime_policy": policy,
            "base_quantization": getattr(model, "base_quantization", None),
            "trainer_commit": _git_commit(),
            "trainable_parameters": parameter_map,
            "optimizer_groups": optimizer_groups,
            "bucket_plan": batch_plan.report(),
            "device": str(accelerator.device),
            "source_forward_validated": source_forward_validated,
            "experimental_surrogate": policy["numerics_profile"] == "train_experimental"
            or training["development_smoke"]
            or not source_forward_validated,
            "temporal_trained": training["mode"] == "temporal",
        }
        baseline = None

        def run_evaluation():
            with evaluation_mode(train_module):
                return evaluate(model, validation, training["seed"], accelerator.device)

        if validation and config["evaluation"]["compare_baseline"]:
            baseline = coordinated_call(accelerator, run_evaluation, main_only=True)
        if restored:

            def restore():
                if network is not None:
                    load_adapter(network, Path(resume) / "adapter.safetensors", base_identity)
                else:
                    model.load_canonical(str(Path(resume) / "model.safetensors"))
                restore_state(restored, optimizer, accelerator=accelerator, scheduler=scheduler)

            coordinated_call(accelerator, restore)

        def write_initial_metadata():
            write_json(output_dir / "run_config.json", metadata)
            if network is not None:
                write_json(output_dir / "lora_report.json", network.report)

        coordinated_call(accelerator, write_initial_metadata, main_only=True)
        if training["development_smoke"]:
            logger.warning("EXPERIMENTAL NR smoke run: native/float compatibility has not been validated")
        elif not source_forward_validated:
            logger.warning("NR source forward compatibility is unvalidated; training does not certify native/DLL compatibility")

        def save_product(folder, update, *, with_state=False):
            coordinated_call(
                accelerator,
                lambda: self._save_product(folder, model, network, source_dir, metadata, base_identity, update),
                main_only=True,
            )
            if with_state:
                save_state(folder, optimizer, identity, update, cursor, accelerator=accelerator, scheduler=scheduler)

        for step in range(completed, steps):
            microbatches = coordinated_call(
                accelerator,
                lambda: [
                    _microbatch(train_data, (step * accum + micro) * world + rank, config, batch_plan) for micro in range(accum)
                ],
            )
            denominators = {name: 0.0 for name in config["loss"]}
            for batch, _ in microbatches:
                for name, count in loss_denominators(batch, training["burn_in"]).items():
                    denominators[name] += count
            denominators = reduce_values(accelerator, denominators)
            attempt_rng = capture_rng(accelerator.device)
            for attempt in range(training["max_overflow_retries"] + 1):
                train_module.train()
                optimizer.zero_grad(set_to_none=True)
                if attempt:
                    restore_rng(attempt_rng)
                metrics = {}
                for batch, seeds in microbatches:
                    with accelerator.accumulate(wrapped):
                        loss, measured = coordinated_call(
                            accelerator, lambda: wrapped(move_batch(batch, accelerator.device), seeds, denominators)
                        )
                        # Undo Accelerate's accumulation division and DDP's gradient averaging.
                        accelerator.backward(loss * accum * world)
                    del loss
                    for name, value in measured.items():
                        metrics[name] = max(metrics.get(name, 0), value) if name == "blend_max" else metrics.get(name, 0) + value
                if not accelerator.sync_gradients:
                    raise RuntimeError("optimizer update is not at an accumulation boundary")
                model.enforce_lane15()
                if optimizer_update(accelerator, train_module, optimizer, optimizer_cfg["max_grad_norm"]):
                    break
                logger.warning("NR FP16 overflow at update %d, retry %d", step + 1, attempt + 1)
            else:
                raise RuntimeError("FP16 gradient overflow exceeded --max_overflow_retries; no successful update saved")
            clear_lane15_state(model, optimizer)
            coordinated_call(accelerator, lambda: assert_finite_parameters(train_module))
            scheduler.step()
            count = reduce_values(accelerator, {"samples": sum(len(seeds) for _, seeds in microbatches)})
            cursor += int(count["samples"])
            update = step + 1
            metrics = reduce_values(accelerator, metrics, max_keys=("blend_max",))
            metrics["overflow_retries"] = attempt
            metrics["learning_rate"] = scheduler.get_last_lr()[0]

            def write_metrics():
                logger.info("NR update %d/%d loss=%.6f", update, steps, metrics["loss"])
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"update": update, "consumed_samples": cursor, **metrics}, allow_nan=False) + "\n")

            coordinated_call(accelerator, write_metrics, main_only=True)
            interval = config["evaluation"]["sample_every_n_steps"]
            if validation and ((interval and update % interval == 0) or update == steps):
                candidate = coordinated_call(accelerator, run_evaluation, main_only=True)
                coordinated_call(
                    accelerator,
                    lambda: write_json(
                        output_dir / "evaluation" / f"step{update:06d}.json",
                        {
                            "update": update,
                            "numerics_profile": policy["numerics_profile"],
                            "runtime_policy": policy,
                            "baseline": baseline,
                            "candidate": candidate,
                        },
                    ),
                    main_only=True,
                )
            save_every = output["save_every_n_steps"]
            if save_every and update % save_every == 0:
                prefix = "state-step" if output["save_state"] else "step"
                save_product(output_dir / f"{prefix}{update:06d}", update, with_state=output["save_state"])
        if output["save_state"] and (not output["save_every_n_steps"] or steps % output["save_every_n_steps"]):
            save_product(output_dir / f"state-step{steps:06d}", steps, with_state=True)
        save_product(output_dir / "final", steps)

    def _initialize_model(self, policy):
        from musubi_tuner.networks.lora_dlssnr import base_target_sha256, inject

        config = self.config
        training, optimizer_cfg = config["training"], config["optimizer"]
        set_seed(training["seed"])
        model = NRModel().to(dtype=torch.float32)
        source_dir = config["model"].get("model_dir")
        if source_dir:
            model.load_canonical(str(Path(source_dir) / "model.safetensors"))
        configure_model_runtime(model, policy, training=True)
        base_identity = base_target_sha256(model, [])
        network = inject(model, config["lora"]) if self.lora else None
        if network is not None:
            network.runtime_policy = dict(policy)
            if policy["fp8_base"]:
                from musubi_tuner.dlssnr.fp8 import quantize_frozen_base

                network.base_quantization = quantize_frozen_base(model, scaled=policy["fp8_scaled"])
        if self.lora:
            optimizer = create_nr_optimizer(network.parameters(), optimizer_cfg)
        else:
            groups = config["parameter_groups"]
            optimizer = build_optimizer(
                model,
                optimizer_cfg["learning_rate"],
                {
                    "priors": groups["prior_lr_multiplier"],
                    "scales": groups["scale_lr_multiplier"],
                    "temporal_blend": groups["temporal_blend_lr_multiplier"],
                },
                0.0,
                include_temporal=training["mode"] == "temporal",
                optimizer_config=optimizer_cfg,
            )
        return model, network, optimizer, base_identity

    @staticmethod
    def _save_product(folder, model, network, source_dir, metadata, base_identity, update):
        from musubi_tuner.networks.lora_dlssnr import save_adapter

        metadata = {
            **metadata,
            "global_update": update,
            "float_validated": False,
            "temporal_validated": False,
            "native_export_validated": False,
        }
        if network is None:
            save_canonical(model, folder, source_dir=source_dir, metadata=metadata)
        else:
            save_adapter(network, folder / "adapter.safetensors", base_identity)
            write_json(folder / "training_metadata.json", metadata)
            write_json(
                folder / "base_identity.json",
                {
                    "base_profile": PROFILE_ID,
                    "base_weight_sha256": base_identity,
                    "identity_scope": "all_canonical_parameters",
                    "targets": network.target_names,
                },
            )
        write_json(folder / "run_config.json", metadata["config"])


def _train_from_args(args, *, lora):
    reject_unsupported_runtime(args.mixed_precision)
    config = build_train_config(args, lora=lora)
    NRSupervisedTrainer(config, lora=lora).train(args.resume)


def train_from_args(args):
    _train_from_args(args, lora=False)


def train_lora_from_args(args):
    _train_from_args(args, lora=True)
