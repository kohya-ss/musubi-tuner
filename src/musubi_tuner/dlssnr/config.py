"""Resolve dataset-only TOML and CLI arguments into a checked run snapshot."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import toml

from musubi_tuner.dlssnr.controls import CONTROL_DEFAULTS, resolve_fixed_controls
from musubi_tuner.dlssnr.control_randomization import encode_control_point, validate_control_settings
from musubi_tuner.dlssnr.filenames import validate_filename
from musubi_tuner.dlssnr.geometry import resolve_geometry
from musubi_tuner.dlssnr.profiles import PROFILE_ID, PROFILES
from musubi_tuner.dlssnr.temporal import AUGMENTATION_SEED_POLICY
from musubi_tuner.dlssnr.synthetic_temporal import validate_max_shift


def _integer(value, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name: str, *, positive: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def _dino_settings(args):
    _number(args.dino_loss_weight, "--dino_loss_weight")
    defaults = {"model_type": "small", "layer": -4, "resize": 224, "use_gram": True, "use_norm": True}
    supplied = {name: getattr(args, f"dino_loss_{name}") for name in defaults}
    if not args.dino_loss_weight:
        if any(value is not None for value in supplied.values()):
            raise ValueError("DINO options require positive --dino_loss_weight")
        return None
    settings = {name: defaults[name] if value is None else value for name, value in supplied.items()}
    if settings["model_type"] not in ("small", "small_plus", "base", "large"):
        raise ValueError("unknown --dino_loss_model_type")
    if type(settings["layer"]) is not int:
        raise ValueError("--dino_loss_layer must be an integer")
    resize = settings["resize"]
    if type(resize) is not int or not 16 <= resize <= 1024 or resize % 16:
        raise ValueError("--dino_loss_resize must be a multiple of 16 between 16 and 1024")
    for name in ("use_gram", "use_norm"):
        if type(settings[name]) is not bool:
            raise ValueError(f"--dino_loss_{name} must be a boolean")
    return settings


def _control_settings(args, data):
    defaults = {"residual_sigma": 6.0, "anchor_probability": 0.25, "corner_probability": 0.25}
    supplied = {name: getattr(args, f"control_{name}", None) for name in defaults}
    enabled = getattr(args, "control_randomization", False)
    if type(enabled) is not bool:
        raise ValueError("control_randomization must be a boolean")
    if not enabled:
        if any(value is not None for value in supplied.values()):
            raise ValueError("control-specific options require --control_randomization")
        return None
    settings = {name: default if supplied[name] is None else supplied[name] for name, default in defaults.items()}
    validate_control_settings(settings)
    for entry in data.get("datasets", [data]):
        if "fixed_controls" not in entry:
            raise ValueError("control_randomization requires nr_controls_mode=fixed for every training dataset")
        encode_control_point(entry["fixed_controls"], (1, 1))
    return {"schema": "dlssnr_control_randomization_v1", "seed_policy": AUGMENTATION_SEED_POLICY, **settings}


def validate_lora(table: dict) -> None:
    if table.get("qkv_mode", "fused_head_major") != "fused_head_major":
        raise ValueError("only qkv_mode = fused_head_major is implemented")
    dropout = table.get("dropout", 0.0)
    _number(dropout, "lora.dropout")
    if dropout >= 1:
        raise ValueError("lora.dropout must be < 1")
    profile = table.get("profile")
    if profile == "vit_only":
        if "rank_by_width" in table or "alpha_by_width" in table:
            raise ValueError("vit_only cannot also set a width rank map")
        _integer(table.get("rank"), "lora.rank")
        if table["rank"] > 1024:
            raise ValueError("vit_only rank cannot exceed 1024")
        _number(table.get("alpha"), "lora.alpha", positive=True)
    elif profile == "multiscale":
        if "rank" in table or "alpha" in table:
            raise ValueError("multiscale uses the width table, not a single rank/alpha")
        defaults = {"32": 2, "64": 4, "128": 8, "256": 8, "512": 16, "1024": 16}
        table.setdefault("rank_by_width", defaults.copy())
        table.setdefault("alpha_by_width", table["rank_by_width"].copy())
        for field in ("rank_by_width", "alpha_by_width"):
            values = table[field]
            if not isinstance(values, dict) or set(values) != set(defaults):
                raise ValueError(f"lora.{field} requires exactly widths {list(defaults)}")
            for width, value in values.items():
                if field == "rank_by_width":
                    _integer(value, f"lora.{field}.{width}")
                    if value > {"512": 64, "1024": 1024}.get(width, 32):
                        raise ValueError(f"rank exceeds a target's input/output width at C={width}")
                else:
                    _number(value, f"lora.{field}.{width}", positive=True)
    else:
        raise ValueError(f"unknown LoRA profile {profile}")
    table.setdefault("dropout", 0.0)
    table.setdefault("qkv_mode", "fused_head_major")


def load_dataset_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = toml.load(path)
    if unknown := set(raw) - {"general", "datasets"}:
        raise ValueError(
            f"dataset TOML accepts only [general] and [[datasets]], not {sorted(unknown)}; "
            "pass model, optimizer and training settings as command-line arguments"
        )
    general = raw.get("general", {})
    datasets = raw.get("datasets")
    settings = {
        "resolution",
        "batch_size",
        "enable_bucket",
        "bucket_no_upscale",
        "num_repeats",
        "caption_extension",
        "synthetic_temporal",
        "synthetic_max_shift_px",
    }
    if not isinstance(general, dict) or (unknown := set(general) - settings):
        raise ValueError(f"dataset [general] supports only {sorted(settings)}")
    if not isinstance(datasets, list) or not datasets or any(not isinstance(item, dict) for item in datasets):
        raise ValueError("DLSS-NR requires at least one [[datasets]] entry")
    if "synthetic_temporal" in general and type(general["synthetic_temporal"]) is not bool:
        raise ValueError("general.synthetic_temporal must be a boolean")
    if "synthetic_max_shift_px" in general:
        validate_max_shift(general["synthetic_max_shift_px"])
    paths = {"train_manifest", "validation_manifest", "sequence_manifest", "image_directory", "control_directory"}
    entries = []
    for dataset in datasets:
        if unknown := set(dataset) - paths - settings - set(CONTROL_DEFAULTS) - {"nr_controls_mode", "cache_directory"}:
            raise ValueError(f"unsupported dataset fields: {sorted(unknown)}; training settings belong on the command line")
        effective = {**general, **dataset}
        if dataset.get("synthetic_temporal") is False and "synthetic_max_shift_px" not in dataset:
            effective.pop("synthetic_max_shift_px", None)
        entries.append(_resolve_dataset_entry(effective, path.parent, paths))
    if "synthetic_max_shift_px" in general and not any(entry.get("synthetic_temporal") for entry in entries):
        raise ValueError("general.synthetic_max_shift_px requires an enabled synthetic_temporal dataset")
    if len(entries) == 1:
        return entries[0]
    return {"datasets": entries, "batch_size": entries[0]["batch_size"]}


def _resolve_dataset_entry(effective, base_directory, paths):
    resolution = effective.get("resolution", [1024, 1024])
    if type(resolution) is int:
        resolution = [resolution, resolution]
    if not isinstance(resolution, list) or len(resolution) != 2 or any(type(value) is not int for value in resolution):
        raise ValueError("dataset resolution must be an integer or [width, height]")
    resolve_geometry(*resolution)
    batch_size = effective.get("batch_size", 1)
    _integer(batch_size, "dataset.batch_size")
    directory = bool(effective.get("image_directory"))
    if directory == bool(effective.get("train_manifest")):
        raise ValueError("dataset requires exactly one image_directory or train_manifest")
    if directory != bool(effective.get("control_directory")):
        raise ValueError("directory pairs require both image_directory and control_directory")
    data = {"bucket_size": resolution, "batch_size": batch_size}
    synthetic = effective.get("synthetic_temporal", False)
    if type(synthetic) is not bool:
        raise ValueError("dataset.synthetic_temporal must be a boolean")
    if synthetic:
        shift = effective.get("synthetic_max_shift_px", 0.5)
        validate_max_shift(shift)
        data["synthetic_temporal"] = {"schema": "dlssnr_synthetic_temporal_v1", "max_shift_px": float(shift)}
    elif "synthetic_max_shift_px" in effective:
        raise ValueError("synthetic_max_shift_px requires synthetic_temporal=true")
    for name in ("enable_bucket", "bucket_no_upscale"):
        value = effective.get(name, False)
        if type(value) is not bool:
            raise ValueError(f"dataset.{name} must be a boolean")
        data[name] = value
    if not data["enable_bucket"]:
        data["bucket_no_upscale"] = False
    for name in paths:
        if name in effective:
            value = effective[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"dataset.{name} must be a nonempty path")
            data[name] = str((base_directory / value).resolve())
    repeats = effective.get("num_repeats", 1)
    _integer(repeats, "dataset.num_repeats")
    if repeats != 1:
        data["num_repeats"] = repeats
    mode = effective.get("nr_controls_mode", "fixed" if directory else "files")
    if mode not in ("fixed", "files") or directory and mode != "fixed":
        raise ValueError("nr_controls_mode must be fixed for directories, or fixed/files for manifests")
    if mode == "fixed":
        data["fixed_controls"] = resolve_fixed_controls(effective)
    elif set(CONTROL_DEFAULTS) & effective.keys():
        raise ValueError("fixed condition parameters require nr_controls_mode=fixed")
    return data


def _key_value_args(values, name, *, allow_strings=False):
    result = {}
    for item in values or []:
        key, separator, text = item.partition("=")
        if not separator or not key.isidentifier() or key in result or not text:
            raise ValueError(f"{name} requires unique key=value arguments, got {item!r}")
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            if not allow_strings:
                raise ValueError(f"{name}: {key} must be a Python literal (for example betas=0.9,0.999)") from None
            value = text
        result[key] = value
    return result


def build_train_config(args, *, lora=False) -> dict:
    data = load_dataset_config(args.dataset_config)
    control_settings = _control_settings(args, data)
    mode = args.training_mode
    if mode != "temporal" and any(entry.get("synthetic_temporal") for entry in data.get("datasets", [data])):
        raise ValueError("synthetic_temporal requires temporal training mode")
    lengths = (args.sequence_length, args.burn_in, args.tbptt_length)
    if mode == "temporal" and any(value is None for value in lengths):
        raise ValueError("temporal mode requires --sequence_length, --burn_in and --tbptt_length")
    lengths = [value if value is not None else default for value, default in zip(lengths, (1, 0, 1))]
    training = {
        "mode": mode,
        "seed": args.seed,
        "shuffle_dataset": args.shuffle_dataset,
        "device": args.device,
        "development_smoke": args.development_smoke,
        "batch_size": data.pop("batch_size"),
        "sequence_length": lengths[0],
        "burn_in": lengths[1],
        "tbptt_length": lengths[2],
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_train_steps": args.max_train_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_min_pixels": args.gradient_checkpointing_min_pixels,
        "max_overflow_retries": args.max_overflow_retries,
    }
    _integer(args.gradient_checkpointing_min_pixels, "--gradient_checkpointing_min_pixels", 0)
    if args.gradient_checkpointing_min_pixels and not args.gradient_checkpointing:
        raise ValueError("--gradient_checkpointing_min_pixels requires --gradient_checkpointing")
    if args.ema_decay is not None:
        _number(args.ema_decay, "--ema_decay", positive=True)
        if args.ema_decay >= 1:
            raise ValueError("--ema_decay must be strictly between 0 and 1")
        training["ema_decay"] = args.ema_decay
    for field in ("batch_size", "sequence_length", "tbptt_length", "gradient_accumulation_steps", "max_train_steps"):
        _integer(training[field], f"--{field}")
    _integer(training["burn_in"], "--burn_in", 0)
    _integer(training["seed"], "--seed", 0)
    _integer(training["max_overflow_retries"], "--max_overflow_retries", 0)
    if mode == "single_frame":
        if lengths != [1, 0, 1]:
            raise ValueError("single-frame mode requires sequence_length=1, burn_in=0, tbptt_length=1")
        if args.loss_temporal != 0:
            raise ValueError("single-frame mode requires --loss_temporal=0")
    elif mode == "temporal":
        if any("image_directory" in entry and not entry.get("synthetic_temporal") for entry in data.get("datasets", [data])):
            raise ValueError("temporal training requires a manifest with explicit frames, motion and validity masks")
        if lengths[0] != lengths[1] + lengths[2] or lengths[1] < 1:
            raise ValueError("sequence_length must equal burn_in + tbptt_length, with both at least 1")
        if args.loss_temporal and lengths[2] < 2:
            raise ValueError("temporal loss requires at least two supervised frames (tbptt_length >= 2)")
    else:
        raise ValueError("--training_mode must be single_frame or temporal")
    if args.device not in ("auto", "cpu", "cuda") or args.mixed_precision not in ("no", "fp16", "bf16"):
        raise ValueError("NR requires device auto/cpu/cuda and precision no/fp16/bf16")
    if args.profile not in ("auto", *PROFILES) or args.numerics_profile not in ("train_surrogate", "train_experimental"):
        raise ValueError("unsupported DLSS-NR profile or numerics")
    if args.mixed_precision != "no":
        if args.numerics_profile != "train_experimental":
            raise ValueError("mixed precision requires --numerics_profile train_experimental")
        if args.device == "cpu":
            raise ValueError("NR mixed precision currently requires CUDA")
    if args.fp8_scaled and not args.fp8_base:
        raise ValueError("--fp8_scaled requires --fp8_base")
    if args.fp8_base:
        if not lora:
            raise ValueError("--fp8_base is only supported for a frozen LoRA base")
        if args.numerics_profile != "train_experimental":
            raise ValueError("FP8 storage requires --numerics_profile train_experimental")
    if args.deployment_target not in ("float_runtime", "native_roundtrip"):
        raise ValueError("unknown deployment_target")
    if args.numerics_profile == "train_experimental" and args.deployment_target != "float_runtime":
        raise ValueError("train_experimental only supports deployment_target float_runtime")
    model = {
        "profile": args.profile,
        "numerics_profile": args.numerics_profile,
        "deployment_target": args.deployment_target,
        "attention_backend": args.attention_backend,
        "attention_scope": args.attention_scope,
    }
    if args.native_weight_qat:
        model["native_weight_qat"] = True
    for name in ("model_dir", "forward_validation_report"):
        value = getattr(args, name)
        if value is not None:
            model[name] = str(Path(value).resolve())
    if not model.get("model_dir") and not args.development_smoke:
        raise ValueError("--model_dir is required; random initialization is only allowed in --development_smoke")
    if model.get("forward_validation_report") and not model.get("model_dir"):
        raise ValueError("--forward_validation_report requires --model_dir to identify the source weights")
    from musubi_tuner.dlssnr.artifacts import read_canonical_profile

    if model.get("model_dir"):
        profile = read_canonical_profile(model["model_dir"])
        if args.profile != "auto" and args.profile != profile:
            raise ValueError(f"requested profile {args.profile} differs from checkpoint profile {profile}")
    else:
        profile = PROFILE_ID if args.profile == "auto" else args.profile
    model["profile"] = profile

    optimizer_type = args.optimizer_type or "AdamW"
    optimizer_args = _key_value_args(args.optimizer_args, "--optimizer_args")
    if {"lr", "params"} & optimizer_args.keys():
        raise ValueError("use --learning_rate for lr; optimizer params are managed by the trainer")
    if optimizer_type.lower() in ("adamw", "torch.optim.adamw", "adamw8bit", "bitsandbytes.optim.adamw8bit"):
        optimizer_args.setdefault("weight_decay", 0.0)
    if "weight_decay" in optimizer_args:
        _number(optimizer_args["weight_decay"], "optimizer weight_decay")
    if optimizer_type.lower() in ("adafactor", "transformers.optimization.adafactor"):
        if optimizer_args.get("relative_step", True) or optimizer_args.get("warmup_init", False):
            raise ValueError("NR Adafactor requires --optimizer_args relative_step=False warmup_init=False")
    _number(args.learning_rate, "--learning_rate", positive=True)
    _number(args.max_grad_norm, "--max_grad_norm")
    optimizer = {
        "type": optimizer_type,
        "args": [f"{key}={value!r}" for key, value in sorted(optimizer_args.items())],
        "learning_rate": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "max_grad_norm": args.max_grad_norm,
        "scheduler": validate_scheduler_config(args),
    }
    loss = {name: getattr(args, f"loss_{name}") for name in ("pre", "out", "edge", "temporal")}
    for name, value in loss.items():
        _number(value, f"--loss_{name}")
    if not any(loss.values()):
        raise ValueError("at least one loss weight must be positive")
    _number(args.base_anchor_weight, "--base_anchor_weight")
    if args.base_anchor_weight:
        loss["base_anchor"] = args.base_anchor_weight
    dino_settings = _dino_settings(args)
    if dino_settings is not None:
        loss["dino"] = args.dino_loss_weight
    loss_profile = None
    if args.loss_profile == "frequency_split":
        sigma = 6.0 if args.loss_lowpass_sigma is None else args.loss_lowpass_sigma
        _number(sigma, "--loss_lowpass_sigma", positive=True)
        if sigma > 32:
            raise ValueError("--loss_lowpass_sigma must be <= 32")
        loss_profile = {"name": "frequency_split", "lowpass_sigma": sigma}
    elif args.loss_profile != "pixel":
        raise ValueError("unknown --loss_profile")
    elif args.loss_lowpass_sigma is not None:
        raise ValueError("--loss_lowpass_sigma requires --loss_profile frequency_split")
    evaluation = {
        "sample_every_n_steps": args.sample_every_n_steps,
        "min_sequence_frames": args.min_sequence_frames,
        "compare_baseline": args.compare_baseline,
    }
    if args.eval_native:
        evaluation["native"] = True
    if args.eval_detail_diagnostics:
        evaluation["detail_diagnostics"] = True
    if args.eval_content_preservation:
        evaluation["content_preservation"] = True
    if "sequence_manifest" in data:
        evaluation["sequence_manifest"] = data.pop("sequence_manifest")
    _integer(args.sample_every_n_steps, "--sample_every_n_steps", 0)
    _integer(args.min_sequence_frames, "--min_sequence_frames")
    if (args.sample_every_n_steps or args.eval_native or args.eval_detail_diagnostics or args.eval_content_preservation) and not (
        any(entry.get("validation_manifest") or entry.get("sequence_manifest") for entry in data.get("datasets", [data]))
        or evaluation.get("sequence_manifest")
    ):
        raise ValueError("evaluation requires a validation_manifest or sequence_manifest in the dataset TOML")
    validate_filename(args.output_name, "--output_name")
    _integer(args.save_every_n_steps, "--save_every_n_steps", 0)
    output = {
        "output_dir": str(Path(args.output_dir).resolve()),
        "output_name": args.output_name,
        "save_every_n_steps": args.save_every_n_steps,
        "save_state": args.save_state,
        "save_precision": args.save_precision,
    }
    config = {
        "schema_version": 2,
        "model": model,
        "data": data,
        "training": training,
        "optimizer": optimizer,
        "precision": {
            "mixed_precision": args.mixed_precision,
            "master_dtype": "float32",
            "fp8_base": args.fp8_base,
            "fp8_scaled": args.fp8_scaled,
        },
        "loss": loss,
        "evaluation": evaluation,
        "output": output,
    }
    if loss_profile is not None:
        config["loss_profile"] = loss_profile
    if dino_settings is not None:
        config["dino_loss"] = dino_settings
    if control_settings is not None:
        config["control_randomization"] = control_settings
    if lora:
        network = _key_value_args(args.network_args, "--network_args", allow_strings=True)
        if unknown := set(network) - {"profile", "qkv_mode", "rank_by_width", "alpha_by_width"}:
            raise ValueError(f"unsupported --network_args: {sorted(unknown)}")
        network.setdefault("profile", "vit_only")
        if network["profile"] == "vit_only":
            network["rank"] = args.network_dim if args.network_dim is not None else 16
            network["alpha"] = args.network_alpha if args.network_alpha is not None else network["rank"]
        elif args.network_dim is not None or args.network_alpha is not None:
            raise ValueError("multiscale uses rank_by_width/alpha_by_width, not --network_dim/--network_alpha")
        network["dropout"] = args.network_dropout
        validate_lora(network)
        if args.native_weight_qat and network["dropout"]:
            raise ValueError("native weight QAT requires --network_dropout 0; split dropout is not a fused export weight")
        config["lora"] = network
    else:
        groups = {
            name: getattr(args, name) for name in ("prior_lr_multiplier", "scale_lr_multiplier", "temporal_blend_lr_multiplier")
        }
        for name, value in groups.items():
            _number(value, f"--{name}")
        config["parameter_groups"] = groups
    from musubi_tuner.dlssnr.runtime import runtime_policy, validate_runtime_policy

    validate_runtime_policy(runtime_policy(config), training=True)
    return config


def config_sha256(config: dict) -> str:
    """Hash effective values, not CLI argument order or dataset TOML comments."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_scheduler_config(args) -> dict:
    names = (
        "lr_scheduler",
        "lr_warmup_steps",
        "lr_decay_steps",
        "lr_scheduler_num_cycles",
        "lr_scheduler_power",
        "lr_scheduler_timescale",
        "lr_scheduler_min_lr_ratio",
        "lr_scheduler_type",
        "lr_scheduler_args",
    )
    values = {name: getattr(args, name) for name in names}
    supported = {
        "constant",
        "constant_with_warmup",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "cosine_with_min_lr",
        "polynomial",
        "inverse_sqrt",
        "warmup_stable_decay",
        "rex",
        "piecewise_constant",
    }
    if not args.lr_scheduler_type and args.lr_scheduler not in supported:
        raise ValueError(f"unsupported NR lr_scheduler {args.lr_scheduler!r}")
    counts = {}
    for name in ("lr_warmup_steps", "lr_decay_steps"):
        value = values[name]
        _number(value, name)
        if value == 0:
            values[name] = 0
        if isinstance(value, float) and value >= 1:
            raise ValueError(f"{name} must be integer steps or a ratio below 1")
        counts[name] = int(value * args.max_train_steps) if isinstance(value, float) else value
        if counts[name] > args.max_train_steps:
            raise ValueError(f"{name} exceeds max_train_steps")
    if args.lr_scheduler == "constant" and values["lr_warmup_steps"]:
        raise ValueError("constant scheduler requires lr_warmup_steps=0")
    if args.lr_scheduler == "warmup_stable_decay" and sum(counts.values()) > args.max_train_steps:
        raise ValueError("scheduler warmup and decay exceed max_train_steps")
    _integer(args.lr_scheduler_num_cycles, "lr_scheduler_num_cycles")
    _number(args.lr_scheduler_power, "lr_scheduler_power", positive=True)
    if args.lr_scheduler_timescale is not None:
        _integer(args.lr_scheduler_timescale, "lr_scheduler_timescale")
    if args.lr_scheduler == "inverse_sqrt" and not args.lr_scheduler_timescale and not counts["lr_warmup_steps"]:
        raise ValueError("inverse_sqrt requires positive warmup or lr_scheduler_timescale")
    if args.lr_scheduler_min_lr_ratio is not None:
        _number(args.lr_scheduler_min_lr_ratio, "lr_scheduler_min_lr_ratio")
        if args.lr_scheduler_min_lr_ratio > 1:
            raise ValueError("lr_scheduler_min_lr_ratio must be <= 1")
    values["lr_scheduler_power"] = float(args.lr_scheduler_power)
    extras = _key_value_args(args.lr_scheduler_args, "lr_scheduler_args")
    values["lr_scheduler_args"] = [f"{key}={value!r}" for key, value in sorted(extras.items())] if extras else None
    return values
