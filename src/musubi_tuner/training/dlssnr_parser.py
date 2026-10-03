"""Dataset-only TOML and explicit training arguments for the NR entry points."""

import argparse
from pathlib import Path

from musubi_tuner.dlssnr.profiles import PROFILE_ID
from musubi_tuner.training.parser_common import add_lr_scheduler_args, add_optimizer_args


def setup_parser(*, lora=False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a DLSS-NR LoRA." if lora else "Fine-tune DLSS-NR on paired frames or finite temporal clips.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--dataset_config",
        type=Path,
        required=True,
        help="Dataset-only TOML: [general] and [[datasets]] directory pairs or manifests.",
    )
    parser.add_argument(
        "--model_dir", type=Path, help="Pretrained canonical model directory; required outside development smoke runs."
    )
    parser.add_argument("--profile", default=PROFILE_ID, choices=[PROFILE_ID])
    parser.add_argument("--numerics_profile", default="train_surrogate", choices=["train_surrogate", "train_experimental"])
    parser.add_argument("--deployment_target", default="float_runtime", choices=["float_runtime", "native_roundtrip"])
    parser.add_argument(
        "--forward_validation_report",
        type=Path,
        help="Optional evidence bound to the source weights and implementation; checked when explicitly supplied.",
    )
    parser.add_argument(
        "--development_smoke",
        action="store_true",
        help="Developer-only: allow incomplete conversion provenance or random initialization without --model_dir.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--mixed_precision", default="no", choices=["no", "fp16", "bf16"], help="CUDA AMP requires train_experimental."
    )
    parser.add_argument("--max_overflow_retries", type=int, default=16, help="Retry the same FP16 batch after gradient overflow.")
    parser.add_argument("--fp8_base", action="store_true", help="Experimental FP8 frozen projection storage (LoRA only).")
    parser.add_argument("--fp8_scaled", action="store_true", help="Use block64/channel scaled storage; requires --fp8_base.")
    backends = parser.add_mutually_exclusive_group()
    backends.add_argument(
        "--attention_backend", choices=["native", "sdpa", "flash_attn", "xformers", "sage_attn"], default="native"
    )
    for backend in ("sdpa", "flash_attn", "xformers", "sage_attn"):
        backends.add_argument(f"--{backend}", dest="attention_backend", action="store_const", const=backend)
    parser.add_argument("--attention_scope", choices=["all", "global"], default="all")
    parser.add_argument("--training_mode", default="single_frame", choices=["single_frame", "temporal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Number of optimizer updates.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Recompute NR blocks during backward to save memory.")
    parser.add_argument("--sequence_length", type=int, default=None, help="Temporal mode requires all three clip-length arguments.")
    parser.add_argument("--burn_in", type=int, default=None)
    parser.add_argument("--tbptt_length", type=int, default=None)

    add_optimizer_args(parser)
    parser.set_defaults(optimizer_type="AdamW", learning_rate=1e-4 if lora else 1e-5, max_grad_norm=0.0)
    add_lr_scheduler_args(parser)
    parser.set_defaults(lr_scheduler_min_lr_ratio=0.1)
    for name, default in (("pre", 1.0), ("out", 1.0), ("edge", 0.05), ("temporal", 0.0)):
        parser.add_argument(f"--loss_{name}", type=float, default=default)
    parser.add_argument("--sample_every_n_steps", type=int, default=0, help="Evaluate the validation manifests every N updates.")
    parser.add_argument("--min_sequence_frames", type=int, default=64)
    parser.add_argument("--compare_baseline", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", required=True, help="Run directory name inside output_dir.")
    parser.add_argument("--save_every_n_steps", type=int, default=100)
    parser.add_argument("--save_state", action="store_true", help="Save optimizer/RNG state as well as weights.")
    parser.add_argument("--resume", type=Path, help="Exact state directory written at an optimizer-update boundary.")
    if lora:
        parser.add_argument("--network_dim", type=int, default=None, help="ViT LoRA rank (default: 16).")
        parser.add_argument("--network_alpha", type=float, default=None, help="LoRA scaling alpha (default: rank).")
        parser.add_argument("--network_dropout", type=float, default=0.0)
        parser.add_argument(
            "--network_args",
            nargs="*",
            default=None,
            help="NR adapter options: profile=vit_only|multiscale, qkv_mode, rank_by_width and alpha_by_width.",
        )
    else:
        for name in ("prior_lr_multiplier", "scale_lr_multiplier", "temporal_blend_lr_multiplier"):
            parser.add_argument(f"--{name}", type=float, default=0.1)
    return parser
