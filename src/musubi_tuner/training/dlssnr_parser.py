"""Dataset-only TOML and explicit training arguments for the NR entry points."""

import argparse
from pathlib import Path

from musubi_tuner.dlssnr.profiles import PROFILES
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
    parser.add_argument(
        "--profile",
        default="auto",
        choices=["auto", *PROFILES],
        help="Detect HT/CC from --model_dir by default; explicit profiles must match the checkpoint.",
    )
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
    parser.add_argument(
        "--native_weight_qat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt into native-export weight fake quantization; keep FP32 masters (disabled by default).",
    )
    backends = parser.add_mutually_exclusive_group()
    backends.add_argument(
        "--attention_backend", choices=["native", "sdpa", "flash_attn", "xformers", "sage_attn"], default="native"
    )
    for backend in ("sdpa", "flash_attn", "xformers", "sage_attn"):
        backends.add_argument(f"--{backend}", dest="attention_backend", action="store_const", const=backend)
    parser.add_argument("--attention_scope", choices=["all", "global"], default="all")
    parser.add_argument("--training_mode", default="single_frame", choices=["single_frame", "temporal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministically shuffle samples within buckets and batch order each epoch, using --seed (enabled by default).",
    )
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Number of optimizer updates.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ema = parser.add_mutually_exclusive_group()
    ema.add_argument(
        "--ema_decay",
        type=float,
        default=None,
        help="Opt into FP32 trainable-parameter EMA with decay strictly between 0 and 1; saved and evaluated separately.",
    )
    ema.add_argument("--no-ema", dest="ema_decay", action="store_const", const=None, help="Disable parameter EMA.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Recompute NR blocks during backward to save memory.")
    parser.add_argument(
        "--gradient_checkpointing_min_pixels",
        type=int,
        default=0,
        help="Checkpoint only blocks with feature-map H*W above this threshold (0: all); requires --gradient_checkpointing.",
    )
    parser.add_argument("--sequence_length", type=int, default=None, help="Temporal mode requires all three clip-length arguments.")
    parser.add_argument("--burn_in", type=int, default=None)
    parser.add_argument("--tbptt_length", type=int, default=None)

    add_optimizer_args(parser)
    parser.set_defaults(optimizer_type="AdamW", learning_rate=1e-4 if lora else 1e-5, max_grad_norm=0.0)
    add_lr_scheduler_args(parser)
    parser.set_defaults(lr_scheduler_min_lr_ratio=0.1)
    parser.add_argument(
        "--loss_profile",
        choices=["pixel", "frequency_split"],
        default="pixel",
        help="pixel keeps the original objective; frequency_split matches low-frequency targets and anchors edges to inputs.",
    )
    parser.add_argument(
        "--loss_lowpass_sigma",
        type=float,
        default=None,
        help="Gaussian sigma in transformed image pixels for frequency_split (default: 6; 0 < sigma <= 32).",
    )
    for name, default in (("pre", 1.0), ("out", 1.0), ("edge", 0.05), ("temporal", 0.0)):
        parser.add_argument(f"--loss_{name}", type=float, default=default)
    parser.add_argument(
        "--base_anchor_weight",
        type=float,
        default=0.0,
        help="Optional rendered-output loss against the initial base with identical inputs, controls and seeds; zero disables it.",
    )
    parser.add_argument(
        "--control_randomization",
        action="store_true",
        help="Sample fixed tone/structure controls with frozen-base-relative targets.",
    )
    parser.add_argument(
        "--control_residual_sigma",
        type=float,
        default=None,
        help="Gaussian sigma for enhancement target bands, independent of loss filtering (default: 6; 0 < sigma <= 32).",
    )
    parser.add_argument(
        "--control_anchor_probability", type=float, default=None, help="Reference-point sampling probability (default: 0.25)."
    )
    parser.add_argument(
        "--control_corner_probability", type=float, default=None, help="Four-corner sampling probability (default: 0.25)."
    )
    parser.add_argument(
        "--dino_loss_weight", type=float, default=0.0, help="Optional frozen DINOv3 patch loss; zero disables loading and compute."
    )
    parser.add_argument("--dino_loss_model_type", choices=["small", "small_plus", "base", "large"], default=None)
    parser.add_argument(
        "--dino_loss_layer", type=int, default=None, help="ViT layer, with Python-style negative indexing (default: -4)."
    )
    parser.add_argument(
        "--dino_loss_resize",
        type=int,
        default=None,
        help="Longest-side limit, without upscaling; multiple of 16, 16-1024 (default: 224).",
    )
    parser.add_argument(
        "--dino_loss_use_gram",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compare patch feature Gram statistics (default: enabled); disable only for aligned feature MSE.",
    )
    parser.add_argument(
        "--dino_loss_use_norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="L2-normalize patch features (default: enabled).",
    )
    parser.add_argument("--sample_every_n_steps", type=int, default=0, help="Evaluate the validation manifests every N updates.")
    parser.add_argument("--min_sequence_frames", type=int, default=64)
    parser.add_argument("--compare_baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--eval_native", action="store_true", help="Also evaluate native-quantized weights with FP32 native attention."
    )
    parser.add_argument(
        "--eval_detail_diagnostics",
        action="store_true",
        help="Opt into masked high-frequency energy and paired-seed noise sensitivity; adds one forward per frame/runtime.",
    )
    parser.add_argument(
        "--eval_content_preservation",
        action="store_true",
        help="Opt into frozen DINOv3 spatial patch MSE against inputs during evaluation only; does not add a training loss.",
    )

    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", required=True, help="Run directory name inside output_dir.")
    parser.add_argument("--save_every_n_steps", type=int, default=100)
    parser.add_argument(
        "--save_precision",
        choices=("native", "fp32"),
        default="native",
        help="Full/EMA model storage (default: native FP8/FP16/FP32); LoRA and exact resume snapshots remain FP32.",
    )
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
            help="NR adapter options: profile=vit_only|multiscale, qkv_mode, rank_by_width and alpha_by_width. "
            "Oversized multiscale ranks are capped per width group; alpha scales proportionally.",
        )
    else:
        for name, default in (("prior_lr_multiplier", 1.0), ("scale_lr_multiplier", 1.0), ("temporal_blend_lr_multiplier", 0.1)):
            parser.add_argument(f"--{name}", type=float, default=default)
    return parser
