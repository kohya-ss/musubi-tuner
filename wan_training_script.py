#!/usr/bin/env python3
"""
WAN 2.2 T2V Training Script for Character LoRA
This script trains both high noise and low noise transformers with character tokens.
"""

import os
import argparse
from pathlib import Path

def create_training_command(
    # Dataset parameters
    train_data_dir: str,
    caption_extension: str = ".txt",

    # Model paths
    pretrained_model_name_or_path: str,
    vae_path: str = None,
    text_encoder_path: str = None,

    # Training parameters
    output_dir: str = "./lora_output",
    output_name: str = "character_lora",
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    lr_scheduler: str = "cosine_with_restarts",
    max_train_steps: int = 1000,
    save_every_n_steps: int = 500,

    # LoRA parameters
    network_module: str = "networks.lora",
    network_dim: int = 64,
    network_alpha: int = 32,
    network_dropout: float = 0.0,

    # Video parameters
    width: int = 512,
    height: int = 512,
    frame_count: int = 81,
    fps: int = 16,

    # Noise scheduling
    noise_scheduler: str = "flow_match",
    flow_shift: float = 3.0,

    # Mixed precision and optimization
    mixed_precision: str = "bf16",
    gradient_checkpointing: bool = True,
    xformers: bool = True,

    # Character token settings
    token_string: str = "CHARACTER_VANRAJ_V1",

    # Advanced options
    cache_latents: bool = True,
    cache_text_encoder_outputs: bool = True,
    caption_dropout_rate: float = 0.05,
    shuffle_caption: bool = True,
    keep_tokens: int = 1,

    # Loss weighting
    loss_type: str = "l2",
    huber_c: float = 0.1,

    # Logging
    logging_dir: str = "./logs",
    log_with: str = "tensorboard",

    # Memory optimization
    lowram: bool = False,
    mem_eff_attn: bool = True,
):
    """
    Create the training command for WAN 2.2 character LoRA.

    Args:
        train_data_dir: Directory containing training videos and captions
        caption_extension: Extension for caption files
        pretrained_model_name_or_path: Path to WAN 2.2 model
        vae_path: Path to VAE model
        text_encoder_path: Path to text encoder
        output_dir: Output directory for LoRA weights
        output_name: Name for output LoRA files
        train_batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        lr_scheduler: Learning rate scheduler
        max_train_steps: Maximum training steps
        save_every_n_steps: Save checkpoint every N steps
        network_module: LoRA network module
        network_dim: LoRA dimension
        network_alpha: LoRA alpha value
        network_dropout: LoRA dropout rate
        width: Video width
        height: Video height
        frame_count: Number of frames per video
        fps: Frames per second
        noise_scheduler: Noise scheduler type
        flow_shift: Flow matching shift parameter
        mixed_precision: Mixed precision training
        gradient_checkpointing: Enable gradient checkpointing
        xformers: Use xformers for attention
        token_string: Character token string
        cache_latents: Cache latents to disk
        cache_text_encoder_outputs: Cache text encoder outputs
        caption_dropout_rate: Caption dropout rate
        shuffle_caption: Shuffle caption tokens
        keep_tokens: Number of tokens to keep at beginning
        loss_type: Loss function type
        huber_c: Huber loss parameter
        logging_dir: Directory for logs
        log_with: Logging backend
        lowram: Enable low RAM mode
        mem_eff_attn: Use memory efficient attention

    Returns:
        str: Complete command line for training
    """

    cmd = ["python", "wan_train_network.py"]

    # Required parameters
    cmd.extend(["--pretrained_model_name_or_path", pretrained_model_name_or_path])
    cmd.extend(["--train_data_dir", train_data_dir])
    cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--output_name", output_name])

    # Dataset parameters
    cmd.extend(["--caption_extension", caption_extension])

    # Model paths
    if vae_path:
        cmd.extend(["--vae", vae_path])
    if text_encoder_path:
        cmd.extend(["--text_encoder", text_encoder_path])

    # Training parameters
    cmd.extend(["--train_batch_size", str(train_batch_size)])
    cmd.extend(["--gradient_accumulation_steps", str(gradient_accumulation_steps)])
    cmd.extend(["--learning_rate", str(learning_rate)])
    cmd.extend(["--lr_scheduler", lr_scheduler])
    cmd.extend(["--max_train_steps", str(max_train_steps)])
    cmd.extend(["--save_every_n_steps", str(save_every_n_steps)])

    # LoRA parameters
    cmd.extend(["--network_module", network_module])
    cmd.extend(["--network_dim", str(network_dim)])
    cmd.extend(["--network_alpha", str(network_alpha)])
    if network_dropout > 0:
        cmd.extend(["--network_dropout", str(network_dropout)])

    # Video parameters
    cmd.extend(["--width", str(width)])
    cmd.extend(["--height", str(height)])
    cmd.extend(["--frame_count", str(frame_count)])
    cmd.extend(["--fps", str(fps)])

    # Noise scheduling
    cmd.extend(["--noise_scheduler", noise_scheduler])
    cmd.extend(["--flow_shift", str(flow_shift)])

    # Mixed precision and optimization
    cmd.extend(["--mixed_precision", mixed_precision])
    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if xformers:
        cmd.append("--xformers")

    # Character token
    cmd.extend(["--token_string", token_string])

    # Advanced options
    if cache_latents:
        cmd.append("--cache_latents")
    if cache_text_encoder_outputs:
        cmd.append("--cache_text_encoder_outputs")

    cmd.extend(["--caption_dropout_rate", str(caption_dropout_rate)])
    if shuffle_caption:
        cmd.append("--shuffle_caption")
    cmd.extend(["--keep_tokens", str(keep_tokens)])

    # Loss parameters
    cmd.extend(["--loss_type", loss_type])
    if loss_type == "huber":
        cmd.extend(["--huber_c", str(huber_c)])

    # Logging
    cmd.extend(["--logging_dir", logging_dir])
    cmd.extend(["--log_with", log_with])

    # Memory optimization
    if lowram:
        cmd.append("--lowram")
    if mem_eff_attn:
        cmd.append("--mem_eff_attn")

    return " ".join(cmd)


def main():
    parser = argparse.ArgumentParser(description="WAN 2.2 T2V Character LoRA Training Script")

    # Required parameters
    parser.add_argument("--pretrained_model", type=str, required=True,
                       help="Path to WAN 2.2 pretrained model")
    parser.add_argument("--train_data_dir", type=str, required=True,
                       help="Directory containing training videos and captions")
    parser.add_argument("--output_dir", type=str, default="./character_lora_output",
                       help="Output directory for trained LoRA")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--max_train_steps", type=int, default=1000,
                       help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")

    # LoRA parameters
    parser.add_argument("--lora_dim", type=int, default=64,
                       help="LoRA dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")

    # Video parameters
    parser.add_argument("--resolution", type=int, nargs=2, default=[512, 512],
                       help="Video resolution (height width)")
    parser.add_argument("--frame_count", type=int, default=81,
                       help="Number of frames")

    # Character token
    parser.add_argument("--character_token", type=str, default="CHARACTER_VANRAJ_V1",
                       help="Character token string")

    # Options
    parser.add_argument("--dry_run", action="store_true",
                       help="Print command without executing")
    parser.add_argument("--execute", action="store_true",
                       help="Execute the command directly")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate training command
    cmd = create_training_command(
        train_data_dir=args.train_data_dir,
        pretrained_model_name_or_path=args.pretrained_model,
        output_dir=args.output_dir,
        output_name="character_vanraj_lora",
        learning_rate=args.learning_rate,
        max_train_steps=args.max_train_steps,
        train_batch_size=args.batch_size,
        network_dim=args.lora_dim,
        network_alpha=args.lora_alpha,
        width=args.resolution[1],
        height=args.resolution[0],
        frame_count=args.frame_count,
        token_string=args.character_token,
    )

    print("Generated training command:")
    print(cmd)
    print()

    if args.dry_run:
        print("Dry run mode - command not executed")
        return

    if args.execute:
        print("Executing training command...")
        os.system(cmd)
    else:
        print("To execute this command, run with --execute flag or copy the command above")


if __name__ == "__main__":
    main()
