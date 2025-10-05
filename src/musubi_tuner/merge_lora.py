import argparse
import re
import sys
import logging
from importlib.util import find_spec
from types import ModuleType
from typing import Callable, List, Optional

import torch
from safetensors.torch import load_file

from musubi_tuner.hunyuan_model.models import load_transformer
from musubi_tuner.hv_generate_video import synchronize_device
from musubi_tuner.networks import lora
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file

lycoris_available = find_spec("lycoris") is not None
if lycoris_available:
    from lycoris.kohya import create_network_from_weights
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def merge_lora_weights(
    lora_module: ModuleType,
    model: torch.nn.Module,
    lora_weights: List[str],
    lora_multipliers: List[float],
    include_patterns: Optional[List[str]],
    exclude_patterns: Optional[List[str]],
    device: torch.device,
    lycoris: bool = False,
    converter: Optional[Callable] = None,
    save_merged_model: Optional[str] = None,
) -> None:
    """merge LoRA weights to the model

    Args:
        lora_module: LoRA module, e.g. lora_wan
        model: DiT model
        lora_weights: paths to LoRA weights
        lora_multipliers: multipliers for LoRA weights
        include_patterns: regex patterns to include LoRA modules
        exclude_patterns: regex patterns to exclude LoRA modules
        device: torch.device
        lycoris: use LyCORIS
        converter: Optional[callable] = None
        save_merged_model: Optional[str] = None
    """
    if lora_weights is None or len(lora_weights) == 0:
        return

    for i, lora_weight in enumerate(lora_weights):
        if lora_multipliers is not None and len(lora_multipliers) > i:
            lora_multiplier = lora_multipliers[i]
        else:
            lora_multiplier = 1.0

        logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {lora_multiplier}")
        weights_sd = load_file(lora_weight)
        if converter is not None:
            weights_sd = converter(weights_sd)

        # apply include/exclude patterns
        original_key_count = len(weights_sd.keys())
        if include_patterns is not None and len(include_patterns) > i:
            include_pattern = include_patterns[i]
            regex_include = re.compile(include_pattern)
            weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
            logger.info(f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}")
        if exclude_patterns is not None and len(exclude_patterns) > i:
            original_key_count_ex = len(weights_sd.keys())
            exclude_pattern = exclude_patterns[i]
            regex_exclude = re.compile(exclude_pattern)
            weights_sd = {k: v for k, v in weights_sd.items() if not regex_exclude.search(k)}
            logger.info(
                f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}"
            )
        if len(weights_sd) != original_key_count:
            remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys()]))
            remaining_keys.sort()
            logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
            if len(weights_sd) == 0:
                logger.warning("No keys left after filtering.")

        if lycoris:
            lycoris_net, _ = create_network_from_weights(
                multiplier=lora_multiplier,
                file=None,
                weights_sd=weights_sd,
                unet=model,
                text_encoder=None,
                vae=None,
                for_inference=True,
            )
            lycoris_net.merge_to(None, model, weights_sd, dtype=None, device=device)
        else:
            network = lora_module.create_arch_network_from_weights(lora_multiplier, weights_sd, unet=model, for_inference=True)
            network.merge_to(None, model, weights_sd, device=device, non_blocking=True)

        synchronize_device(device)
        logger.info("LoRA weights loaded")

    # if we only want to save the model, we can skip the rest
    if save_merged_model:
        model.eval().requires_grad_(False)
        clean_memory_on_device(device)
        logger.info(f"Saving merged model to {save_merged_model}")
        mem_eff_save_file(model.state_dict(), save_merged_model)  # save_file needs a lot of memory
        logger.info("Model merged and saved. Exiting...")
        sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanVideo model merger script")

    parser.add_argument("--dit", type=str, required=True, help="DiT checkpoint path or directory")
    parser.add_argument("--dit_in_channels", type=int, default=16, help="input channels for DiT, default is 16, skyreels I2V is 32")
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument(
        "--lora_multiplier", type=float, nargs="*", default=[1.0], help="LoRA multiplier (can specify multiple values)"
    )
    parser.add_argument("--save_merged_model", type=str, required=True, help="Path to save the merged model")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for merging"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load DiT model
    logger.info(f"Loading DiT model from {args.dit}")
    transformer = load_transformer(args.dit, "torch", False, "cpu", torch.bfloat16, in_channels=args.dit_in_channels)
    transformer.eval()

    # Load LoRA weights and merge
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        for i, lora_weight in enumerate(args.lora_weight):
            # Use the corresponding lora_multiplier or default to 1.0
            if args.lora_multiplier is not None and len(args.lora_multiplier) > i:
                lora_multiplier = args.lora_multiplier[i]
            else:
                lora_multiplier = 1.0

            logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {lora_multiplier}")
            weights_sd = load_file(lora_weight)
            network = lora.create_arch_network_from_weights(lora_multiplier, weights_sd, unet=transformer, for_inference=True)
            logger.info("Merging LoRA weights to DiT model")
            network.merge_to(None, transformer, weights_sd, device=device, non_blocking=True)

            logger.info("LoRA weights loaded")

    # Save the merged model
    logger.info(f"Saving merged model to {args.save_merged_model}")
    mem_eff_save_file(transformer.state_dict(), args.save_merged_model)
    logger.info("Merged model saved")


if __name__ == "__main__":
    main()
