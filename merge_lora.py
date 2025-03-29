import argparse
import logging
import torch
from safetensors.torch import load_file
from networks import lora
from utils.safetensors_utils import mem_eff_save_file
from hunyuan_model.models import load_transformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanVideo model merger script")

    parser.add_argument("--dit", type=str, required=True, help="DiT checkpoint path or directory")
    parser.add_argument("--dit_in_channels", type=int, default=16, help="input channels for DiT, default is 16, skyreels I2V is 32")
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=[1.0], help="LoRA multiplier (can specify multiple values)")
    parser.add_argument("--save_merged_model", type=str, required=True, help="Path to save the merged model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for merging")

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
            network = lora.create_arch_network_from_weights(
                lora_multiplier, weights_sd, unet=transformer, for_inference=True
            )
            logger.info("Merging LoRA weights to DiT model")
            network.merge_to(None, transformer, weights_sd, device=device, non_blocking=True)

            logger.info("LoRA weights loaded")

    # Save the merged model
    logger.info(f"Saving merged model to {args.save_merged_model}")
    mem_eff_save_file(transformer.state_dict(), args.save_merged_model)
    logger.info("Merged model saved")


if __name__ == "__main__":
    main()