"""HunyuanVideo training entry point.

NetworkTrainer base class and shared helpers have been moved to the
musubi_tuner.training package for clearer file naming. This module:

- re-exports those symbols so existing imports from architecture-specific
  training scripts (wan_train_network, fpack_train_network, ...) keep working,
- defines HunyuanVideo-specific CLI argument additions (hv_setup_parser),
- provides the main() entry point.
"""

import argparse
import logging

from musubi_tuner.training.trainer_base import (
    NetworkTrainer,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
    SS_METADATA_MINIMUM_KEYS,
)
from musubi_tuner.training.accelerator_setup import (
    clean_memory_on_device,
    collator_class,
    prepare_accelerator,
)
from musubi_tuner.training.sampling_prompts import (
    line_to_prompt_dict,
    load_prompts,
    should_sample_images,
)
from musubi_tuner.training.timesteps import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    get_sigmas,
)
from musubi_tuner.training.parser_common import (
    setup_parser_common,
    read_config_from_file,
)

# accelerate.set_seed is re-exported because qwen_image_train.py and
# zimage_train.py import it from this module.
from accelerate.utils import set_seed

__all__ = [
    # trainer_base
    "NetworkTrainer",
    "SS_METADATA_KEY_BASE_MODEL_VERSION",
    "SS_METADATA_KEY_NETWORK_MODULE",
    "SS_METADATA_KEY_NETWORK_DIM",
    "SS_METADATA_KEY_NETWORK_ALPHA",
    "SS_METADATA_KEY_NETWORK_ARGS",
    "SS_METADATA_MINIMUM_KEYS",
    # accelerator_setup
    "clean_memory_on_device",
    "collator_class",
    "prepare_accelerator",
    # sampling_prompts
    "line_to_prompt_dict",
    "load_prompts",
    "should_sample_images",
    # timesteps
    "compute_density_for_timestep_sampling",
    "compute_loss_weighting_for_sd3",
    "get_sigmas",
    # parser_common
    "setup_parser_common",
    "read_config_from_file",
    # accelerate
    "set_seed",
]

logger = logging.getLogger(__name__)


def hv_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """HunyuanVideo specific parser setup"""
    # model settings
    parser.add_argument("--dit_dtype", type=str, default=None, help="data type for DiT, default is bfloat16")
    parser.add_argument("--dit_in_channels", type=int, default=16, help="input channels for DiT, default is 16, skyreels I2V is 32")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for LLM / LLMにfp8を使う")
    parser.add_argument("--text_encoder1", type=str, help="Text Encoder 1 directory / テキストエンコーダ1のディレクトリ")
    parser.add_argument("--text_encoder2", type=str, help="Text Encoder 2 directory / テキストエンコーダ2のディレクトリ")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for Text Encoder, default is float16")
    parser.add_argument(
        "--vae_tiling",
        action="store_true",
        help="enable spatial tiling for VAE, default is False. If vae_spatial_tile_sample_min_size is set, this is automatically enabled."
        " / VAEの空間タイリングを有効にする、デフォルトはFalse。vae_spatial_tile_sample_min_sizeが設定されている場合、自動的に有効になります。",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = hv_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.fp8_scaled = False  # HunyuanVideo does not support this yet

    trainer = NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
