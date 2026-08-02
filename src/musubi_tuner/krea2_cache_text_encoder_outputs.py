"""Cache text encoder (Qwen3-VL-4B) outputs for Krea 2 (K2) training.

K2's text encoder returns a stack of *selected* Qwen3-VL hidden-state layers
(shape (B, seq, 12, 2560)) plus an attention mask. The layerwise fusion
(TextFusionTransformer) is trainable and lives inside the DiT, so we cache the raw
selected-layer stack. Padding tokens are dropped per item (varlen) — K2 gives text
tokens zero RoPE position and masks padding in attention, so this is lossless for
the image outputs.
"""

import argparse
import logging

import numpy as np
import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache_krea2
from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2, ARCHITECTURE_KREA2_EDIT
from musubi_tuner.krea2 import krea2_utils
from musubi_tuner.krea2 import krea2_sampling

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(encoder, batch: list[ItemInfo], *, edit: bool = False):
    prompts = [item.caption for item in batch]
    images = None
    if edit:
        images = []
        for item in batch:
            if item.control_content is None or len(item.control_content) == 0:
                raise ValueError(f"Krea 2 Edit item {item.item_key} has no reference/control image")
            refs = [
                torch.from_numpy(np.array(control[..., :3], copy=True)).permute(2, 0, 1).float().div_(255.0)
                for control in item.control_content
            ]
            images.append(krea2_sampling.prepare_vlm_reference_images(refs))

    for i, item in enumerate(batch):
        ref_shapes = None if images is None else [tuple(image.shape) for image in images[i]]
        print(f"Item {i}: {item.item_key}, prompt: {item.caption}, references: {ref_shapes}")

    hiddens, mask = krea2_utils.get_krea2_prompt_embeds(
        encoder,
        prompts,
        images=images,
    )  # (B, seq, L, D), (B, seq)

    # Save per item, dropping padding tokens (varlen).
    for item, hidden_i, mask_i in zip(batch, hiddens, mask):
        valid = mask_i.bool()
        embed_i = hidden_i[valid]  # (valid_len, L, D)
        save_text_encoder_output_cache_krea2(item, embed_i, edit=edit)


def krea2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="Qwen3-VL-4B text encoder safetensors path (official or ComfyUI key layout)",
    )
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for the text encoder, default is bfloat16")
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = krea2_setup_parser(parser)
    parser.add_argument("--edit", action="store_true", help="cache image-conditioned Qwen3-VL outputs for Krea 2 Edit")

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    te_dtype = torch.bfloat16
    if args.text_encoder_dtype is not None:
        from musubi_tuner.utils.model_utils import str_to_dtype

        te_dtype = str_to_dtype(args.text_encoder_dtype)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    architecture = ARCHITECTURE_KREA2_EDIT if args.edit else ARCHITECTURE_KREA2
    blueprint = blueprint_generator.generate(user_config, args, architecture=architecture)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets
    if args.edit:
        missing_control = [index for index, dataset in enumerate(datasets) if not dataset.has_control]
        if missing_control:
            raise ValueError(f"Krea 2 --edit requires control images in every dataset; missing in datasets {missing_control}")

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    encoder = krea2_utils.load_krea2_text_encoder(args.text_encoder, dtype=te_dtype, device=device)

    logger.info("Encoding with Qwen3-VL")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal encoder
        encode_and_save_batch(encoder, batch, edit=args.edit)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
        requires_content=args.edit,
    )
    del encoder

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


if __name__ == "__main__":
    main()
