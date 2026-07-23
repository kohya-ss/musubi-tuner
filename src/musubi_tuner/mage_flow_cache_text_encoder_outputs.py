import argparse
import logging

import numpy as np
import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MAGE_FLOW, ARCHITECTURE_MAGE_FLOW_EDIT
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_mage_flow
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.mage_flow.text_encoder import (
    QWEN3_VL_4B_INSTRUCT_REPO_ID,
    encode_conditioning,
    load_mage_flow_text_encoder,
)
from musubi_tuner.utils.model_utils import str_to_dtype


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _item_references(item: ItemInfo, is_edit: bool) -> list[np.ndarray]:
    if item.control_content is None:
        controls = []
    elif isinstance(item.control_content, np.ndarray) and item.control_content.ndim == 3:
        controls = [item.control_content]
    else:
        controls = list(item.control_content)
    if is_edit and not 1 <= len(controls) <= 3:
        raise ValueError(f"Mage-Flow-Edit item {item.item_key} requires between 1 and 3 ordered references")
    if not is_edit and controls:
        raise ValueError(f"Mage-Flow T2I item {item.item_key} contains reference images")
    return controls


def encode_and_save_batch(encoder, batch: list[ItemInfo], *, is_edit: bool) -> None:
    prompts = [item.caption for item in batch]
    references = [_item_references(item, is_edit) for item in batch] if is_edit else None
    embeddings = encode_conditioning(encoder, prompts, references=references, is_edit=is_edit)
    for item, embedding in zip(batch, embeddings):
        save_text_encoder_output_cache_mage_flow(item, embedding.to(torch.bfloat16), is_edit=is_edit)


def setup_parser() -> argparse.ArgumentParser:
    parser = cache_text.setup_parser_common()
    parser.add_argument("--text_encoder", type=str, required=True, help="single Qwen3-VL-4B safetensors component")
    parser.add_argument("--text_encoder_dtype", type=str, default="bfloat16")
    parser.add_argument("--processor", type=str, default=QWEN3_VL_4B_INSTRUCT_REPO_ID)
    parser.add_argument("--is_edit", action="store_true", help="condition on one to three ordered reference images")
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = str_to_dtype(args.text_encoder_dtype, torch.bfloat16)
    architecture = ARCHITECTURE_MAGE_FLOW_EDIT if args.is_edit else ARCHITECTURE_MAGE_FLOW
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=architecture)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    existing, expected = cache_text.prepare_cache_files_and_paths(dataset_group.datasets)
    encoder = load_mage_flow_text_encoder(
        args.text_encoder,
        device=device,
        dtype=dtype,
        processor_source=args.processor,
    )
    cache_text.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        dataset_group.datasets,
        existing,
        expected,
        lambda batch: encode_and_save_batch(encoder, batch, is_edit=args.is_edit),
        requires_content=args.is_edit,
    )
    cache_text.post_process_cache_files(dataset_group.datasets, existing, expected, args.keep_cache)


if __name__ == "__main__":
    main()
