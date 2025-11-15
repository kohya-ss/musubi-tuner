import argparse
import os
from typing import List, Optional, Sequence, Tuple

import torch
from contextlib import nullcontext

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_LONGCAT,
    ItemInfo,
    save_text_encoder_output_cache_longcat,
)
from musubi_tuner.wan.modules.t5 import T5EncoderModel
from musubi_tuner.wan.configs import wan_t2v_14B
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    text_encoder: "LongcatTextEncoder",
    batch: list[ItemInfo],
    device: torch.device,
    autocast_dtype: Optional[torch.dtype] = None,
) -> None:
    prompts = [item.caption for item in batch]
    outputs = text_encoder.encode(prompts, device, autocast_dtype)

    for item, (embed, mask) in zip(batch, outputs):
        save_text_encoder_output_cache_longcat(item, embed, mask)


class LongcatTextEncoder:
    def __init__(self, t5_model: T5EncoderModel) -> None:
        self.t5 = t5_model

    def to(self, device: torch.device) -> "LongcatTextEncoder":
        self.device = device
        return self

    def eval(self) -> "LongcatTextEncoder":
        # WAN T5 encoder is already inference-only
        return self

    def encode(
        self,
        prompts: List[str],
        device: torch.device,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
        if autocast_dtype is not None and device.type == "cuda":
            autocast_context = torch.cuda.amp.autocast(dtype=autocast_dtype)
        else:
            autocast_context = nullcontext()
        with torch.no_grad(), autocast_context:
            contexts = self.t5(prompts, device=device)

        results: list[Tuple[torch.Tensor, torch.Tensor]] = []
        for context in contexts:
            context_cpu = context.to("cpu")
            mask = torch.ones(context_cpu.shape[0], dtype=torch.bool)
            results.append((context_cpu, mask))
        return results


def main() -> None:
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = longcat_setup_parser(parser)
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_LONGCAT)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    if not args.text_encoder.endswith((".pth", ".safetensors")) and not os.path.isfile(args.text_encoder):
        raise ValueError(
            "LongCat text encoder caching expects WAN T5 weights (.pth/.safetensors). "
            "Provide WAN T5 encoder weights via --text_encoder."
        )

    config = wan_t2v_14B.t2v_14B
    logger.info(f"Loading WAN T5 weights: {args.text_encoder}")
    t5_model = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=device,
        weight_path=args.text_encoder,
        fp8=args.fp8_t5,
    )

    encoder = LongcatTextEncoder(t5_model).eval().to(device)

    autocast_dtype = torch.bfloat16 if args.fp8_t5 and torch.cuda.is_available() else None

    def encode_fn(batch: list[ItemInfo]) -> None:
        encode_and_save_batch(encoder, batch, device, autocast_dtype)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_fn,
    )

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


def longcat_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="WAN T5 encoder weights (.pth/.safetensors)")
    parser.add_argument("--fp8_t5", action="store_true", help="Enable bf16 autocast for T5 forward pass")
    return parser


if __name__ == "__main__":
    main()
