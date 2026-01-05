import logging
import os
from types import SimpleNamespace

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_KANDINSKY5,
    ARCHITECTURE_KANDINSKY5_FULL,
    ItemInfo,
    save_text_encoder_output_cache_kandinsky5,
)
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.kandinsky5.models.text_embedders import get_text_embedder
from musubi_tuner.utils import safetensors_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _ensure_cache_architecture(item: ItemInfo):
    path = item.text_encoder_output_cache_path
    if not path or not os.path.exists(path):
        return
    try:
        with safetensors_utils.MemoryEfficientSafeOpen(path) as f:
            meta = f.metadata()
        if meta.get("architecture") != ARCHITECTURE_KANDINSKY5_FULL:
            logger.warning(
                f"Removing text-encoder cache with mismatched architecture: {path} "
                f"(found {meta.get('architecture')}, expected {ARCHITECTURE_KANDINSKY5_FULL})"
            )
            os.remove(path)
    except Exception as e:
        logger.warning(f"Failed to read existing cache {path} ({e}); removing to regenerate.")
        os.remove(path)


def encode_and_save_batch(text_embedder, batch: list[ItemInfo], device: torch.device):
    prompts = [item.caption for item in batch]
    # Keep the cache encoder aligned with training/inference: use video template when the batch contains videos.
    is_video_batch = any((item.frame_count or 1) > 1 for item in batch)
    content_type = "video" if is_video_batch else "image"
    embeds, cu_seqlens, attention_mask = text_embedder.encode(prompts, type_of_content=content_type)

    text_embeds = embeds["text_embeds"].to("cpu")
    pooled_embed = embeds["pooled_embed"].to("cpu")
    attention_mask = attention_mask.to("cpu")

    if text_embeds.dim() == 2 and attention_mask.dim() == 2 and cu_seqlens is not None and cu_seqlens.numel() == len(batch) + 1:
        # Variable-length packed embeds: slice by cu_seqlens per item.
        for idx, item in enumerate(batch):
            start = int(cu_seqlens[idx].item())
            end = int(cu_seqlens[idx + 1].item())
            te = text_embeds[start:end]
            pe = pooled_embed[idx]
            am = attention_mask[idx].bool().flatten()
            if am.numel() != te.shape[0]:
                if am.sum().item() == te.shape[0]:
                    am = am[am]
                else:
                    am = torch.ones((te.shape[0],), dtype=torch.bool)
            _ensure_cache_architecture(item)
            save_text_encoder_output_cache_kandinsky5(item, te, pe, am)
    else:
        # Fallback: per-item tensors already aligned on batch dim.
        for item, te, pe, am in zip(batch, text_embeds, pooled_embed, attention_mask):
            _ensure_cache_architecture(item)
            save_text_encoder_output_cache_kandinsky5(item, te, pe, am)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser.add_argument("--text_encoder_qwen", type=str, required=True, help="Qwen2.5-VL checkpoint path")
    parser.add_argument("--text_encoder_clip", type=str, required=True, help="CLIP text encoder checkpoint path")
    parser.add_argument("--qwen_max_length", type=int, default=512, help="Max length for Qwen tokenizer")
    parser.add_argument("--clip_max_length", type=int, default=77, help="Max length for CLIP tokenizer")
    parser.add_argument("--quantized_qwen", action="store_true", help="Load Qwen text encoder in 4bit mode")

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_KANDINSKY5)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    text_embedder_conf = SimpleNamespace(
        qwen=SimpleNamespace(checkpoint_path=args.text_encoder_qwen, max_length=args.qwen_max_length),
        clip=SimpleNamespace(checkpoint_path=args.text_encoder_clip, max_length=args.clip_max_length),
    )
    text_embedder = get_text_embedder(
        text_embedder_conf,
        device=device,
        quantized_qwen=args.quantized_qwen,
    )

    def encode_for_text_encoder(batch: list[ItemInfo]):
        encode_and_save_batch(text_embedder, batch, device)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )

    # remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


if __name__ == "__main__":
    main()
