import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_FLUX_2_DEV,
    ARCHITECTURE_FLUX_2_DEV_FULL,
    ARCHITECTURE_FLUX_2_KLEIN_4B,
    ARCHITECTURE_FLUX_2_KLEIN_4B_FULL,
    ARCHITECTURE_FLUX_2_KLEIN_9B,
    ARCHITECTURE_FLUX_2_KLEIN_9B_FULL,
    ARCHITECTURE_FLUX_KONTEXT,
    ARCHITECTURE_FLUX_KONTEXT_FULL,
    ARCHITECTURE_FRAMEPACK,
    ARCHITECTURE_FRAMEPACK_FULL,
    ARCHITECTURE_HIDREAM_O1,
    ARCHITECTURE_HIDREAM_O1_FULL,
    ARCHITECTURE_HUNYUAN_VIDEO,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL,
    ARCHITECTURE_HUNYUAN_VIDEO_FULL,
    ARCHITECTURE_IDEOGRAM4,
    ARCHITECTURE_IDEOGRAM4_FULL,
    ARCHITECTURE_KANDINSKY5,
    ARCHITECTURE_KANDINSKY5_FULL,
    ARCHITECTURE_KREA2,
    ARCHITECTURE_KREA2_FULL,
    ARCHITECTURE_QWEN_IMAGE,
    ARCHITECTURE_QWEN_IMAGE_EDIT,
    ARCHITECTURE_QWEN_IMAGE_EDIT_FULL,
    ARCHITECTURE_QWEN_IMAGE_FULL,
    ARCHITECTURE_QWEN_IMAGE_LAYERED,
    ARCHITECTURE_QWEN_IMAGE_LAYERED_FULL,
    ARCHITECTURE_WAN,
    ARCHITECTURE_Z_IMAGE,
    ARCHITECTURE_Z_IMAGE_FULL,
)
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# short name -> full/short aliases accepted on the CLI
ARCHITECTURE_CHOICES = {
    ARCHITECTURE_HUNYUAN_VIDEO: [ARCHITECTURE_HUNYUAN_VIDEO, ARCHITECTURE_HUNYUAN_VIDEO_FULL],
    ARCHITECTURE_WAN: [ARCHITECTURE_WAN, "wan2.1", "wan2.2"],
    ARCHITECTURE_FRAMEPACK: [ARCHITECTURE_FRAMEPACK, ARCHITECTURE_FRAMEPACK_FULL],
    ARCHITECTURE_FLUX_KONTEXT: [ARCHITECTURE_FLUX_KONTEXT, ARCHITECTURE_FLUX_KONTEXT_FULL],
    ARCHITECTURE_FLUX_2_DEV: [ARCHITECTURE_FLUX_2_DEV, ARCHITECTURE_FLUX_2_DEV_FULL],
    ARCHITECTURE_FLUX_2_KLEIN_4B: [ARCHITECTURE_FLUX_2_KLEIN_4B, ARCHITECTURE_FLUX_2_KLEIN_4B_FULL],
    ARCHITECTURE_FLUX_2_KLEIN_9B: [ARCHITECTURE_FLUX_2_KLEIN_9B, ARCHITECTURE_FLUX_2_KLEIN_9B_FULL],
    ARCHITECTURE_QWEN_IMAGE: [ARCHITECTURE_QWEN_IMAGE, ARCHITECTURE_QWEN_IMAGE_FULL],
    ARCHITECTURE_QWEN_IMAGE_EDIT: [ARCHITECTURE_QWEN_IMAGE_EDIT, ARCHITECTURE_QWEN_IMAGE_EDIT_FULL],
    ARCHITECTURE_QWEN_IMAGE_LAYERED: [ARCHITECTURE_QWEN_IMAGE_LAYERED, ARCHITECTURE_QWEN_IMAGE_LAYERED_FULL],
    ARCHITECTURE_KANDINSKY5: [ARCHITECTURE_KANDINSKY5, ARCHITECTURE_KANDINSKY5_FULL],
    ARCHITECTURE_HUNYUAN_VIDEO_1_5: [ARCHITECTURE_HUNYUAN_VIDEO_1_5, ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL],
    ARCHITECTURE_Z_IMAGE: [ARCHITECTURE_Z_IMAGE, ARCHITECTURE_Z_IMAGE_FULL],
    ARCHITECTURE_HIDREAM_O1: [ARCHITECTURE_HIDREAM_O1, ARCHITECTURE_HIDREAM_O1_FULL],
    ARCHITECTURE_IDEOGRAM4: [ARCHITECTURE_IDEOGRAM4, ARCHITECTURE_IDEOGRAM4_FULL],
    ARCHITECTURE_KREA2: [ARCHITECTURE_KREA2, ARCHITECTURE_KREA2_FULL],
}


def resolve_architecture(name: str) -> str:
    for short_name, aliases in ARCHITECTURE_CHOICES.items():
        if name in aliases:
            return short_name
    all_aliases = sorted({alias for aliases in ARCHITECTURE_CHOICES.values() for alias in aliases})
    raise ValueError(f"Unknown architecture: {name}. Valid options: {all_aliases}")


def check_file(path: str) -> tuple[str, bool, str]:
    try:
        BucketBatchManager._load_cache_file(path)
        return path, True, ""
    except Exception as e:  # noqa: BLE001 - any load failure means the cache file is corrupted/unreadable
        return path, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Verify cached latent/text-encoder .safetensors files are loadable.")
    parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
    parser.add_argument("--architecture", type=str, required=True, help="architecture short/full name, e.g. wan, hv, qi")
    parser.add_argument("--num_workers", type=int, default=None, help="number of workers, default is cpu count-1")
    parser.add_argument(
        "--delete_corrupted", action="store_true", help="delete corrupted cache files after reporting them (irreversible)"
    )
    args = parser.parse_args()

    architecture = resolve_architecture(args.architecture)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=architecture)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    cache_files = []
    for dataset in dataset_group.datasets:
        cache_files.extend(dataset.get_all_latent_cache_files())
        cache_files.extend(dataset.get_all_text_encoder_output_cache_files())
    cache_files = sorted(set(cache_files))

    logger.info(f"Found {len(cache_files)} cache files to verify")
    if not cache_files:
        return

    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
    corrupted = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(check_file, path) for path in cache_files]
        for i, future in enumerate(as_completed(futures)):
            path, ok, error = future.result()
            if not ok:
                corrupted.append(path)
                logger.error(f"Corrupted cache file: {path}: {error}")
            if (i + 1) % 500 == 0:
                logger.info(f"Checked {i + 1}/{len(cache_files)} cache files")

    logger.info(f"Checked {len(cache_files)} cache files, found {len(corrupted)} corrupted")
    for path in corrupted:
        print(path)

    if corrupted and args.delete_corrupted:
        for path in corrupted:
            os.remove(path)
            logger.info(f"Deleted corrupted cache file: {path}")


if __name__ == "__main__":
    main()
