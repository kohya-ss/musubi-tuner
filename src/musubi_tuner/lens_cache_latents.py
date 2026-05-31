import logging
from typing import List

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LENS, ItemInfo, save_latent_cache_lens
from musubi_tuner.lens import lens_utils
import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_lens(batch: List[ItemInfo]) -> torch.Tensor:
    contents = []
    for item in batch:
        if item.control_content is not None:
            raise ValueError("Lens MVP supports text-to-image only; control images are not supported.")
        content = item.content
        content = content[0] if isinstance(content, list) else content
        contents.append(torch.from_numpy(content[..., :3]))

    contents = torch.stack(contents, dim=0)
    contents = contents.permute(0, 3, 1, 2)
    contents = contents / 127.5 - 1.0
    return contents


def encode_and_save_batch(ae, batch: List[ItemInfo]):
    contents = preprocess_contents_lens(batch)
    with torch.no_grad():
        latents = ae.encode(contents.to(ae.device, dtype=ae.dtype))

    for b, item in enumerate(batch):
        target_latent = latents[b]
        logger.info(f"Saving Lens latent cache for {item.item_key}: {target_latent.shape}")
        save_latent_cache_lens(item_info=item, latent=target_latent)


def main():
    parser = cache_latents.setup_parser_common()
    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_LENS)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=1
        )
        return

    assert args.vae is not None, "vae checkpoint is required"
    vae_dtype = torch.float32 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    ae = lens_utils.load_lens_vae(args.vae, dtype=vae_dtype, device=device)
    ae.to(device)
    ae.eval()

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(ae, batch)

    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
