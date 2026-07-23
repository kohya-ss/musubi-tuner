import argparse
import logging
from typing import Sequence

import numpy as np
import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MAGE_FLOW, ARCHITECTURE_MAGE_FLOW_EDIT
from musubi_tuner.dataset.cache_io import save_latent_cache_mage_flow
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.mage_flow.mage_vae import load_mage_vae, posterior_seed, sample_posterior
from musubi_tuner.mage_flow.utils import architecture_for_mode
from musubi_tuner.utils.model_utils import str_to_dtype


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _image_tensor(image: np.ndarray, *, item_key: str, role: str) -> torch.Tensor:
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"{role} image for {item_key} must have shape [H,W,3+]")
    height, width = image.shape[:2]
    if height % 16 or width % 16:
        raise ValueError(f"{role} image for {item_key} must have dimensions divisible by 16, got {height}x{width}")
    return torch.from_numpy(image[..., :3]).permute(2, 0, 1).float().div(127.5).sub(1.0)


def _controls_for_item(item: ItemInfo, is_edit: bool) -> list[np.ndarray]:
    if item.control_content is None:
        controls = []
    elif isinstance(item.control_content, np.ndarray) and item.control_content.ndim == 3:
        controls = [item.control_content]
    else:
        controls = list(item.control_content)
    if is_edit and not 1 <= len(controls) <= 3:
        raise ValueError(f"Mage-Flow-Edit item {item.item_key} requires between 1 and 3 ordered references")
    if not is_edit and controls:
        raise ValueError(f"Mage-Flow T2I item {item.item_key} contains controls; T2I mode must reject them")
    return controls


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def encode_and_save_batch(vae, batch: Sequence[ItemInfo], *, is_edit: bool, seed: int) -> None:
    if not batch:
        return
    _, architecture_full = architecture_for_mode(is_edit)
    target_tensors = []
    controls_by_item = []
    for item in batch:
        content = item.content[0] if isinstance(item.content, list) else item.content
        target_tensors.append(_image_tensor(content, item_key=item.item_key, role="target"))
        controls_by_item.append(_controls_for_item(item, is_edit))
    if len({tuple(tensor.shape) for tensor in target_tensors}) != 1:
        raise ValueError("one MageVAE cache batch must contain one target resolution")

    pixels = torch.stack(target_tensors).to(device=vae.device, dtype=vae.dtype)
    with torch.no_grad():
        means, logvars = vae.encode_moments(pixels)
    target_latents = []
    for index, item in enumerate(batch):
        generator = _generator(
            means.device,
            posterior_seed(architecture_full, item.item_key, "target", seed),
        )
        target_latents.append(
            sample_posterior(means[index : index + 1], logvars[index : index + 1], generator)[0].to(torch.bfloat16)
        )

    for item_index, item in enumerate(batch):
        control_latents = []
        for control_index, control in enumerate(controls_by_item[item_index]):
            control_pixels = _image_tensor(
                control,
                item_key=item.item_key,
                role=f"control:{control_index}",
            ).unsqueeze(0)
            control_pixels = control_pixels.to(device=vae.device, dtype=vae.dtype)
            with torch.no_grad():
                control_mean, control_logvar = vae.encode_moments(control_pixels)
            generator = _generator(
                control_mean.device,
                posterior_seed(architecture_full, item.item_key, f"control:{control_index}", seed),
            )
            control_latents.append(sample_posterior(control_mean, control_logvar, generator)[0].to(torch.bfloat16))
        save_latent_cache_mage_flow(
            item,
            target_latents[item_index],
            control_latents if is_edit else None,
            is_edit=is_edit,
        )


def setup_parser() -> argparse.ArgumentParser:
    parser = cache_latents.setup_parser_common()
    parser.add_argument("--is_edit", action="store_true", help="cache Mage-Flow-Edit targets and one to three ordered references")
    parser.add_argument("--seed", type=int, default=0, help="base seed mixed with each stable item/role identity")
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    architecture = ARCHITECTURE_MAGE_FLOW_EDIT if args.is_edit else ARCHITECTURE_MAGE_FLOW
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=architecture)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    if args.debug_mode is not None:
        cache_latents.show_datasets(
            dataset_group.datasets,
            args.debug_mode,
            args.console_width,
            args.console_back,
            args.console_num_images,
            fps=16,
        )
        return
    if args.vae is None:
        parser.error("--vae is required")
    vae_dtype = str_to_dtype(args.vae_dtype, torch.bfloat16)
    vae = load_mage_vae(args.vae, device=device, dtype=vae_dtype, require_decoder=False)
    cache_latents.encode_datasets(
        dataset_group.datasets,
        lambda batch: encode_and_save_batch(vae, batch, is_edit=args.is_edit, seed=args.seed),
        args,
    )


if __name__ == "__main__":
    main()
