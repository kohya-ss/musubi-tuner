import logging

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from musubi_tuner.dataset.architectures import ARCHITECTURE_QWEN_IMAGE_21
from musubi_tuner.dataset.bucket import BucketSelector
from musubi_tuner.dataset.media_utils import resize_image_to_bucket
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.qwen_image_21 import qwen_image_21_utils

logger = logging.getLogger(__name__)


def encode_sample_prompts(processor, encoder, prompts: list[dict], control_image_size: tuple[int, int] | None = None) -> list[dict]:
    """Encode sample prompts, reusing captions with the same ordered control images."""
    image_cache = {}
    text_cache = {}
    for prompt in prompts:
        paths = tuple(prompt.get("control_image_path") or [])
        for path in paths:
            if path not in image_cache:
                with Image.open(path) as source:
                    source = source.convert("RGBA")
                    size = control_image_size or BucketSelector.calculate_bucket_resolution(
                        source.size, (1024, 1024), architecture=ARCHITECTURE_QWEN_IMAGE_21
                    )
                    image_cache[path] = resize_image_to_bucket(source, size)
        images = [image_cache[path] for path in paths]
        prompt["reference_images"] = images
        captions = [("positive", prompt.get("prompt", ""))]
        if prompt.get("negative_prompt") is not None and (prompt.get("cfg_scale") or 1.0) > 1.0:
            captions.append(("negative", prompt["negative_prompt"]))
        for name, caption in captions:
            key = (caption, paths)
            if key not in text_cache:
                logger.info(f"cache Text Encoder outputs for prompt: {caption} with image: {paths}")
                text_cache[key] = tuple(t.cpu() for t in qwen_image_21_utils.encode_prompt(processor, encoder, caption, images))
            prompt[name] = text_cache[key]
    return prompts


@torch.no_grad()
def sample_image(
    transformer,
    vae,
    sample_parameter: dict,
    device: torch.device,
    dit_dtype: torch.dtype,
    width: int,
    height: int,
    sample_steps: int,
    generator: torch.Generator,
    discrete_flow_shift: float | None = None,
    cfg_scale: float | None = None,
    do_classifier_free_guidance: bool = True,
    return_latents: bool = False,
) -> torch.Tensor:
    """Return normalized latents or an RGBA image in [0, 1] on CPU."""
    width, height = max(32, width // 32 * 32), max(32, height // 32 * 32)
    vae.to(device)
    if sample_parameter["reference_images"]:
        logger.info("Encoding control images with VAE")
    reference_latents = [qwen_image_21_utils.encode_image(vae, image) for image in sample_parameter["reference_images"]]
    vae.to("cpu")
    img_shapes = [[(1, ref.shape[-2], ref.shape[-1]) for ref in reference_latents] + [(1, height // 16, width // 16)]]
    reference_latents = [qwen_image_21_utils.pack_latents(ref).to(device=device, dtype=dit_dtype) for ref in reference_latents]
    latents = torch.randn((1, height // 16 * (width // 16), 64), generator=generator, device=device, dtype=dit_dtype)
    # Qwen-Image 2.1 uses the same scheduler configuration as Qwen-Image.
    scheduler = qwen_image_utils.get_scheduler(discrete_flow_shift)
    sigmas = np.linspace(1.0, 1 / sample_steps, sample_steps)
    mu = qwen_image_utils.calculate_shift_qwen_image(latents.shape[1])
    scheduler.set_timesteps(sample_steps, device=device, sigmas=sigmas, mu=mu)
    scheduler.set_begin_index(0)
    cfg_scale = 1.0 if cfg_scale is None else cfg_scale
    do_cfg = do_classifier_free_guidance and cfg_scale > 1.0

    def predict(name, timestep):
        prompt_embeds, image_slots, _ = sample_parameter[name]
        return transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds[None].to(device=device, dtype=dit_dtype),
            timestep=timestep.expand(1).to(dit_dtype) / 1000,
            img_shapes=img_shapes,
            reference_latents=reference_latents,
            image_slots=[image_slots.tolist()],
        )

    with tqdm(total=sample_steps, desc="Denoising steps") as pbar:
        for timestep in scheduler.timesteps:
            transformer.prepare_block_swap_before_forward()
            noise_pred = predict("positive", timestep)
            if do_cfg:
                transformer.prepare_block_swap_before_forward()
                noise_pred_uncond = predict("negative", timestep)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)
            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            pbar.update()
        latents = qwen_image_21_utils.unpack_latents(latents, height // 16, width // 16)
        if return_latents:
            return latents.cpu()
        vae.to(device)
        logger.info(f"Decoding image from latents: {latents.shape}")
        pixels = qwen_image_21_utils.decode_latents(vae, latents)
    logger.info("Decoding complete")
    return pixels.float().cpu()
