"""Model loading, image preprocessing, and text encoding for Qwen-Image 2.1."""

import hashlib
import logging
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from safetensors.torch import load_file

from musubi_tuner.dataset.architectures import ARCHITECTURE_QWEN_IMAGE_21
from musubi_tuner.dataset.config_utils import (
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
    load_user_config,
)
from musubi_tuner.qwen_image_21.qwen_image_21_text_encoder import (
    QWEN3_VL_8B_INSTRUCT_CONFIG,
    normalize_qwen3_vl_state_dict_for_base_model,
)
from musubi_tuner.utils.safetensors_utils import load_safetensors

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from argparse import Namespace

    from musubi_tuner.dataset.image_video_dataset import ImageDataset
    from musubi_tuner.qwen_image_21.qwen_image_21_autoencoder_kl import AutoencoderKLQwenImage21

ImageInput = Union[Image.Image, np.ndarray]

QWEN_IMAGE_21_ID = "Qwen/Qwen-Image-2.1"

SYSTEM_PROMPT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
VAE_SCALE_FACTOR = 16
LATENT_CHANNELS = 64


def load_datasets(args: "Namespace") -> list["ImageDataset"]:
    """Load image datasets for caching."""
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        load_user_config(args.dataset_config), args, architecture=ARCHITECTURE_QWEN_IMAGE_21
    )
    datasets = generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets
    for dataset in datasets:
        if getattr(dataset, "multiple_target", False) or not hasattr(dataset, "image_directory"):
            raise ValueError("Qwen-Image 2.1 expects image datasets with one target image per item")
    return datasets


def reference_fingerprints(images: Optional[list[ImageInput]]) -> torch.Tensor:
    """Return SHA256 hashes of resized RGBA images in reference order."""
    digests = []
    for image in images or []:
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        image = image.convert("RGBA")
        payload = f"{image.width}x{image.height}:".encode() + image.tobytes()
        digests.append(list(hashlib.sha256(payload).digest()))
    return torch.tensor(digests, dtype=torch.uint8).reshape(-1, 32)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Flatten [B, 64, 1, H, W] latents to [B, H*W, 64] without patch packing."""
    if latents.ndim != 5 or latents.shape[1:3] != (LATENT_CHANNELS, 1):
        raise ValueError("Expected 2.1 latents [B, 64, 1, H, W]; regenerate caches with qwen_image_21_cache_latents.py")
    return latents.flatten(2).transpose(1, 2)


def unpack_latents(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Reshape [B, H*W, 64] tokens to [B, 64, 1, H, W] latents."""
    return tokens.transpose(1, 2).reshape(tokens.shape[0], LATENT_CHANNELS, 1, height, width)


def load_vae(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    tiling: bool = False,
) -> "AutoencoderKLQwenImage21":
    """Load local safetensors weights using the embedded Qwen-Image 2.1 VAE configuration."""
    from accelerate import init_empty_weights

    from musubi_tuner.qwen_image_21.qwen_image_21_autoencoder_kl import AutoencoderKLQwenImage21

    vae_path = Path(path)
    if vae_path.is_dir():
        vae_path = vae_path / "diffusion_pytorch_model.safetensors"
    logger.info("Initializing VAE")
    with init_empty_weights():
        vae = AutoencoderKLQwenImage21()
    logger.info(f"Loading VAE from {vae_path}")
    state_dict = load_safetensors(str(vae_path), device="cpu")
    info = vae.load_state_dict(state_dict, strict=True, assign=True)
    logger.info(f"Loaded VAE: {info}")
    vae.eval().requires_grad_(False).to(device=device, dtype=dtype)
    if tiling:
        vae.enable_tiling()
    return vae


def latent_stats(vae: "AutoencoderKLQwenImage21", latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return latent mean and standard deviation as [1, C, 1, 1, 1] tensors."""
    mean = latents.new_tensor(vae.latents_mean).view(1, -1, 1, 1, 1)
    std = latents.new_tensor(vae.latents_std).view(1, -1, 1, 1, 1)
    return mean, std


def image_tensor(image: ImageInput, channels: int = 4) -> torch.Tensor:
    """Convert an image to a normalized [1, C, 1, H, W] VAE input."""
    image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    if image.width % 32 or image.height % 32:
        raise ValueError("Qwen-Image 2.1 image dimensions must be divisible by 32; enable bucketing")
    image = np.array(image.convert("RGBA" if channels == 4 else "RGB"), copy=True)
    return torch.from_numpy(image).permute(2, 0, 1)[None, :, None].float() / 127.5 - 1


@torch.no_grad()
def encode_image(vae: "AutoencoderKLQwenImage21", image: ImageInput) -> torch.Tensor:
    """Encode an image and apply the VAE's per-channel latent normalization."""
    return encode_images(vae, [image])[0].unsqueeze(0)


@torch.no_grad()
def encode_images(vae: "AutoencoderKLQwenImage21", images: list[ImageInput]) -> list[torch.Tensor]:
    """Encode equal-sized images together and return latents in input order."""
    groups = {}
    for index, image in enumerate(images):
        pixels = image_tensor(image, vae.in_channels)
        groups.setdefault(pixels.shape[-2:], []).append((index, pixels))
    results = [None] * len(images)
    for group in groups.values():
        pixels = torch.cat([pixels for _, pixels in group]).to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode_pixels_to_latents(pixels)
        for (index, _), latent in zip(group, latents):
            results[index] = latent
    return results


@torch.no_grad()
def decode_latents(vae: "AutoencoderKLQwenImage21", latents: torch.Tensor) -> torch.Tensor:
    """Decode normalized latents to [B, C, H, W] pixels in [0, 1]."""
    latents = latents.to(device=vae.device, dtype=vae.dtype)
    return vae.decode_to_pixels(latents)


def _load_single_file_text_encoder(path, dtype):
    from accelerate import init_empty_weights
    from transformers import Qwen3VLConfig, Qwen3VLModel

    path = Path(path)
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"Qwen3-VL single-file text encoder must be a safetensors file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Qwen3-VL text encoder file not found: {path}")

    config = Qwen3VLConfig.from_dict(QWEN3_VL_8B_INSTRUCT_CONFIG)
    config._attn_implementation = "sdpa"
    with init_empty_weights():
        encoder = Qwen3VLModel(config)

    logger.info("Reading Qwen3-VL single-file checkpoint: %s", path)
    state_dict = normalize_qwen3_vl_state_dict_for_base_model(load_file(str(path), device="cpu"))
    logger.info("Assigning Qwen3-VL checkpoint weights")
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            "Qwen3-VL single-file checkpoint does not match Qwen3-VL-8B-Instruct after key normalization: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    logger.info("Qwen3-VL single-file checkpoint loaded")
    return encoder.to(dtype=dtype)


def load_text_encoder(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    fp8_vl: bool = False,
):
    """Load the Qwen3-VL text encoder and processor."""
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    logger.info("Loading processor from %s", QWEN_IMAGE_21_ID)
    processor = Qwen3VLProcessor.from_pretrained(QWEN_IMAGE_21_ID, subfolder="processor")
    if Path(path).suffix.lower() == ".safetensors":
        encoder = _load_single_file_text_encoder(path, dtype)
    else:
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=dtype, local_files_only=True, attn_implementation="sdpa"
        )
    encoder.eval().requires_grad_(False)
    encoder_model = getattr(encoder, "model", encoder)
    if fp8_vl:
        # Quantize before moving to the device to reduce peak VRAM usage.
        store_linears_in_fp8(encoder_model.language_model.layers)
    encoder.to(device)
    return processor, encoder


@torch.no_grad()
def encode_prompt(
    processor, encoder, prompt: str, images: Optional[list[ImageInput]] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a prompt and its reference images.

    Returns text features before the final RMSNorm, reference insertion positions,
    and reference grid sizes. Vision tokens are removed from the text features.
    """
    return encode_prompts(processor, encoder, [prompt], [images or []])[0]


@torch.no_grad()
def encode_prompts(
    processor, encoder, prompts: list[str], images: Optional[list[list[ImageInput]]] = None
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Encode a batch of captions and ordered reference images without retaining padding."""
    if images is None:
        images = [[] for _ in prompts]
    if len(images) != len(prompts):
        raise ValueError("Each prompt must have a reference image list")
    if not prompts:
        return []
    texts, image_batches, grids = [], [], []
    for prompt, references in zip(prompts, images):
        refs = " ".join(f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(len(references)))
        texts.append(SYSTEM_PROMPT + "<|im_start|>user\n" + refs + (prompt or " ") + "<|im_end|>\n<|im_start|>assistant\n")
        rgb = []
        for image in references:
            image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            if image.mode == "RGBA":
                white = Image.new("RGB", image.size, "white")
                white.paste(image, mask=image.getchannel("A"))
                image = white
            rgb.append(image.convert("RGB"))
        image_batches.extend(rgb)
        grids.append(torch.tensor([[im.height // 16, im.width // 16] for im in rgb], dtype=torch.int64).reshape(-1, 2))
    kwargs = dict(text=texts, padding=True, padding_side="left", return_tensors="pt")
    if image_batches:
        # Both encoders must use the same reference image size.
        kwargs.update(images=image_batches, do_resize=False)
    inputs = processor(**kwargs).to(encoder.device)
    encoder_model = getattr(encoder, "model", encoder)
    language_model = getattr(encoder_model, "language_model", encoder_model)
    pre_norm_hidden = []
    handle = language_model.norm.register_forward_pre_hook(lambda module, args: pre_norm_hidden.append(args[0]))
    try:
        # The norm pre-hook captures the features required by the DiT.
        encoder_model(**inputs, output_hidden_states=False, return_dict=True, use_cache=False)
    finally:
        handle.remove()
    if len(pre_norm_hidden) != 1:
        raise RuntimeError("Expected one Qwen3-VL final language norm invocation")
    if image_batches and not torch.equal(inputs.image_grid_thw[:, 1:].cpu(), torch.cat(grids)):
        raise ValueError("VLM resized reference images: reference VAE and VLM grids must agree")
    im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    results = []
    for row, references in enumerate(images):
        valid = inputs.attention_mask[row].bool()
        ids = inputs.input_ids[row, valid]
        hidden = pre_norm_hidden[0][row, valid]
        starts = (ids == im_start).nonzero(as_tuple=True)[0]
        if len(starts) < 2:
            raise ValueError("Unexpected Qwen3-VL template: missing user turn")
        drop = int(starts[1])
        ids, hidden = ids[drop:], hidden[drop:]
        keep = ids != image_id
        spans = []
        previous = False
        for i, is_image in enumerate((~keep).tolist()):
            if is_image and not previous:
                spans.append(int(keep[:i].sum()))
            previous = is_image
        if len(spans) != len(references):
            raise ValueError("VLM image spans do not match reference images")
        results.append((hidden[keep].contiguous(), torch.tensor(spans, dtype=torch.int64), grids[row]))
    return results


def _linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


def store_linears_in_fp8(module: nn.Module) -> None:
    """Store Linear weights in FP8 and cast to the input dtype during forward.

    Norms, embeddings, and biases keep their dtype. F.linear preserves input
    gradients so LoRA layers before these frozen layers can be trained.
    """
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            layer.weight = nn.Parameter(layer.weight.to(torch.float8_e4m3fn), requires_grad=False)
            layer.forward = MethodType(_linear_forward, layer)
