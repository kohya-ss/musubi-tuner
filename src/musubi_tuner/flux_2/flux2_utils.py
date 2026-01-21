import argparse
import json
import torch
import torchvision
import math


from accelerate import init_empty_weights
from einops import rearrange
from PIL import Image
from typing import Optional, Union
from torch import Tensor
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
    Mistral3Config,
    AutoProcessor,
)
from tqdm import tqdm

from .flux2_models import Flux2, Flux2Params, Klein4BParams, Klein9BParams

from musubi_tuner.flux_2 import flux2_models
from musubi_tuner.utils.safetensors_utils import load_split_weights

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

M3_TOKENIZER_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
OUTPUT_LAYERS_MISTRAL = [10, 20, 30]
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
MAX_LENGTH = 512
UPSAMPLING_MAX_IMAGE_SIZE = 768**2
SYSTEM_MESSAGE = """You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object
attribution and actions without speculation."""


FLUX2_MODEL_INFO = {
    "flux.2-klein-4b": {
        "params": Klein4BParams(),
        "qwen_variant": "4B",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},
        "guidance_distilled": True,
    },
    "flux.2-klein-base-4b": {
        "params": Klein4BParams(),
        "qwen_variant": "4B",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-klein-9b": {
        "params": Klein9BParams(),
        "qwen_variant": "8B",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},
        "guidance_distilled": True,
    },
    "flux.2-klein-base-9b": {
        "params": Klein9BParams(),
        "qwen_variant": "8B",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-dev": {
        "params": Flux2Params(),
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": True,
    },
}


def add_model_version_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model_version",
        type=str,
        default="flux.2-dev",
        choices=list(FLUX2_MODEL_INFO.keys()),
        help="model version",
    )


def is_fp8(dt):
    return dt in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]


def compress_time(t_ids: Tensor) -> Tensor:
    assert t_ids.ndim == 1
    t_ids_max = torch.max(t_ids)
    t_remap = torch.zeros((t_ids_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    t_unique_sorted_ids = torch.unique(t_ids, sorted=True)
    t_remap[t_unique_sorted_ids] = torch.arange(len(t_unique_sorted_ids), device=t_ids.device, dtype=t_ids.dtype)
    t_ids_compressed = t_remap[t_ids]
    return t_ids_compressed


def scatter_ids(x: Tensor, x_ids: Tensor) -> list[Tensor]:
    """
    using position ids to scatter tokens into place
    """
    x_list = []
    t_coords = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape  # noqa: F841
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_cmpr = compress_time(t_ids)

        t = torch.max(t_ids_cmpr) + 1
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids

        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        x_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
        t_coords.append(torch.unique(t_ids, sorted=True))
    return x_list


def encode_image_refs(ae, img_ctx: list[Image.Image]):
    scale = 10

    if len(img_ctx) > 1:
        limit_pixels = 1024**2
    elif len(img_ctx) == 1:
        limit_pixels = 2024**2
    else:
        limit_pixels = None

    if not img_ctx:
        return None, None

    img_ctx_prep = default_prep(img=img_ctx, limit_pixels=limit_pixels)
    if not isinstance(img_ctx_prep, list):
        img_ctx_prep = [img_ctx_prep]

    # Encode each reference image
    encoded_refs = []
    for img in img_ctx_prep:
        encoded = ae.encode(img[None].cuda())[0]
        encoded_refs.append(encoded)

    # Create time offsets for each reference
    t_off = [scale + scale * t for t in torch.arange(0, len(encoded_refs))]
    t_off = [t.view(-1) for t in t_off]

    # Process with position IDs
    ref_tokens, ref_ids = listed_prc_img(encoded_refs, t_coord=t_off)

    # Concatenate all references along sequence dimension
    ref_tokens = torch.cat(ref_tokens, dim=0)  # (total_ref_tokens, C)
    ref_ids = torch.cat(ref_ids, dim=0)  # (total_ref_tokens, 4)

    # Add batch dimension
    ref_tokens = ref_tokens.unsqueeze(0)  # (1, total_ref_tokens, C)
    ref_ids = ref_ids.unsqueeze(0)  # (1, total_ref_tokens, 4)

    return ref_tokens.to(torch.bfloat16), ref_ids


def prc_txt(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _l, _ = x.shape  # noqa: F841

    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(1),  # dummy dimension
        "w": torch.arange(1),  # dummy dimension
        "l": torch.arange(_l),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids.to(x.device)


def batched_wrapper(fn):
    def batched_prc(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
        results = []
        for i in range(len(x)):
            results.append(
                fn(
                    x[i],
                    t_coord[i] if t_coord is not None else None,
                )
            )
        x, x_ids = zip(*results)
        return torch.stack(x), torch.stack(x_ids)

    return batched_prc


def listed_wrapper(fn):
    def listed_prc(
        x: list[Tensor],
        t_coord: list[Tensor] | None = None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        results = []
        for i in range(len(x)):
            results.append(
                fn(
                    x[i],
                    t_coord[i] if t_coord is not None else None,
                )
            )
        x, x_ids = zip(*results)
        return list(x), list(x_ids)

    return listed_prc


def prc_img(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _, h, w = x.shape  # noqa: F841
    x_coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(h),
        "w": torch.arange(w),
        "l": torch.arange(1),
    }
    x_ids = torch.cartesian_prod(x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"])
    x = rearrange(x, "c h w -> (h w) c")
    return x, x_ids.to(x.device)


listed_prc_img = listed_wrapper(prc_img)
batched_prc_img = batched_wrapper(prc_img)
batched_prc_txt = batched_wrapper(prc_txt)


def center_crop_to_multiple_of_x(img: Image.Image | list[Image.Image], x: int) -> Image.Image | list[Image.Image]:
    if isinstance(img, list):
        return [center_crop_to_multiple_of_x(_img, x) for _img in img]  # type: ignore

    w, h = img.size
    new_w = (w // x) * x
    new_h = (h // x) * x

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    resized = img.crop((left, top, right, bottom))
    return resized


def cap_pixels(img: Image.Image | list[Image.Image], k):
    if isinstance(img, list):
        return [cap_pixels(_img, k) for _img in img]
    w, h = img.size
    pixel_count = w * h

    if pixel_count <= k:
        return img

    # Scaling factor to reduce total pixels below K
    scale = math.sqrt(k / pixel_count)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def cap_min_pixels(img: Image.Image | list[Image.Image], max_ar=8, min_sidelength=64):
    if isinstance(img, list):
        return [cap_min_pixels(_img, max_ar=max_ar, min_sidelength=min_sidelength) for _img in img]
    w, h = img.size
    if w < min_sidelength or h < min_sidelength:
        raise ValueError(f"Skipping due to minimal sidelength underschritten h {h} w {w}")
    if w / h > max_ar or h / w > max_ar:
        raise ValueError(f"Skipping due to maximal ar overschritten h {h} w {w}")
    return img


def to_rgb(img: Image.Image | list[Image.Image]):
    if isinstance(img, list):
        return [
            to_rgb(
                _img,
            )
            for _img in img
        ]
    return img.convert("RGB")


def default_images_prep(
    x: Image.Image | list[Image.Image],
) -> torch.Tensor | list[torch.Tensor]:
    if isinstance(x, list):
        return [default_images_prep(e) for e in x]  # type: ignore
    x_tensor = torchvision.transforms.ToTensor()(x)
    return 2 * x_tensor - 1


def default_prep(
    img: Image.Image | list[Image.Image], limit_pixels: int | None, ensure_multiple: int = 16
) -> torch.Tensor | list[torch.Tensor]:
    img_rgb = to_rgb(img)
    img_min = cap_min_pixels(img_rgb)  # type: ignore
    if limit_pixels is not None:
        img_cap = cap_pixels(img_min, limit_pixels)  # type: ignore
    else:
        img_cap = img_min
    img_crop = center_crop_to_multiple_of_x(img_cap, ensure_multiple)  # type: ignore
    img_tensor = default_images_prep(img_crop)
    return img_tensor


def generalized_time_snr_shift(t: Tensor, mu: float, sigma: float) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def denoise(
    model: Flux2,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            assert img_cond_seq_ids is not None, "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
        )
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

    return img


def vanilla_guidance(x: torch.Tensor, cfg_val: float) -> torch.Tensor:
    x_u, x_c = x.chunk(2)
    x = x_u + cfg_val * (x_c - x_u)
    return x


def denoise_cfg(
    model: Flux2,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,  # Already cat([txt_empty, txt_prompt])
    txt_ids: Tensor,
    timesteps: list[float],
    guidance: float,
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    img = torch.cat([img, img], dim=0)
    img_ids = torch.cat([img_ids, img_ids], dim=0)

    if img_cond_seq is not None:
        assert img_cond_seq_ids is not None
        img_cond_seq = torch.cat([img_cond_seq, img_cond_seq], dim=0)
        img_cond_seq_ids = torch.cat([img_cond_seq_ids, img_cond_seq_ids], dim=0)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=None,
        )

        if img_cond_seq is not None:
            pred = pred[:, : img.shape[1]]

        pred_uncond, pred_cond = pred.chunk(2)
        pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        pred = torch.cat([pred, pred], dim=0)

        img = img + (t_prev - t_curr) * pred

    return img.chunk(2)[0]


def concatenate_images(
    images: list[Image.Image],
) -> Image.Image:
    """
    Concatenate a list of PIL images horizontally with center alignment and white background.
    """

    # If only one image, return a copy of it
    if len(images) == 1:
        return images[0].copy()

    # Convert all images to RGB if not already
    images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

    # Calculate dimensions for horizontal concatenation
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create new image with white background
    background_color = (255, 255, 255)
    new_img = Image.new("RGB", (total_width, max_height), background_color)

    # Paste images with center alignment
    x_offset = 0
    for img in images:
        y_offset = (max_height - img.height) // 2
        new_img.paste(img, (x_offset, y_offset))
        x_offset += img.width

    return new_img


def load_flow_model(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    attn_mode: str = "torch",
    split_attn: bool = False,
    loading_device: Optional[Union[str, torch.device]] = None,
    fp8_scaled: bool = False,
) -> flux2_models.Flux2:
    if loading_device is None:
        loading_device = device

    device = torch.device(device)
    loading_device = torch.device(loading_device) if loading_device is not None else device
    flux_2_loading_device = loading_device if not fp8_scaled else torch.device("cpu")

    # build model
    with init_empty_weights():
        params = flux2_models.configs_flux_2_dev.params

        model = flux2_models.Flux2(params, attn_mode, split_attn)
        if dtype is not None:
            model = model.to(dtype)

    # load_sft doesn't support torch.device
    logger.info(f"Loading state dict from {ckpt_path} to {flux_2_loading_device}")
    sd = load_split_weights(ckpt_path, device=flux_2_loading_device, disable_mmap=disable_mmap, dtype=dtype)

    # # if the key has annoying prefix, remove it
    # for key in list(sd.keys()):
    #     new_key = key.replace("model.diffusion_model.", "")
    #     if new_key == key:
    #         break  # the model doesn't have annoying prefix
    #     sd[new_key] = sd.pop(key)

    # if fp8_scaled is True, convert the model to fp8
    if fp8_scaled:
        # fp8 optimization: calculate on CUDA, move back to CPU if loading_device is CPU (block swap)
        logger.info("Optimizing model weights to fp8. This may take a while.")
        sd = model.fp8_optimization(sd, device, move_to_device=loading_device.type == "cpu")

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded Flux 2: {info}")
    return model


def load_ae(
    ckpt_path: str, dtype: torch.dtype, device: Union[str, torch.device], disable_mmap: bool = False
) -> flux2_models.AutoEncoder:
    logger.info("Building AutoEncoder")
    with init_empty_weights():
        # dev and schnell have the same AE params
        ae = flux2_models.AutoEncoder(flux2_models.configs_flux_2_dev.ae_params).to(dtype)

    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = ae.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded AE: {info}")
    return ae


class Mistral3Embedder(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        dtype: Optional[torch.dtype],
        device: Union[str, torch.device],
        disable_mmap: bool = False,
        state_dict: Optional[dict] = None,
    ) -> tuple[AutoProcessor, Mistral3ForConditionalGeneration]:
        super().__init__()

        M3_CONFIG_JSON = """
{
  "architectures": [
    "Mistral3ForConditionalGeneration"
  ],
  "dtype": "bfloat16",
  "image_token_index": 10,
  "model_type": "mistral3",
  "multimodal_projector_bias": false,
  "projector_hidden_act": "gelu",
  "spatial_merge_size": 2,
  "text_config": {
    "attention_dropout": 0.0,
    "dtype": "bfloat16",
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 32768,
    "max_position_embeddings": 131072,
    "model_type": "mistral",
    "num_attention_heads": 32,
    "num_hidden_layers": 40,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000000.0,
    "sliding_window": null,
    "use_cache": true,
    "vocab_size": 131072
  },
  "transformers_version": "4.57.1",
  "vision_config": {
    "attention_dropout": 0.0,
    "dtype": "bfloat16",
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "image_size": 1540,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "model_type": "pixtral",
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "rope_theta": 10000.0
  },
  "vision_feature_layer": -1
}
"""
        config = json.loads(M3_CONFIG_JSON)
        config = Mistral3Config(**config)
        with init_empty_weights():
            self.mistral3 = Mistral3ForConditionalGeneration._from_config(config)

        if state_dict is not None:
            sd = state_dict
        else:
            logger.info(f"Loading state dict from {ckpt_path}")
            sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

        # if the key has annoying prefix, remove it
        for key in list(sd.keys()):
            new_key = key.replace("language_model.lm_", "lm_")
            new_key = new_key.replace("language_model.model.", "model.language_model.")
            new_key = new_key.replace("multi_modal_projector.", "model.multi_modal_projector.")
            new_key = new_key.replace("vision_tower.", "model.vision_tower.")
            sd[new_key] = sd.pop(key)

        info = self.mistral3.load_state_dict(sd, strict=True, assign=True)
        logger.info(f"Loaded Mistral 3: {info}")
        self.mistral3.to(device)

        if dtype is not None:
            if is_fp8(dtype):
                logger.info(f"prepare Mistral 3 for fp8: set to {dtype}")
                raise NotImplementedError(f"Mistral 3 {dtype}")  # TODO
            else:
                logger.info(f"Setting Mistral 3 to dtype: {dtype}")
                self.mistral3.to(dtype)

        # Load tokenizer
        self.tokenizer = AutoProcessor.from_pretrained(M3_TOKENIZER_ID, use_fast=False)

    @property
    def dtype(self):
        return self.mistral3.dtype

    @property
    def device(self):
        return self.mistral3.device

    def to(self, *args, **kwargs):
        return self.mistral3.to(*args, **kwargs)

    def forward(self, txt: list[str]):
        if not isinstance(txt, list):
            txt = [txt]

        # Format input messages
        messages_batch = self.format_input(txt=txt)

        # Process all messages at once
        # with image processing a too short max length can throw an error in here.
        inputs = self.tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(self.mistral3.device)
        attention_mask = inputs["attention_mask"].to(self.mistral3.device)

        # Forward pass through the model
        output = self.mistral3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_MISTRAL], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")

    @staticmethod
    def _validate_and_process_images(img: list[list[Image.Image]] | list[Image.Image]) -> list[list[Image.Image]]:
        # Simple validation: ensure it's a list of PIL images or list of lists of PIL images
        if not img:
            return []

        # Check if it's a list of lists or a list of images
        if isinstance(img[0], Image.Image):
            # It's a list of images, convert to list of lists
            img = [[im] for im in img]

        # potentially concatenate multiple images to reduce the size
        img = [[concatenate_images(img_i)] if len(img_i) > 1 else img_i for img_i in img]

        # cap the pixels
        img = [[cap_pixels(img_i, UPSAMPLING_MAX_IMAGE_SIZE) for img_i in img_i] for img_i in img]
        return img

    def format_input(
        self,
        txt: list[str],
        system_message: str = SYSTEM_MESSAGE,
        img: list[Image.Image] | list[list[Image.Image]] | None = None,
    ) -> list[list[dict]]:
        """
        Format a batch of text prompts into the conversation format expected by apply_chat_template.
        Optionally, add images to the input.

        Args:
            txt: List of text prompts
            system_message: System message to use (default: CREATIVE_SYSTEM_MESSAGE)
            img: List of images to add to the input.

        Returns:
            List of conversations, where each conversation is a list of message dicts
        """
        # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
        # when truncation is enabled. The processor counts [IMG] tokens and fails
        # if the count changes after truncation.
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in txt]

        if img is None or len(img) == 0:
            return [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_message}],
                    },
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ]
                for prompt in cleaned_txt
            ]
        else:
            assert len(img) == len(txt), "Number of images must match number of prompts"
            img = self._validate_and_process_images(img)

            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_message}],
                    },
                ]
                for _ in cleaned_txt
            ]

            for i, (el, images) in enumerate(zip(messages, img)):
                # optionally add the images per batch element.
                if images is not None:
                    el.append(
                        {
                            "role": "user",
                            "content": [{"type": "image", "image": image_obj} for image_obj in images],
                        }
                    )
                # add the text.
                el.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": cleaned_txt[i]}],
                    }
                )

            return messages


class Qwen3Embedder(nn.Module):
    def __init__(
        self,
        model_spec: str,
        ckpt_path: str,
        dtype: Optional[torch.dtype],
        device: Union[str, torch.device],
        disable_mmap: bool = False,
        state_dict: Optional[dict] = None,
    ):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            torch_dtype=None,
            device_map=str(device),
        )

        # config = json.loads(M3_CONFIG_JSON)
        # config = Mistral3Config(**config)
        # with init_empty_weights():
        #     self.mistral3 = Mistral3ForConditionalGeneration._from_config(config)
        #
        # if state_dict is not None:
        #     sd = state_dict
        # else:
        #     logger.info(f"Loading state dict from {ckpt_path}")
        #     sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
        #
        #
        # info = self.mistral3.load_state_dict(sd, strict=True, assign=True)
        # logger.info(f"Loaded Mistral 3: {info}")
        # self.mistral3.to(device)
        #
        # if dtype is not None:
        #     if is_fp8(dtype):
        #         logger.info(f"prepare Mistral 3 for fp8: set to {dtype}")
        #         raise NotImplemented(f"Mistral 3 {dtype}")  # TODO
        #     else:
        #         logger.info(f"Setting Mistral 3 to dtype: {dtype}")
        #         self.mistral3.to(dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_spec)
        self.max_length = MAX_LENGTH

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)

    def forward(self, txt: list[str]):
        all_input_ids = []
        all_attention_masks = []

        for prompt in txt:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.model.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")


def load_textembedder(
    model_version: str,
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> tuple[AutoProcessor, Mistral3ForConditionalGeneration]:
    if model_version == "flux.2-dev":
        return Mistral3Embedder(ckpt_path, dtype, device, disable_mmap, state_dict)
    else:
        variant = FLUX2_MODEL_INFO[model_version]["qwen_variant"]
        return Qwen3Embedder(
            f"Qwen/Qwen3-{variant}-FP8",
            ckpt_path,
            dtype,
            device,
            disable_mmap,
            state_dict,
        )
