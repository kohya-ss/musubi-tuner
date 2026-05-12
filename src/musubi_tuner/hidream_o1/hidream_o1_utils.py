import glob
import json
import logging
import os

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoProcessor, PreTrainedTokenizerBase

from musubi_tuner.hidream_o1.pipeline import PATCH_SIZE, build_t2i_text_sample


logger = logging.getLogger(__name__)

HIDREAM_O1_REPO_FULL = "HiDream-ai/HiDream-O1-Image"
HIDREAM_O1_REPO_DEV = "HiDream-ai/HiDream-O1-Image-Dev"


def default_repo_for_model_type(model_type: str = "full") -> str:
    return HIDREAM_O1_REPO_DEV if model_type == "dev" else HIDREAM_O1_REPO_FULL


def add_special_tokens(tokenizer):
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def get_tokenizer(processor):
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def load_processor(model_path: str | None = None, model_type: str = "full"):
    if model_path is None:
        model_path = default_repo_for_model_type(model_type)
    processor = AutoProcessor.from_pretrained(model_path)
    add_special_tokens(get_tokenizer(processor))
    return processor


def normalize_single_checkpoint_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert Comfy-style bare HiDream-O1 checkpoint keys to the local HF-style wrapper keys."""
    normalized = {}
    for key, value in sd.items():
        if key.startswith(("vae.", "text_encoders.")):
            continue
        if key.startswith("diffusion_model."):
            key = key[len("diffusion_model.") :]
        if "visual.deepstack_merger_list" in key:
            continue
        if key.startswith("model.") or key.startswith("lm_head."):
            normalized[key] = value
        else:
            normalized[f"model.{key}"] = value
    return normalized


def load_single_checkpoint_model(
    checkpoint_path: str,
    config_path: str | None,
    dtype: torch.dtype,
    device: str | torch.device = "cpu",
    model_type: str = "full",
):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    from musubi_tuner.hidream_o1.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    if config_path is None:
        config_path = default_repo_for_model_type(model_type)
    config = Qwen3VLConfig.from_pretrained(config_path)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)
    # Comfy single checkpoints omit the LM head and unused Qwen3-VL deepstack merger weights.
    model.lm_head = nn.Identity()
    model.model.visual.deepstack_visual_indexes = []
    model.model.visual.deepstack_merger_list = nn.ModuleList()

    sd = load_file(checkpoint_path, device="cpu")
    sd = normalize_single_checkpoint_state_dict(sd)
    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded HiDream-O1 single checkpoint: {info}")
    del sd

    model.eval()
    model.to(device=device, dtype=dtype)
    return model


def load_model(
    model_path: str,
    dtype: torch.dtype,
    device: str | torch.device = "cpu",
    device_map=None,
    config_path: str | None = None,
    model_type: str = "full",
):
    from musubi_tuner.hidream_o1.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
        return load_single_checkpoint_model(model_path, config_path, dtype, device, model_type=model_type)

    kwargs = {"torch_dtype": dtype}
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs).eval()
    if device_map is None:
        model.to(device)
    return model


def _safe_open_get_tensor(path: str, key: str, device: str | torch.device) -> torch.Tensor | None:
    with safe_open(path, framework="pt", device=str(device)) as f:
        if key in f.keys():
            return f.get_tensor(key)
    return None


def load_text_embedding_weight(weight_path: str, dtype: torch.dtype, device: str | torch.device) -> torch.Tensor:
    """Load only the Qwen3VL token embedding weight when possible."""
    candidate_keys = [
        "model.language_model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    ]

    if os.path.isfile(weight_path) and weight_path.endswith(".safetensors"):
        for key in candidate_keys:
            weight = _safe_open_get_tensor(weight_path, key, device)
            if weight is not None:
                return weight.to(dtype=dtype)

    if os.path.isdir(weight_path):
        index_path = os.path.join(weight_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map", {})
            for key in candidate_keys:
                shard = weight_map.get(key)
                if shard is None:
                    continue
                weight = _safe_open_get_tensor(os.path.join(weight_path, shard), key, device)
                if weight is not None:
                    return weight.to(dtype=dtype)

        for path in sorted(glob.glob(os.path.join(weight_path, "*.safetensors"))):
            for key in candidate_keys:
                weight = _safe_open_get_tensor(path, key, device)
                if weight is not None:
                    return weight.to(dtype=dtype)

    model = load_model(weight_path, dtype=torch.bfloat16, device=device)
    weight = model.get_input_embeddings().weight.detach().to(device=device, dtype=dtype).contiguous()
    del model
    return weight


def build_text_input_embeds(input_ids: torch.Tensor, embedding_weight: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    input_ids = input_ids.to(device=device, dtype=torch.long)
    return F.embedding(input_ids, embedding_weight).detach()


def patchify_pixels(image: torch.Tensor) -> torch.Tensor:
    """Convert BCHW pixels normalized to [-1, 1] into HiDream-O1 32x32 patch tokens."""
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor shape B,C,H,W, got {tuple(image.shape)}")
    height, width = image.shape[-2:]
    if height % PATCH_SIZE != 0 or width % PATCH_SIZE != 0:
        raise ValueError(f"Image size must be divisible by {PATCH_SIZE}, got {width}x{height}")
    return einops.rearrange(image, "B C (H p1) (W p2) -> B (H W) (C p1 p2)", p1=PATCH_SIZE, p2=PATCH_SIZE)


def patchify_pixels_grid(image: torch.Tensor) -> torch.Tensor:
    """Convert BCHW pixels normalized to [-1, 1] into B,H,W,D HiDream-O1 patch tokens."""
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor shape B,C,H,W, got {tuple(image.shape)}")
    height, width = image.shape[-2:]
    if height % PATCH_SIZE != 0 or width % PATCH_SIZE != 0:
        raise ValueError(f"Image size must be divisible by {PATCH_SIZE}, got {width}x{height}")
    return einops.rearrange(image, "B C (H p1) (W p2) -> B H W (C p1 p2)", p1=PATCH_SIZE, p2=PATCH_SIZE)


def unpatchify_pixels(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert HiDream-O1 32x32 patch tokens back to BCHW pixels in [-1, 1]."""
    if height % PATCH_SIZE != 0 or width % PATCH_SIZE != 0:
        raise ValueError(f"Image size must be divisible by {PATCH_SIZE}, got {width}x{height}")
    return einops.rearrange(
        tokens,
        "B (H W) (C p1 p2) -> B C (H p1) (W p2)",
        H=height // PATCH_SIZE,
        W=width // PATCH_SIZE,
        p1=PATCH_SIZE,
        p2=PATCH_SIZE,
        C=3,
    )


def preprocess_image_tensor(contents: torch.Tensor) -> torch.Tensor:
    """Convert BHWC uint8 RGB/RGBA image data to BCHW float pixels in [-1, 1]."""
    if contents.ndim != 4:
        raise ValueError(f"Expected contents shape B,H,W,C, got {tuple(contents.shape)}")
    if contents.shape[-1] == 4:
        contents = contents[..., :3]
    contents = contents.permute(0, 3, 1, 2).float()
    return contents / 127.5 - 1.0


def build_t2i_input_ids(prompt: str, processor) -> torch.Tensor:
    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)
    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")
    messages = [{"role": "user", "content": prompt}]
    template_caption = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + boi_token + tms_token
    return tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False).squeeze(0).to(torch.long)


def build_t2i_cache_tensors(prompt: str, height: int, width: int, processor, model_config):
    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)
    sample = build_t2i_text_sample(prompt, height, width, tokenizer, processor, model_config)
    return (
        sample["input_ids"].squeeze(0).to(torch.long),
        sample["position_ids"].squeeze(1).to(torch.long),
        sample["token_types"].squeeze(0).to(torch.long),
    )
