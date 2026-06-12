from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from musubi_tuner.ideogram4.pipeline import get_qwen3_vl_features, pad_text_features


DEFAULT_QWEN3_VL_PATH = "Qwen/Qwen3-VL-8B-Instruct"


def load_qwen3_vl_text_encoder(
    text_encoder_path: str = DEFAULT_QWEN3_VL_PATH,
    dtype: torch.dtype = torch.bfloat16,
    device_map="auto",
):
    """
    Load the frozen Qwen3-VL text encoder used by Ideogram4.

    Ideogram4 does not use Qwen3-VL to generate text.
    It uses hidden states from Qwen3-VL as prompt-conditioning features.
    """
    print(f"Loading Qwen3-VL tokenizer from: {text_encoder_path}")
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)

    print(f"Loading Qwen3-VL text encoder from: {text_encoder_path}")
    text_encoder = AutoModel.from_pretrained(
        text_encoder_path,
        torch_dtype=dtype,
        device_map=device_map,
    )

    text_encoder.eval()
    text_encoder.requires_grad_(False)

    return tokenizer, text_encoder


def caption_to_token_ids(
    tokenizer,
    caption: str,
    max_text_length: int = 3072,
) -> List[int]:
    """
    Convert a caption/JSON prompt into Qwen3-VL token IDs using the Qwen chat template.

    Ideogram4 expects prompts to pass through Qwen3-VL in chat-template format.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": caption}]}]

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_text_length,
    )["input_ids"]

    if len(ids) == 0:
        ids = [tokenizer.eos_token_id or 0]

    return ids


@torch.no_grad()
def encode_caption_to_features(
    tokenizer,
    text_encoder,
    caption: str,
    max_text_length: int = 3072,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Encode one caption into Ideogram4 Qwen3-VL features.

    Returns:
      (num_tokens, feature_dim)

    For Ideogram4, feature_dim is usually 53248 because multiple Qwen3-VL
    hidden layers are concatenated.
    """
    ids = caption_to_token_ids(tokenizer, caption, max_text_length=max_text_length)

    device = next(text_encoder.parameters()).device

    token_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(token_ids)

    # Qwen3-VL expects 2D position ids for text-only mode here.
    pos_2d = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0).to(torch.long)

    features = get_qwen3_vl_features(
        text_encoder,
        token_ids,
        attention_mask,
        pos_2d,
    )

    return features[0].to(dtype)


@torch.no_grad()
def encode_captions_to_padded_features(
    tokenizer,
    text_encoder,
    captions: List[str],
    max_text_length: int = 3072,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a list of captions and pad them into a batch.

    Returns:
      llm_features: (B, max_tokens, feature_dim)
      text_mask:    (B, max_tokens)
    """
    features_list = [
        encode_caption_to_features(
            tokenizer,
            text_encoder,
            caption,
            max_text_length=max_text_length,
            dtype=dtype,
        )
        for caption in captions
    ]

    return pad_text_features(features_list, torch.device(device), dtype)
