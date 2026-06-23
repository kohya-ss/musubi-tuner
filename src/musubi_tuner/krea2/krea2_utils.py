import logging
import math
from typing import List, Tuple, Union

import numpy as np
import torch
from transformers import AutoProcessor

from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.qwen_image.qwen_image_autoencoder_kl import AutoencoderKLQwenImage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VAE_SCALE_FACTOR = 8
PATCH_SIZE = 2
VAE_CHANNELS = 16
IN_CHANNELS = VAE_CHANNELS * PATCH_SIZE * PATCH_SIZE  # 64

# Krea2 time-shift parameters (from official sampling.py)
TIME_SHIFT_X1 = (256 / 16) ** 2  # 256
TIME_SHIFT_Y1 = 0.5
TIME_SHIFT_X2 = (1280 / 16) ** 2  # 6400
TIME_SHIFT_Y2 = 1.15
TIME_SHIFT_SIGMA = 1.0
TURBO_MU = 1.15

# Text encoder constants (from official encoder.py)
QWEN3_VL_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
NUM_SELECT_LAYERS = len(SELECT_LAYERS)  # 12
TEXT_HIDDEN_DIM = 2560
MAX_LENGTH = 512
PROMPT_TEMPLATE_ENCODE_START_IDX = 34  # tokens to drop from the front (prefix)
PROMPT_TEMPLATE_ENCODE_SUFFIX_START_IDX = 5  # leading tokens of the separately-tokenized suffix
SYSTEM_PROMPT = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"
# Official encoder.py tokenizes (prefix + user_text) and (suffix) SEPARATELY, padding the
# former to a fixed length and only then concatenating the suffix at the very end. Keep the
# two pieces apart so we can reproduce that scheme exactly.
PROMPT_TEMPLATE_ENCODE_PREFIX = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n"
PROMPT_TEMPLATE_ENCODE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def load_vae(vae_path: str, device: Union[str, torch.device] = "cpu", disable_mmap: bool = False) -> AutoencoderKLQwenImage:
    return qwen_image_utils.load_vae(vae_path, input_channels=3, device=device, disable_mmap=disable_mmap)


def load_text_encoder(
    text_encoder_path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[AutoProcessor, torch.nn.Module]:
    from musubi_tuner.hidream_o1 import hidream_o1_utils

    model = hidream_o1_utils.load_model(
        model_path=text_encoder_path,
        dtype=dtype,
        device=device,
    )
    processor = AutoProcessor.from_pretrained(QWEN3_VL_MODEL_ID)
    return processor, model


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    return qwen_image_utils.pack_latents(latents)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return qwen_image_utils.unpack_latents(latents, height, width, vae_scale_factor=VAE_SCALE_FACTOR)


def prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator):
    num_layers = 1
    return qwen_image_utils.prepare_latents(batch_size, num_layers, num_channels_latents, height, width, dtype, device, generator)


def calculate_shift_krea2(image_seq_len: int) -> float:
    if image_seq_len <= TIME_SHIFT_X1:
        return TIME_SHIFT_Y1
    if image_seq_len >= TIME_SHIFT_X2:
        return TIME_SHIFT_Y2
    m = (TIME_SHIFT_Y2 - TIME_SHIFT_Y1) / (TIME_SHIFT_X2 - TIME_SHIFT_X1)
    b = TIME_SHIFT_Y1 - m * TIME_SHIFT_X1
    return image_seq_len * m + b


def get_timesteps(num_inference_steps: int, image_seq_len: int, device: torch.device, is_turbo: bool = False) -> torch.Tensor:
    ts = np.linspace(1.0, 0.0, num_inference_steps + 1)
    if is_turbo:
        mu = TURBO_MU
    else:
        mu = calculate_shift_krea2(image_seq_len)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** TIME_SHIFT_SIGMA)
    ts = ts[:-1]
    return torch.from_numpy(ts).to(dtype=torch.float32, device=device)


def get_krea2_prompt_embeds(
    processor,
    text_encoder: torch.nn.Module,
    prompt: Union[str, List[str]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts with Qwen3-VL, reproducing official krea-2 encoder.py exactly.

    The prefix (system + user header) + user text is tokenized and padded to a fixed
    ``max_length``, then the suffix (assistant header) is tokenized separately and
    concatenated at the very end. Finally the leading ``PROMPT_TEMPLATE_ENCODE_START_IDX``
    (34) prefix tokens are dropped. Output: hidden ``(B, T, 12, 2560)`` + mask ``(B, T)``.
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    if isinstance(prompt, str):
        prompt = [prompt]

    prefix_idx = PROMPT_TEMPLATE_ENCODE_START_IDX

    # Prefix + user text, padded to a fixed length (suffix is NOT included here).
    texts = [PROMPT_TEMPLATE_ENCODE_PREFIX + p for p in prompt]
    main_max_length = MAX_LENGTH + prefix_idx - PROMPT_TEMPLATE_ENCODE_SUFFIX_START_IDX
    main = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=main_max_length,
        return_tensors="pt",
    )

    # Suffix tokenized separately and appended after the padded main block.
    suffix = tokenizer([PROMPT_TEMPLATE_ENCODE_SUFFIX] * len(texts), return_tensors="pt")

    input_ids = torch.cat([main.input_ids, suffix.input_ids], dim=1).to(device)
    attention_mask = torch.cat([main.attention_mask, suffix.attention_mask], dim=1).to(device)

    with torch.no_grad():
        outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    selected = [hidden_states[i] for i in SELECT_LAYERS]
    stacked = torch.stack(selected, dim=2)  # (B, T, 12, 2560)

    # Drop the leading prefix tokens.
    stacked = stacked[:, prefix_idx:]
    mask = attention_mask[:, prefix_idx:]

    # extract per-sample valid tokens, then pad to the batch max length
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    max_len = valid_lengths.max().item()

    result = []
    result_mask = []
    for b in range(stacked.shape[0]):
        valid = stacked[b, : valid_lengths[b]]
        padded = torch.cat([valid, valid.new_zeros(max_len - valid_lengths[b], NUM_SELECT_LAYERS, TEXT_HIDDEN_DIM)], dim=0)
        result.append(padded)
        m = torch.cat(
            [
                torch.ones(valid_lengths[b], dtype=torch.long, device=device),
                torch.zeros(max_len - valid_lengths[b], dtype=torch.long, device=device),
            ]
        )
        result_mask.append(m)

    hidden = torch.stack(result, dim=0).to(dtype=dtype, device=device)
    attn_mask = torch.stack(result_mask, dim=0)
    return hidden, attn_mask
