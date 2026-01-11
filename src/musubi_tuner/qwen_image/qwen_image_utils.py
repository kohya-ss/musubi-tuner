import argparse
import json
import logging
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from transformers.image_utils import ImageInput
from accelerate import init_empty_weights
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from musubi_tuner.dataset.image_video_dataset import BucketSelector, ARCHITECTURE_QWEN_IMAGE_EDIT
from musubi_tuner.flux.flux_utils import is_fp8
from musubi_tuner.qwen_image.qwen_image_autoencoder_kl import AutoencoderKLQwenImage
from musubi_tuner.utils import image_utils
from musubi_tuner.utils.safetensors_utils import load_safetensors, load_split_weights

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# region constants

SCHEDULER_BASE_IMAGE_SEQ_LEN = 256
SCHEDULER_BASE_SHIFT = 0.5
SCHEDULER_MAX_IMAGE_SEQ_LEN = 8192
SCHEDULER_MAX_SHIFT = 0.9

VAE_SCALE_FACTOR = 8  # Qwen Image uses 8x compression

# endregion constants

# region text encoder
QWEN_IMAGE_ID = "Qwen/Qwen-Image"
QWEN_IMAGE_EDIT_ID = "Qwen/Qwen-Image-Edit"

GENERATION_CONFIG_JSON = """
{
  "bos_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [
    151645,
    151643
  ],
  "pad_token_id": 151643,
  "repetition_penalty": 1.05,
  "temperature": 0.1,
  "top_k": 1,
  "top_p": 0.001,
  "transformers_version": "4.53.1"
}
"""


def load_qwen2_5_vl(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> tuple[Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration]:
    QWEN2_5_VL_CONFIG_JSON = """
{
  "architectures": [
    "Qwen2_5_VLForConditionalGeneration"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "image_token_id": 151655,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 128000,
  "max_window_layers": 28,
  "model_type": "qwen2_5_vl",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "mrope_section": [
      16,
      24,
      24
    ],
    "rope_type": "default",
    "type": "default"
  },
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "text_config": {
    "architectures": [
      "Qwen2_5_VLForConditionalGeneration"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "image_token_id": null,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "layer_types": [
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention"
    ],
    "max_position_embeddings": 128000,
    "max_window_layers": 28,
    "model_type": "qwen2_5_vl_text",
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
      "mrope_section": [
        16,
        24,
        24
      ],
      "rope_type": "default",
      "type": "default"
    },
    "rope_theta": 1000000.0,
    "sliding_window": null,
    "torch_dtype": "float32",
    "use_cache": true,
    "use_sliding_window": false,
    "video_token_id": null,
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
    "vision_token_id": 151654,
    "vocab_size": 152064
  },
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.53.1",
  "use_cache": true,
  "use_sliding_window": false,
  "video_token_id": 151656,
  "vision_config": {
    "depth": 32,
    "fullatt_block_indexes": [
      7,
      15,
      23,
      31
    ],
    "hidden_act": "silu",
    "hidden_size": 1280,
    "in_channels": 3,
    "in_chans": 3,
    "initializer_range": 0.02,
    "intermediate_size": 3420,
    "model_type": "qwen2_5_vl",
    "num_heads": 16,
    "out_hidden_size": 3584,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "temporal_patch_size": 2,
    "tokens_per_second": 2,
    "torch_dtype": "float32",
    "window_size": 112
  },
  "vision_end_token_id": 151653,
  "vision_start_token_id": 151652,
  "vision_token_id": 151654,
  "vocab_size": 152064
}
"""
    config = json.loads(QWEN2_5_VL_CONFIG_JSON)
    config = Qwen2_5_VLConfig(**config)
    with init_empty_weights():
        qwen2_5_vl = Qwen2_5_VLForConditionalGeneration._from_config(config)

    if state_dict is not None:
        sd = state_dict
    else:
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

    # convert prefixes
    for key in list(sd.keys()):
        if key.startswith("model."):
            new_key = key.replace("model.", "model.language_model.", 1)
        elif key.startswith("visual."):
            new_key = key.replace("visual.", "model.visual.", 1)
        else:
            continue
        if key not in sd:
            logger.warning(f"Key {key} not found in state dict, skipping.")
            continue
        sd[new_key] = sd.pop(key)

    info = qwen2_5_vl.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded Qwen2.5-VL: {info}")
    qwen2_5_vl.to(device)

    if dtype is not None:
        if is_fp8(dtype):
            org_dtype = torch.bfloat16  # model weight is fp8 in loading, but original dtype is bfloat16
            logger.info(f"prepare Qwen2.5-VL for fp8: set to {dtype} from {org_dtype}")
            qwen2_5_vl.to(dtype)

            # prepare LLM for fp8
            def prepare_fp8(vl_model: Qwen2_5_VLForConditionalGeneration, target_dtype):
                def forward_hook(module):
                    def forward(hidden_states):
                        input_dtype = hidden_states.dtype
                        hidden_states = hidden_states.to(torch.float32)
                        variance = hidden_states.pow(2).mean(-1, keepdim=True)
                        hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
                        # return module.weight.to(input_dtype) * hidden_states.to(input_dtype)
                        return (module.weight.to(torch.float32) * hidden_states.to(torch.float32)).to(input_dtype)

                    return forward

                def decoder_forward_hook(module):
                    def forward(
                        hidden_states: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None,
                        position_ids: Optional[torch.LongTensor] = None,
                        past_key_value: Optional[tuple[torch.Tensor]] = None,
                        output_attentions: Optional[bool] = False,
                        use_cache: Optional[bool] = False,
                        cache_position: Optional[torch.LongTensor] = None,
                        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
                        **kwargs,
                    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
                        residual = hidden_states

                        hidden_states = module.input_layernorm(hidden_states)

                        # Self Attention
                        hidden_states, self_attn_weights = module.self_attn(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_value,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            **kwargs,
                        )
                        input_dtype = hidden_states.dtype
                        hidden_states = residual.to(torch.float32) + hidden_states.to(torch.float32)
                        hidden_states = hidden_states.to(input_dtype)

                        # Fully Connected
                        residual = hidden_states
                        hidden_states = module.post_attention_layernorm(hidden_states)
                        hidden_states = module.mlp(hidden_states)
                        hidden_states = residual + hidden_states

                        outputs = (hidden_states,)

                        if output_attentions:
                            outputs += (self_attn_weights,)

                        return outputs

                    return forward

                for module in vl_model.modules():
                    if module.__class__.__name__ in ["Embedding"]:
                        # print("set", module.__class__.__name__, "to", target_dtype)
                        module.to(target_dtype)
                    if module.__class__.__name__ in ["Qwen2RMSNorm"]:
                        # print("set", module.__class__.__name__, "hooks")
                        module.forward = forward_hook(module)
                    if module.__class__.__name__ in ["Qwen2_5_VLDecoderLayer"]:
                        # print("set", module.__class__.__name__, "hooks")
                        module.forward = decoder_forward_hook(module)
                    if module.__class__.__name__ in ["Qwen2_5_VisionRotaryEmbedding"]:
                        # print("set", module.__class__.__name__, "hooks")
                        module.to(target_dtype)

            prepare_fp8(qwen2_5_vl, org_dtype)

        else:
            logger.info(f"Setting Qwen2.5-VL to dtype: {dtype}")
            qwen2_5_vl.to(dtype)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {QWEN_IMAGE_ID}")
    tokenizer = Qwen2Tokenizer.from_pretrained(QWEN_IMAGE_ID, subfolder="tokenizer")
    return tokenizer, qwen2_5_vl


def load_vl_processor() -> Qwen2VLProcessor:
    logger.info(f"Loading VL processor from {QWEN_IMAGE_EDIT_ID}")
    return Qwen2VLProcessor.from_pretrained(QWEN_IMAGE_EDIT_ID, subfolder="processor")


def extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

    return split_result


def get_qwen_prompt_embeds(
    tokenizer: Qwen2Tokenizer, vlm: Qwen2_5_VLForConditionalGeneration, prompt: Union[str, List[str]] = None
):
    tokenizer_max_length = 1024
    prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx = 34
    # default_sample_size = 128

    device = vlm.device
    dtype = vlm.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt

    template = prompt_template_encode
    drop_idx = prompt_template_encode_start_idx
    txt = [template.format(e) for e in prompt]
    txt_tokens = tokenizer(txt, max_length=tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt").to(
        device
    )

    if is_fp8(dtype):
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            encoder_hidden_states = vlm(
                input_ids=txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask, output_hidden_states=True
            )
    else:
        with torch.no_grad():
            encoder_hidden_states = vlm(
                input_ids=txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask, output_hidden_states=True
            )
    hidden_states = encoder_hidden_states.hidden_states[-1]
    if hidden_states.shape[1] > tokenizer_max_length + drop_idx:
        logger.warning(f"Hidden states shape {hidden_states.shape} exceeds max length {tokenizer_max_length + drop_idx}")

    split_hidden_states = extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
    encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds, encoder_attention_mask


def get_qwen_prompt_embeds_with_image(
    vl_processor: Qwen2VLProcessor,
    vlm: Qwen2_5_VLForConditionalGeneration,
    prompt: Union[str, List[str]],
    image: Union[List[ImageInput], ImageInput] = None,
    model_version: str = "edit",
):
    r"""
    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`):
            image to be encoded
        model_version (`str`, *optional*, defaults to "edit"):
            version of the prompt, can be "edit", "edit-2509" or "edit-2511"
    """
    if model_version == "edit":
        prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
    elif model_version == "edit-2509" or model_version == "edit-2511":
        prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx = 64
    # default_sample_size = 128

    device = vlm.device
    dtype = vlm.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if isinstance(image, list):
        if len(image) == 0:
            image = None
        else:
            if isinstance(image[0], list):
                pass  # list of list, it's ok
            else:
                assert len(prompt) == 1, "Image must be a list of list when multiple prompts are provided."
                image = [image]  # wrap to list of list

    elif image is not None:
        image = [[image]]  # wrap to list of list, not necessary, but for consistency

    # RGB conversion
    if image is not None:
        for i in range(len(image)):
            for j in range(len(image[i])):
                img = image[i][j]
                if isinstance(img, np.ndarray):
                    if img.shape[2] == 4:
                        img = img[:, :, :3]
                    image[i][j] = img
                elif isinstance(img, Image.Image):
                    if img.mode == "RGBA":
                        img = img.convert("RGB")
                    image[i][j] = img

    assert image is None or len(image) == len(prompt), (
        f"Number of images {len(image) if image is not None else 0} must match number of prompts {len(prompt)} for batch processing"
    )

    base_img_prompts = [""] * len(prompt)
    if image is not None:
        vl_image_inputs = []  # flat list of images
        if model_version == "edit":
            for i, img in enumerate(image):
                if img is None or len(img) == 0:
                    logger.warning(f"No image provided for prompt {i}, but version is {model_version}, this may cause issues.")
                    continue
                if len(img) > 1:
                    logger.warning(
                        f"Multiple images {len(img)} provided for prompt {i}, but version is {model_version}, 2nd and later images will be ignored."
                    )
                vl_image_inputs.append(img[0])
        else:
            img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
            for i, img in enumerate(image):
                if img is None or len(img) == 0:
                    continue
                for j in range(len(img)):
                    base_img_prompts[i] += img_prompt_template.format(j + 1)
                vl_image_inputs.extend(img)
    else:
        vl_image_inputs = None

    template = prompt_template_encode
    drop_idx = prompt_template_encode_start_idx
    txt = [template.format(base + e) for base, e in zip(base_img_prompts, prompt)]

    model_inputs = vl_processor(text=txt, images=vl_image_inputs, padding=True, return_tensors="pt").to(device)

    if is_fp8(dtype):
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            encoder_hidden_states = vlm(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )
    else:
        with torch.no_grad():
            encoder_hidden_states = vlm(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values if vl_image_inputs is not None else None,
                image_grid_thw=model_inputs.image_grid_thw if vl_image_inputs is not None else None,
                output_hidden_states=True,
            )

    hidden_states = encoder_hidden_states.hidden_states[-1]
    # if hidden_states.shape[1] > tokenizer_max_length + drop_idx:
    #     logger.warning(f"Hidden states shape {hidden_states.shape} exceeds max length {tokenizer_max_length + drop_idx}")

    split_hidden_states = extract_masked_hidden(hidden_states, model_inputs.attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
    encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds, encoder_attention_mask


def get_image_caption(
    vl_processor: Qwen2VLProcessor,
    vlm: Qwen2_5_VLForConditionalGeneration,
    prompt_image: Union[List[ImageInput], ImageInput] = None,
    use_en_prompt: bool = True,
) -> str:
    image_caption_prompt_cn = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n# 图像标注器\n你是一个专业的图像标注器。请基于输入图像，撰写图注:\n1.
使用自然、描述性的语言撰写图注，不要使用结构化形式或富文本形式。\n2. 通过加入以下内容，丰富图注细节：\n - 对象的属性：如数量、颜色、形状、大小、位置、材质、状态、动作等\n -
对象间的视觉关系：如空间关系、功能关系、动作关系、从属关系、比较关系、因果关系等\n - 环境细节：例如天气、光照、颜色、纹理、气氛等\n - 文字内容：识别图像中清晰可见的文字，不做翻译和解释，用引号在图注中强调\n3.
保持真实性与准确性：\n - 不要使用笼统的描述\n -
描述图像中所有可见的信息，但不要加入没有在图像中出现的内容\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"""
    image_caption_prompt_en = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n# Image Annotator\nYou are a professional
image annotator. Please write an image caption based on the input image:\n1. Write the caption using natural,
descriptive language without structured formats or rich text.\n2. Enrich caption details by including: \n - Object
attributes, such as quantity, color, shape, size, material, state, position, actions, and so on\n - Vision Relations
between objects, such as spatial relations, functional relations, possessive relations, attachment relations, action
relations, comparative relations, causal relations, and so on\n - Environmental details, such as weather, lighting,
colors, textures, atmosphere, and so on\n - Identify the text clearly visible in the image, without translation or
explanation, and highlight it in the caption with quotation marks\n3. Maintain authenticity and accuracy:\n - Avoid
generalizations\n - Describe all visible information in the image, while do not add information not explicitly shown in
the image\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"""

    if use_en_prompt:
        prompt = image_caption_prompt_en
    else:
        prompt = image_caption_prompt_cn

    # Remove alpha channel if present
    if isinstance(prompt_image, list) and isinstance(prompt_image[0], np.ndarray):
        prompt_image = [img[:, :, :3] if img.shape[2] == 4 else img for img in prompt_image]
    elif isinstance(prompt_image, np.ndarray):
        if prompt_image.shape[2] == 4:
            prompt_image = prompt_image[:, :, :3]

    model_inputs = vl_processor(
        text=prompt,
        images=prompt_image,
        padding=True,
        return_tensors="pt",
    ).to(vlm.device)
    generated_ids = vlm.generate(**model_inputs, max_new_tokens=512)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
    output_text = vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text.strip()


"""
def encode_prompt(
    vlm: Qwen2_5_VLForConditionalGeneration,
    prompt: Union[str, List[str]],
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
):
    r""

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
    ""
    # max_sequence_length: int = 1024,
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds, prompt_embeds_mask = get_qwen_prompt_embeds(vlm, prompt)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)
    prompt_embeds_mask = prompt_embeds_mask.view(batch_size, seq_len)

    return prompt_embeds, prompt_embeds_mask
"""

# endregion text encoder

# region vae and latents


# Convert ComfyUI keys to standard keys if necessary
def convert_comfyui_state_dict(sd):
    if "conv1.bias" not in sd:
        return sd

    # Key mapping from ComfyUI VAE to official VAE, auto-generated by a script
    key_map = {
        "conv1": "quant_conv",
        "conv2": "post_quant_conv",
        "decoder.conv1": "decoder.conv_in",
        "decoder.head.0": "decoder.norm_out",
        "decoder.head.2": "decoder.conv_out",
        "decoder.middle.0.residual.0": "decoder.mid_block.resnets.0.norm1",
        "decoder.middle.0.residual.2": "decoder.mid_block.resnets.0.conv1",
        "decoder.middle.0.residual.3": "decoder.mid_block.resnets.0.norm2",
        "decoder.middle.0.residual.6": "decoder.mid_block.resnets.0.conv2",
        "decoder.middle.1.norm": "decoder.mid_block.attentions.0.norm",
        "decoder.middle.1.proj": "decoder.mid_block.attentions.0.proj",
        "decoder.middle.1.to_qkv": "decoder.mid_block.attentions.0.to_qkv",
        "decoder.middle.2.residual.0": "decoder.mid_block.resnets.1.norm1",
        "decoder.middle.2.residual.2": "decoder.mid_block.resnets.1.conv1",
        "decoder.middle.2.residual.3": "decoder.mid_block.resnets.1.norm2",
        "decoder.middle.2.residual.6": "decoder.mid_block.resnets.1.conv2",
        "decoder.upsamples.0.residual.0": "decoder.up_blocks.0.resnets.0.norm1",
        "decoder.upsamples.0.residual.2": "decoder.up_blocks.0.resnets.0.conv1",
        "decoder.upsamples.0.residual.3": "decoder.up_blocks.0.resnets.0.norm2",
        "decoder.upsamples.0.residual.6": "decoder.up_blocks.0.resnets.0.conv2",
        "decoder.upsamples.1.residual.0": "decoder.up_blocks.0.resnets.1.norm1",
        "decoder.upsamples.1.residual.2": "decoder.up_blocks.0.resnets.1.conv1",
        "decoder.upsamples.1.residual.3": "decoder.up_blocks.0.resnets.1.norm2",
        "decoder.upsamples.1.residual.6": "decoder.up_blocks.0.resnets.1.conv2",
        "decoder.upsamples.10.residual.0": "decoder.up_blocks.2.resnets.2.norm1",
        "decoder.upsamples.10.residual.2": "decoder.up_blocks.2.resnets.2.conv1",
        "decoder.upsamples.10.residual.3": "decoder.up_blocks.2.resnets.2.norm2",
        "decoder.upsamples.10.residual.6": "decoder.up_blocks.2.resnets.2.conv2",
        "decoder.upsamples.11.resample.1": "decoder.up_blocks.2.upsamplers.0.resample.1",
        "decoder.upsamples.12.residual.0": "decoder.up_blocks.3.resnets.0.norm1",
        "decoder.upsamples.12.residual.2": "decoder.up_blocks.3.resnets.0.conv1",
        "decoder.upsamples.12.residual.3": "decoder.up_blocks.3.resnets.0.norm2",
        "decoder.upsamples.12.residual.6": "decoder.up_blocks.3.resnets.0.conv2",
        "decoder.upsamples.13.residual.0": "decoder.up_blocks.3.resnets.1.norm1",
        "decoder.upsamples.13.residual.2": "decoder.up_blocks.3.resnets.1.conv1",
        "decoder.upsamples.13.residual.3": "decoder.up_blocks.3.resnets.1.norm2",
        "decoder.upsamples.13.residual.6": "decoder.up_blocks.3.resnets.1.conv2",
        "decoder.upsamples.14.residual.0": "decoder.up_blocks.3.resnets.2.norm1",
        "decoder.upsamples.14.residual.2": "decoder.up_blocks.3.resnets.2.conv1",
        "decoder.upsamples.14.residual.3": "decoder.up_blocks.3.resnets.2.norm2",
        "decoder.upsamples.14.residual.6": "decoder.up_blocks.3.resnets.2.conv2",
        "decoder.upsamples.2.residual.0": "decoder.up_blocks.0.resnets.2.norm1",
        "decoder.upsamples.2.residual.2": "decoder.up_blocks.0.resnets.2.conv1",
        "decoder.upsamples.2.residual.3": "decoder.up_blocks.0.resnets.2.norm2",
        "decoder.upsamples.2.residual.6": "decoder.up_blocks.0.resnets.2.conv2",
        "decoder.upsamples.3.resample.1": "decoder.up_blocks.0.upsamplers.0.resample.1",
        "decoder.upsamples.3.time_conv": "decoder.up_blocks.0.upsamplers.0.time_conv",
        "decoder.upsamples.4.residual.0": "decoder.up_blocks.1.resnets.0.norm1",
        "decoder.upsamples.4.residual.2": "decoder.up_blocks.1.resnets.0.conv1",
        "decoder.upsamples.4.residual.3": "decoder.up_blocks.1.resnets.0.norm2",
        "decoder.upsamples.4.residual.6": "decoder.up_blocks.1.resnets.0.conv2",
        "decoder.upsamples.4.shortcut": "decoder.up_blocks.1.resnets.0.conv_shortcut",
        "decoder.upsamples.5.residual.0": "decoder.up_blocks.1.resnets.1.norm1",
        "decoder.upsamples.5.residual.2": "decoder.up_blocks.1.resnets.1.conv1",
        "decoder.upsamples.5.residual.3": "decoder.up_blocks.1.resnets.1.norm2",
        "decoder.upsamples.5.residual.6": "decoder.up_blocks.1.resnets.1.conv2",
        "decoder.upsamples.6.residual.0": "decoder.up_blocks.1.resnets.2.norm1",
        "decoder.upsamples.6.residual.2": "decoder.up_blocks.1.resnets.2.conv1",
        "decoder.upsamples.6.residual.3": "decoder.up_blocks.1.resnets.2.norm2",
        "decoder.upsamples.6.residual.6": "decoder.up_blocks.1.resnets.2.conv2",
        "decoder.upsamples.7.resample.1": "decoder.up_blocks.1.upsamplers.0.resample.1",
        "decoder.upsamples.7.time_conv": "decoder.up_blocks.1.upsamplers.0.time_conv",
        "decoder.upsamples.8.residual.0": "decoder.up_blocks.2.resnets.0.norm1",
        "decoder.upsamples.8.residual.2": "decoder.up_blocks.2.resnets.0.conv1",
        "decoder.upsamples.8.residual.3": "decoder.up_blocks.2.resnets.0.norm2",
        "decoder.upsamples.8.residual.6": "decoder.up_blocks.2.resnets.0.conv2",
        "decoder.upsamples.9.residual.0": "decoder.up_blocks.2.resnets.1.norm1",
        "decoder.upsamples.9.residual.2": "decoder.up_blocks.2.resnets.1.conv1",
        "decoder.upsamples.9.residual.3": "decoder.up_blocks.2.resnets.1.norm2",
        "decoder.upsamples.9.residual.6": "decoder.up_blocks.2.resnets.1.conv2",
        "encoder.conv1": "encoder.conv_in",
        "encoder.downsamples.0.residual.0": "encoder.down_blocks.0.norm1",
        "encoder.downsamples.0.residual.2": "encoder.down_blocks.0.conv1",
        "encoder.downsamples.0.residual.3": "encoder.down_blocks.0.norm2",
        "encoder.downsamples.0.residual.6": "encoder.down_blocks.0.conv2",
        "encoder.downsamples.1.residual.0": "encoder.down_blocks.1.norm1",
        "encoder.downsamples.1.residual.2": "encoder.down_blocks.1.conv1",
        "encoder.downsamples.1.residual.3": "encoder.down_blocks.1.norm2",
        "encoder.downsamples.1.residual.6": "encoder.down_blocks.1.conv2",
        "encoder.downsamples.10.residual.0": "encoder.down_blocks.10.norm1",
        "encoder.downsamples.10.residual.2": "encoder.down_blocks.10.conv1",
        "encoder.downsamples.10.residual.3": "encoder.down_blocks.10.norm2",
        "encoder.downsamples.10.residual.6": "encoder.down_blocks.10.conv2",
        "encoder.downsamples.2.resample.1": "encoder.down_blocks.2.resample.1",
        "encoder.downsamples.3.residual.0": "encoder.down_blocks.3.norm1",
        "encoder.downsamples.3.residual.2": "encoder.down_blocks.3.conv1",
        "encoder.downsamples.3.residual.3": "encoder.down_blocks.3.norm2",
        "encoder.downsamples.3.residual.6": "encoder.down_blocks.3.conv2",
        "encoder.downsamples.3.shortcut": "encoder.down_blocks.3.conv_shortcut",
        "encoder.downsamples.4.residual.0": "encoder.down_blocks.4.norm1",
        "encoder.downsamples.4.residual.2": "encoder.down_blocks.4.conv1",
        "encoder.downsamples.4.residual.3": "encoder.down_blocks.4.norm2",
        "encoder.downsamples.4.residual.6": "encoder.down_blocks.4.conv2",
        "encoder.downsamples.5.resample.1": "encoder.down_blocks.5.resample.1",
        "encoder.downsamples.5.time_conv": "encoder.down_blocks.5.time_conv",
        "encoder.downsamples.6.residual.0": "encoder.down_blocks.6.norm1",
        "encoder.downsamples.6.residual.2": "encoder.down_blocks.6.conv1",
        "encoder.downsamples.6.residual.3": "encoder.down_blocks.6.norm2",
        "encoder.downsamples.6.residual.6": "encoder.down_blocks.6.conv2",
        "encoder.downsamples.6.shortcut": "encoder.down_blocks.6.conv_shortcut",
        "encoder.downsamples.7.residual.0": "encoder.down_blocks.7.norm1",
        "encoder.downsamples.7.residual.2": "encoder.down_blocks.7.conv1",
        "encoder.downsamples.7.residual.3": "encoder.down_blocks.7.norm2",
        "encoder.downsamples.7.residual.6": "encoder.down_blocks.7.conv2",
        "encoder.downsamples.8.resample.1": "encoder.down_blocks.8.resample.1",
        "encoder.downsamples.8.time_conv": "encoder.down_blocks.8.time_conv",
        "encoder.downsamples.9.residual.0": "encoder.down_blocks.9.norm1",
        "encoder.downsamples.9.residual.2": "encoder.down_blocks.9.conv1",
        "encoder.downsamples.9.residual.3": "encoder.down_blocks.9.norm2",
        "encoder.downsamples.9.residual.6": "encoder.down_blocks.9.conv2",
        "encoder.head.0": "encoder.norm_out",
        "encoder.head.2": "encoder.conv_out",
        "encoder.middle.0.residual.0": "encoder.mid_block.resnets.0.norm1",
        "encoder.middle.0.residual.2": "encoder.mid_block.resnets.0.conv1",
        "encoder.middle.0.residual.3": "encoder.mid_block.resnets.0.norm2",
        "encoder.middle.0.residual.6": "encoder.mid_block.resnets.0.conv2",
        "encoder.middle.1.norm": "encoder.mid_block.attentions.0.norm",
        "encoder.middle.1.proj": "encoder.mid_block.attentions.0.proj",
        "encoder.middle.1.to_qkv": "encoder.mid_block.attentions.0.to_qkv",
        "encoder.middle.2.residual.0": "encoder.mid_block.resnets.1.norm1",
        "encoder.middle.2.residual.2": "encoder.mid_block.resnets.1.conv1",
        "encoder.middle.2.residual.3": "encoder.mid_block.resnets.1.norm2",
        "encoder.middle.2.residual.6": "encoder.mid_block.resnets.1.conv2",
    }

    new_state_dict = {}
    for key in sd.keys():
        new_key = key
        key_without_suffix = key.rsplit(".", 1)[0]
        if key_without_suffix in key_map:
            new_key = key.replace(key_without_suffix, key_map[key_without_suffix])
        new_state_dict[new_key] = sd[key]

    logger.info("Converted ComfyUI AutoencoderKL state dict keys to official format")
    return new_state_dict


def load_vae(
    vae_path: str, input_channels: int = 3, device: Union[str, torch.device] = "cpu", disable_mmap: bool = False
) -> AutoencoderKLQwenImage:
    """Load VAE from a given path."""
    VAE_CONFIG_JSON = """
{
  "_class_name": "AutoencoderKLQwenImage",
  "_diffusers_version": "0.34.0.dev0",
  "attn_scales": [],
  "base_dim": 96,
  "dim_mult": [
    1,
    2,
    4,
    4
  ],
  "dropout": 0.0,
  "latents_mean": [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921
  ],
  "latents_std": [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.916
  ],
  "num_res_blocks": 2,
  "temperal_downsample": [
    false,
    true,
    true
  ],
  "z_dim": 16
}
"""
    logger.info("Initializing VAE")
    config = json.loads(VAE_CONFIG_JSON)
    vae = AutoencoderKLQwenImage(
        base_dim=config["base_dim"],
        z_dim=config["z_dim"],
        dim_mult=config["dim_mult"],
        num_res_blocks=config["num_res_blocks"],
        attn_scales=config["attn_scales"],
        temperal_downsample=config["temperal_downsample"],
        dropout=config["dropout"],
        latents_mean=config["latents_mean"],
        latents_std=config["latents_std"],
        input_channels=input_channels,
    )

    logger.info(f"Loading VAE from {vae_path}")
    state_dict = load_safetensors(vae_path, device=device, disable_mmap=disable_mmap)

    # Convert ComfyUI VAE keys to official VAE keys
    state_dict = convert_comfyui_state_dict(state_dict)

    info = vae.load_state_dict(state_dict, strict=True, assign=True)
    logger.info(f"Loaded VAE: {info}")

    vae.to(device)
    return vae


def unpack_latents(latents, height, width, vae_scale_factor=VAE_SCALE_FACTOR) -> torch.Tensor:
    """
    Returns layered (B, L, C, H, W) or single frame (B, C, 1, H, W) latents from (B, N, C) packed latents,
    where L is number of layers, N = (H/2)*(W/2)*L.
    """
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    num_layers = num_patches // ((height // 2) * (width // 2))

    latents = latents.view(batch_size, num_layers, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 1, 4, 2, 5, 3, 6)
    latents = latents.reshape(batch_size, num_layers, channels // (2 * 2), height, width)
    if num_layers == 1:
        latents = latents.permute(0, 2, 1, 3, 4)  # (B, C, 1, H, W)
    return latents


# not used in the current implementation, but kept for reference
# def prepare_latent_image_ids(batch_size, height, width, device, dtype):
#     latent_image_ids = torch.zeros(height, width, 3)
#     latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
#     latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
#     latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
#     latent_image_ids = latent_image_ids.reshape(latent_image_id_height * latent_image_id_width, latent_image_id_channels)
#     return latent_image_ids.to(device=device, dtype=dtype)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    This function handles layered (B, L, C, H, W), single frame (B, C, 1, H, W) and normal (B, C, H, W) latents. So the logic is a bit weird.
    If latents have 4 dimensions or the 3rd dimension is 1, it assumes it's single frame or normal latents.
    It packs the latents into a shape of (B, H/2, W/2, C, 2, 2) and then reshapes it to (B, H/2 * W/2, C*4) = (B, Seq, In-Channels)
    If latents have 5 dimensions and the 3rd dimension is not 1, it assumes it's layered latents.
    It packs the latents into a shape of (B, L, H/2, W/2, C, 2, 2) and then reshapes it to (B, L * H/2 * W/2, C*4) = (B, Seq, In-Channels)
    """
    batch_size = latents.shape[0]
    if latents.ndim == 4 or latents.shape[2] == 1:
        # single frame or normal latents
        num_channels_latents = latents.shape[1]
        height = latents.shape[-2]
        width = latents.shape[-1]

        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    else:
        # layered latents: if num_layers == 1, it's equivalent to single frame latents
        num_layers = latents.shape[1]
        num_channels_latents = latents.shape[2]
        height = latents.shape[-2]
        width = latents.shape[-1]

        latents = latents.view(batch_size, num_layers, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4, 6)
        latents = latents.reshape(batch_size, num_layers * (height // 2) * (width // 2), num_channels_latents * 4)

    return latents


def prepare_latents(batch_size, num_layers, num_channels_latents, height, width, dtype, device, generator):
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    vae_scale_factor = VAE_SCALE_FACTOR
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    shape = (batch_size, num_layers, num_channels_latents, height, width)

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = pack_latents(latents)
    return latents


CONDITION_IMAGE_RESOLUTION = (384, 384)
VAE_IMAGE_RESOLUTION = (1024, 1024)


def preprocess_control_image(
    control_image_path: str, resize_to_prefered: bool = True, resize_size: Optional[Tuple[int, int]] = None
) -> tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess the control image for the model. See `preprocess_image` for details.

    Args:
        control_image_path (str): Path to the control image.
        resize_to_prefered (bool): Whether to resize the image to the preferred resolution (based on the model's requirements).
        resize_size (Optional[Tuple[int, int]]): Override target size for resizing if resize_to_prefered is False, with (width, height).

    Returns:
        Tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]: A tuple containing:
            - control_image_tensor (torch.Tensor): The preprocessed control image tensor for the model. NCHW format.
            - control_image_np (np.ndarray): The preprocessed control image as a NumPy array for conditioning. HWC format.
            - None: Placeholder for compatibility (no additional data returned).
    """
    # See:
    # https://github.com/huggingface/diffusers/pull/12188
    # https://github.com/huggingface/diffusers/pull/12190

    control_image = Image.open(control_image_path)

    if resize_to_prefered or resize_size is None:
        resolution = VAE_IMAGE_RESOLUTION if resize_to_prefered else control_image.size
        resize_size = BucketSelector.calculate_bucket_resolution(
            control_image.size, resolution, architecture=ARCHITECTURE_QWEN_IMAGE_EDIT
        )

        cond_resolution = CONDITION_IMAGE_RESOLUTION if resize_to_prefered else control_image.size
        cond_resize_size = BucketSelector.calculate_bucket_resolution(
            control_image.size, cond_resolution, architecture=ARCHITECTURE_QWEN_IMAGE_EDIT
        )
    else:
        cond_resize_size = resize_size

    control_image_tensor, _, _ = image_utils.preprocess_image(control_image, *resize_size, handle_alpha=True)
    _, control_image_np, _ = image_utils.preprocess_image(control_image, *cond_resize_size, handle_alpha=True)
    return control_image_tensor, control_image_np, None


# endregion vae and latents

# region scheduler


def calculate_shift_qwen_image(
    image_seq_len: int,
    base_seq_len: int = SCHEDULER_BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = SCHEDULER_MAX_IMAGE_SEQ_LEN,
    base_shift: float = SCHEDULER_BASE_SHIFT,
    max_shift: float = SCHEDULER_MAX_SHIFT,
) -> float:
    return calculate_shift(
        image_seq_len, base_seq_len=base_seq_len, max_seq_len=max_seq_len, base_shift=base_shift, max_shift=max_shift
    )


def calculate_shift(
    image_seq_len: int, base_seq_len: int = 256, max_seq_len: int = 4096, base_shift: float = 0.5, max_shift: float = 1.15
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Following code is minimal implementation of retrieve_timesteps
# def retrieve_timesteps(
#     sigmas: np.ndarray, device: torch.device, mu: Optional[float] = None, shift: Optional[float] = None
# ) -> tuple[np.ndarray, int]:
#     # Copy from FlowMatchDiscreteScheduler in Diffusers
#     def _time_shift_exponential(mu, sigma, t):
#         return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
#     def _time_shift_linear(mu, sigma, t):
#         return mu / (mu + (1 / t - 1) ** sigma)
#     if mu is not None:
#         sigmas = _time_shift_exponential(mu, 1.0, sigmas)
#     elif shift is not None:
#         sigmas = _time_shift_linear(shift, 1.0, sigmas)
#     else:
#         pass  # sigmas is already in the correct form
#     sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
#     timesteps = sigmas * 1000  # num_train_timesteps
#     return timesteps, len(timesteps)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        # accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        accepts_timesteps = True
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        # accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        accept_sigmas = True
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copy from diffusers.schedulers.scheduling_flow_match_discrete.FlowMatchDiscreteScheduler


class FlowMatchEulerDiscreteScheduler:  # SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        use_dynamic_shifting (`bool`, defaults to False):
            Whether to apply timestep shifting on-the-fly based on the image resolution.
        base_shift (`float`, defaults to 0.5):
            Value to stabilize image generation. Increasing `base_shift` reduces variation and image is more consistent
            with desired output.
        max_shift (`float`, defaults to 1.15):
            Value change allowed to latent vectors. Increasing `max_shift` encourages more variation and image may be
            more exaggerated or stylized.
        base_image_seq_len (`int`, defaults to 256):
            The base image sequence length.
        max_image_seq_len (`int`, defaults to 4096):
            The maximum image sequence length.
        invert_sigmas (`bool`, defaults to False):
            Whether to invert the sigmas.
        shift_terminal (`float`, defaults to None):
            The end value of the shifted timestep schedule.
        use_karras_sigmas (`bool`, defaults to False):
            Whether to use Karras sigmas for step sizes in the noise schedule during sampling.
        use_exponential_sigmas (`bool`, defaults to False):
            Whether to use exponential sigmas for step sizes in the noise schedule during sampling.
        use_beta_sigmas (`bool`, defaults to False):
            Whether to use beta sigmas for step sizes in the noise schedule during sampling.
        time_shift_type (`str`, defaults to "exponential"):
            The type of dynamic resolution-dependent timestep shifting to apply. Either "exponential" or "linear".
        stochastic_sampling (`bool`, defaults to False):
            Whether to use stochastic sampling.
    """

    _compatibles = []
    order = 1

    # @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
    ):
        assert not use_beta_sigmas, "Beta sigmas are not supported in this minimal implementation."
        assert not use_karras_sigmas, "Karras sigmas are not supported in this minimal implementation."
        assert not use_exponential_sigmas, "Exponential sigmas are not supported in this minimal implementation."
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")

        self.num_train_timesteps = num_train_timesteps
        self.use_dynamic_shifting = use_dynamic_shifting
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.invert_sigmas = invert_sigmas
        self.shift_terminal = shift_terminal
        self.use_karras_sigmas = use_karras_sigmas
        self.use_exponential_sigmas = use_exponential_sigmas
        self.use_beta_sigmas = use_beta_sigmas
        self.time_shift_type = time_shift_type
        self.stochastic_sampling = stochastic_sampling

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self._shift = shift

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def shift(self):
        """
        The value used for shifting.
        """
        return self._shift

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_shift(self, shift: float):
        self._shift = shift

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        if self.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        elif self.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)

    def stretch_shift_to_terminal(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Stretches and shifts the timestep schedule to ensure it terminates at the configured `shift_terminal` config
        value.

        Reference:
        https://github.com/Lightricks/LTX-Video/blob/a01a171f8fe3d99dce2728d60a73fecf4d4238ae/ltx_video/schedulers/rf.py#L51

        Args:
            t (`torch.Tensor`):
                A tensor of timesteps to be stretched and shifted.

        Returns:
            `torch.Tensor`:
                A tensor of adjusted timesteps such that the final value equals `self.shift_terminal`.
        """
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
        stretched_t = 1 - (one_minus_z / scale_factor)
        return stretched_t

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[List[float]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            sigmas (`List[float]`, *optional*):
                Custom values for sigmas to be used for each diffusion step. If `None`, the sigmas are computed
                automatically.
            mu (`float`, *optional*):
                Determines the amount of shifting applied to sigmas when performing resolution-dependent timestep
                shifting.
            timesteps (`List[float]`, *optional*):
                Custom values for timesteps to be used for each diffusion step. If `None`, the timesteps are computed
                automatically.
        """
        if self.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")

        if sigmas is not None and timesteps is not None:
            if len(sigmas) != len(timesteps):
                raise ValueError("`sigmas` and `timesteps` should have the same length")

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length as num_inference_steps, if `num_inference_steps` is provided"
                )
        else:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)

        self.num_inference_steps = num_inference_steps

        # 1. Prepare default sigmas
        is_timesteps_provided = timesteps is not None

        if is_timesteps_provided:
            timesteps = np.array(timesteps).astype(np.float32)

        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps)
            sigmas = timesteps / self.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)

        # 2. Perform timestep shifting. Either no shifting is applied, or resolution-dependent shifting of
        #    "exponential" or "linear" type is applied
        if self.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        # 3. If required, stretch the sigmas schedule to terminate at the configured `shift_terminal` value
        if self.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        # # 4. If required, convert sigmas to one of karras, exponential, or beta sigma schedules
        # if self.use_karras_sigmas:
        #     sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        # elif self.use_exponential_sigmas:
        #     sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        # elif self.use_beta_sigmas:
        #     sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        # 5. Convert sigmas and timesteps to tensors and move to specified device
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        if not is_timesteps_provided:
            timesteps = sigmas * self.num_train_timesteps
        else:
            timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)

        # 6. Append the terminal sigma value.
        #    If a model requires inverted sigma schedule for denoising but timesteps without inversion, the
        #    `invert_sigmas` flag can be set to `True`. This case is only required in Mochi
        if self.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        else:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = timesteps
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            per_token_timesteps (`torch.Tensor`, *optional*):
                The timesteps for each token in the sample.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """

        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.num_train_timesteps

            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(dim=0)

            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            sigma_idx = self.step_index
            sigma = self.sigmas[sigma_idx]
            sigma_next = self.sigmas[sigma_idx + 1]

            current_sigma = sigma
            next_sigma = sigma_next
            dt = sigma_next - sigma

        if self.stochastic_sampling:
            x0 = sample - current_sigma * model_output
            noise = torch.randn_like(sample)
            prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
        else:
            prev_sample = sample + dt * model_output

        # upon completion increase step index by one
        self._step_index += 1
        if per_token_timesteps is None:
            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.to(model_output.dtype)

        # if not return_dict:
        return (prev_sample,)

        # return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def _time_shift_exponential(self, mu, sigma, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(self, mu, sigma, t):
        return mu / (mu + (1 / t - 1) ** sigma)

    def __len__(self):
        return self.num_train_timesteps


def get_scheduler(shift: Optional[float] = None) -> FlowMatchEulerDiscreteScheduler:
    SCHEDULER_CONFIG_JSON = """
{
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "_diffusers_version": "0.34.0.dev0",
  "base_image_seq_len": 256,
  "base_shift": 0.5,
  "invert_sigmas": false,
  "max_image_seq_len": 8192,
  "max_shift": 0.9,
  "num_train_timesteps": 1000,
  "shift": 1.0,
  "shift_terminal": 0.02,
  "stochastic_sampling": false,
  "time_shift_type": "exponential",
  "use_beta_sigmas": false,
  "use_dynamic_shifting": true,
  "use_exponential_sigmas": false,
  "use_karras_sigmas": false
}
"""
    config = json.loads(SCHEDULER_CONFIG_JSON)

    scheduler = FlowMatchEulerDiscreteScheduler(
        config["num_train_timesteps"],
        shift=config["shift"] if shift is None else shift,
        # use_dynamic_shifting=config["use_dynamic_shifting"],
        use_dynamic_shifting=True if shift is None else False,
        base_shift=config["base_shift"],
        max_shift=config["max_shift"],
        base_image_seq_len=config["base_image_seq_len"],
        max_image_seq_len=config["max_image_seq_len"],
        invert_sigmas=config["invert_sigmas"],
        shift_terminal=config["shift_terminal"],
        stochastic_sampling=config["stochastic_sampling"],
        time_shift_type="exponential" if shift is None else "linear",
        use_beta_sigmas=config["use_beta_sigmas"],
        use_exponential_sigmas=config["use_exponential_sigmas"],
        use_karras_sigmas=config["use_karras_sigmas"],
    )
    return scheduler


# endregion scheduler

# region model utils


def add_model_version_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--edit", action="store_true", help="Enable Qwen-Image-Edit original, recommend `--model_version edit` instead"
    )
    parser.add_argument(
        "--edit_plus", action="store_true", help="Enable Qwen-Image-Edit-2509 (plus), recommend `--model_version edit-2509` instead"
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default=None,
        help="training for Qwen-Image model version, e.g., 'original', 'layered', 'edit', 'edit-2509', 'edit-2511' etc.",
    )


def resolve_model_version_args(args: argparse.Namespace) -> str:
    if args.model_version is not None:
        args.model_version = args.model_version.lower()
    elif getattr(args, "edit_plus", False):
        args.model_version = "edit-2509"
    elif getattr(args, "edit", False):
        args.model_version = "edit"
    else:
        args.model_version = "original"  # Not specified, use original (non-edit) model

    valid_model_versions = {"original", "layered", "edit", "edit-2509", "edit-2511"}
    if args.model_version not in valid_model_versions:
        valid_str = "', '".join(sorted(valid_model_versions))
        raise ValueError(f"Invalid model_version '{args.model_version}'. Valid options are: '{valid_str}'.")

    args.is_edit = args.model_version in {"edit", "edit-2509", "edit-2511"}
    args.is_layered = args.model_version == "layered"
    return args.model_version


# endregion model utils
