import json
from typing import Optional, Union
import logging

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
from accelerate import init_empty_weights

from musubi_tuner.utils.safetensors_utils import load_split_weights

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


ZIMAGE_ID = "Tongyi-MAI/Z-Image-Turbo"


def load_qwen3(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> tuple[AutoTokenizer, Qwen3ForCausalLM]:
    QWEN3_CONFIG_JSON = """
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "initializer_range": 0.02,
  "intermediate_size": 9728,
  "max_position_embeddings": 40960,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
"""
    config = json.loads(QWEN3_CONFIG_JSON)
    config = Qwen3Config(**config)
    with init_empty_weights():
        qwen3 = Qwen3ForCausalLM._from_config(config)

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

    info = qwen3.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded Qwen3: {info}")
    qwen3.to(device)

    if dtype is not None:
        if dtype.itemsize() == 1:  # torch.float8
            # prepare Qwen3 for fp8
            org_dtype = torch.bfloat16  # model weight is fp8 in loading, but original dtype is bfloat16
            logger.info(f"prepare Qwen3 for fp8: set to {dtype} from {org_dtype}")
            qwen3.to(dtype)

            # prepare LLM for fp8
            def prepare_fp8(vl_model: Qwen3ForCausalLM, target_dtype):
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

            prepare_fp8(qwen3, org_dtype)

        else:
            logger.info(f"Setting Qwen3 to dtype: {dtype}")
            qwen3.to(dtype)
    # Load tokenizer
    # TODO change to specific tokenizer class
    logger.info(f"Loading tokenizer from {ZIMAGE_ID}")
    tokenizer = AutoTokenizer.from_pretrained(ZIMAGE_ID, subfolder="tokenizer")
    print(tokenizer)
    return tokenizer, qwen3
