# Original work: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
# Re-implemented for license compliance for sd-scripts.

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights

import logging

from musubi_tuner.modules.attention import AttentionParams
from musubi_tuner.modules.custom_offloading_utils import ModelOffloader
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, WeightTransformHooks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


from musubi_tuner.hunyuan_video_1_5.hunyuan_video_1_5_modules import (
    PatchEmbed,
    SingleTokenRefiner,
    ByT5Mapper,
    TimestepEmbedder,
    MMDoubleStreamBlock,
    FinalLayer,
    VisionProjection,
)
from musubi_tuner.hunyuan_video_1_5.hunyuan_video_1_5_utils import get_nd_rotary_pos_embed

FP8_OPTIMIZATION_TARGET_KEYS = ["double_blocks"]
# FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "_mod", "_emb"]  # , "modulation"
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "_emb"]  # , "modulation", "_mod"


# region DiT Model
class HunyuanVideo_1_5_DiffusionTransformer(nn.Module):
    """
    HunyuanVideo-1.5 Diffusion Transformer.

    A multimodal transformer for image generation with text conditioning,
    featuring separate double-stream and single-stream processing blocks.

    Args:
        attn_mode: Attention implementation mode.
    """

    def __init__(self, task_type: str = "t2v", attn_mode: str = "torch", split_attn: bool = False):
        super().__init__()
        assert task_type in ["t2v", "i2v"], f"Unsupported task type: {task_type}"
        self.task_type = task_type

        # Fixed architecture parameters for HunyuanVideo-1.5
        self.patch_size = [1, 1, 1]  # 1x1 patch size (no spatial downsampling)
        self.in_channels = 32  # Input latent channels
        self.out_channels = 32  # Output latent channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = False  # Guidance embedding disabled
        self.rope_dim_list = [16, 56, 56]  # RoPE dimensions for 2D positional encoding
        self.rope_theta = 256  # RoPE frequency scaling
        self.use_attention_mask = True
        self.vision_projection = "linear"
        self.vision_states_dim = 1152
        self.text_projection = "single_refiner"
        self.hidden_size = 2048  # Model dimension
        self.heads_num = 16  # Number of attention heads

        # Architecture configuration
        mm_double_blocks_depth = 54  # Double-stream transformer blocks
        # mm_single_blocks_depth = 0  # Single-stream transformer blocks
        mlp_width_ratio = 4  # MLP expansion ratio
        text_states_dim = 3584  # Text encoder output dimension
        guidance_embed = False  # No guidance embedding

        # Layer configuration
        mlp_act_type: str = "gelu_tanh"  # MLP activation function
        qkv_bias: bool = True  # Use bias in QKV projections
        qk_norm: bool = True  # Apply QK normalization
        qk_norm_type: str = "rms"  # RMS normalization type

        self.attn_mode = attn_mode
        self.split_attn = split_attn

        # ByT5 character-level text encoder mapping
        self.byt5_in = ByT5Mapper(in_dim=1472, out_dim=2048, hidden_dim=2048, out_dim1=self.hidden_size, use_residual=False)

        # Image latent patch embedding
        self.img_in = PatchEmbed(self.patch_size, self.in_channels, self.hidden_size)

        # Vision feature projection
        self.vision_in = VisionProjection(input_dim=self.vision_states_dim, output_dim=self.hidden_size)

        # Text token refinement with cross-attention
        self.txt_in = SingleTokenRefiner(text_states_dim, self.hidden_size, self.heads_num, depth=2)

        # Timestep embedding for diffusion process
        self.time_in = TimestepEmbedder(self.hidden_size, nn.SiLU)

        # MeanFlow not supported in this implementation
        self.time_r_in = None

        # Guidance embedding (disabled for non-distilled model)
        self.guidance_in = TimestepEmbedder(self.hidden_size, nn.SiLU) if guidance_embed else None

        # Double-stream blocks: separate image and text processing
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels, nn.SiLU)

        self.cond_type_embedding = nn.Embedding(3, self.hidden_size)

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.blocks_to_swap = None

        self.offloader_double = None
        self.num_double_blocks = len(self.double_blocks)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

        for block in self.double_blocks:
            block.enable_gradient_checkpointing(cpu_offload=cpu_offload)

        print(f"HunyuanVideo-1.5: Gradient checkpointing enabled. CPU offload: {cpu_offload}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

        for block in self.double_blocks:
            block.disable_gradient_checkpointing()

        print("HunyuanVideo-1.5: Gradient checkpointing disabled.")

    def enable_block_swap(self, num_blocks: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False):
        self.blocks_to_swap = num_blocks

        assert self.blocks_to_swap < self.num_double_blocks - 2, (
            f"Cannot swap more than {self.num_double_blocks - 2} double blocks. Requested {self.blocks_to_swap} double blocks."
        )

        self.offloader_double = ModelOffloader(
            "double", self.double_blocks, len(self.double_blocks), self.blocks_to_swap, supports_backward, device, use_pinned_memory
        )
        print(
            f"HunyuanVideo-1.5: Block swap enabled. Swapping {num_blocks} blocks to device {device}. Supports backward: {supports_backward}"
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader_double.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print("HunyuanVideo-1.5: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader_double.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print("HunyuanVideo-1.5: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        if self.blocks_to_swap:
            save_double_blocks = self.double_blocks
            self.double_blocks = nn.ModuleList()

        self.to(device)

        if self.blocks_to_swap:
            self.double_blocks = save_double_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader_double.prepare_block_devices_before_forward(self.double_blocks)

    def get_rotary_pos_embed(self, rope_sizes):
        """
        Generate 3D rotary position embeddings for image tokens.

        Args:
            rope_sizes: Tuple of (height, width) for spatial dimensions.

        Returns:
            Tuple of (freqs_cos, freqs_sin) tensors for rotary position encoding.
        """
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(self.rope_dim_list, rope_sizes, theta=self.rope_theta)
        return freqs_cos, freqs_sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        vision_states: Optional[torch.Tensor] = None,
        byt5_text_states: Optional[torch.Tensor] = None,
        byt5_text_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb_cache: Optional[Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the HunyuanVideo diffusion transformer.

        Args:
            hidden_states: Input image latents [B, C, H, W].
            timestep: Diffusion timestep [B].
            text_states: Word-level text embeddings [B, L, D].
            encoder_attention_mask: Text attention mask [B, L].
            byt5_text_states: ByT5 character-level embeddings [B, L_byt5, D_byt5].
            byt5_text_mask: ByT5 attention mask [B, L_byt5].

        Returns:
            Tuple of (denoised_image, spatial_shape).
        """
        bs = hidden_states.shape[0]
        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        txt = text_states

        # Calculate spatial dimensions for rotary position embeddings
        _, _, ot, oh, ow = x.shape
        tt, th, tw = ot, oh, ow  # frame, height and width (patch_size=[1,1,1] means no temporal and spatial downsampling)
        if rotary_pos_emb_cache is not None:
            if (th, th, tw) in rotary_pos_emb_cache:
                freqs_cis = rotary_pos_emb_cache[(tt, th, tw)]
                freqs_cis = (freqs_cis[0].to(img.device), freqs_cis[1].to(img.device))
            else:
                freqs_cis = self.get_rotary_pos_embed((tt, th, tw))
                rotary_pos_emb_cache[(tt, th, tw)] = (freqs_cis[0].cpu(), freqs_cis[1].cpu())
        else:
            freqs_cis = self.get_rotary_pos_embed((tt, th, tw))

        # Reshape image latents to sequence format: [B, C, H, W] -> [B, H*W, C]
        img = self.img_in(img)

        # Generate timestep conditioning vector
        vec = self.time_in(t)

        # MeanFlow and guidance embedding not used in this configuration

        # Process text tokens through refinement layers
        txt_attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, 0, text_mask)
        txt = self.txt_in(txt, t, txt_attn_params)

        cond_emb = self.cond_type_embedding(torch.zeros_like(txt[:, :, 0], device=txt.device, dtype=torch.long))
        txt = txt + cond_emb

        # Process ByT5 character-level text embeddings if provided
        byt5_txt = self.byt5_in(byt5_text_states)

        cond_emb = self.cond_type_embedding(torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long))
        byt5_txt = byt5_txt + cond_emb

        # Project vision features if provided
        if vision_states is not None:
            if self.task_type == "t2v" and torch.all(vision_states == 0):
                # This does not affect the output because of the masks. Kept for reference.
                # # If t2v, set extra_attention_mask to zeros when vision_states is all zeros
                # extra_attention_mask = torch.zeros(
                #     bs, extra_encoder_hidden_states.shape[1], device=text_mask.device, dtype=text_mask.dtype
                # )
                # # Original impl comment: Set vision tokens to zero to mitigate potential block mask error in SSTA
                # extra_encoder_hidden_states = extra_encoder_hidden_states * 0
                vision_states = None
            else:
                extra_encoder_hidden_states = self.vision_in(vision_states)
                cond_emb = self.cond_type_embedding(
                    torch.full_like(
                        extra_encoder_hidden_states[:, :, 0], 2, device=extra_encoder_hidden_states.device, dtype=torch.long
                    )
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb
        else:
            extra_encoder_hidden_states = None

        # concatenate txt tokens in the order of [vision (if any), ByT5, word tokens]
        concatenated_txt = []
        txt_lens = []
        for i in range(bs):
            txt_length = text_mask[i].to(dtype=torch.bool).sum()
            byt5_txt_length = byt5_text_mask[i].to(dtype=torch.bool).sum()
            total_length = txt_length + byt5_txt_length

            txt_i = txt[i, :txt_length, :]
            byt5_txt_i = byt5_txt[i, :byt5_txt_length, :]

            if vision_states is not None:
                extra_encoder_hidden_states_i = extra_encoder_hidden_states[i]
                concatenated_txt_i = torch.cat([extra_encoder_hidden_states_i, byt5_txt_i, txt_i], dim=0)
                total_length += extra_encoder_hidden_states_i.shape[0]
            else:
                concatenated_txt_i = torch.cat([byt5_txt_i, txt_i], dim=0)

            concatenated_txt.append(concatenated_txt_i)
            txt_lens.append(total_length)

        # pad to max length in the batch
        max_txt_len = max(txt_lens)
        txt = torch.stack(
            [
                torch.cat(
                    [concatenated_txt[i], torch.zeros(max_txt_len - txt_lens[i], concatenated_txt[i].shape[-1], device=txt.device)]
                )
                for i in range(bs)
            ]
        )

        # create combined text mask
        if bs == 1:
            text_mask = None  # for single batch, no need to pass mask
        else:
            text_mask = torch.stack(
                [
                    torch.cat(
                        [
                            torch.ones(txt_lens[i], device=text_mask.device, dtype=text_mask.dtype),
                            torch.zeros(max_txt_len - txt_lens[i], device=text_mask.device, dtype=text_mask.dtype),
                        ]
                    )
                    for i in range(bs)
                ]
            )
        img_seq_len = img.shape[1]

        attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, img_seq_len, text_mask)

        # Process through double-stream blocks (separate image/text attention)
        for index, block in enumerate(self.double_blocks):
            if self.blocks_to_swap:
                self.offloader_double.wait_for_block(index)
            img, txt = block(img, txt, vec, freqs_cis, attn_params)
            if self.blocks_to_swap:
                self.offloader_double.submit_move_blocks_forward(self.double_blocks, index)
        del txt, attn_params, freqs_cis

        # Apply final projection to output space
        img = self.final_layer(img, vec)
        del vec

        # Reshape from sequence to spatial format: [B, L, C] -> [B, C, T, H, W]
        img = self.unpatchify(img, tt, th, tw)
        return img

    def unpatchify(self, x, t, h, w):
        """
        Convert sequence format back to spatial image format.

        Args:
            x: Input tensor [B, T*H*W, C].
            h: Height dimension.
            w: Width dimension.

        Returns:
            Spatial tensor [B, C, T, H, W].
        """
        c = self.unpatchify_channels

        x = x.reshape(shape=(x.shape[0], t, h, w, c))
        imgs = x.permute(0, 4, 1, 2, 3)  # .contiguous()
        return imgs


# endregion

# region Model Utils


def detect_hunyuan_video_1_5_sd_dtype(path: str) -> torch.dtype:
    # get dtype from model weights
    with MemoryEfficientSafeOpen(path) as f:
        keys = set(f.keys())
        key1 = "double_blocks.0.img_attn_k.weight"  # Official
        key2 = "double_blocks.0.img_attn_qkv.weight"  # ComfyUI repackaged
        if key1 in keys:
            dit_dtype = f.get_tensor(key1).dtype
        elif key2 in keys:
            dit_dtype = f.get_tensor(key2).dtype
        else:
            raise ValueError(f"Could not find the dtype in the model weights: {path}")
    logger.info(f"Detected DiT dtype: {dit_dtype}")
    return dit_dtype


def create_model(
    task_type: str, attn_mode: str, split_attn: bool, dtype: Optional[torch.dtype]
) -> HunyuanVideo_1_5_DiffusionTransformer:
    with init_empty_weights():
        model = HunyuanVideo_1_5_DiffusionTransformer(task_type=task_type, attn_mode=attn_mode, split_attn=split_attn)
        if dtype is not None:
            model.to(dtype)
    return model


def load_hunyuan_video_1_5_model(
    device: Union[str, torch.device],
    task_type: str,
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[Dict[str, torch.Tensor]] = None,
    lora_multipliers: Optional[list[float]] = None,
) -> HunyuanVideo_1_5_DiffusionTransformer:
    """
    Load a HunyuanVideo model from the specified checkpoint.

    Args:
        device (Union[str, torch.device]): Device for optimization or merging
        task_type (str): Task type, either "t2v" or "i2v".
        dit_path (str): Path to the DiT model checkpoint.
        attn_mode (str): Attention mode to use, e.g., "torch", "flash", etc.
        split_attn (bool): Whether to use split attention.
        loading_device (Union[str, torch.device]): Device to load the model weights on.
        dit_weight_dtype (Optional[torch.dtype]): Data type of the DiT weights.
            If None, it will be loaded as is (same as the state_dict) or scaled for fp8. if not None, model weights will be casted to this dtype.
        fp8_scaled (bool): Whether to use fp8 scaling for the model weights.
        lora_weights_list (Optional[Dict[str, torch.Tensor]]): LoRA weights to apply, if any.
        lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
    """
    # dit_weight_dtype is None for fp8_scaled
    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    model = create_model(task_type, attn_mode, split_attn, dit_weight_dtype)

    # load model weights with dynamic fp8 optimization and LoRA merging if needed
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")

    def comfyui_hunyuan_video_1_5_weight_split_hook(
        key: str, value: Optional[torch.Tensor]
    ) -> Tuple[Optional[list[str]], Optional[list[torch.Tensor]]]:
        # ComfyUI repackaged HunyuanVideo-1.5 uses packed QKV weights for double blocks
        if "img_attn_qkv.weight" in key or "img_attn_qkv.bias" in key or "txt_attn_qkv.weight" in key or "txt_attn_qkv.bias" in key:
            # convert to separate Q, K, V weights/biases
            new_keys = [key.replace("_qkv", suffix) for suffix in ["_q", "_k", "_v"]]
            return new_keys, list(torch.chunk(value, 3, dim=0)) if value is not None else None

        return None, None

    hooks = WeightTransformHooks(split_hook=comfyui_hunyuan_video_1_5_weight_split_hook, concat_hook=None)

    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
        weight_transform_hooks=hooks,
    )

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded DiT model from {dit_path}, info={info}")

    return model


# endregion
