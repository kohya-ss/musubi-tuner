# Inference model for Hunyuan Video Packed
# We do not want to break the training accidentally, so we use a separate file for inference model.

# MagCache: modified from https://github.com/Zehong-Ma/MagCache/blob/main/MagCache4HunyuanVideo/magcache_sample_video.py

from types import SimpleNamespace
from typing import Optional
import einops
import numpy as np
import torch
from torch.nn import functional as F
from musubi_tuner.frame_pack.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked, get_cu_seqlens

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HunyuanVideoTransformer3DModelPackedInference(HunyuanVideoTransformer3DModelPacked):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_magcache = False

    def initialize_magcache(
        self,
        enable: bool = True,
        retention_ratio: float = 0.2,
        mag_ratios: Optional[list[float]] = None,
        magcache_thresh: float = 0.24,
        K: int = 6,
        calibration: bool = False,
    ):
        if mag_ratios is None:
            # Copy from original MagCache
            mag_ratios = np.array(
                [1.0]
                + [
                    1.06971,
                    1.29073,
                    1.11245,
                    1.09596,
                    1.05233,
                    1.01415,
                    1.05672,
                    1.00848,
                    1.03632,
                    1.02974,
                    1.00984,
                    1.03028,
                    1.00681,
                    1.06614,
                    1.05022,
                    1.02592,
                    1.01776,
                    1.02985,
                    1.00726,
                    1.03727,
                    1.01502,
                    1.00992,
                    1.03371,
                    0.9976,
                    1.02742,
                    1.0093,
                    1.01869,
                    1.00815,
                    1.01461,
                    1.01152,
                    1.03082,
                    1.0061,
                    1.02162,
                    1.01999,
                    0.99063,
                    1.01186,
                    1.0217,
                    0.99947,
                    1.01711,
                    0.9904,
                    1.00258,
                    1.00878,
                    0.97039,
                    0.97686,
                    0.94315,
                    0.97728,
                    0.91154,
                    0.86139,
                    0.76592,
                ]
            )
        self.enable_magcache = enable
        self.calibration = calibration
        self.retention_ratio = retention_ratio
        self.default_mag_ratios = mag_ratios
        self.magcache_thresh = magcache_thresh
        self.K = K
        self.reset_magcache()

    def reset_magcache(self, num_steps: int = 50):
        if not self.enable_magcache:
            return

        def nearest_interp(src_array, target_length):
            src_length = len(src_array)
            if target_length == 1:
                return np.array([src_array[-1]])

            scale = (src_length - 1) / (target_length - 1)
            mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
            return np.array(src_array)[mapped_indices]

        if not self.calibration and num_steps != len(self.default_mag_ratios):
            logger.info(f"Interpolating mag_ratios from {len(self.default_mag_ratios)} to {num_steps} steps.")
            self.mag_ratios = nearest_interp(self.default_mag_ratios, num_steps)
        else:
            self.mag_ratios = self.default_mag_ratios

        self.cnt = 0
        self.num_steps = num_steps
        self.residual_cache = None
        self.accumulated_ratio = 1.0
        self.accumulated_steps = 0
        self.accumulated_err = 0
        self.norm_ratio = []
        self.norm_std = []
        self.cos_dis = []

    def get_calibration_data(self) -> tuple[list[float], list[float], list[float]]:
        if not self.enable_magcache or not self.calibration:
            raise ValueError("MagCache is not enabled or calibration is not set.")
        return self.norm_ratio, self.norm_std, self.cos_dis

    def forward(self, *args, **kwargs):
        # Forward pass for inference
        if self.enable_magcache:
            return self.magcache_forward(*args, **kwargs, calibration=self.calibration)
        else:
            return super().forward(*args, **kwargs)

    def magcache_forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
        latent_indices=None,
        clean_latents=None,
        clean_latent_indices=None,
        clean_latents_2x=None,
        clean_latent_2x_indices=None,
        clean_latents_4x=None,
        clean_latent_4x_indices=None,
        image_embeddings=None,
        attention_kwargs=None,
        return_dict=True,
        calibration=False,
    ):
        if attention_kwargs is None:
            attention_kwargs = {}

        # RoPE scaling: must be done before processing hidden states
        if self.rope_scaling_timestep_threshold is not None:
            if timestep >= self.rope_scaling_timestep_threshold:
                self.rope.h_w_scaling_factor = self.rope_scaling_factor
            else:
                self.rope.h_w_scaling_factor = 1.0

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config_patch_size, self.config_patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

        hidden_states, rope_freqs = self.process_input_hidden_states(
            hidden_states,
            latent_indices,
            clean_latents,
            clean_latent_indices,
            clean_latents_2x,
            clean_latent_2x_indices,
            clean_latents_4x,
            clean_latent_4x_indices,
        )
        del (
            latent_indices,
            clean_latents,
            clean_latent_indices,
            clean_latents_2x,
            clean_latent_2x_indices,
            clean_latents_4x,
            clean_latent_4x_indices,
        )  # free memory

        temb = self.gradient_checkpointing_method(self.time_text_embed, timestep, guidance, pooled_projections)
        encoder_hidden_states = self.gradient_checkpointing_method(
            self.context_embedder, encoder_hidden_states, timestep, encoder_attention_mask
        )

        if self.image_projection is not None:
            assert image_embeddings is not None, "You must use image embeddings!"
            extra_encoder_hidden_states = self.gradient_checkpointing_method(self.image_projection, image_embeddings)
            extra_attention_mask = torch.ones(
                (batch_size, extra_encoder_hidden_states.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )

            # must cat before (not after) encoder_hidden_states, due to attn masking
            encoder_hidden_states = torch.cat([extra_encoder_hidden_states, encoder_hidden_states], dim=1)
            encoder_attention_mask = torch.cat([extra_attention_mask, encoder_attention_mask], dim=1)
            del extra_encoder_hidden_states, extra_attention_mask  # free memory

        with torch.no_grad():
            if batch_size == 1:
                # When batch size is 1, we do not need any masks or var-len funcs since cropping is mathematically same to what we want
                # If they are not same, then their impls are wrong. Ours are always the correct one.
                text_len = encoder_attention_mask.sum().item()
                encoder_hidden_states = encoder_hidden_states[:, :text_len]
                attention_mask = None, None, None, None, None
            else:
                img_seq_len = hidden_states.shape[1]
                txt_seq_len = encoder_hidden_states.shape[1]

                cu_seqlens_q, seq_len = get_cu_seqlens(encoder_attention_mask, img_seq_len)
                cu_seqlens_kv = cu_seqlens_q
                max_seqlen_q = img_seq_len + txt_seq_len
                max_seqlen_kv = max_seqlen_q

                attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, seq_len
                del cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, seq_len  # free memory
        del encoder_attention_mask  # free memory

        if self.enable_teacache:
            raise NotImplementedError("TEACache is not implemented for inference model.")

        skip_forward = False
        if (
            self.enable_magcache
            and not calibration
            and self.cnt >= max(int(self.retention_ratio * self.num_steps), 1)
            and self.cnt < self.num_steps - 1
        ):
            cur_mag_ratio = self.mag_ratios[self.cnt]
            self.accumulated_ratio = self.accumulated_ratio * cur_mag_ratio
            cur_skip_err = np.abs(1 - self.accumulated_ratio)
            self.accumulated_err += cur_skip_err
            self.accumulated_steps += 1
            if self.accumulated_err <= self.magcache_thresh and self.accumulated_steps <= self.K:
                skip_forward = True
            else:
                self.accumulated_ratio = 1.0
                self.accumulated_steps = 0
                self.accumulated_err = 0

        if skip_forward:
            # uncomment the following line to debug
            # print(
            #     f"Skipping forward pass at step {self.cnt}, accumulated ratio: {self.accumulated_ratio:.4f}, "
            #     f"accumulated error: {self.accumulated_err:.4f}, accumulated steps: {self.accumulated_steps}"
            # )
            hidden_states = hidden_states + self.residual_cache
        else:
            ori_hidden_states = hidden_states

            for block_id, block in enumerate(self.transformer_blocks):
                if self.blocks_to_swap:
                    self.offloader_double.wait_for_block(block_id)

                hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                    block, hidden_states, encoder_hidden_states, temb, attention_mask, rope_freqs
                )

                if self.blocks_to_swap:
                    self.offloader_double.submit_move_blocks_forward(self.transformer_blocks, block_id)

            for block_id, block in enumerate(self.single_transformer_blocks):
                if self.blocks_to_swap:
                    self.offloader_single.wait_for_block(block_id)

                hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                    block, hidden_states, encoder_hidden_states, temb, attention_mask, rope_freqs
                )

                if self.blocks_to_swap:
                    self.offloader_single.submit_move_blocks_forward(self.single_transformer_blocks, block_id)

            if self.enable_magcache:
                cur_residual = hidden_states - ori_hidden_states
                if calibration and self.cnt >= 1:
                    norm_ratio = ((cur_residual.norm(dim=-1) / self.residual_cache.norm(dim=-1)).mean()).item()
                    norm_std = (cur_residual.norm(dim=-1) / self.residual_cache.norm(dim=-1)).std().item()
                    cos_dis = (1 - F.cosine_similarity(cur_residual, self.residual_cache, dim=-1, eps=1e-8)).mean().item()
                    self.norm_ratio.append(round(norm_ratio, 5))
                    self.norm_std.append(round(norm_std, 5))
                    self.cos_dis.append(round(cos_dis, 5))
                    logger.info(f"time: {self.cnt}, norm_ratio: {norm_ratio}, norm_std: {norm_std}, cos_dis: {cos_dis}")
                self.residual_cache = cur_residual

            del ori_hidden_states  # free memory

        del attention_mask, rope_freqs  # free memory
        del encoder_hidden_states  # free memory

        hidden_states = self.gradient_checkpointing_method(self.norm_out, hidden_states, temb)

        hidden_states = hidden_states[:, -original_context_length:, :]

        if self.high_quality_fp32_output_for_inference:
            hidden_states = hidden_states.to(dtype=torch.float32)
            if self.proj_out.weight.dtype != torch.float32:
                self.proj_out.to(dtype=torch.float32)

        hidden_states = self.gradient_checkpointing_method(self.proj_out, hidden_states)

        hidden_states = einops.rearrange(
            hidden_states,
            "b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)",
            t=post_patch_num_frames,
            h=post_patch_height,
            w=post_patch_width,
            pt=p_t,
            ph=p,
            pw=p,
        )

        if self.enable_magcache:
            self.cnt += 1
            if self.cnt >= self.num_steps:
                self.cnt = 0
                self.accumulated_ratio = 1.0
                self.accumulated_steps = 0
                self.accumulated_err = 0

        if return_dict:
            # return Transformer2DModelOutput(sample=hidden_states)
            return SimpleNamespace(sample=hidden_states)

        return (hidden_states,)
