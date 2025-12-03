import argparse
import gc
import math
import time
from typing import Optional
from PIL import Image


import numpy as np
import torch
from accelerate import Accelerator

from musubi_tuner.dataset import image_video_dataset
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_FRAMEPACK, ARCHITECTURE_FRAMEPACK_FULL
from musubi_tuner.fpack_generate_video import decode_latent
from musubi_tuner.frame_pack import hunyuan
from musubi_tuner.frame_pack.clip_vision import hf_clip_vision_encode
from musubi_tuner.frame_pack.framepack_utils import load_image_encoders, load_text_encoder1, load_text_encoder2
from musubi_tuner.frame_pack.framepack_utils import load_vae as load_framepack_vae
from musubi_tuner.frame_pack.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked, load_packed_model
from musubi_tuner.frame_pack.k_diffusion_hunyuan import sample_hunyuan
from musubi_tuner.frame_pack.utils import crop_or_pad_yield_mask
from musubi_tuner.dataset.image_video_dataset import resize_image_to_bucket
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
)

import logging

from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FramePackNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_FRAMEPACK

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_FRAMEPACK_FULL

    def handle_model_specific_args(self, args):
        self._i2v_training = True
        self._control_training = False
        self.default_guidance_scale = 10.0  # embeded guidance scale

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        device = accelerator.device

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        # load text encoder
        tokenizer1, text_encoder1 = load_text_encoder1(args, args.fp8_llm, device)
        tokenizer2, text_encoder2 = load_text_encoder2(args)
        text_encoder2.to(device)

        sample_prompts_te_outputs = {}  # (prompt) -> (t1 embeds, t1 mask, t2 embeds)
        for prompt_dict in prompts:
            for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                if p is None or p in sample_prompts_te_outputs:
                    continue
                logger.info(f"cache Text Encoder outputs for prompt: {p}")
                with torch.amp.autocast(device_type=device.type, dtype=text_encoder1.dtype), torch.no_grad():
                    llama_vec, clip_l_pooler = hunyuan.encode_prompt_conds(p, text_encoder1, text_encoder2, tokenizer1, tokenizer2)
                llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)

                llama_vec = llama_vec.to("cpu")
                llama_attention_mask = llama_attention_mask.to("cpu")
                clip_l_pooler = clip_l_pooler.to("cpu")
                sample_prompts_te_outputs[p] = (llama_vec, llama_attention_mask, clip_l_pooler)
        del text_encoder1, text_encoder2
        clean_memory_on_device(device)

        # image embedding for I2V training
        feature_extractor, image_encoder = load_image_encoders(args)
        image_encoder.to(device)

        # encode image with image encoder
        sample_prompts_image_embs = {}
        for prompt_dict in prompts:
            image_path = prompt_dict.get("image_path", None)
            assert image_path is not None, "image_path should be set for I2V training"
            if image_path in sample_prompts_image_embs:
                continue

            logger.info(f"Encoding image to image encoder context: {image_path}")

            height = prompt_dict.get("height", 256)
            width = prompt_dict.get("width", 256)

            img = Image.open(image_path).convert("RGB")
            img_np = np.array(img)  # PIL to numpy, HWC
            img_np = image_video_dataset.resize_image_to_bucket(img_np, (width, height))  # returns a numpy array

            with torch.no_grad():
                image_encoder_output = hf_clip_vision_encode(img_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to("cpu")
            sample_prompts_image_embs[image_path] = image_encoder_last_hidden_state

        del image_encoder
        clean_memory_on_device(device)

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            p = prompt_dict.get("prompt", "")
            llama_vec, llama_attention_mask, clip_l_pooler = sample_prompts_te_outputs[p]
            prompt_dict_copy["llama_vec"] = llama_vec
            prompt_dict_copy["llama_attention_mask"] = llama_attention_mask
            prompt_dict_copy["clip_l_pooler"] = clip_l_pooler

            p = prompt_dict.get("negative_prompt", "")
            llama_vec, llama_attention_mask, clip_l_pooler = sample_prompts_te_outputs[p]
            prompt_dict_copy["negative_llama_vec"] = llama_vec
            prompt_dict_copy["negative_llama_attention_mask"] = llama_attention_mask
            prompt_dict_copy["negative_clip_l_pooler"] = clip_l_pooler

            p = prompt_dict.get("image_path", None)
            prompt_dict_copy["image_encoder_last_hidden_state"] = sample_prompts_image_embs[p]

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)
        return sample_parameters

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        """architecture dependent inference"""
        model: HunyuanVideoTransformer3DModelPacked = transformer
        device = accelerator.device
        if cfg_scale is None:
            cfg_scale = 1.0
        do_classifier_free_guidance = do_classifier_free_guidance and cfg_scale != 1.0

        # prepare parameters
        one_frame_mode = args.one_frame
        if one_frame_mode:
            one_frame_inference = set()
            for mode in sample_parameter["one_frame"].split(","):
                one_frame_inference.add(mode.strip())
        else:
            one_frame_inference = None

        latent_window_size = args.latent_window_size  # default is 9
        latent_f = (frame_count - 1) // 4 + 1
        total_latent_sections = math.floor((latent_f - 1) / latent_window_size)
        if total_latent_sections < 1 and not one_frame_mode:
            logger.warning(f"Not enough frames for FramePack: {latent_f}, minimum: {latent_window_size * 4 + 1}")
            return None

        latent_f = total_latent_sections * latent_window_size + 1
        actual_frame_count = (latent_f - 1) * 4 + 1
        if actual_frame_count != frame_count:
            logger.info(f"Frame count mismatch: {actual_frame_count} != {frame_count}, trimming to {actual_frame_count}")
        frame_count = actual_frame_count
        num_frames = latent_window_size * 4 - 3

        # prepare start and control latent
        def encode_image(path):
            image = Image.open(path)
            if image.mode == "RGBA":
                alpha = image.split()[-1]
                image = image.convert("RGB")
            else:
                alpha = None
            image = resize_image_to_bucket(image, (width, height))  # returns a numpy array
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(1).unsqueeze(0).float()  # 1, C, 1, H, W
            image = image / 127.5 - 1  # -1 to 1
            return hunyuan.vae_encode(image, vae).to("cpu"), alpha

        # VAE encoding
        logger.info("Encoding image to latent space")
        vae.to(device)

        start_latent, _ = (
            encode_image(image_path) if image_path else torch.zeros((1, 16, 1, height // 8, width // 8), dtype=torch.float32)
        )

        if one_frame_mode:
            control_latents = []
            control_alphas = []
            if "control_image_path" in sample_parameter:
                for control_image_path in sample_parameter["control_image_path"]:
                    control_latent, control_alpha = encode_image(control_image_path)
                    control_latents.append(control_latent)
                    control_alphas.append(control_alpha)
        else:
            control_latents = None
            control_alphas = None

        vae.to("cpu")  # move VAE to CPU to save memory
        clean_memory_on_device(device)

        # sampilng
        if not one_frame_mode:
            f1_mode = args.f1
            history_latents = torch.zeros((1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32)

            if not f1_mode:
                total_generated_latent_frames = 0
                latent_paddings = reversed(range(total_latent_sections))
            else:
                total_generated_latent_frames = 1
                history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
                latent_paddings = [0] * total_latent_sections

            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

            latent_paddings = list(latent_paddings)
            for loop_index in range(total_latent_sections):
                latent_padding = latent_paddings[loop_index]

                if not f1_mode:
                    is_last_section = latent_padding == 0
                    latent_padding_size = latent_padding * latent_window_size

                    logger.info(f"latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}")

                    indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                    (
                        clean_latent_indices_pre,
                        blank_indices,
                        latent_indices,
                        clean_latent_indices_post,
                        clean_latent_2x_indices,
                        clean_latent_4x_indices,
                    ) = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                    clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                    clean_latents_pre = start_latent.to(history_latents)
                    clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, : 1 + 2 + 16, :, :].split(
                        [1, 2, 16], dim=2
                    )
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                else:
                    indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
                    (
                        clean_latent_indices_start,
                        clean_latent_4x_indices,
                        clean_latent_2x_indices,
                        clean_latent_1x_indices,
                        latent_indices,
                    ) = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
                    clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

                    clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]) :, :, :].split(
                        [16, 2, 1], dim=2
                    )
                    clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

                # if use_teacache:
                #     transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                # else:
                #     transformer.initialize_teacache(enable_teacache=False)

                llama_vec = sample_parameter["llama_vec"].to(device, dtype=torch.bfloat16)
                llama_attention_mask = sample_parameter["llama_attention_mask"].to(device)
                clip_l_pooler = sample_parameter["clip_l_pooler"].to(device, dtype=torch.bfloat16)
                if cfg_scale == 1.0:
                    llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
                    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
                else:
                    llama_vec_n = sample_parameter["negative_llama_vec"].to(device, dtype=torch.bfloat16)
                    llama_attention_mask_n = sample_parameter["negative_llama_attention_mask"].to(device)
                    clip_l_pooler_n = sample_parameter["negative_clip_l_pooler"].to(device, dtype=torch.bfloat16)
                image_encoder_last_hidden_state = sample_parameter["image_encoder_last_hidden_state"].to(
                    device, dtype=torch.bfloat16
                )

                generated_latents = sample_hunyuan(
                    transformer=model,
                    sampler=args.sample_solver,
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg_scale,
                    distilled_guidance_scale=guidance_scale,
                    guidance_rescale=0.0,
                    # shift=3.0,
                    num_inference_steps=sample_steps,
                    generator=generator,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=device,
                    dtype=torch.bfloat16,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                )

                total_generated_latent_frames += int(generated_latents.shape[2])
                if not f1_mode:
                    if is_last_section:
                        generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
                        total_generated_latent_frames += 1
                    history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
                    real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
                else:
                    history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
                    real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

                logger.info(f"Generated. Latent shape {real_history_latents.shape}")
        else:
            # one frame mode
            sample_num_frames = 1
            latent_indices = torch.zeros((1, 1), dtype=torch.int64)  # 1x1 latent index for target image
            latent_indices[:, 0] = latent_window_size  # last of latent_window

            def get_latent_mask(mask_image: Image.Image):
                mask_image = mask_image.resize((width // 8, height // 8), Image.LANCZOS)
                mask_image = np.array(mask_image)  # PIL to numpy, HWC
                mask_image = torch.from_numpy(mask_image).float() / 255.0  # 0 to 1.0, HWC
                mask_image = mask_image.squeeze(-1)  # HWC -> HW
                mask_image = mask_image.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # HW -> 111HW (B, C, F, H, W)
                mask_image = mask_image.to(torch.float32)
                return mask_image

            if control_latents is None or len(control_latents) == 0:
                logger.info("No control images provided for one frame inference. Use zero latents for control images.")
                control_latents = [torch.zeros(1, 16, 1, height // 8, width // 8, dtype=torch.float32)]

            if "no_post" not in one_frame_inference:
                # add zero latents as clean latents post
                control_latents.append(torch.zeros((1, 16, 1, height // 8, width // 8), dtype=torch.float32))
                logger.info("Add zero latents as clean latents post for one frame inference.")

            # kisekaeichi and 1f-mc: both are using control images, but indices are different
            clean_latents = torch.cat(control_latents, dim=2)  # (1, 16, num_control_images, H//8, W//8)
            clean_latent_indices = torch.zeros((1, len(control_latents)), dtype=torch.int64)
            if "no_post" not in one_frame_inference:
                clean_latent_indices[:, -1] = 1 + latent_window_size  # default index for clean latents post

            # apply mask for control latents (clean latents)
            for i in range(len(control_alphas)):
                control_alpha = control_alphas[i]
                if control_alpha is not None:
                    latent_mask = get_latent_mask(control_alpha)
                    logger.info(f"Apply mask for clean latents 1x for {i + 1}: shape: {latent_mask.shape}")
                    clean_latents[:, :, i : i + 1, :, :] = clean_latents[:, :, i : i + 1, :, :] * latent_mask

            for one_frame_param in one_frame_inference:
                if one_frame_param.startswith("target_index="):
                    target_index = int(one_frame_param.split("=")[1])
                    latent_indices[:, 0] = target_index
                    logger.info(f"Set index for target: {target_index}")
                elif one_frame_param.startswith("control_index="):
                    control_indices = one_frame_param.split("=")[1].split(";")
                    i = 0
                    while i < len(control_indices) and i < clean_latent_indices.shape[1]:
                        control_index = int(control_indices[i])
                        clean_latent_indices[:, i] = control_index
                        i += 1
                    logger.info(f"Set index for clean latent 1x: {control_indices}")

            if "no_2x" in one_frame_inference:
                clean_latents_2x = None
                clean_latent_2x_indices = None
                logger.info("No clean_latents_2x")
            else:
                clean_latents_2x = torch.zeros((1, 16, 2, height // 8, width // 8), dtype=torch.float32)
                index = 1 + latent_window_size + 1
                clean_latent_2x_indices = torch.arange(index, index + 2).unsqueeze(0)  #  2

            if "no_4x" in one_frame_inference:
                clean_latents_4x = None
                clean_latent_4x_indices = None
                logger.info("No clean_latents_4x")
            else:
                clean_latents_4x = torch.zeros((1, 16, 16, height // 8, width // 8), dtype=torch.float32)
                index = 1 + latent_window_size + 1 + 2
                clean_latent_4x_indices = torch.arange(index, index + 16).unsqueeze(0)  #  16

            logger.info(
                f"One frame inference. clean_latent: {clean_latents.shape} latent_indices: {latent_indices}, clean_latent_indices: {clean_latent_indices}, num_frames: {sample_num_frames}"
            )

            # prepare conditioning inputs
            llama_vec = sample_parameter["llama_vec"].to(device, dtype=torch.bfloat16)
            llama_attention_mask = sample_parameter["llama_attention_mask"].to(device)
            clip_l_pooler = sample_parameter["clip_l_pooler"].to(device, dtype=torch.bfloat16)
            if cfg_scale == 1.0:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
                llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
            else:
                llama_vec_n = sample_parameter["negative_llama_vec"].to(device, dtype=torch.bfloat16)
                llama_attention_mask_n = sample_parameter["negative_llama_attention_mask"].to(device)
                clip_l_pooler_n = sample_parameter["negative_clip_l_pooler"].to(device, dtype=torch.bfloat16)
            image_encoder_last_hidden_state = sample_parameter["image_encoder_last_hidden_state"].to(device, dtype=torch.bfloat16)

            generated_latents = sample_hunyuan(
                transformer=model,
                sampler=args.sample_solver,
                width=width,
                height=height,
                frames=1,
                real_guidance_scale=cfg_scale,
                distilled_guidance_scale=guidance_scale,
                guidance_rescale=0.0,
                # shift=3.0,
                num_inference_steps=sample_steps,
                generator=generator,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=device,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
            )

            real_history_latents = generated_latents.to(clean_latents)

        # wait for 5 seconds until block swap is done
        logger.info("Waiting for 5 seconds to finish block swap")
        time.sleep(5)

        gc.collect()
        clean_memory_on_device(device)

        video = decode_latent(
            latent_window_size, total_latent_sections, args.bulk_decode, vae, real_history_latents, device, one_frame_mode
        )
        video = video.to("cpu", dtype=torch.float32).unsqueeze(0)  # add batch dimension
        video = (video / 2 + 0.5).clamp(0, 1)  # -1 to 1 -> 0 to 1
        clean_memory_on_device(device)

        return video

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae
        logger.info(f"Loading VAE model from {vae_path}")
        vae = load_framepack_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, args.vae_tiling, "cpu")
        return vae

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        logger.info(f"Loading DiT model from {dit_path}")
        device = accelerator.device
        model = load_packed_model(
            device, dit_path, attn_mode, loading_device, args.fp8_scaled, split_attn, disable_numpy_memmap=args.disable_numpy_memmap
        )
        return model

    def compile_transformer(self, args, transformer):
        transformer: HunyuanVideoTransformer3DModelPacked = transformer
        return model_utils.compile_transformer(
            args,
            transformer,
            [transformer.transformer_blocks, transformer.single_transformer_blocks],
            disable_linear=self.blocks_to_swap > 0,
        )

    def scale_shift_latents(self, latents):
        # FramePack VAE includes scaling
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        model: HunyuanVideoTransformer3DModelPacked = transformer
        device = accelerator.device
        batch_size = latents.shape[0]

        # maybe model.dtype is better than network_dtype...
        distilled_guidance = torch.tensor([args.guidance_scale * 1000.0] * batch_size).to(device=device, dtype=network_dtype)
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        # for k, v in batch.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: {v.shape} {v.dtype} {v.device}")
        with accelerator.autocast():
            clean_latent_2x_indices = batch["clean_latent_2x_indices"] if "clean_latent_2x_indices" in batch else None
            if clean_latent_2x_indices is not None:
                clean_latent_2x = batch["latents_clean_2x"] if "latents_clean_2x" in batch else None
                if clean_latent_2x is None:
                    clean_latent_2x = torch.zeros(
                        (batch_size, 16, 2, latents.shape[3], latents.shape[4]), dtype=latents.dtype, device=latents.device
                    )
            else:
                clean_latent_2x = None

            clean_latent_4x_indices = batch["clean_latent_4x_indices"] if "clean_latent_4x_indices" in batch else None
            if clean_latent_4x_indices is not None:
                clean_latent_4x = batch["latents_clean_4x"] if "latents_clean_4x" in batch else None
                if clean_latent_4x is None:
                    clean_latent_4x = torch.zeros(
                        (batch_size, 16, 16, latents.shape[3], latents.shape[4]), dtype=latents.dtype, device=latents.device
                    )
            else:
                clean_latent_4x = None

            model_pred = model(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=batch["llama_vec"],
                encoder_attention_mask=batch["llama_attention_mask"],
                pooled_projections=batch["clip_l_pooler"],
                guidance=distilled_guidance,
                latent_indices=batch["latent_indices"],
                clean_latents=batch["latents_clean"],
                clean_latent_indices=batch["clean_latent_indices"],
                clean_latents_2x=clean_latent_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latent_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                image_embeddings=batch["image_embeddings"],
                return_dict=False,
            )
            model_pred = model_pred[0]  # returns tuple (model_pred, )

        # flow matching loss
        target = noise - latents

        return model_pred, target

    # endregion model specific


def framepack_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """FramePack specific parser setup"""
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for LLM / LLMにfp8を使う")
    parser.add_argument("--text_encoder1", type=str, help="Text Encoder 1 directory / テキストエンコーダ1のディレクトリ")
    parser.add_argument("--text_encoder2", type=str, help="Text Encoder 2 directory / テキストエンコーダ2のディレクトリ")
    parser.add_argument(
        "--vae_tiling",
        action="store_true",
        help="enable spatial tiling for VAE, default is False. If vae_spatial_tile_sample_min_size is set, this is automatically enabled",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    parser.add_argument("--image_encoder", type=str, required=True, help="Image encoder (CLIP) checkpoint path or directory")
    parser.add_argument("--latent_window_size", type=int, default=9, help="FramePack latent window size (default 9)")
    parser.add_argument("--bulk_decode", action="store_true", help="decode all frames at once in sample generation")
    parser.add_argument("--f1", action="store_true", help="Use F1 sampling method for sample generation")
    parser.add_argument("--one_frame", action="store_true", help="Use one frame sampling method for sample generation")
    return parser


def main():
    parser = setup_parser_common()
    parser = framepack_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    assert args.vae_dtype is None or args.vae_dtype == "float16", (
        "VAE dtype must be float16 / VAEのdtypeはfloat16でなければなりません"
    )
    args.vae_dtype = "float16"  # fixed
    args.dit_dtype = "bfloat16"  # fixed
    args.sample_solver = "unipc"  # for sample generation, fixed to unipc

    trainer = FramePackNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
