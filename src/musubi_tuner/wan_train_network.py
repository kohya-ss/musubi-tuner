import argparse
from typing import List, Optional
from PIL import Image

import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_WAN, ARCHITECTURE_WAN_FULL, load_video
from musubi_tuner.hv_generate_video import resize_image_to_bucket
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.utils.device_utils import synchronize_device
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.wan_generate_video import parse_one_frame_inference_args

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.utils import model_utils
from musubi_tuner.wan.configs import WAN_CONFIGS
from musubi_tuner.wan.modules.clip import CLIPModel
from musubi_tuner.wan.modules.model import WanModel, detect_wan_sd_dtype, load_wan_model
from musubi_tuner.wan.modules.t5 import T5EncoderModel
from musubi_tuner.wan.modules.vae import WanVAE
from musubi_tuner.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_WAN

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_WAN_FULL

    def handle_model_specific_args(self, args):
        self.config = WAN_CONFIGS[args.task]
        # we cannot use config.i2v because Fun-Control T2V has i2v flag TODO refactor this
        self._i2v_training = "i2v" in args.task or "flf2v" in args.task
        self._control_training = self.config.is_fun_control

        self.dit_dtype = detect_wan_sd_dtype(args.dit)

        if self.dit_dtype == torch.float16:
            assert args.mixed_precision in ["fp16", "no"], "DiT weights are in fp16, mixed precision must be fp16 or no"
        elif self.dit_dtype == torch.bfloat16:
            assert args.mixed_precision in ["bf16", "no"], "DiT weights are in bf16, mixed precision must be bf16 or no"

        if args.fp8_scaled and self.dit_dtype.itemsize == 1:
            raise ValueError(
                "DiT weights is already in fp8 format, cannot scale to fp8. Please use fp16/bf16 weights / DiTの重みはすでにfp8形式です。fp8にスケーリングできません。fp16/bf16の重みを使用してください"
            )

        # dit_dtype cannot be fp8, so we select the appropriate dtype
        if self.dit_dtype.itemsize == 1:
            self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)

        # Wan2.2: Store timestep boundary
        self.dit_high_noise_path = args.dit_high_noise
        self.high_low_training = self.dit_high_noise_path is not None

        if self.high_low_training:
            if args.blocks_to_swap is not None and args.blocks_to_swap > 0:
                assert not args.offload_inactive_dit, (
                    "Block swap is not supported with offloading inactive DiT / 非アクティブDiTをオフロードする設定ではブロックスワップはサポートされていません"
                )
            if args.num_timestep_buckets is not None:
                logger.warning(
                    "num_timestep_buckets is not working well with high and low models training / high and lowモデルのトレーニングではnum_timestep_bucketsがうまく機能しません"
                )

        self.timestep_boundary = (
            args.timestep_boundary if args.timestep_boundary is not None else self.config.boundary
        )  # may be None
        if self.timestep_boundary is None and self.high_low_training:
            raise ValueError(
                "timestep_boundary is not specified for high noise model"
                + " / high noiseモデルを使用する場合は、timestep_boundaryを指定する必要があります。"
            )
        if self.timestep_boundary is not None:
            if self.timestep_boundary > 1:
                self.timestep_boundary /= 1000.0  # convert to 0 to 1 range
            logger.info(f"Converted timestep_boundary to 0 to 1 range: {self.timestep_boundary}")

        self.default_guidance_scale = 1.0  # not used

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        config = self.config
        device = accelerator.device
        t5_path, clip_path, fp8_t5 = args.t5, args.clip, args.fp8_t5

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        def encode_for_text_encoder(text_encoder):
            sample_prompts_te_outputs = {}  # (prompt) -> (embeds, mask)
            # with accelerator.autocast(), torch.no_grad(): # this causes NaN if dit_dtype is fp16
            t5_dtype = config.t5_dtype
            with torch.amp.autocast(device_type=device.type, dtype=t5_dtype), torch.no_grad():
                for prompt_dict in prompts:
                    if "negative_prompt" not in prompt_dict:
                        prompt_dict["negative_prompt"] = self.config["sample_neg_prompt"]
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", None)]:
                        if p is None:
                            continue
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"cache Text Encoder outputs for prompt: {p}")

                            prompt_outputs = text_encoder([p], device)
                            sample_prompts_te_outputs[p] = prompt_outputs

            return sample_prompts_te_outputs

        # Load Text Encoder 1 and encode
        logger.info(f"loading T5: {t5_path}")
        t5 = T5EncoderModel(text_len=config.text_len, dtype=config.t5_dtype, device=device, weight_path=t5_path, fp8=fp8_t5)

        logger.info("encoding with Text Encoder 1")
        te_outputs_1 = encode_for_text_encoder(t5)
        del t5

        # load CLIP and encode image (for I2V training)
        # Note: VAE encoding is done in do_inference() for I2V training, because we have VAE in the pipeline. Control video is also done in do_inference()
        sample_prompts_image_embs = {}
        for prompt_dict in prompts:
            if prompt_dict.get("image_path", None) is not None and self.i2v_training:
                sample_prompts_image_embs[prompt_dict["image_path"]] = None  # this will be replaced with CLIP context
            if prompt_dict.get("end_image_path", None) is not None and self.i2v_training:
                sample_prompts_image_embs[prompt_dict["end_image_path"]] = None

        if len(sample_prompts_image_embs) > 0 and not self.config.v2_2:  # Wan2.2 does not use CLIP for I2V training
            logger.info(f"loading CLIP: {clip_path}")
            assert clip_path is not None, "CLIP path is required for I2V training / I2V学習にはCLIPのパスが必要です"
            clip = CLIPModel(dtype=config.clip_dtype, device=device, weight_path=clip_path)
            clip.model.to(device)

            logger.info("Encoding image to CLIP context")
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
                for image_path in sample_prompts_image_embs:
                    logger.info(f"Encoding image: {image_path}")
                    img = Image.open(image_path).convert("RGB")
                    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)  # -1 to 1
                    clip_context = clip.visual([img[:, None, :, :]])
                    sample_prompts_image_embs[image_path] = clip_context

            del clip
            clean_memory_on_device(device)

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            p = prompt_dict.get("prompt", "")
            prompt_dict_copy["t5_embeds"] = te_outputs_1[p][0]

            p = prompt_dict.get("negative_prompt", None)
            if p is not None:
                prompt_dict_copy["negative_t5_embeds"] = te_outputs_1[p][0]

            p = prompt_dict.get("image_path", None)
            if p is not None and self.i2v_training:
                prompt_dict_copy["clip_embeds"] = sample_prompts_image_embs[p]

            p = prompt_dict.get("end_image_path", None)
            if p is not None and self.i2v_training:
                prompt_dict_copy["end_image_clip_embeds"] = sample_prompts_image_embs[p]

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
        model: WanModel = transformer
        device = accelerator.device

        if self.high_low_training:
            self.next_model_is_high_noise = False  # We use low noise model to sample the video
            self.swap_high_low_weights(args, accelerator, model)

        # TODO support different cfg_scale for low and high noise models
        if cfg_scale is None:
            cfg_scale = self.config.sample_guide_scale[0]  # use low noise guide scale by default
        do_classifier_free_guidance = do_classifier_free_guidance and cfg_scale != 1.0

        # prepare parameters
        one_frame_mode = args.one_frame
        if one_frame_mode:
            target_index, control_indices, f_indices, one_frame_inference_index = parse_one_frame_inference_args(
                sample_parameter["one_frame"]
            )
            latent_video_length = len(f_indices)  # number of frames in the video
        else:
            target_index, control_indices, f_indices, one_frame_inference_index = None, None, None, None

            # Calculate latent video length based on VAE version
            latent_video_length = (frame_count - 1) // self.config["vae_stride"][0] + 1

        # Get embeddings
        context = sample_parameter["t5_embeds"].to(device=device)
        if do_classifier_free_guidance:
            context_null = sample_parameter["negative_t5_embeds"].to(device=device)
        else:
            context_null = None

        num_channels_latents = 16  # model.in_dim
        vae_scale_factor = self.config["vae_stride"][1]

        # Initialize latents
        lat_h = height // vae_scale_factor
        lat_w = width // vae_scale_factor
        shape_or_frame = (1, num_channels_latents, 1, lat_h, lat_w)
        latents = []
        for _ in range(latent_video_length):
            latents.append(torch.randn(shape_or_frame, generator=generator, device=device, dtype=torch.float32))
        latents = torch.cat(latents, dim=2)

        image_latents = None

        if one_frame_mode:
            # One frame inference mode
            logger.info(
                f"One frame inference mode: target_index={target_index}, control_indices={control_indices}, f_indices={f_indices}"
            )
            vae.to(device)
            vae.eval()

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
                with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
                    image = image.to(device=device)
                    latent = vae.encode(image)[0]
                return latent, alpha

            control_latents = []
            control_alphas = []
            if "control_image_path" in sample_parameter:
                for control_image_path in sample_parameter["control_image_path"]:
                    control_latent, control_alpha = encode_image(control_image_path)
                    control_latents.append(control_latent)
                    control_alphas.append(control_alpha)

            with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
                black_image_latent = vae.encode([torch.zeros((3, 1, height, width), dtype=torch.float32, device=device)])[0]

            # Create latent and mask for the required number of frames
            image_latents = torch.zeros(4 + 16, len(f_indices), lat_h, lat_w, dtype=torch.float32, device=device)
            ci = 0
            for j, index in enumerate(f_indices):
                if index == target_index:
                    image_latents[4:, j : j + 1, :, :] = black_image_latent  # set black latent for the target frame
                else:
                    image_latents[:4, j, :, :] = 1.0  # set mask to 1.0 for the clean latent frames
                    image_latents[4:, j : j + 1, :, :] = control_latents[ci]  # set control latent
                    ci += 1
            image_latents = image_latents.unsqueeze(0)  # add batch dim

            vae.to("cpu")
            clean_memory_on_device(device)

        elif self.i2v_training or self.control_training:
            # Move VAE to the appropriate device for sampling: consider to cache image latents in CPU in advance
            vae.to(device)
            vae.eval()

            if self.i2v_training:
                image = Image.open(image_path)
                image = resize_image_to_bucket(image, (width, height))  # returns a numpy array
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(1).float()  # C, 1, H, W
                image = image / 127.5 - 1  # -1 to 1

                # Create mask for the required number of frames
                msk = torch.ones(1, frame_count, lat_h, lat_w, device=device)
                msk[:, 1:] = 0
                msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
                msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
                msk = msk.transpose(1, 2)  # B, C, T, H, W

                with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
                    # Zero padding for the required number of frames only
                    padding_frames = frame_count - 1  # The first frame is the input image
                    image = torch.concat([image, torch.zeros(3, padding_frames, height, width)], dim=1).to(device=device)
                    y = vae.encode([image])[0]

                y = y[:, :latent_video_length]  # may be not needed
                y = y.unsqueeze(0)  # add batch dim
                image_latents = torch.concat([msk, y], dim=1)

            if self.control_training:
                # Control video
                video = load_video(control_video_path, 0, frame_count, bucket_reso=(width, height))  # list of frames
                video = np.stack(video, axis=0)  # F, H, W, C
                video = torch.from_numpy(video).permute(3, 0, 1, 2).float()  # C, F, H, W
                video = video / 127.5 - 1  # -1 to 1
                video = video.to(device=device)

                with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
                    control_latents = vae.encode([video])[0]
                    control_latents = control_latents[:, :latent_video_length]
                    control_latents = control_latents.unsqueeze(0)  # add batch dim

                # We supports Wan2.1-Fun-Control only
                if image_latents is not None:
                    image_latents = image_latents[:, 4:]  # remove mask for Wan2.1-Fun-Control
                    image_latents[:, :, 1:] = 0  # remove except the first frame
                else:
                    image_latents = torch.zeros_like(control_latents)  # B, C, F, H, W

                image_latents = torch.concat([control_latents, image_latents], dim=1)  # B, C, F, H, W

            vae.to("cpu")
            clean_memory_on_device(device)

        # use the default value for num_train_timesteps (1000)
        scheduler = FlowUniPCMultistepScheduler(shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(sample_steps, device=device, shift=discrete_flow_shift)
        timesteps = scheduler.timesteps

        # Generate noise for the required number of frames only
        noise = torch.randn(16, latent_video_length, lat_h, lat_w, dtype=torch.float32, generator=generator, device=device).to(
            "cpu"
        )

        # prepare the model input
        max_seq_len = latent_video_length * lat_h * lat_w // (self.config.patch_size[1] * self.config.patch_size[2])
        arg_c = {"context": [context], "seq_len": max_seq_len}
        arg_null = {"context": [context_null], "seq_len": max_seq_len}

        if self.i2v_training and not one_frame_mode:
            if not self.config.v2_2:
                arg_c["clip_fea"] = sample_parameter["clip_embeds"].to(device=device, dtype=dit_dtype)
                arg_null["clip_fea"] = arg_c["clip_fea"]

        if one_frame_mode:
            if not self.config.v2_2:
                if "end_image_clip_embeds" in sample_parameter:
                    arg_c["clip_fea"] = torch.cat(
                        [sample_parameter["clip_embeds"], sample_parameter["end_image_clip_embeds"]], dim=0
                    ).to(device=device, dtype=dit_dtype)
                else:
                    arg_c["clip_fea"] = sample_parameter["clip_embeds"].to(device=device, dtype=dit_dtype)
                arg_null["clip_fea"] = arg_c["clip_fea"]

            arg_c["f_indices"] = [f_indices]
            arg_null["f_indices"] = arg_c["f_indices"]
            # print(f"One arg_c: {arg_c}, arg_null: {arg_null}")

        if self.i2v_training or self.control_training:
            arg_c["y"] = image_latents
            arg_null["y"] = image_latents

        # Wrap the inner loop with tqdm to track progress over timesteps
        prompt_idx = sample_parameter.get("enum", 0)
        latent = noise
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc=f"Sampling timesteps for prompt {prompt_idx + 1}")):
                latent_model_input = [latent.to(device=device)]
                timestep = t.unsqueeze(0)

                with accelerator.autocast():
                    noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0].to("cpu")
                    if do_classifier_free_guidance:
                        noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0].to("cpu")
                    else:
                        noise_pred_uncond = None

                if do_classifier_free_guidance:
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                temp_x0 = scheduler.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=generator)[0]
                latent = temp_x0.squeeze(0)

        # Move VAE to the appropriate device for sampling
        vae.to(device)
        vae.eval()

        # Decode latents to video
        logger.info(f"Decoding video from latents: {latent.shape}")
        latent = latent.unsqueeze(0)  # add batch dim
        latent = latent.to(device=device)

        if one_frame_mode:
            latent = latent[:, :, one_frame_inference_index : one_frame_inference_index + 1, :, :]  # select the one frame
        with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
            video = vae.decode(latent)[0]  # vae returns list
        video = video.unsqueeze(0)  # add batch dim
        del latent

        logger.info("Decoding complete")
        video = video.to(torch.float32).cpu()
        video = (video / 2 + 0.5).clamp(0, 1)  # -1 to 1 -> 0 to 1

        vae.to("cpu")
        clean_memory_on_device(device)

        return video

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae

        logger.info(f"Loading VAE model from {vae_path}")
        cache_device = torch.device("cpu") if args.vae_cache_cpu else None
        vae = WanVAE(vae_path=vae_path, device="cpu", dtype=vae_dtype, cache_device=cache_device)
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
        model = load_wan_model(
            self.config,
            accelerator.device,
            dit_path,
            attn_mode,
            split_attn,
            loading_device,
            dit_weight_dtype,
            args.fp8_scaled,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )
        if args.force_v2_1_time_embedding:
            model.set_time_embedding_v2_1(True)

        if self.high_low_training:
            # load high noise model
            logger.info(f"Loading high noise model from {self.dit_high_noise_path}")
            model_high_noise = load_wan_model(
                self.config,
                accelerator.device,
                self.dit_high_noise_path,
                attn_mode,
                split_attn,
                "cpu" if args.offload_inactive_dit else loading_device,
                dit_weight_dtype,
                args.fp8_scaled,
                disable_numpy_memmap=args.disable_numpy_memmap,
            )
            if args.force_v2_1_time_embedding:
                model_high_noise.set_time_embedding_v2_1(True)
            if self.blocks_to_swap > 0:
                # This moves the weights to the appropriate device
                logger.info(f"Prepare block swap for high noise model, blocks_to_swap={self.blocks_to_swap}")
                model_high_noise.enable_block_swap(self.blocks_to_swap, accelerator.device, supports_backward=True)
                model_high_noise.move_to_device_except_swap_blocks(accelerator.device)
                model_high_noise.prepare_block_swap_before_forward()

            self.dit_inactive_state_dict = model_high_noise.state_dict()

            self.current_model_is_high_noise = False
            self.next_model_is_high_noise = False
        else:
            self.dit_inactive_state_dict = None
            self.current_model_is_high_noise = False
            self.next_model_is_high_noise = False

        return model

    def compile_transformer(self, args, transformer):
        transformer: WanModel = transformer
        return model_utils.compile_transformer(args, transformer, [transformer.blocks], disable_linear=self.blocks_to_swap > 0)

    def scale_shift_latents(self, latents):
        return latents

    def get_noisy_model_input_and_timesteps(
        self,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: Optional[List[float]],
        noise_scheduler: FlowMatchDiscreteScheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if not self.high_low_training:
            return super().get_noisy_model_input_and_timesteps(args, noise, latents, timesteps, noise_scheduler, device, dtype)

        # high-low training case
        # call super to get the noisy model input and timesteps, and sample only the first one, and choose the model we want based on the timestep
        noisy_model_input, sample_timesteps = super().get_noisy_model_input_and_timesteps(
            args, noise[0:1], latents[0:1], timesteps[0:1] if timesteps is not None else None, noise_scheduler, device, dtype
        )
        high_noise = sample_timesteps[0] / 1000.0 >= self.timestep_boundary
        self.next_model_is_high_noise = high_noise

        # choose each member of latents for high or low noise model. because we want to train all the latents
        num_max_calls = 100
        final_noisy_model_inputs = []
        final_timesteps_list = []
        bsize = latents.shape[0]
        for i in range(bsize):
            for _ in range(num_max_calls):
                ts_i = [self.get_bucketed_timestep()] if self.num_timestep_buckets is not None else None

                noisy_model_input, ts_i = super().get_noisy_model_input_and_timesteps(
                    args, noise[i : i + 1], latents[i : i + 1], ts_i, noise_scheduler, device, dtype
                )
                if (high_noise and ts_i[0] / 1000.0 >= self.timestep_boundary) or (
                    not high_noise and ts_i[0] / 1000.0 < self.timestep_boundary
                ):
                    final_noisy_model_inputs.append(noisy_model_input)
                    final_timesteps_list.append(ts_i)
                    break

        if len(final_noisy_model_inputs) < bsize:
            logger.warning(
                f"No valid noisy model inputs found for bsize={bsize}, high_noise={high_noise}, timestep_boundary={self.timestep_boundary}"
            )
            # fall back to the original method
            return super().get_noisy_model_input_and_timesteps(args, noise, latents, timesteps, noise_scheduler, device, dtype)

        # final noisy model input may have less than bsize elements, it will be fine for training
        final_noisy_model_input = torch.cat(final_noisy_model_inputs, dim=0)
        final_timesteps = torch.cat(final_timesteps_list, dim=0)

        return final_noisy_model_input, final_timesteps

    def swap_high_low_weights(self, args: argparse.Namespace, accelerator: Accelerator, model: WanModel):
        if self.current_model_is_high_noise != self.next_model_is_high_noise:

            def patch_fn(state_dict):
                if not args.compile:
                    return state_dict
                for key in list(state_dict.keys()):
                    if key.startswith("blocks.") and "._orig_mod." not in key:
                        tokens = key.split(".")
                        new_key = ".".join(tokens[:2] + ["_orig_mod"] + tokens[2:])
                        state_dict[new_key] = state_dict.pop(key)
                return state_dict

            if self.blocks_to_swap == 0:
                # If offloading inactive DiT, move the model to CPU first
                if args.offload_inactive_dit:
                    model.to("cpu", non_blocking=True)
                    synchronize_device(accelerator.device)  # wait for the CPU to finish
                    clean_memory_on_device(accelerator.device)

                state_dict = model.state_dict()  # CPU or accelerator.device

                info = model.load_state_dict(patch_fn(self.dit_inactive_state_dict), strict=True, assign=True)
                assert len(info.missing_keys) == 0, f"Missing keys: {info.missing_keys}"
                assert len(info.unexpected_keys) == 0, f"Unexpected keys: {info.unexpected_keys}"

                if args.offload_inactive_dit:
                    model.to(accelerator.device, non_blocking=True)
                    synchronize_device(accelerator.device)

                self.dit_inactive_state_dict = state_dict  # swap the state dict
            else:
                # If block swap is enabled, we cannot use offloading inactive DiT, because weights are partially on CPU
                state_dict = model.state_dict()  # CPU or accelerator.device

                info = model.load_state_dict(patch_fn(self.dit_inactive_state_dict), strict=True, assign=True)
                assert len(info.missing_keys) == 0, f"Missing keys: {info.missing_keys}"
                assert len(info.unexpected_keys) == 0, f"Unexpected keys: {info.unexpected_keys}"

                self.dit_inactive_state_dict = state_dict  # swap the state dict

            self.current_model_is_high_noise = self.next_model_is_high_noise

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
        if self.high_low_training:
            # high-low training case
            self.swap_high_low_weights(args, accelerator, transformer)

        # Call the DiT model
        return self._call_dit(args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype)

    def _call_dit(
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
        model: WanModel = transformer

        # I2V training and Control training
        image_latents = None
        clip_fea = None
        if self.i2v_training:
            image_latents = batch["latents_image"]
            image_latents = image_latents.to(device=accelerator.device, dtype=network_dtype)

            if not self.config.v2_2:
                clip_fea = batch["clip"]
                clip_fea = clip_fea.to(device=accelerator.device, dtype=network_dtype)

                # clip_fea is [B, N, D] (normal) or [B, 1, N, D] (one frame) for I2V, and [B, 2, N, D] for FLF2V, we need to reshape it to [B, N, D] for I2V and [B*2, N, D] for FLF2V
                if clip_fea.shape[1] == 1:
                    clip_fea = clip_fea.squeeze(1)
                elif clip_fea.shape[1] == 2:
                    clip_fea = clip_fea.view(-1, clip_fea.shape[2], clip_fea.shape[3])

        if self.control_training:
            control_latents = batch["latents_control"]
            control_latents = control_latents.to(device=accelerator.device, dtype=network_dtype)
            if image_latents is not None:
                image_latents = image_latents[:, 4:]  # remove mask for Wan2.1-Fun-Control
                image_latents[:, :, 1:] = 0  # remove except the first frame
            else:
                image_latents = torch.zeros_like(control_latents)  # B, C, F, H, W
            image_latents = torch.concat([control_latents, image_latents], dim=1)  # B, C, F, H, W
            control_latents = None

        context = [t.to(device=accelerator.device, dtype=network_dtype) for t in batch["t5"]]

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in context:
                t.requires_grad_(True)
            if image_latents is not None:
                image_latents.requires_grad_(True)
            if clip_fea is not None:
                clip_fea.requires_grad_(True)

        # call DiT
        lat_f, lat_h, lat_w = latents.shape[2:5]
        seq_len = lat_f * lat_h * lat_w // (self.config.patch_size[0] * self.config.patch_size[1] * self.config.patch_size[2])
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        with accelerator.autocast():
            model_pred = model(noisy_model_input, t=timesteps, context=context, clip_fea=clip_fea, seq_len=seq_len, y=image_latents)
        model_pred = torch.stack(model_pred, dim=0)  # list to tensor

        # flow matching loss
        target = noise - latents

        return model_pred, target

    # endregion model specific


def wan_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Wan2.1/2.2 specific parser setup"""
    parser.add_argument("--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--t5", type=str, default=None, help="text encoder (T5) checkpoint path")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        help="text encoder (CLIP) checkpoint path, optional. If training Wan2.1 I2V model, this is required",
    )
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    parser.add_argument("--one_frame", action="store_true", help="Use one frame sampling method for sample generation")

    # Wan2.2 specific arguments
    parser.add_argument(
        "--force_v2_1_time_embedding", action="store_true", help="Force using 2.1 style time embedding even for Wan 2.2"
    )
    parser.add_argument("--dit_high_noise", type=str, required=False, default=None, help="DiT checkpoint path for high noise model")
    parser.add_argument(
        "--timestep_boundary",
        type=int,
        default=None,
        help="Timestep boundary for switching between high and low noise models, defaults to None (task specific) / 高ノイズモデルと低ノイズモデルを切り替えるタイムステップ境界。デフォルトはNone（タスク固有）",
    )
    parser.add_argument(
        "--offload_inactive_dit",
        action="store_true",
        help="Offload inactive DiT model to CPU. Cannot be used with block swap / アクティブではないDiTモデルをCPUにオフロードします。ブロックスワップと併用できません",
    )

    return parser


def main():
    parser = setup_parser_common()
    parser = wan_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # automatically detected
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"  # make bfloat16 as default for VAE

    trainer = WanNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
