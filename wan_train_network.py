import argparse
from typing import Optional
from PIL import Image


import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from accelerate import Accelerator, init_empty_weights

from dataset.image_video_dataset import ARCHITECTURE_WAN, ARCHITECTURE_WAN_FULL, load_video
from hv_generate_video import resize_image_to_bucket
from hv_train_network import NetworkTrainer, load_prompts, clean_memory_on_device, setup_parser_common, read_config_from_file

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from utils import model_utils
from utils.safetensors_utils import load_safetensors, MemoryEfficientSafeOpen
from wan.configs import WAN_CONFIGS
from wan.modules.clip import CLIPModel
from wan.modules.model import WanModel, detect_wan_sd_dtype, load_wan_model
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


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
        self._i2v_training = "i2v" in args.task  # we cannot use config.i2v because Fun-Control T2V has i2v flag TODO refactor this
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

        if len(sample_prompts_image_embs) > 0:
            logger.info(f"loading CLIP: {clip_path}")
            assert clip_path is not None, "CLIP path is required for I2V training / I2V学習にはCLIPのパスが必要です"
            clip = CLIPModel(dtype=config.clip_dtype, device=device, weight_path=clip_path)
            clip.model.to(device)

            logger.info(f"Encoding image to CLIP context")
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
        if cfg_scale is None:
            cfg_scale = 5.0
        do_classifier_free_guidance = do_classifier_free_guidance and cfg_scale != 1.0

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
        if self.i2v_training or self.control_training:
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

        if self.i2v_training:
            arg_c["clip_fea"] = sample_parameter["clip_embeds"].to(device=device, dtype=dit_dtype)
            arg_null["clip_fea"] = arg_c["clip_fea"]
        if self.i2v_training or self.control_training:
            arg_c["y"] = image_latents
            arg_null["y"] = image_latents

        # Wrap the inner loop with tqdm to track progress over timesteps
        prompt_idx = sample_parameter.get("enum", 0)
        latent = noise
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc=f"Sampling timesteps for prompt {prompt_idx+1}")):
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

        with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
            video = vae.decode(latent)[0]  # vae returns list
        video = video.unsqueeze(0)  # add batch dim
        del latent

        logger.info(f"Decoding complete")
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
            self.config, accelerator.device, dit_path, attn_mode, split_attn, loading_device, dit_weight_dtype, args.fp8_scaled
        )
        return model

    def scale_shift_latents(self, latents):
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
        model: WanModel = transformer

        # I2V training and Control training
        image_latents = None
        clip_fea = None
        if self.i2v_training:
            image_latents = batch["latents_image"]
            image_latents = image_latents.to(device=accelerator.device, dtype=network_dtype)
            clip_fea = batch["clip"]
            clip_fea = clip_fea.to(device=accelerator.device, dtype=network_dtype)
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
    """Wan2.1 specific parser setup"""
    parser.add_argument("--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--t5", type=str, default=None, help="text encoder (T5) checkpoint path")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        help="text encoder (CLIP) checkpoint path, optional. If training I2V model, this is required",
    )
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    return parser


if __name__ == "__main__":
    parser = setup_parser_common()
    parser = wan_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # automatically detected
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"  # make bfloat16 as default for VAE

    trainer = WanNetworkTrainer()
    trainer.train(args)
