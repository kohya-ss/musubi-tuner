import argparse
import logging
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_HUNYUAN_VIDEO_1_5,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL,
    resize_image_to_bucket,
)
from musubi_tuner.frame_pack.clip_vision import hf_clip_vision_encode
from musubi_tuner.frame_pack.framepack_utils import load_image_encoders
from musubi_tuner.hunyuan_video_1_5 import (
    hunyuan_video_1_5_models,
    hunyuan_video_1_5_text_encoder,
    hunyuan_video_1_5_utils,
    hunyuan_video_1_5_vae,
)
from musubi_tuner.hunyuan_video_1_5.hunyuan_video_1_5_models import (
    HunyuanVideo_1_5_DiffusionTransformer,
    detect_hunyuan_video_1_5_sd_dtype,
)
from musubi_tuner.hunyuan_video_1_5.hunyuan_video_1_5_vae import VAE_LATENT_CHANNELS
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    clean_memory_on_device,
    load_prompts,
    read_config_from_file,
    setup_parser_common,
)
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HunyuanVideo15NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 6.0

    # region model specific
    @property
    def architecture(self) -> str:
        return ARCHITECTURE_HUNYUAN_VIDEO_1_5

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        self._i2v_training = args.task == "i2v"
        self._control_training = False

        # Detect the original dtype from checkpoint to prevent incompatible dtype conversions
        # float16 checkpoints cannot be safely converted to bfloat16/float32
        sd_dit_dtype = detect_hunyuan_video_1_5_sd_dtype(args.dit)
        assert not (sd_dit_dtype is torch.float16 and args.dit_dtype in ["bfloat16", "float32"]), (
            "Loaded DiT checkpoint is float16, cannot override dit_dtype to bfloat16 or float32."
            " / DiTの重みがfloat16のため、dit_dtypeをbfloat16またはfloat32に設定できません。"
        )
        # Use checkpoint's native dtype if not explicitly specified to preserve precision
        if args.dit_dtype is None:
            args.dit_dtype = "float16" if sd_dit_dtype == torch.float16 else "bfloat16"
        # VAE defaults to float16 for VRAM efficiency while maintaining acceptable quality
        if args.vae_dtype is None:
            args.vae_dtype = "float16"

    @property
    def i2v_training(self) -> bool:
        return self._i2v_training

    @property
    def control_training(self) -> bool:
        return self._control_training

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        device = accelerator.device
        logger.info("cache Text Encoder outputs for sample prompt: %s", sample_prompts)
        prompts = load_prompts(sample_prompts)

        # HV1.5 uses Qwen2.5-VL as the primary text encoder; fp8 optional for VRAM savings
        vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
        tokenizer_vlm, text_encoder_vlm = qwen_image_utils.load_qwen2_5_vl(args.text_encoder, vl_dtype, device, disable_mmap=True)
        # BYT5 is used as a secondary encoder for glyph/character-level understanding
        tokenizer_byt5, text_encoder_byt5 = hunyuan_video_1_5_text_encoder.load_byt5(
            args.byt5, dtype=torch.float16, device=device, disable_mmap=True
        )

        sample_prompts_te_outputs = {}
        with torch.no_grad():
            for prompt_dict in prompts:
                if "negative_prompt" not in prompt_dict:
                    # empty negative prompt if not provided, this can be ignored with cfg_scale=1.0
                    prompt_dict["negative_prompt"] = ""
                for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                    if p is None or p in sample_prompts_te_outputs:
                        continue
                    embed_vlm, mask_vlm = hunyuan_video_1_5_text_encoder.get_qwen_prompt_embeds(tokenizer_vlm, text_encoder_vlm, p)
                    embed_byt5, mask_byt5 = hunyuan_video_1_5_text_encoder.get_glyph_prompt_embeds(
                        tokenizer_byt5, text_encoder_byt5, p
                    )
                    embed_vlm = embed_vlm.to("cpu")
                    mask_vlm = mask_vlm.to("cpu")
                    embed_byt5 = embed_byt5.to("cpu")
                    mask_byt5 = mask_byt5.to("cpu")
                    sample_prompts_te_outputs[p] = (embed_vlm, mask_vlm, embed_byt5, mask_byt5)

        # Release text encoders immediately after caching to free VRAM for DiT inference
        del tokenizer_vlm, text_encoder_vlm, tokenizer_byt5, text_encoder_byt5
        clean_memory_on_device(device)

        # image embedding for I2V training
        sample_prompts_image_embs = {}
        if self.i2v_training:
            feature_extractor, image_encoder = load_image_encoders(args)
            image_encoder.to(device)

            # encode image with image encoder
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
                img_np = resize_image_to_bucket(img_np, (width, height))  # returns a numpy array

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

            p = prompt_dict_copy.get("prompt", "")
            embed_vlm, mask_vlm, embed_byt5, mask_byt5 = sample_prompts_te_outputs[p]
            prompt_dict_copy["vl_embed"] = embed_vlm
            prompt_dict_copy["vl_mask"] = mask_vlm
            prompt_dict_copy["byt5_embed"] = embed_byt5
            prompt_dict_copy["byt5_mask"] = mask_byt5

            p = prompt_dict_copy.get("negative_prompt", "")
            neg_embed_vlm, neg_mask_vlm, neg_embed_byt5, neg_mask_byt5 = sample_prompts_te_outputs[p]
            prompt_dict_copy["negative_vl_embed"] = neg_embed_vlm
            prompt_dict_copy["negative_vl_mask"] = neg_mask_vlm
            prompt_dict_copy["negative_byt5_embed"] = neg_embed_byt5
            prompt_dict_copy["negative_byt5_mask"] = neg_mask_byt5

            p = prompt_dict_copy.get("image_path", None)  # for I2V, None for T2V
            prompt_dict_copy["image_encoder_last_hidden_state"] = sample_prompts_image_embs.get(p, None)

            sample_parameters.append(prompt_dict_copy)

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
        """architecture dependent inference for sampling"""
        device = accelerator.device
        if do_classifier_free_guidance and cfg_scale is None:
            logger.info(f"Using default guidance scale: {self.default_guidance_scale}")
        cfg_scale = cfg_scale if cfg_scale is not None else self.default_guidance_scale

        # Skip CFG computation entirely when scale is 1.0 to save inference time
        do_cfg = do_classifier_free_guidance and cfg_scale != 1.0

        # Latent dimensions are 1/16 of image dimensions spatially, and (frames-1)/4 + 1 temporally
        # This matches the VAE's compression ratio
        lat_f = 1 + (frame_count - 1) // 4
        lat_h = height // 16
        lat_w = width // 16

        if self.i2v_training:
            # Move VAE to the appropriate device for sampling: consider to cache image latents in CPU in advance
            logger.info("Encoding image to latent space")
            vae_original_device = vae.device
            vae.to(device)
            vae.eval()

            img = Image.open(image_path).convert("RGB")
            img_np = resize_image_to_bucket(img, (width, height))  # returns a numpy array

            # convert to tensor (-1 to 1)
            img_tensor = TF.to_tensor(img_np).sub_(0.5).div_(0.5).to(device)
            img_tensor = img_tensor[None, :, None, :, :]  # BCFHW, B=1, F=1

            # encode image to latent space
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True), torch.no_grad():
                cond_latents = vae.encode(img_tensor)[0].mode()
                cond_latents = cond_latents * vae.scaling_factor

            # prepare mask for image latent
            latent_mask = torch.zeros(1, 1, lat_f, lat_h, lat_w, device=device)
            latent_mask[0, 0, 0, :, :] = 1.0  # first frame is image

            latents_concat = torch.zeros(
                1, hunyuan_video_1_5_vae.VAE_LATENT_CHANNELS, lat_f, lat_h, lat_w, dtype=torch.float32, device=device
            )
            latents_concat[:, :, 0:1, :, :] = cond_latents

            cond_latents = torch.concat([latents_concat, latent_mask], dim=1)

            vae.to(vae_original_device)
            if vae_original_device != device:
                clean_memory_on_device(device)
        else:
            # T2V mode
            cond_latents = torch.zeros(
                1, hunyuan_video_1_5_vae.VAE_LATENT_CHANNELS + 1, lat_f, lat_h, lat_w, dtype=torch.float32, device=device
            )

        timesteps, sigmas = hunyuan_video_1_5_utils.get_timesteps_sigmas(sample_steps, discrete_flow_shift, device)
        latents = torch.randn(
            (1, hunyuan_video_1_5_vae.VAE_LATENT_CHANNELS, lat_f, lat_h, lat_w), generator=generator, device=device, dtype=dit_dtype
        )

        vl_embed = sample_parameter["vl_embed"].to(device, dtype=torch.bfloat16)
        vl_mask = sample_parameter["vl_mask"].to(device, dtype=torch.bool)
        byt5_embed = sample_parameter["byt5_embed"].to(device, dtype=torch.bfloat16)
        byt5_mask = sample_parameter["byt5_mask"].to(device, dtype=torch.bool)

        if do_cfg:
            negative_vl_embed = sample_parameter["negative_vl_embed"].to(device, dtype=torch.bfloat16)
            negative_vl_mask = sample_parameter["negative_vl_mask"].to(device, dtype=torch.bool)
            negative_byt5_embed = sample_parameter["negative_byt5_embed"].to(device, dtype=torch.bfloat16)
            negative_byt5_mask = sample_parameter["negative_byt5_mask"].to(device, dtype=torch.bool)
        else:
            negative_vl_embed = negative_vl_mask = negative_byt5_embed = negative_byt5_mask = None

        image_encoder_last_hidden_state = sample_parameter["image_encoder_last_hidden_state"]
        if image_encoder_last_hidden_state is not None:
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(device)

        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                timestep = t.expand(latents.shape[0])
                # Concatenate noise latents with conditioning latents along channel dimension
                # This is how HV1.5 architecture handles I2V conditioning
                latents_concat = torch.cat([latents, cond_latents], dim=1)
                with accelerator.autocast():
                    noise_pred = transformer(
                        hidden_states=latents_concat,
                        timestep=timestep,
                        text_states=vl_embed,
                        encoder_attention_mask=vl_mask,
                        vision_states=image_encoder_last_hidden_state,
                        byt5_text_states=byt5_embed,
                        byt5_text_mask=byt5_mask,
                        rotary_pos_emb_cache=None,
                    )

                    if do_cfg:
                        # CFG: predict noise for negative prompt, then interpolate
                        # noise_pred = negative + scale * (positive - negative)
                        latents_concat = torch.cat([latents, cond_latents], dim=1)
                        neg_noise_pred = transformer(
                            hidden_states=latents_concat,
                            timestep=timestep,
                            text_states=negative_vl_embed,
                            encoder_attention_mask=negative_vl_mask,
                            vision_states=image_encoder_last_hidden_state,
                            byt5_text_states=negative_byt5_embed,
                            byt5_text_mask=negative_byt5_mask,
                            rotary_pos_emb_cache=None,
                        )
                        noise_pred = neg_noise_pred + cfg_scale * (noise_pred - neg_noise_pred)

                latents = hunyuan_video_1_5_utils.step(latents, noise_pred, sigmas, i)

        # VAE decode: move to device just before use to minimize VRAM usage during denoising
        vae_original_device = vae.device
        vae.to(device)
        with torch.autocast(device_type=device.type, dtype=model_utils.str_to_dtype(args.vae_dtype)), torch.no_grad():
            decoded = vae.decode(latents / vae.scaling_factor)[0]
        vae.to(vae_original_device)

        # Convert to float32 for video saving to avoid precision issues
        decoded = decoded.to(torch.float32).cpu() * 0.5 + 0.5  # scale to [0, 1]
        return decoded

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        logger.info(f"Loading VAE model from {vae_path}")
        vae = hunyuan_video_1_5_vae.load_vae_from_checkpoint(
            vae_path, device="cpu", dtype=vae_dtype, sample_size=args.vae_sample_size, enable_patch_conv=args.vae_enable_patch_conv
        )
        vae.eval()
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
        # Select T2V or I2V model variant based on training mode
        task_type = "i2v" if self._i2v_training else "t2v"
        transformer = hunyuan_video_1_5_models.load_hunyuan_video_1_5_model(
            device=accelerator.device,
            task_type=task_type,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=args.fp8_scaled,
        )
        return transformer

    def compile_transformer(self, args, transformer):
        transformer: HunyuanVideo_1_5_DiffusionTransformer = transformer
        # Disable linear compilation when block swapping is enabled
        # because torch.compile doesn't work well with dynamic module movement
        return model_utils.compile_transformer(
            args, transformer, [transformer.double_blocks], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
        latents = latents * hunyuan_video_1_5_vae.VAE_SCALING_FACTOR
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer_arg,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        transformer: HunyuanVideo_1_5_DiffusionTransformer = transformer_arg

        # Check if this batch has I2V conditioning (first frame latents)
        cond_latents = batch.get("latents_image", None)
        if cond_latents is None:
            assert not self.i2v_training, (
                "Expected latents_image for I2V training. Add `--i2v` and `--image_encoder` arguments for `hv_1_5_cache_latents` script."
                + " / I2V学習ではlatents_imageが必要です。`hv_1_5_cache_latents`スクリプトに`--i2v`と`--image_encoder`引数を追加してください。"
            )
            # For T2V batches, create zero conditioning tensor
            # Extra channel (+1) is the conditioning mask, all zeros means "no conditioning"
            cond_latents = torch.zeros(
                (latents.shape[0], VAE_LATENT_CHANNELS + 1, *latents.shape[2:]), device=latents.device, dtype=latents.dtype
            )

        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        cond_latents = cond_latents.to(device=accelerator.device, dtype=network_dtype)

        # HV1.5 concatenates noisy latents with conditioning along channel dim
        latents_concat = torch.cat([noisy_model_input, cond_latents], dim=1)

        def pad_varlen(seq_list: list[torch.Tensor]):
            """Pad variable-length sequences in batch to the maximum length.

            Different prompts have different token counts, so we need to pad
            to create uniform tensors for batched processing.
            """
            lengths = [t.shape[0] for t in seq_list]
            max_len = max(lengths)
            padded = []
            for t in seq_list:
                if t.shape[0] < max_len:
                    t = torch.nn.functional.pad(t, (0, 0, 0, max_len - t.shape[0]))
                padded.append(t)
            stacked = torch.stack(padded, dim=0)

            # Create attention mask: True for valid positions, False for padding
            mask = torch.zeros((len(seq_list), max_len), device=accelerator.device, dtype=torch.bool)
            for i, l in enumerate(lengths):
                mask[i, :l] = True
            return stacked.to(device=accelerator.device, dtype=network_dtype), mask

        vl_embed, vl_mask = pad_varlen(batch["vl_embed"])
        byt5_embed, byt5_mask = pad_varlen(batch["byt5_embed"])
        # SigLIP vision states for I2V image understanding (optional)
        vision_states = batch.get("siglip", None)
        if vision_states is not None:
            vision_states = vision_states.to(device=accelerator.device, dtype=network_dtype)

        # Enable gradient computation for inputs when using gradient checkpointing
        # Required because checkpointing recomputes forward pass during backward
        if args.gradient_checkpointing:
            latents_concat.requires_grad_(True)
            vl_embed.requires_grad_(True)
            byt5_embed.requires_grad_(True)
            if vision_states is not None:
                vision_states.requires_grad_(True)

        with accelerator.autocast():
            model_pred = transformer(
                hidden_states=latents_concat,
                timestep=timesteps,
                text_states=vl_embed,
                encoder_attention_mask=vl_mask,
                vision_states=vision_states,
                byt5_text_states=byt5_embed,
                byt5_text_mask=byt5_mask,
                rotary_pos_emb_cache=None,
            )

        # Flow matching target: predict the velocity (noise - clean)
        # This is different from DDPM which predicts noise directly
        target = noise - latents
        return model_pred, target

    # endregion model specific


def hv1_5_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """HunyuanVideo-1.5 specific parser setup"""
    parser.add_argument(
        "--task",
        type=str,
        default="t2v",
        choices=["t2v", "i2v"],
        help="training task type: text-to-video (t2v) or image-to-video (i2v)",
    )
    parser.add_argument("--dit_dtype", type=str, default=None, help="data type for DiT, default is bfloat16")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT")
    parser.add_argument("--text_encoder", type=str, default=None, help="text encoder (Qwen2.5-VL) checkpoint path")
    parser.add_argument("--fp8_vl", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument("--byt5", type=str, default=None, help="BYT5 checkpoint path")
    parser.add_argument("--image_encoder", type=str, default=None, help="SigLIP image encoder path (for I2V cache compatibility)")
    parser.add_argument(
        "--vae_sample_size",
        type=int,
        default=128,
        help="VAE sample size (height/width). Default 128; set 256 if VRAM is sufficient for better quality. Set 0 to disable tiling.",
    )
    parser.add_argument(
        "--vae_enable_patch_conv",
        action="store_true",
        help="Enable patch-based convolution in VAE for memory optimization",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = hv1_5_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    trainer = HunyuanVideo15NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
