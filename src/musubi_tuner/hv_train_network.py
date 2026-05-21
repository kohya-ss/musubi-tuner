"""HunyuanVideo training entry point.

Defines HunyuanVideoNetworkTrainer, the concrete NetworkTrainer subclass for
HunyuanVideo. The shared NetworkTrainer base class and helpers live under the
musubi_tuner.training package; this module re-exports those symbols so existing
imports from architecture-specific training scripts (wan_train_network,
fpack_train_network, ...) keep working.
"""

import argparse
import logging
from typing import Optional

import torch
from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator

from musubi_tuner import convert_lora
from musubi_tuner.training.trainer_base import (
    DiTOutput,
    NetworkTrainer,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
    SS_METADATA_MINIMUM_KEYS,
)
from musubi_tuner.training.accelerator_setup import (
    clean_memory_on_device,
    collator_class,
    prepare_accelerator,
)
from musubi_tuner.training.sampling_prompts import (
    line_to_prompt_dict,
    load_prompts,
    should_sample_images,
)
from musubi_tuner.training.timesteps import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    get_sigmas,
)
from musubi_tuner.training.parser_common import (
    setup_parser_common,
    read_config_from_file,
)

from musubi_tuner.hunyuan_model.models import (
    load_transformer as hv_load_transformer,
    get_rotary_pos_embed_by_shape,
    HYVideoDiffusionTransformer,
)
import musubi_tuner.hunyuan_model.text_encoder as text_encoder_module
from musubi_tuner.hunyuan_model.vae import load_vae as hv_load_vae, VAE_VER
import musubi_tuner.hunyuan_model.vae as vae_module
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
import musubi_tuner.networks.lora as lora_module
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_HUNYUAN_VIDEO,
    ARCHITECTURE_HUNYUAN_VIDEO_FULL,
)
from musubi_tuner.hv_generate_video import resize_image_to_bucket, encode_to_latents
from musubi_tuner.utils import model_utils

# accelerate.set_seed is re-exported because qwen_image_train.py and
# zimage_train.py import it from this module.
from accelerate.utils import set_seed

__all__ = [
    # trainer_base
    "DiTOutput",
    "NetworkTrainer",
    "HunyuanVideoNetworkTrainer",
    "SS_METADATA_KEY_BASE_MODEL_VERSION",
    "SS_METADATA_KEY_NETWORK_MODULE",
    "SS_METADATA_KEY_NETWORK_DIM",
    "SS_METADATA_KEY_NETWORK_ALPHA",
    "SS_METADATA_KEY_NETWORK_ARGS",
    "SS_METADATA_MINIMUM_KEYS",
    # accelerator_setup
    "clean_memory_on_device",
    "collator_class",
    "prepare_accelerator",
    # sampling_prompts
    "line_to_prompt_dict",
    "load_prompts",
    "should_sample_images",
    # timesteps
    "compute_density_for_timestep_sampling",
    "compute_loss_weighting_for_sd3",
    "get_sigmas",
    # parser_common
    "setup_parser_common",
    "read_config_from_file",
    # accelerate
    "set_seed",
]

logger = logging.getLogger(__name__)


class HunyuanVideoNetworkTrainer(NetworkTrainer):
    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_HUNYUAN_VIDEO

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_HUNYUAN_VIDEO_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        self.pos_embed_cache = {}

        self._i2v_training = args.dit_in_channels == 32  # may be changed in the future
        if self._i2v_training:
            logger.info("I2V training mode")

        self._control_training = False  # HunyuanVideo does not support control training yet

        self.default_guidance_scale = 6.0

    def convert_weight_keys(self, weights_sd: dict[str, torch.Tensor], network_module: lora_module):
        keys = list(weights_sd.keys())
        if keys[0].startswith("lora_"):
            return weights_sd  # default format
        if keys[0].startswith("diffusion_model.") or keys[0].startswith("transformer."):
            # Diffusers? format
            logger.info("converting LoRA weights from diffusers format to default format")
            return convert_lora.convert_from_diffusers("lora_unet_", weights_sd)
        return weights_sd  # unknown format, return as is

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        text_encoder1, text_encoder2, fp8_llm = args.text_encoder1, args.text_encoder2, args.fp8_llm

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        def encode_for_text_encoder(text_encoder, is_llm=True):
            sample_prompts_te_outputs = {}  # (prompt) -> (embeds, mask)
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", None)]:
                        if p is None:
                            continue
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"cache Text Encoder outputs for prompt: {p}")

                            data_type = "video"
                            text_inputs = text_encoder.text2tokens(p, data_type=data_type)

                            prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)
                            sample_prompts_te_outputs[p] = (prompt_outputs.hidden_state, prompt_outputs.attention_mask)

            return sample_prompts_te_outputs

        # Load Text Encoder 1 and encode
        text_encoder_dtype = torch.float16 if args.text_encoder_dtype is None else model_utils.str_to_dtype(args.text_encoder_dtype)
        logger.info(f"loading text encoder 1: {text_encoder1}")
        text_encoder_1 = text_encoder_module.load_text_encoder_1(text_encoder1, accelerator.device, fp8_llm, text_encoder_dtype)

        logger.info("encoding with Text Encoder 1")
        te_outputs_1 = encode_for_text_encoder(text_encoder_1)
        del text_encoder_1

        # Load Text Encoder 2 and encode
        logger.info(f"loading text encoder 2: {text_encoder2}")
        text_encoder_2 = text_encoder_module.load_text_encoder_2(text_encoder2, accelerator.device, text_encoder_dtype)

        logger.info("encoding with Text Encoder 2")
        te_outputs_2 = encode_for_text_encoder(text_encoder_2, is_llm=False)
        del text_encoder_2

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            p = prompt_dict.get("prompt", "")
            prompt_dict_copy["llm_embeds"] = te_outputs_1[p][0]
            prompt_dict_copy["llm_mask"] = te_outputs_1[p][1]
            prompt_dict_copy["clipL_embeds"] = te_outputs_2[p][0]
            prompt_dict_copy["clipL_mask"] = te_outputs_2[p][1]

            p = prompt_dict.get("negative_prompt", None)
            if p is not None:
                prompt_dict_copy["negative_llm_embeds"] = te_outputs_1[p][0]
                prompt_dict_copy["negative_llm_mask"] = te_outputs_1[p][1]
                prompt_dict_copy["negative_clipL_embeds"] = te_outputs_2[p][0]
                prompt_dict_copy["negative_clipL_mask"] = te_outputs_2[p][1]

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
        device = accelerator.device
        if cfg_scale is None:
            cfg_scale = 1.0
        do_classifier_free_guidance = do_classifier_free_guidance and cfg_scale != 1.0

        # Prepare scheduler for each prompt
        scheduler = FlowMatchDiscreteScheduler(shift=discrete_flow_shift, reverse=True, solver="euler")

        # Number of inference steps for sampling
        scheduler.set_timesteps(sample_steps, device=device)
        timesteps = scheduler.timesteps

        # Calculate latent video length based on VAE version
        if "884" in VAE_VER:
            latent_video_length = (frame_count - 1) // 4 + 1
        elif "888" in VAE_VER:
            latent_video_length = (frame_count - 1) // 8 + 1
        else:
            latent_video_length = frame_count

        # Get embeddings
        prompt_embeds = sample_parameter["llm_embeds"].to(device=device, dtype=dit_dtype)
        prompt_mask = sample_parameter["llm_mask"].to(device=device)
        prompt_embeds_2 = sample_parameter["clipL_embeds"].to(device=device, dtype=dit_dtype)

        if do_classifier_free_guidance:
            negative_prompt_embeds = sample_parameter["negative_llm_embeds"].to(device=device, dtype=dit_dtype)
            negative_prompt_mask = sample_parameter["negative_llm_mask"].to(device=device)
            negative_prompt_embeds_2 = sample_parameter["negative_clipL_embeds"].to(device=device, dtype=dit_dtype)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_mask = torch.cat([negative_prompt_mask, prompt_mask], dim=0)
            prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2], dim=0)

        num_channels_latents = 16  # transformer.config.in_channels
        vae_scale_factor = 2 ** (4 - 1)  # Assuming 4 VAE blocks

        # Initialize latents
        shape_or_frame = (
            1,
            num_channels_latents,
            1,
            height // vae_scale_factor,
            width // vae_scale_factor,
        )
        latents = []
        for _ in range(latent_video_length):
            latents.append(torch.randn(shape_or_frame, generator=generator, device=device, dtype=dit_dtype))
        latents = torch.cat(latents, dim=2)

        if self.i2v_training:
            # Move VAE to the appropriate device for sampling
            vae.to(device)
            vae.eval()

            image = Image.open(image_path)
            image = resize_image_to_bucket(image, (width, height))  # returns a numpy array
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float()  # 1, C, 1, H, W
            image = image / 255.0

            logger.info("Encoding image to latents")
            image_latents = encode_to_latents(args, image, device)  # 1, C, 1, H, W
            image_latents = image_latents.to(device=device, dtype=dit_dtype)

            vae.to("cpu")
            clean_memory_on_device(device)

            zero_latents = torch.zeros_like(latents)
            zero_latents[:, :, :1, :, :] = image_latents
            image_latents = zero_latents
        else:
            image_latents = None

        # Guidance scale
        guidance_expand = torch.tensor([guidance_scale * 1000.0], dtype=torch.float32, device=device).to(dit_dtype)

        # Get rotary positional embeddings
        freqs_cos, freqs_sin = get_rotary_pos_embed_by_shape(transformer, latents.shape[2:])
        freqs_cos = freqs_cos.to(device=device, dtype=dit_dtype)
        freqs_sin = freqs_sin.to(device=device, dtype=dit_dtype)

        # Wrap the inner loop with tqdm to track progress over timesteps
        prompt_idx = sample_parameter.get("enum", 0)
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc=f"Sampling timesteps for prompt {prompt_idx + 1}")):
                latents_input = scheduler.scale_model_input(latents, t)

                if do_classifier_free_guidance:
                    latents_input = torch.cat([latents_input, latents_input], dim=0)  # 2, C, F, H, W

                if image_latents is not None:
                    latents_image_input = (
                        image_latents if not do_classifier_free_guidance else torch.cat([image_latents, image_latents], dim=0)
                    )
                    latents_input = torch.cat([latents_input, latents_image_input], dim=1)  # 1 or 2, C*2, F, H, W

                noise_pred = transformer(
                    latents_input,
                    t.repeat(latents.shape[0]).to(device=device, dtype=dit_dtype),
                    text_states=prompt_embeds,
                    text_mask=prompt_mask,
                    text_states_2=prompt_embeds_2,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    guidance=guidance_expand,
                    return_dict=True,
                )["x"]

                # perform classifier free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

                # Compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Move VAE to the appropriate device for sampling
        vae.to(device)
        vae.eval()

        # Decode latents to video
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
            latents = latents / vae.config.scaling_factor + vae.config.shift_factor
        else:
            latents = latents / vae.config.scaling_factor

        latents = latents.to(device=device, dtype=vae.dtype)
        with torch.no_grad():
            video = vae.decode(latents, return_dict=False)[0]
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.cpu().float()

        return video

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae, _, s_ratio, t_ratio = hv_load_vae(vae_dtype=vae_dtype, device="cpu", vae_path=vae_path)

        if args.vae_chunk_size is not None:
            vae.set_chunk_size_for_causal_conv_3d(args.vae_chunk_size)
            logger.info(f"Set chunk_size to {args.vae_chunk_size} for CausalConv3d in VAE")
        if args.vae_spatial_tile_sample_min_size is not None:
            vae.enable_spatial_tiling(True)
            vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size
            vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8
        elif args.vae_tiling:
            vae.enable_spatial_tiling(True)

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
        transformer = hv_load_transformer(dit_path, attn_mode, split_attn, loading_device, dit_weight_dtype, args.dit_in_channels)

        if args.img_in_txt_in_offloading:
            logger.info("Enable offloading img_in and txt_in to CPU")
            transformer.enable_img_in_txt_in_offloading()

        return transformer

    def compile_transformer(self, args, transformer):
        transformer: HYVideoDiffusionTransformer = transformer
        return model_utils.compile_transformer(
            args, transformer, [transformer.double_blocks, transformer.single_blocks], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
        latents = latents * vae_module.SCALING_FACTOR
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
        **kwargs,
    ) -> DiTOutput:
        transformer: HYVideoDiffusionTransformer = transformer_arg
        bsz = latents.shape[0]

        # I2V training
        if self.i2v_training:
            image_latents = torch.zeros_like(latents)
            image_latents[:, :, :1, :, :] = latents[:, :, :1, :, :]
            noisy_model_input = torch.cat([noisy_model_input, image_latents], dim=1)  # concat along channel dim

        # ensure guidance_scale in args is float
        guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device)  # , dtype=dit_dtype)

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            guidance_vec.requires_grad_(True)

        pos_emb_shape = latents.shape[1:]
        if pos_emb_shape not in self.pos_embed_cache:
            freqs_cos, freqs_sin = get_rotary_pos_embed_by_shape(transformer, latents.shape[2:])
            # freqs_cos = freqs_cos.to(device=accelerator.device, dtype=dit_dtype)
            # freqs_sin = freqs_sin.to(device=accelerator.device, dtype=dit_dtype)
            self.pos_embed_cache[pos_emb_shape] = (freqs_cos, freqs_sin)
        else:
            freqs_cos, freqs_sin = self.pos_embed_cache[pos_emb_shape]

        # call DiT
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        with accelerator.autocast():
            model_pred = transformer(
                noisy_model_input,
                timesteps,
                text_states=batch["llm"],
                text_mask=batch["llm_mask"],
                text_states_2=batch["clipL"],
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                guidance=guidance_vec,
                return_dict=False,
            )

        # flow matching loss
        target = noise - latents

        return DiTOutput(pred=model_pred, target=target)

    # endregion model specific


def hv_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """HunyuanVideo specific parser setup"""
    # model settings
    parser.add_argument("--dit_dtype", type=str, default=None, help="data type for DiT, default is bfloat16")
    parser.add_argument("--dit_in_channels", type=int, default=16, help="input channels for DiT, default is 16, skyreels I2V is 32")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for LLM / LLMにfp8を使う")
    parser.add_argument("--text_encoder1", type=str, help="Text Encoder 1 directory / テキストエンコーダ1のディレクトリ")
    parser.add_argument("--text_encoder2", type=str, help="Text Encoder 2 directory / テキストエンコーダ2のディレクトリ")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for Text Encoder, default is float16")
    parser.add_argument(
        "--vae_tiling",
        action="store_true",
        help="enable spatial tiling for VAE, default is False. If vae_spatial_tile_sample_min_size is set, this is automatically enabled."
        " / VAEの空間タイリングを有効にする、デフォルトはFalse。vae_spatial_tile_sample_min_sizeが設定されている場合、自動的に有効になります。",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = hv_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.fp8_scaled = False  # HunyuanVideo does not support this yet

    trainer = HunyuanVideoNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
