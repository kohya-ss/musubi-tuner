import argparse
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from transformers import AutoTokenizer, UMT5EncoderModel
import logging
from tqdm import tqdm

from musubi_tuner.longcat_video.configs import (
    LONGCAT_BASE_CONFIG,
    LONGCAT_LATENTS_MEAN,
    LONGCAT_LATENTS_STD,
)
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, load_split_weights

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LONGCAT, ARCHITECTURE_LONGCAT_FULL
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    read_config_from_file,
    setup_parser_common,
)
from musubi_tuner.longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from musubi_tuner.longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from musubi_tuner.utils import model_utils
from musubi_tuner.wan.configs import wan_t2v_14B
from musubi_tuner.wan.modules.t5 import T5EncoderModel
from musubi_tuner.wan.modules.vae import WanVAE


KEEP_FP8_HIGH_PRECISION_TOKENS = (
    "norm",
    "bias",
    "scale_shift_table",
    "patchify_proj",
    "proj_out",
    "adaln_single",
    "caption_projection",
    "adaLN_modulation",
    "q_norm",
    "k_norm",
    "t_embedder",
    "y_embedder",
    "x_embedder",
    "final_layer",
)


class WanT5EncoderWrapper(torch.nn.Module):
    def __init__(self, core: T5EncoderModel) -> None:
        super().__init__()
        self._core = core
        self.model = core.model
        self.tokenizer = core.tokenizer
        self.config = SimpleNamespace(d_model=self.model.dim)
        self._refresh_device_dtype()

    def _refresh_device_dtype(self) -> None:
        try:
            param = next(self.model.parameters())
            self.device = param.device
            self.dtype = param.dtype
        except StopIteration:
            self.device = torch.device("cpu")
            self.dtype = torch.float32

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model.to(*args, **kwargs)
        self._refresh_device_dtype()
        return self

    def eval(self):
        self.model.eval()
        return self

    def requires_grad_(self, requires_grad: bool = True):
        self.model.requires_grad_(requires_grad)
        return self

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.model(input_ids.to(self.device), attention_mask.to(self.device))
        return SimpleNamespace(last_hidden_state=hidden_states)


def detect_longcat_sd_dtype(dit_path: str) -> torch.dtype:
    if not os.path.isfile(dit_path):
        raise FileNotFoundError(f"LongCat DiT weights must be a .safetensors file. Got: {dit_path}")

    with MemoryEfficientSafeOpen(dit_path) as handle:
        keys = list(handle.keys())
        if not keys:
            raise ValueError(f"Unable to detect LongCat dtype; no tensors found in {dit_path}")

        for key in keys:
            tensor = handle.get_tensor(key)
            if tensor.is_floating_point():
                dtype = tensor.dtype
                break
        else:
            dtype = handle.get_tensor(keys[0]).dtype

    logger.info("Detected LongCat DiT dtype: %s", dtype)
    return dtype


def load_longcat_model(
    dit_path: str,
    device: Union[str, torch.device] = "cpu",
    load_device: Union[str, torch.device] = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
    fp8_scaled: bool = False,
    use_scaled_mm: bool = False,
    disable_numpy_memmap: bool = False,
    attn_mode: str = "torch",
    **_: Any,
) -> LongCatVideoTransformer3DModel:
    if not os.path.isfile(dit_path):
        raise FileNotFoundError(f"LongCat DiT weights must be provided as a safetensors file. Got: {dit_path}")

    target_device = torch.device(device)
    load_device = torch.device(load_device)

    base_config = dict(LONGCAT_BASE_CONFIG)
    attn_mode_normalized = (attn_mode or "torch").lower()
    if attn_mode_normalized == "torch":
        base_config["enable_flashattn2"] = False
        base_config["enable_flashattn3"] = False
        base_config["enable_xformers"] = False
    elif attn_mode_normalized == "flash":
        base_config["enable_flashattn2"] = True
        base_config["enable_flashattn3"] = False
        base_config["enable_xformers"] = False
    elif attn_mode_normalized == "flash3":
        base_config["enable_flashattn3"] = True
        base_config["enable_flashattn2"] = False
        base_config["enable_xformers"] = False
    elif attn_mode_normalized == "xformers":
        base_config["enable_flashattn2"] = False
        base_config["enable_flashattn3"] = False
        base_config["enable_xformers"] = True
    else:
        raise ValueError(f"Unsupported attention backend '{attn_mode}' for LongCat")

    config = base_config
    config_source = "built-in LONGCAT_BASE_CONFIG"

    logger.info("Creating LongCatVideoTransformer3DModel from %s", config_source)
    model = LongCatVideoTransformer3DModel(**config)

    logger.info("Loading LongCat weights from %s", dit_path)
    state_dict = load_split_weights(
        dit_path,
        device="cpu",
        disable_mmap=disable_numpy_memmap,
        dtype=None,
    )

    if fp8_scaled:
        state_dict = model.fp8_optimization(
            state_dict,
            load_device,
            move_to_device=(load_device.type != "cpu"),
            use_scaled_mm=use_scaled_mm,
        )
        load_info = model.load_state_dict(state_dict, strict=True, assign=True)
        missing, unexpected = load_info.missing_keys, load_info.unexpected_keys
    else:
        load_info = model.load_state_dict(state_dict, strict=False)
        missing, unexpected = load_info

    if missing:
        logger.warning("Missing LongCat weights: %s", missing)
    if unexpected:
        logger.warning("Unexpected LongCat weights: %s", unexpected)

    del state_dict

    model = model.to(load_device)

    if fp8_scaled:
        model = model.to(target_device)
        return model

    if torch_dtype is not None:
        if torch_dtype == torch.float8_e4m3fn:
            if load_device.type != "cuda":
                raise ValueError("FP8 base casting requires loading the model on a CUDA device.")
            for name, parameter in model.named_parameters():
                if not parameter.is_floating_point():
                    continue
                if any(token in name for token in KEEP_FP8_HIGH_PRECISION_TOKENS):
                    continue
                try:
                    parameter.data = parameter.data.to(torch_dtype)
                except RuntimeError as exc:
                    logger.warning("Skipping FP8 cast for %s due to: %s", name, exc)
            for name, buffer in model.named_buffers():
                if not buffer.is_floating_point():
                    continue
                if any(token in name for token in KEEP_FP8_HIGH_PRECISION_TOKENS):
                    continue
                model.register_buffer(name, buffer.to(torch_dtype))
        else:
            for name, parameter in model.named_parameters():
                if not parameter.is_floating_point():
                    continue
                if any(token in name for token in KEEP_FP8_HIGH_PRECISION_TOKENS):
                    continue
                parameter.data = parameter.data.to(torch_dtype)
            for name, buffer in model.named_buffers():
                if not buffer.is_floating_point():
                    continue
                if any(token in name for token in KEEP_FP8_HIGH_PRECISION_TOKENS):
                    continue
                model.register_buffer(name, buffer.to(torch_dtype))

    model = model.to(device=target_device)

    return model


class LongCatNetworkTrainer(NetworkTrainer):
    def __init__(self) -> None:
        super().__init__()
        self._tokenizer: Optional[AutoTokenizer] = None
        self._text_encoder: Optional[Union[UMT5EncoderModel, WanT5EncoderWrapper]] = None
        self._dit_attn_mode: Optional[str] = None
        self._latent_norm_cache: dict[tuple[str, Optional[int], torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
        mean = torch.tensor(LONGCAT_LATENTS_MEAN, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(LONGCAT_LATENTS_STD, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = std.clamp_min(1e-6)
        self._latent_norm_base: tuple[torch.Tensor, torch.Tensor] = (mean, std.reciprocal())
        self._flow_target: str = "x1_minus_x0"
        self._vae_tiling_enabled: bool = False
        self._vae_tiling_warning_emitted: bool = False
        self._missing_i2v_cache_warning: bool = False
        self._num_timesteps: int = 1000
        self._num_distill_sample_steps: int = 50

    # region model specific -------------------------------------------------

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_LONGCAT

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_LONGCAT_FULL

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        self.dit_dtype = detect_longcat_sd_dtype(args.dit)

        if self.dit_dtype == torch.float16:
            assert args.mixed_precision in ["fp16", "no"], "DiT weights are fp16; mixed precision must be fp16 or no"
        elif self.dit_dtype == torch.bfloat16:
            assert args.mixed_precision in ["bf16", "no"], "DiT weights are bf16; mixed precision must be bf16 or no"

        if args.fp8_scaled and self.dit_dtype.itemsize == 1:
            raise ValueError("DiT weights are already fp8; do not combine with --fp8_scaled")

        if args.fp8_scaled:
            args.dit_dtype = None
        else:
            args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)

        self._i2v_training = bool(args.longcat_i2v)
        self._control_training = False
        self.default_guidance_scale = 4.0
        self._flow_target = getattr(args, "flow_target", "x1_minus_x0")

        if args.control_video:
            raise ValueError("LongCat training does not support control video conditioning; omit --control_video options")

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
    ) -> Optional[List[Dict]]:
        prompts = load_prompts(sample_prompts)
        if not prompts:
            return None

        sample_parameters: List[Dict] = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            sample_parameters.append(prompt_dict_copy)

        text_encoder = self._ensure_text_encoder_loaded(args)
        tokenizer = self._tokenizer

        if tokenizer is None or text_encoder is None:
            return sample_parameters

        encode_device = torch.device("cpu")
        original_device = getattr(text_encoder, "device", torch.device("cpu"))
        original_dtype = getattr(text_encoder, "dtype", None)

        if original_dtype is not None and encode_device.type == "cpu":
            text_encoder = text_encoder.to(device=encode_device, dtype=torch.float32)
        else:
            text_encoder = text_encoder.to(device=encode_device)
        text_encoder.eval()

        def encode_text(strings: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
            tokenized = tokenizer(
                strings,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            if isinstance(tokenized, torch.Tensor):
                input_ids = tokenized.to(encode_device)
                pad_token_id = getattr(tokenizer, "pad_token_id", 0)
                attention_mask = (input_ids != pad_token_id).long()
            elif isinstance(tokenized, (tuple, list)) and len(tokenized) == 2:
                input_ids = tokenized[0].to(encode_device)
                attention_mask = tokenized[1].to(encode_device)
            else:
                input_ids = tokenized.input_ids.to(encode_device)
                attention_mask = tokenized.attention_mask.to(encode_device)

            with torch.no_grad():
                outputs = text_encoder(input_ids, attention_mask)
                if hasattr(outputs, "last_hidden_state"):
                    hidden_states = outputs.last_hidden_state
                else:
                    hidden_states = outputs[0]

            return hidden_states.detach().to("cpu"), attention_mask.detach().to("cpu")

        positive_texts = [param.get("prompt", "") or "" for param in sample_parameters]
        positive_embeds, positive_masks = encode_text(positive_texts)

        negative_texts = [param.get("negative_prompt", "") or "" for param in sample_parameters]
        negative_embeds, negative_masks = encode_text(negative_texts)

        for idx, param in enumerate(sample_parameters):
            param["prompt_embeds"] = positive_embeds[idx]
            param["prompt_attention_mask"] = positive_masks[idx]
            param["negative_prompt"] = negative_texts[idx]
            if negative_embeds is not None and negative_masks is not None:
                param["negative_prompt_embeds"] = negative_embeds[idx]
                param["negative_prompt_attention_mask"] = negative_masks[idx]

        if original_dtype is not None:
            text_encoder = text_encoder.to(device=original_device, dtype=original_dtype)
        else:
            text_encoder = text_encoder.to(device=original_device)
        text_encoder.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return sample_parameters

    def do_inference(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        sample_parameter: Dict,
        vae: WanVAE,
        dit_dtype: torch.dtype,
        transformer,
        discrete_flow_shift: float,
        sample_steps: int,
        width: int,
        height: int,
        frame_count: int,
        generator: torch.Generator,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        cfg_scale: Optional[float],
        image_path: Optional[str] = None,
        control_video_path: Optional[str] = None,
    ):
        scheduler = FlowMatchEulerDiscreteScheduler(
            shift=12.0,
            use_dynamic_shifting=False,
            invert_sigmas=False,
            time_shift_type="linear",
        )
        if discrete_flow_shift is not None:
            scheduler.set_shift(discrete_flow_shift)

        transformer_device = next(transformer.parameters()).device
        original_vae_device = getattr(vae, "device", torch.device("cpu"))
        original_vae_dtype = getattr(vae, "dtype", torch.float32)
        vae.to_device(transformer_device)
        vae.to_dtype(original_vae_dtype)

        guidance_scale = guidance_scale if guidance_scale is not None else self.default_guidance_scale
        prompt_embeds = sample_parameter.get("prompt_embeds")
        if prompt_embeds is None:
            prompt_embeds = sample_parameter.get("t5_embeds")
        if prompt_embeds is None:
            raise ValueError("Sample parameter missing prompt embeddings for preview.")
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if prompt_embeds.dim() == 3:
            prompt_embeds = prompt_embeds.unsqueeze(1)
        prompt_embeds = prompt_embeds.to(device=transformer_device, dtype=dit_dtype)

        prompt_attention_mask = sample_parameter.get("prompt_attention_mask")
        if prompt_attention_mask is None:
            prompt_attention_mask = sample_parameter.get("t5_mask")
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones(
                (prompt_embeds.shape[0], prompt_embeds.shape[2]),
                dtype=torch.int64,
            )
        if prompt_attention_mask.dim() == 1:
            prompt_attention_mask = prompt_attention_mask.unsqueeze(0)
        prompt_attention_mask = prompt_attention_mask.to(device=transformer_device, dtype=torch.int64)

        negative_prompt_embeds = sample_parameter.get("negative_prompt_embeds")
        if negative_prompt_embeds is None:
            negative_prompt_embeds = sample_parameter.get("negative_t5_embeds")
        negative_prompt_attention_mask = sample_parameter.get("negative_prompt_attention_mask")
        if negative_prompt_attention_mask is None:
            negative_prompt_attention_mask = sample_parameter.get("negative_t5_mask")

        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            else:
                if negative_prompt_embeds.dim() == 2:
                    negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0)
                if negative_prompt_embeds.dim() == 3:
                    negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1)
                negative_prompt_embeds = negative_prompt_embeds.to(device=transformer_device, dtype=dit_dtype)

            neg_batch = negative_prompt_embeds.shape[0]
            neg_seq_len = negative_prompt_embeds.shape[2]
            if negative_prompt_attention_mask is None:
                negative_prompt_attention_mask = torch.ones(
                    (neg_batch, neg_seq_len),
                    dtype=torch.int64,
                    device=transformer_device,
                )
            else:
                if negative_prompt_attention_mask.dim() == 1:
                    negative_prompt_attention_mask = negative_prompt_attention_mask.unsqueeze(0)
                negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                    device=transformer_device,
                    dtype=torch.int64,
                )

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        vae_scale_factor_temporal, vae_scale_factor_spatial = self._resolve_vae_scale_factors(vae)
        latent_video_length = (frame_count - 1) // vae_scale_factor_temporal + 1
        latent_height = height // vae_scale_factor_spatial
        latent_width = width // vae_scale_factor_spatial

        num_channels_latents = getattr(transformer, "config", None)
        if num_channels_latents is not None:
            num_channels_latents = getattr(transformer.config, "in_channels", 16)
        else:
            num_channels_latents = 16

        use_distill = bool(sample_parameter.get("use_distill", False))

        sigmas = self._get_timesteps_sigmas(sample_steps, use_distill=use_distill)
        scheduler.set_timesteps(sample_steps, sigmas=sigmas, device=transformer_device)
        timesteps = scheduler.timesteps

        latents = torch.randn(
            (1, num_channels_latents, latent_video_length, latent_height, latent_width),
            generator=generator,
            device=transformer_device,
            dtype=torch.float32,
        )

        num_cond_latents = 0
        image_latents = None
        if self.i2v_training and image_path is not None:
            with Image.open(image_path) as pil_image:
                pil_image = pil_image.convert("RGB")
                pil_image = pil_image.resize((width, height))
                image_np = np.asarray(pil_image, dtype=np.float32)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
            image_tensor = image_tensor / 127.5 - 1.0

            with torch.no_grad():
                encoded_latents = vae.encode([image_tensor.squeeze(0).to(transformer_device)])
            if not encoded_latents:
                raise ValueError("Failed to encode conditioning image for i2v sampling.")
            image_latents = encoded_latents[0].unsqueeze(0).to(transformer_device, dtype=dit_dtype)
            image_latents = self._normalize_latents_tensor(image_latents)
            image_latents = image_latents.to(latents.dtype)
            num_cond_latents = min(1, image_latents.shape[2])
            latents[:, :, :num_cond_latents] = image_latents[:, :, :num_cond_latents]

        with torch.no_grad():
            if not self.i2v_training:
                for t in tqdm(timesteps, desc="LongCat preview", leave=False):
                    latent_model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
                    latent_model_input = latent_model_input.to(device=transformer_device, dtype=dit_dtype)

                    timestep_tensor = t.expand(latent_model_input.shape[0]).to(device=transformer_device, dtype=dit_dtype)

                    model_pred = transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep_tensor,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        num_cond_latents=0,
                    )

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = model_pred.chunk(2)
                        batch = noise_pred_cond.shape[0]
                        positive = noise_pred_cond.reshape(batch, -1).to(torch.float32)
                        negative = noise_pred_uncond.reshape(batch, -1).to(torch.float32)
                        st_star = self._optimized_scale(positive, negative)
                        reshape_dims = [batch] + [1] * (noise_pred_cond.ndim - 1)
                        st_star = st_star.view(*reshape_dims).to(noise_pred_uncond.dtype)
                        model_pred = noise_pred_uncond * st_star + guidance_scale * (noise_pred_cond - noise_pred_uncond * st_star)

                    model_pred = -model_pred
                    model_pred = model_pred.to(latents.dtype)
                    latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]

            else:
                for t in tqdm(timesteps, desc="LongCat preview", leave=False):
                    latent_model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
                    latent_model_input = latent_model_input.to(device=transformer_device, dtype=dit_dtype)

                    timestep_tensor = t.expand(latent_model_input.shape[0]).to(device=transformer_device, dtype=dit_dtype)
                    timestep_tensor = timestep_tensor.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                    timestep_tensor[:, :num_cond_latents] = 0

                    model_pred = transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep_tensor,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        num_cond_latents=num_cond_latents,
                    )

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = model_pred.chunk(2)
                        batch = noise_pred_cond.shape[0]
                        positive = noise_pred_cond.reshape(batch, -1).to(torch.float32)
                        negative = noise_pred_uncond.reshape(batch, -1).to(torch.float32)
                        st_star = self._optimized_scale(positive, negative)
                        reshape_dims = [batch] + [1] * (noise_pred_cond.ndim - 1)
                        st_star = st_star.view(*reshape_dims).to(noise_pred_uncond.dtype)
                        model_pred = noise_pred_uncond * st_star + guidance_scale * (noise_pred_cond - noise_pred_uncond * st_star)

                    model_pred = -model_pred
                    model_pred = model_pred.to(latents.dtype)
                    denoised = scheduler.step(
                        model_pred[:, :, num_cond_latents:], t, latents[:, :, num_cond_latents:], return_dict=False
                    )[0]
                    latents[:, :, num_cond_latents:] = denoised

        latents_decode = self._denormalize_latents_tensor(latents.to(vae.dtype))
        latents_decode = latents_decode.to(transformer_device)

        with torch.no_grad():
            if isinstance(vae, WanVAE):
                decoded = vae.decode([latents_decode.squeeze(0)])
                if not decoded:
                    raise ValueError("VAE decoding returned no frames.")
                video = decoded[0].unsqueeze(0)
            else:
                decoded = vae.decode(latents_decode)
                if isinstance(decoded, (list, tuple)):
                    decoded = decoded[0]
                if decoded.ndim == 4:
                    decoded = decoded.unsqueeze(0)
                video = decoded
        video = (video / 2 + 0.5).clamp(0, 1)
        video_tensor = video.to(torch.float32).to("cpu")

        vae.to_device(original_vae_device)
        vae.to_dtype(original_vae_dtype)

        return video_tensor

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str) -> WanVAE:
        vae_path = args.vae

        logger.info(f"Loading VAE model from {vae_path}")
        cache_device = torch.device("cpu") if args.vae_cache_cpu else None

        vae = WanVAE(vae_path=vae_path, device="cpu", dtype=vae_dtype, cache_device=cache_device)
        self._update_latent_norm_base_from_vae(vae)
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
        del split_attn  # LongCat currently ignores split attention hints

        if args.sdpa:
            attn_mode = "torch"
        elif args.flash_attn:
            attn_mode = "flash"
        elif args.flash3:
            attn_mode = "flash3"
        elif args.xformers:
            attn_mode = "xformers"
        elif args.sage_attn:
            logger.warning("LongCat does not support SageAttention; falling back to PyTorch SDPA.")
            attn_mode = "torch"
        else:
            attn_mode = "torch"

        self._dit_attn_mode = attn_mode

        transformer = load_longcat_model(
            dit_path=dit_path,
            device=accelerator.device,
            load_device=loading_device,
            torch_dtype=dit_weight_dtype,
            fp8_scaled=args.fp8_scaled,
            use_scaled_mm=args.fp8_fast,
            disable_numpy_memmap=args.disable_numpy_memmap,
            attn_mode=attn_mode,
        )

        transformer.eval()
        transformer.requires_grad_(False)

        if args.blocks_to_swap:
            transformer.enable_block_swap(args.blocks_to_swap, accelerator.device, supports_backward=True)
            transformer.move_to_device_except_swap_blocks(accelerator.device)

        return transformer

    def scale_shift_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents

    def _update_latent_norm_base_from_vae(self, vae: WanVAE) -> None:
        config = getattr(vae, "config", None)
        if config is None:
            return
        latents_mean = getattr(config, "latents_mean", None)
        latents_std = getattr(config, "latents_std", None)
        if latents_mean is None or latents_std is None:
            return
        mean = torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1).clamp_min(1e-6)
        self._latent_norm_base = (mean, std.reciprocal())
        self._latent_norm_cache.clear()

    def _get_latent_norm(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device.type, getattr(device, "index", None), dtype)
        cached = self._latent_norm_cache.get(key)
        if cached is None:
            base_mean, base_inv_std = self._latent_norm_base
            latents_mean = base_mean.to(device=device, dtype=torch.float32)
            latents_inv_std = base_inv_std.to(device=device, dtype=torch.float32)
            cached = (latents_mean.to(dtype=dtype), latents_inv_std.to(dtype=dtype))
            self._latent_norm_cache[key] = cached
        return cached

    def _resolve_vae_scale_factors(self, vae: WanVAE) -> tuple[int, int]:
        config = getattr(vae, "config", None)
        temporal = getattr(config, "scale_factor_temporal", None) if config is not None else None
        spatial = getattr(config, "scale_factor_spatial", None) if config is not None else None
        if temporal is None:
            temporal = getattr(vae, "temporal_downsample_factor", None)
        if spatial is None:
            spatial = getattr(vae, "spatial_downsample_factor", None)
        if temporal is None:
            temporal = 4
        if spatial is None:
            spatial = 8
        return temporal, spatial

    def _normalize_latents_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        mean, inv_std = self._get_latent_norm(latents.device, torch.float32)
        latents_fp32 = latents.to(torch.float32)
        normalized = (latents_fp32 - mean) * inv_std
        return normalized.to(latents.dtype)

    def _denormalize_latents_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        mean, inv_std = self._get_latent_norm(latents.device, torch.float32)
        latents_fp32 = latents.to(torch.float32)
        denorm = latents_fp32 / inv_std + mean
        return denorm.to(latents.dtype)

    def _optimized_scale(self, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        dot_product = torch.sum(positive * negative, dim=1, keepdim=True)
        squared_norm = torch.sum(negative**2, dim=1, keepdim=True) + 1e-8
        return dot_product / squared_norm

    def _get_timesteps_sigmas(self, sampling_steps: int, use_distill: bool = False) -> torch.Tensor:
        if use_distill:
            distill_indices = torch.arange(1, self._num_distill_sample_steps + 1, dtype=torch.float32)
            distill_indices = (distill_indices * (self._num_timesteps // self._num_distill_sample_steps)).round().long()
            inference_indices = np.linspace(0, self._num_distill_sample_steps, num=sampling_steps, endpoint=False)
            inference_indices = np.floor(inference_indices).astype(np.int64)
            sigmas = torch.flip(distill_indices, [0])[inference_indices].float() / self._num_timesteps
        else:
            sigmas = torch.linspace(1.0, 0.001, sampling_steps)
        return sigmas.to(torch.float32)

    # endregion -------------------------------------------------------------

    # region helpers --------------------------------------------------------

    def on_load_text_encoder(
        self,
        args: argparse.Namespace,
        tokenizer,
        text_encoder,
    ) -> None:
        # LongCat training relies on cached embeddings; tokenizer/text encoder kept for inference only.
        self._tokenizer = tokenizer
        self._text_encoder = text_encoder

    def _ensure_text_encoder_loaded(self, args: argparse.Namespace):
        if self._text_encoder is not None:
            return self._text_encoder

        tokenizer = None
        text_encoder = None

        text_encoder_path = os.path.normpath(args.text_encoder)
        use_wan_t5 = text_encoder_path.endswith((".pth", ".safetensors")) or os.path.isfile(text_encoder_path)

        if use_wan_t5:
            config = wan_t2v_14B.t2v_14B
            text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=torch.device("cpu"),
                weight_path=text_encoder_path,
                fp8=args.fp8_t5,
            )
            tokenizer = text_encoder.tokenizer
            text_encoder = WanT5EncoderWrapper(text_encoder)
        else:
            tokenizer_path = os.path.normpath(args.tokenizer)
            tokenizer_is_dir = os.path.isdir(tokenizer_path)
            tokenizer_kwargs = {"local_files_only": tokenizer_is_dir}
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
            tokenizer.padding_side = "right"

            text_encoder_kwargs = {"local_files_only": os.path.isdir(text_encoder_path)}
            text_encoder = UMT5EncoderModel.from_pretrained(text_encoder_path, **text_encoder_kwargs)

        self.on_load_text_encoder(args, tokenizer, text_encoder)
        return self._text_encoder

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
        batched_t5 = batch.get("t5")
        batched_mask = batch.get("t5_mask")
        if batched_t5 is None:
            raise ValueError("Cached text encoder embeddings 't5' missing from batch; ensure cache was generated.")

        prompt_embeds = batched_t5.to(device=accelerator.device, dtype=network_dtype)

        # Debug: Check for NaN in cached T5 embeddings
        if torch.isnan(prompt_embeds).any():
            raise ValueError("NaN detected in cached T5 embeddings!")

        prompt_mask = None
        if batched_mask is not None:
            prompt_mask = batched_mask.to(device=accelerator.device)

        if args.gradient_checkpointing:
            prompt_embeds.requires_grad_(True)
            if prompt_mask is not None:
                prompt_mask = prompt_mask.to(torch.bool)

        latents = latents.to(device=accelerator.device, dtype=network_dtype)

        # Debug: Check for NaN in cached latents
        if torch.isnan(latents).any():
            raise ValueError("NaN detected in cached latents!")

        noise = noise.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)

        num_cond_latents = 0
        if self.i2v_training:
            num_cond_latents = 1
            image_latents = batch.get("latents_image")
            if image_latents is None:
                if not self._missing_i2v_cache_warning:
                    logger.warning(
                        "I2V training is enabled but 'latents_image' is missing from the dataset; "
                        "falling back to the first latent frame as conditioning."
                    )
                    self._missing_i2v_cache_warning = True
                image_latents = latents[:, :, :num_cond_latents].clone()
            else:
                image_latents = image_latents.to(device=accelerator.device, dtype=network_dtype)
                if image_latents.dim() == 4:
                    image_latents = image_latents.unsqueeze(2)
            image_latents = image_latents[:, :, :num_cond_latents]
            latents[:, :, :num_cond_latents] = image_latents
            noisy_model_input[:, :, :num_cond_latents] = image_latents
            noise[:, :, :num_cond_latents] = 0

        timesteps_input = timesteps.to(device=accelerator.device)
        if num_cond_latents > 0:
            total_latent_frames = noisy_model_input.shape[2]
            if timesteps_input.dim() == 1:
                timesteps_expanded = timesteps_input.unsqueeze(1).repeat(1, total_latent_frames)
            else:
                timesteps_expanded = timesteps_input.clone()
                if timesteps_expanded.shape[1] != total_latent_frames:
                    raise ValueError(
                        f"Per-token timestep shape mismatch: expected {total_latent_frames}, got {timesteps_expanded.shape[1]}"
                    )
            timesteps_expanded = timesteps_expanded.clone()
            timesteps_expanded[:, :num_cond_latents] = 0
        else:
            timesteps_expanded = timesteps_input

        timesteps_expanded = timesteps_expanded.to(noisy_model_input.dtype)

        with accelerator.autocast():
            model_pred = transformer(
                hidden_states=noisy_model_input,
                timestep=timesteps_expanded,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_mask,
                num_cond_latents=num_cond_latents,
            )

        flow_target = getattr(self, "_flow_target", "x1_minus_x0")
        if flow_target == "x1_minus_x0":
            target = latents - noise
        elif flow_target == "x0_minus_x1":
            target = noise - latents
        else:
            raise ValueError(f"Unknown flow_target setting: {flow_target}")
        return model_pred, target

    # endregion -------------------------------------------------------------


def longcat_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    def _ensure_arg(name: str, *, required: bool = False, **updates) -> None:
        action = parser._option_string_actions.get(name)
        if action is None:
            parser.add_argument(name, required=required, **updates)
        else:
            if required:
                action.required = True
            for key, value in updates.items():
                setattr(action, key, value)

    _ensure_arg(
        "--dit",
        required=True,
        type=str,
        help="Path to LongCat DiT weights (directory or .safetensors)",
    )
    _ensure_arg(
        "--vae",
        required=True,
        type=str,
        help="Path to Wan VAE weights directory",
    )
    _ensure_arg(
        "--tokenizer",
        type=str,
        default="google/umt5-xxl",
        help="Tokenizer repo or path",
    )
    _ensure_arg(
        "--text_encoder",
        required=True,
        type=str,
        help="UMT5 encoder weights (safetensors/HF folder)",
    )
    _ensure_arg(
        "--fp8_t5",
        action="store_true",
        help="Enable fp8 mode when loading WAN T5 weights",
    )
    parser.add_argument("--longcat_i2v", action="store_true", help="Enable Image-to-Video conditioning")
    parser.add_argument("--vae_cache_cpu", action="store_true", help="Cache WanVAE feature maps on CPU (Wan parity)")
    _ensure_arg(
        "--disable_numpy_memmap",
        action="store_true",
        help="Disable numpy memmap when loading weights",
    )
    _ensure_arg(
        "--fp8_scaled",
        action="store_true",
        help="Enable scaled FP8 weight optimization for the DiT",
    )
    _ensure_arg(
        "--fp8_fast",
        action="store_true",
        help="Enable scaled FP8 matmul kernels (requires SM 8.9+); only effective with --fp8_scaled",
    )
    _ensure_arg(
        "--blocks_to_swap",
        type=int,
        default=0,
        help="Number of transformer blocks to offload to CPU for block swapping (0 disables offloading)",
    )
    _ensure_arg(
        "--flow_target",
        type=str,
        default="x1_minus_x0",
        choices=("x1_minus_x0", "x0_minus_x1"),
        help="Flow matching supervision target: predict x1-x0 (default) or x0-x1",
    )
    parser.add_argument(
        "--control_video",
        action="store_true",
        help="(Unsupported) Placeholder for Wan parity; LongCat will raise if enabled",
    )
    return parser


def main() -> None:
    parser = setup_parser_common()
    parser = longcat_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = LongCatNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
