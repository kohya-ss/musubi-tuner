import argparse
import os
import json
import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator, init_empty_weights
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_KANDINSKY5, ARCHITECTURE_KANDINSKY5_FULL
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
    should_sample_images,
    load_prompts,
)
from musubi_tuner.kandinsky5.configs import TASK_CONFIGS, TaskConfig
from musubi_tuner.kandinsky5.models.dit import DiffusionTransformer3D, get_dit
from musubi_tuner.kandinsky5.models.vae import build_vae
from musubi_tuner.kandinsky5.models.text_embedders import get_text_embedder
from musubi_tuner.kandinsky5 import generation_utils
from musubi_tuner.kandinsky5.models.utils import fast_sta_nabla
from musubi_tuner.kandinsky5.generation_utils import get_first_frame_from_image
from musubi_tuner.kandinsky5.models import attention as k5_attention
from musubi_tuner.kandinsky5.models import nn as k5_nn
from musubi_tuner.modules.fp8_optimization_utils import (
    optimize_state_dict_with_fp8,
    apply_fp8_monkey_patch,
)
from musubi_tuner.utils import model_utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Kandinsky5NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.dit_conf = None
        self.task_conf: TaskConfig = None
        self.vae_scaling_factor = 1.0
        self._vae_checkpoint_path: str | None = None
        self._cached_sample_vae = None
        self._text_encoder_qwen_path: str | None = None
        self._text_encoder_clip_path: str | None = None
        self._nabla_mask_cache: dict[tuple[int, int, int, torch.device], torch.Tensor] = {}
        self._i2v_training = False
        self._i2v_mode = "first"
        self._control_training = False
        self.visual_cond_prob: float = 1.0

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_KANDINSKY5

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_KANDINSKY5_FULL

    def handle_model_specific_args(self, args):
        if args.task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task '{args.task}'. Available: {list(TASK_CONFIGS.keys())}")
        self.task_conf = TASK_CONFIGS[args.task]
        # Override scheduler_scale from CLI if provided
        if getattr(args, "scheduler_scale", None) is not None:
            # TaskConfig is a dataclass, so we can replace the field
            from dataclasses import replace

            self.task_conf = replace(self.task_conf, scheduler_scale=args.scheduler_scale)
            logger.info(f"Overriding scheduler_scale to {args.scheduler_scale}")
        if getattr(args, "force_nabla_attention", False):
            from dataclasses import replace

            self.task_conf = replace(
                self.task_conf,
                attention=replace(
                    self.task_conf.attention,
                    type="nabla",
                    method=getattr(args, "nabla_method", "topcdf"),
                    P=getattr(args, "nabla_P", 0.9),
                    add_sta=getattr(args, "nabla_add_sta", True),
                    wT=getattr(args, "nabla_wT", 11),
                    wH=getattr(args, "nabla_wH", 3),
                    wW=getattr(args, "nabla_wW", 3),
                ),
            )
            logger.info(
                "Forcing nabla attention for training: "
                f"method={self.task_conf.attention.method}, P={self.task_conf.attention.P}, "
                f"wT={self.task_conf.attention.wT}, wH={self.task_conf.attention.wH}, wW={self.task_conf.attention.wW}, "
                f"add_sta={self.task_conf.attention.add_sta}"
            )
        self.dit_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
        self._i2v_training = "i2v" in args.task
        if self._i2v_training:
            logger.info("I2V training mode enabled")
        self._i2v_mode = getattr(args, "i2v_mode", "first") or "first"
        self._control_training = False
        self.visual_cond_prob = float(getattr(args, "visual_cond_prob", 1.0) or 0.0)
        if self.visual_cond_prob < 0.0 or self.visual_cond_prob > 1.0:
            logger.warning(f"visual_cond_prob {self.visual_cond_prob} out of [0,1]; clamping.")
            self.visual_cond_prob = min(1.0, max(0.0, self.visual_cond_prob))
        self.default_guidance_scale = self.task_conf.guidance_weight
        # Text token padding is always enabled to mirror the working inference path.
        self._text_encoder_qwen_path = getattr(args, "text_encoder_qwen", None)
        self._text_encoder_clip_path = getattr(args, "text_encoder_clip", None)
        self._vae_checkpoint_path = getattr(args, "vae", None) or self.task_conf.vae.checkpoint_path

    @property
    def i2v_training(self) -> bool:
        return self._i2v_training

    @property
    def control_training(self) -> bool:
        return self._control_training

    def _build_sparse_params(self, x: torch.Tensor, device: torch.device):
        """Create (and cache) nabla sparse attention params for the current visual grid."""
        attn_conf = getattr(self.task_conf, "attention", None)
        if attn_conf is None or getattr(attn_conf, "type", None) != "nabla":
            return None
        if not self.dit_conf:
            return None

        patch_size = self.dit_conf.get("patch_size", (1, 2, 2))
        # Enforce geometry assumptions required by NABLA/STA masks and fractal flattening.
        if patch_size[0] != 1:
            raise ValueError("NABLA requires temporal patch size == 1 (got patch_size[0] != 1)")

        duration, height, width = x.shape[:3]
        if height % patch_size[1] != 0 or width % patch_size[2] != 0:
            raise ValueError(f"NABLA requires spatial dims divisible by patch_size; got H={height}, W={width}, patch={patch_size}")
        T = duration // patch_size[0]
        H = height // patch_size[1]
        W = width // patch_size[2]
        if H % 8 != 0 or W % 8 != 0:
            raise ValueError(f"NABLA requires H//patch and W//patch divisible by 8 for fractal flattening; got H={H}, W={W}")

        # Cache STA masks per (T, H/8, W/8, device) to avoid recomputing every step.
        sta_key = (T, H // 8, W // 8, device)
        sta_mask = self._nabla_mask_cache.get(sta_key)
        if sta_mask is None:
            sta_mask = fast_sta_nabla(
                T,
                H // 8,
                W // 8,
                attn_conf.wT,
                attn_conf.wH,
                attn_conf.wW,
                device=device,
            )
            self._nabla_mask_cache[sta_key] = sta_mask

        return {
            "sta_mask": sta_mask.unsqueeze(0).unsqueeze(0),
            "attention_type": attn_conf.type,
            "to_fractal": True,
            "P": attn_conf.P,
            "wT": attn_conf.wT,
            "wW": attn_conf.wW,
            "wH": attn_conf.wH,
            "add_sta": attn_conf.add_sta,
            "visual_shape": (T, H, W),
            "method": getattr(attn_conf, "method", "topcdf"),
        }

    def sample_images(self, accelerator: Accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype):
        """Use kandinsky5.generation_utils for quick qualitative checks with on-demand loading/offload."""
        if not should_sample_images(args, steps, epoch):
            return
        if not sample_parameters:
            logger.warning("No sample prompts provided; skipping sampling.")
            return

        # unwrap and optionally offload DiT before loading encoders
        transformer = accelerator.unwrap_model(transformer)
        was_training = transformer.training
        transformer.eval()
        transformer.switch_block_swap_for_inference()
        original_device = next(transformer.parameters()).device
        offload = bool(getattr(args, "offload_dit_during_sampling", False))
        transformer_offloaded = offload and accelerator.device.type == "cuda"
        if transformer_offloaded:
            transformer.to("cpu")
            clean_memory_on_device(accelerator.device)
        # When block swap is enabled the first param can live on CPU; ensure the rest of the model is on the sampling device.
        if getattr(transformer, "blocks_to_swap", 0) and original_device.type == "cpu" and not transformer_offloaded:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(accelerator.device)
            else:
                transformer.to(accelerator.device)
            clean_memory_on_device(accelerator.device)
            original_device = accelerator.device

        save_dir = os.path.join(args.output_dir, "sample")
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for idx, sample in enumerate(sample_parameters):
                prompt = sample.get("prompt", "")
                neg_prompt = sample.get("negative_prompt", "")
                seed = sample.get("seed", 42)
                image_path = sample.get("image_path", None)
                # honor per-prompt overrides from sample file
                width = sample.get("width", sample.get("w", self.task_conf.resolution))
                height = sample.get("height", sample.get("h", self.task_conf.resolution))
                duration = sample.get("frame_count", sample.get("f", 1 if not self.task_conf.dit_params.visual_cond else 5))
                steps_to_use = sample.get("sample_steps", sample.get("s", self.task_conf.num_steps))
                guidance = sample.get("guidance_scale", self.task_conf.guidance_weight)
                logger.info(
                    f"[sampling {idx}] w={width}, h={height}, f={duration}, steps={steps_to_use}, guidance={guidance}, "
                    f"offload_dit={transformer_offloaded}, blocks_to_swap={getattr(transformer, 'blocks_to_swap', 0)}"
                )

                # latent resolution (already /8 in cache)
                latent_w = max(1, width // 8)
                latent_h = max(1, height // 8)
                shape = (1, duration, latent_h, latent_w, self.task_conf.dit_params.in_visual_dim)

                text_embedder_conf = SimpleNamespace(
                    qwen=SimpleNamespace(
                        checkpoint_path=self._text_encoder_qwen_path or self.task_conf.text.qwen_checkpoint,
                        max_length=self.task_conf.text.qwen_max_length,
                    ),
                    clip=SimpleNamespace(
                        checkpoint_path=self._text_encoder_clip_path or self.task_conf.text.clip_checkpoint,
                        max_length=self.task_conf.text.clip_max_length,
                    ),
                )
                # 1) Encode text, then free encoder
                text_embedder = get_text_embedder(
                    text_embedder_conf,
                    device=accelerator.device,
                    quantized_qwen=False,
                )
                # default negative prompt if none provided
                neg_text = neg_prompt if neg_prompt else "low quality, bad quality"
                enc_out, _, attention_mask = text_embedder.encode([prompt], type_of_content=("video" if duration > 1 else "image"))
                neg_out, _, neg_attention_mask = text_embedder.encode(
                    [neg_text], type_of_content=("video" if duration > 1 else "image")
                )
                text_embeds = enc_out["text_embeds"]
                pooled_embed = enc_out["pooled_embed"]
                null_text_embeds = neg_out["text_embeds"]
                null_pooled_embed = neg_out["pooled_embed"]
                if attention_mask is not None:
                    mask = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
                    mask = mask.bool().flatten()
                    if mask.shape[0] != text_embeds.shape[0]:
                        # Processor returns padded masks; embeds are packed to valid tokens.
                        # Don't raise here: cache writing already aligns masks to packed embeds.
                        if mask.sum().item() == text_embeds.shape[0]:
                            mask = mask[mask]
                        else:
                            mask = torch.ones((text_embeds.shape[0],), dtype=torch.bool)
                    text_embeds = text_embeds[mask]
                    attention_mask = None
                if neg_attention_mask is not None:
                    mask = neg_attention_mask[0] if neg_attention_mask.dim() > 1 else neg_attention_mask
                    mask = mask.bool().flatten()
                    if mask.shape[0] != null_text_embeds.shape[0]:
                        if mask.sum().item() == null_text_embeds.shape[0]:
                            mask = mask[mask]
                        else:
                            mask = torch.ones((null_text_embeds.shape[0],), dtype=torch.bool)
                    null_text_embeds = null_text_embeds[mask]
                    neg_attention_mask = None
                # move encoder off GPU promptly
                try:
                    text_embedder.to("cpu")
                except Exception:
                    pass
                del text_embedder
                clean_memory_on_device(accelerator.device)

                # 2) Bring DiT back; keep swap/offload settings (aligns with inference path)
                if not transformer_offloaded and next(transformer.parameters()).device != accelerator.device:
                    if hasattr(transformer, "move_to_device_except_swap_blocks"):
                        transformer.move_to_device_except_swap_blocks(accelerator.device)
                    else:
                        transformer.to(accelerator.device)
                    clean_memory_on_device(accelerator.device)

                conf_ns = SimpleNamespace(model=self.task_conf, metrics=SimpleNamespace(scale_factor=self.task_conf.scale_factor))

                # Load VAE only when needed for init-frame encoding / decoding
                vae_for_sampling = None
                first_frames = None
                if image_path:
                    try:
                        vae_for_sampling = self._load_vae_for_sampling(args, accelerator.device)
                        max_area = 512 * 768 if int(self.task_conf.resolution) == 512 else 1024 * 1024
                        divisibility = 16 if int(self.task_conf.resolution) == 512 else 128
                        _, lat_image, _ = get_first_frame_from_image(
                            image_path,
                            vae_for_sampling,
                            accelerator.device,
                            max_area=max_area,
                            divisibility=divisibility,
                        )
                        first_frames = lat_image[:1]
                        if first_frames is not None:
                            latent_h = int(first_frames.shape[1])
                            latent_w = int(first_frames.shape[2])
                            shape = (1, duration, latent_h, latent_w, self.task_conf.dit_params.in_visual_dim)
                    except Exception as ex:
                        logger.warning(f"Failed to load i2v first frame from image_path='{image_path}': {ex}")

                # If DiT offloading is enabled, immediately unload VAE after init-frame encoding.
                # We'll reload it later only for decoding.
                if transformer_offloaded and vae_for_sampling is not None:
                    try:
                        vae_for_sampling.to("cpu")
                    except Exception:
                        pass
                    del vae_for_sampling
                    vae_for_sampling = None
                    clean_memory_on_device(accelerator.device)

                # Ensure DiT is only on GPU during the actual sampling step when offloading is enabled.
                if transformer_offloaded and next(transformer.parameters()).device.type == "cpu":
                    if hasattr(transformer, "move_to_device_except_swap_blocks"):
                        transformer.move_to_device_except_swap_blocks(accelerator.device)
                    else:
                        transformer.to(accelerator.device)
                    clean_memory_on_device(accelerator.device)
                if hasattr(transformer, "prepare_block_swap_before_forward"):
                    transformer.prepare_block_swap_before_forward()

                autocast_dtype = torch.bfloat16 if accelerator.device.type == "cuda" else None
                with torch.autocast(
                    device_type=accelerator.device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_dtype is not None,
                ):
                    images_latent = generation_utils.generate_sample_latents_only(
                        shape=shape,
                        dit=transformer,
                        text_embeds=text_embeds,
                        pooled_embed=pooled_embed,
                        attention_mask=attention_mask,
                        null_text_embeds=null_text_embeds,
                        null_pooled_embed=null_pooled_embed,
                        null_attention_mask=neg_attention_mask,
                        first_frames=first_frames,
                        num_steps=steps_to_use,
                        guidance_weight=guidance,
                        scheduler_scale=self.task_conf.scheduler_scale or 1,
                        seed=seed,
                        device=accelerator.device,
                        conf=conf_ns,
                        progress=False,
                    )

                # offload DiT between steps if requested
                if transformer_offloaded:
                    transformer.to("cpu")
                    clean_memory_on_device(accelerator.device)

                if vae_for_sampling is None:
                    vae_for_sampling = self._load_vae_for_sampling(args, accelerator.device)

                try:
                    images = generation_utils.decode_latents(
                        images_latent,
                        vae_for_sampling,
                        device=accelerator.device,
                        batch_size=shape[0],
                        num_frames=duration,
                    )
                except Exception as ex:
                    logger.warning(f"VAE decode on {accelerator.device} failed ({ex}); retrying on CPU for sampling.")
                    clean_memory_on_device(accelerator.device)
                    vae_cpu = vae_for_sampling.to("cpu")
                    images = generation_utils.decode_latents(
                        images_latent.to("cpu"),
                        vae_cpu,
                        device=torch.device("cpu"),
                        batch_size=shape[0],
                        num_frames=duration,
                    )
                del vae_for_sampling
                clean_memory_on_device(accelerator.device)

                # save video and first frame
                from musubi_tuner.hv_generate_video import save_videos_grid

                video_out = os.path.join(save_dir, f"sample_{steps:06d}_{idx:02d}.mp4")
                # move to CPU before saving to avoid numpy conversion on CUDA tensors
                video_tensor = images.permute(0, 4, 1, 2, 3).float().cpu() / 255.0
                save_videos_grid(video_tensor, video_out, rescale=False, n_rows=1)

                out_path = os.path.join(save_dir, f"sample_{steps:06d}_{idx:02d}.png")
                import torchvision.utils as vutils

                frame = images[0, 0] if images.dim() == 5 else images[0]
                frame = frame.float().cpu() / 255.0
                frame = frame.permute(2, 0, 1)  # C, H, W
                vutils.save_image(frame, out_path)
                logger.info(f"Saved sample to {out_path} and {video_out}")

        # move DiT back to training device if we offloaded it
        if transformer_offloaded and original_device != next(transformer.parameters()).device:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(original_device)
            else:
                transformer.to(original_device)
        transformer.switch_block_swap_for_training()
        if was_training:
            transformer.train()
        clean_memory_on_device(accelerator.device)

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        logger.info(f"Loading Kandinsky5 sample prompts from {sample_prompts}")
        return load_prompts(sample_prompts)

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
        prompt = sample_parameter.get("prompt", "")
        neg_prompt = sample_parameter.get("negative_prompt", "")
        duration = frame_count if frame_count is not None else 1
        # convert pixel dims to latent dims
        latent_h = height // 8
        latent_w = width // 8

        text_embedder_conf = SimpleNamespace(
            qwen=SimpleNamespace(
                checkpoint_path=self.task_conf.text.qwen_checkpoint,
                max_length=self.task_conf.text.qwen_max_length,
            ),
            clip=SimpleNamespace(
                checkpoint_path=self.task_conf.text.clip_checkpoint,
                max_length=self.task_conf.text.clip_max_length,
            ),
        )
        text_embedder = get_text_embedder(
            text_embedder_conf,
            device=accelerator.device,
            quantized_qwen=False,
        )

        images = generation_utils.generate_sample(
            shape=(1, duration, latent_h, latent_w, self.task_conf.dit_params.in_visual_dim),
            caption=prompt,
            dit=transformer,
            vae=vae,
            conf=SimpleNamespace(metrics=SimpleNamespace(scale_factor=self.task_conf.scale_factor)),
            text_embedder=text_embedder,
            num_steps=sample_steps or self.task_conf.num_steps,
            guidance_weight=guidance_scale or self.task_conf.guidance_weight,
            scheduler_scale=self.task_conf.scheduler_scale or 1,
            negative_caption=neg_prompt,
            seed=sample_parameter.get("seed", 42),
            device=accelerator.device,
            vae_device=accelerator.device,
            text_embedder_device=accelerator.device,
            progress=False,
            offload=False,
        )

        return images

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        # Training always uses cached latents, so defer VAE load until sampling time.
        self._vae_checkpoint_path = vae_path or self.task_conf.vae.checkpoint_path
        self.vae_scaling_factor = 1.0

        class _VaeStub:
            def requires_grad_(self, *_, **__):
                return self

            def eval(self):
                return self

            def to(self, *_, **__):
                return self

        # Return a stub to satisfy base trainer expectations; real VAE is loaded only inside sample_images.
        self.vae = _VaeStub()
        return self.vae

    def _load_dit_config(self, args: argparse.Namespace) -> dict:
        conf = self.task_conf.dit_params
        if args.override_dit:
            conf = SimpleNamespace(**json.loads(args.override_dit))
        conf_dict = conf.__dict__ if isinstance(conf, SimpleNamespace) else conf.__dict__
        # Respect global attention flags for Kandinsky5
        if getattr(args, "flash_attn", False):
            conf_dict["attention_engine"] = "flash_attention_2"
        elif getattr(args, "flash3", False):
            conf_dict["attention_engine"] = "flash_attention_3"
        elif getattr(args, "sage_attn", False):
            conf_dict["attention_engine"] = "sage"
        elif getattr(args, "xformers", False):
            conf_dict["attention_engine"] = "xformers"
        elif getattr(args, "sdpa", False):
            conf_dict["attention_engine"] = "sdpa"
        else:
            conf_dict.setdefault("attention_engine", "auto")
        return conf_dict

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
        def _detect_fp8_checkpoint(path: str, disable_numpy_memmap: bool = False) -> bool:
            with MemoryEfficientSafeOpen(path, disable_numpy_memmap=disable_numpy_memmap) as f:
                for key in f.keys():
                    if key.endswith(".scale_weight"):
                        return True
            return False

        def _load_state_dict_stream(
            path: str, device: str | torch.device = "cpu", dtype: torch.dtype | None = None, disable_numpy_memmap: bool = False
        ):
            dev = torch.device(device)
            sd: dict[str, torch.Tensor] = {}
            with MemoryEfficientSafeOpen(path, disable_numpy_memmap=disable_numpy_memmap) as f:
                for key in f.keys():
                    tensor = f.get_tensor(key, device=dev)
                    if dtype is not None and tensor.dtype != dtype:
                        tensor = tensor.to(dtype)
                    sd[key] = tensor
            return sd

        self.dit_conf = self._load_dit_config(args)
        # Keep the base model dtype at standard precision even when fp8_base is requested,
        # to match the stable inference loader behavior (fp8 is applied via monkey patch/quantization later).
        if args.fp8_base and not args.fp8_scaled:
            dit_weight_dtype = None
        with init_empty_weights():
            model = get_dit(self.dit_conf)
            if dit_weight_dtype is not None:
                model.to(dit_weight_dtype)

        # fp8 weights must live on GPU when possible; adjust quantization device based on swap usage.
        use_fp8 = args.fp8_scaled or args.fp8_base
        blocks_to_swap = getattr(args, "blocks_to_swap", 0) or 0
        quant_device = accelerator.device if use_fp8 else loading_device

        ckpt_path = dit_path or self.task_conf.checkpoint_path
        logger.info(f"Loading DiT from {ckpt_path}")
        disable_memmap = getattr(args, "disable_numpy_memmap", False)
        is_fp8_ckpt = _detect_fp8_checkpoint(ckpt_path, disable_numpy_memmap=disable_memmap)

        # Load state dict
        state_dict = _load_state_dict_stream(ckpt_path, device="cpu", dtype=None, disable_numpy_memmap=disable_memmap)

        if is_fp8_ckpt:
            # fp8 checkpoint: use as-is, just apply monkey patch (like HunyuanVideo)
            logger.info("Checkpoint contains fp8 weights; using as-is.")
            apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=args.fp8_fast)
            dit_weight_dtype = None
            use_fp8 = False  # skip re-quantization below
        elif use_fp8:
            # Limit fp8 to the heavy transformer blocks and output layer to reduce slow per-channel fallbacks on small embeddings.
            # Keep the target set consistent even when block swap is used so all transformer blocks are quantized.
            target_keys = [
                "visual_transformer_blocks",
                "text_transformer_blocks",
                "out_layer",
            ]
            exclude_keys: list[str] = ["norm"]  # skip LayerNorm-like weights to avoid unmatched scale_weight buffers
            logger.info(f"Applying fp8 optimization (scaled={args.fp8_scaled}, base={args.fp8_base}) on {quant_device}")
            # If block swap is disabled, keep weights on GPU for speed; otherwise keep them on CPU to avoid OOM.
            if blocks_to_swap == 0:
                move_to_device = True
                fp8_quant_device = quant_device  # GPU quant/keep
                block_size = 16  # smaller block for better coverage when everything stays on GPU
            else:
                move_to_device = False
                # quantize on GPU even when block swap is on, but keep weights on CPU afterwards for swap
                fp8_quant_device = accelerator.device if quant_device == "cpu" else quant_device
                logger.info(
                    "blocks_to_swap > 0, quantizing fp8 on GPU and keeping weights on CPU for block swap (all transformer blocks)."
                )
                block_size = 64  # larger block to reduce scale tensor size in CPU path
            try:
                state_dict = optimize_state_dict_with_fp8(
                    state_dict,
                    calc_device=fp8_quant_device,
                    target_layer_keys=target_keys,
                    exclude_layer_keys=exclude_keys,
                    quantization_mode="block",
                    block_size=block_size,
                    move_to_device=move_to_device,
                )
                apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=args.fp8_fast)
                dit_weight_dtype = None
            except Exception as ex:
                logger.warning(f"fp8 optimization failed ({ex}); proceeding without fp8 for this run.")

        # keep original loading device logic when not using fp8 (supports block swap on CPU load)
        target_device = "cpu" if blocks_to_swap > 0 else accelerator.device
        loading_device = target_device
        quant_device = target_device

        info = model.load_state_dict(state_dict, strict=False, assign=True)
        logger.info(f"Loaded DiT weights: {info}")
        # free CPU copy ASAP
        del state_dict

        model.attention = SimpleNamespace(**self.task_conf.attention.__dict__)
        model.to(target_device)
        model.dtype = next(model.parameters()).dtype  # align hv_train_network logging expectation
        model.device = target_device  # align hv_train_network logging expectation

        # Ensure norm params are not fp8 (fp8 norms trigger unsupported ops).
        import torch.nn as nn  # local import to avoid cyclic issues

        def _upcast_stable_params(m: nn.Module):
            # Keep numerically sensitive pieces in float32.
            for name, p in m.named_parameters(recurse=False):
                if any(key in name for key in ["embedding", "embeddings", "rope"]):
                    p.data = p.data.to(torch.float32)
                if isinstance(m, (nn.LayerNorm, getattr(nn, "RMSNorm", nn.LayerNorm))):
                    p.data = p.data.to(torch.float32)
            for name, b in m.named_buffers(recurse=False):
                if isinstance(b, torch.Tensor) and any(key in name for key in ["embedding", "embeddings", "rope"]):
                    setattr(m, name, b.to(torch.float32))

        for mod in model.modules():
            if isinstance(mod, (nn.LayerNorm, getattr(nn, "RMSNorm", nn.LayerNorm))):
                if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
                    mod.weight.data = mod.weight.data.to(torch.bfloat16)
                if hasattr(mod, "bias") and isinstance(mod.bias, torch.Tensor) and mod.bias is not None:
                    mod.bias.data = mod.bias.data.to(torch.bfloat16)
            _upcast_stable_params(mod)
        for name, param in model.named_parameters():
            if "norm" in name:
                param.data = param.data.to(torch.bfloat16)

        # Cast any stray fp8 params/buffers outside Linear fp8 modules back to bf16 to avoid unsupported ops.
        for mod in model.modules():
            is_fp8_linear = isinstance(mod, nn.Linear) and hasattr(mod, "scale_weight")
            if not is_fp8_linear:
                for p in mod.parameters(recurse=False):
                    if p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        p.data = p.data.to(torch.bfloat16)
                for b_name, b in mod.named_buffers(recurse=False):
                    if isinstance(b, torch.Tensor) and b.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        setattr(mod, b_name, b.to(torch.bfloat16))

        # Ensure fp8 linears use safe dequant (float32 scale/weight) to avoid unsupported float8 ops.
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module, "scale_weight"):
                module.scale_weight = module.scale_weight.to(torch.float32)

                def _safe_forward(self, x):
                    target_device = x.device
                    weight = self.weight.to(device=target_device, dtype=torch.float32)
                    scale = self.scale_weight.to(device=target_device, dtype=torch.float32)
                    if scale.ndim < 3:
                        w = weight * scale
                    else:
                        out_features, num_blocks, _ = scale.shape
                        w = weight.contiguous().view(out_features, num_blocks, -1)
                        w = w * scale
                        w = w.view(self.weight.shape)
                    bias = self.bias.to(target_device) if self.bias is not None else None
                    out = F.linear(x, w, bias)
                    return out.to(x.dtype)

                module.forward = _safe_forward.__get__(module, type(module))
        if getattr(args, "compile", False):
            model = self.compile_transformer(args, model)
        return model

    def compile_transformer(self, args, transformer):
        transformer: DiffusionTransformer3D = transformer
        return model_utils.compile_transformer(
            args,
            transformer,
            [transformer.text_transformer_blocks, transformer.visual_transformer_blocks],
            disable_linear=self.blocks_to_swap > 0,
        )

    def scale_shift_latents(self, latents):
        # Latents were scaled during caching; avoid re-scaling during training.
        return latents

    def _load_vae_for_sampling(self, args: argparse.Namespace, device: torch.device):
        vae_conf = SimpleNamespace(name=self.task_conf.vae.name, checkpoint_path=self._vae_checkpoint_path)
        # Decode has been unstable in fp16 on some GPUs; prefer float32 for sampling.
        target_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        vae = build_vae(vae_conf, vae_dtype=target_dtype)
        # Enable VAE tiling to reduce VRAM during sampling when available.
        if hasattr(vae, "apply_tiling"):
            tile = (1, 17, 256, 256)
            stride = (8, 192, 192)
            try:
                vae.apply_tiling(tile, stride)
            except Exception:
                pass
        vae = vae.to(device=device, dtype=target_dtype)
        vae.eval()
        return vae

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: DiffusionTransformer3D,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        bsz = latents.shape[0]
        preds = []
        targets = []

        patch_size = self.dit_conf["patch_size"] if self.dit_conf else (1, 2, 2)
        attention_conf = getattr(self.task_conf, "attention", SimpleNamespace(chunk=False, chunk_len=None))
        chunk_len = getattr(attention_conf, "chunk_len", None) or None
        chunk_mode = bool(attention_conf.chunk and chunk_len and chunk_len > 0)

        for b in range(bsz):
            latent_b = latents[b].to(accelerator.device, dtype=network_dtype)
            noise_b = noise[b].to(accelerator.device, dtype=network_dtype)
            noisy_input_b = noisy_model_input[b].to(accelerator.device, dtype=network_dtype)

            text_embed = batch["text_embeds"][b].to(accelerator.device, dtype=network_dtype)
            pooled_embed = batch["pooled_embed"][b].to(accelerator.device, dtype=network_dtype)
            attention_mask = (
                batch["attention_mask"][b].to(accelerator.device) if batch.get("attention_mask", None) is not None else None
            )
            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                if attention_mask.dim() > 1:
                    attention_mask = attention_mask.reshape(attention_mask.shape[0], -1)
                else:
                    attention_mask = attention_mask.unsqueeze(0)

                mask = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
                mask = mask.bool().flatten()
                if mask.shape[0] != text_embed.shape[0]:
                    # Processor returns padded masks; embeds are packed to valid tokens.
                    # Don't raise here: cache writing already aligns masks to packed embeds.
                    if mask.sum().item() == text_embed.shape[0]:
                        mask = mask[mask]
                    else:
                        mask = torch.ones((text_embed.shape[0],), dtype=torch.bool)
                text_embed = text_embed[mask]
                attention_mask = None

            # latents can be image (C, H, W) or video (C, F, H, W)
            if latent_b.dim() == 4:
                duration = latent_b.shape[-3]
                height, width = latent_b.shape[-2:]
                x = noisy_input_b.permute(1, 2, 3, 0)  # C, F, H, W -> F, H, W, C

                # append visual conditioning channels if model expects them (zeros by default)
                if transformer.visual_cond:
                    visual_cond = torch.zeros_like(x)
                    visual_cond_mask = torch.zeros((*x.shape[:-1], 1), device=accelerator.device, dtype=x.dtype)

                    cond_lat = None
                    if self._i2v_training:
                        assert "latents_image" in batch, (
                            "latents_image not found in batch; run kandinsky5_cache_latents to populate I2V caches"
                        )
                        cond_lat = batch["latents_image"][b].to(accelerator.device, dtype=network_dtype)  # C,F,H,W
                        cond_lat = cond_lat.permute(1, 2, 3, 0)  # F,H,W,C
                    # Align conditioning frames to match video duration.
                    if cond_lat is not None and self._i2v_mode == "first_last" and cond_lat.shape[0] >= 2 and x.shape[0] > 1:
                        # Place first frame at index 0 and last frame at the final index; zero elsewhere.
                        aligned = torch.zeros((x.shape[0], *cond_lat.shape[1:]), device=cond_lat.device, dtype=cond_lat.dtype)
                        aligned[0] = cond_lat[0]
                        aligned[-1] = cond_lat[-1]
                        cond_lat = aligned
                    elif cond_lat is not None:
                        # pad/truncate to match duration
                        if cond_lat.shape[0] < x.shape[0]:
                            pad = x.shape[0] - cond_lat.shape[0]
                            cond_lat = torch.cat(
                                [
                                    cond_lat,
                                    torch.zeros((pad, *cond_lat.shape[1:]), device=cond_lat.device, dtype=cond_lat.dtype),
                                ],
                                dim=0,
                            )
                        elif cond_lat.shape[0] > x.shape[0]:
                            cond_lat = cond_lat[: x.shape[0]]

                    if cond_lat is not None:
                        if self._i2v_mode == "first_last" and cond_lat.shape[0] >= 2:
                            frame_mask = torch.zeros(cond_lat.shape[0], device=cond_lat.device, dtype=torch.bool)
                            frame_mask[0] = True
                            frame_mask[-1] = True
                        else:
                            frame_mask = torch.rand(cond_lat.shape[0], device=cond_lat.device) < self.visual_cond_prob
                            if not frame_mask.any():
                                frame_mask[0] = True  # ensure at least first frame conditioned
                        visual_cond[frame_mask] = cond_lat[frame_mask]
                        visual_cond_mask[frame_mask] = 1

                    x = torch.cat([x, visual_cond, visual_cond_mask], dim=-1)

                # repeat text embeddings and masks per frame (default) or per chunk
                repeat_units = math.ceil(duration / chunk_len) if chunk_mode else duration
                if text_embed.dim() == 3:
                    text_embed = text_embed.view(-1, text_embed.shape[-1])
                text_embed = text_embed.unsqueeze(0).repeat(repeat_units, 1, 1).view(-1, text_embed.shape[-1])

                if pooled_embed.dim() == 1:
                    pooled_embed = pooled_embed.unsqueeze(0)
                pooled_embed = pooled_embed.repeat(repeat_units, 1)

                if attention_mask is not None:
                    attention_mask = attention_mask.repeat(repeat_units, 1).reshape(1, 1, -1)
            else:
                duration = 1
                height, width = latent_b.shape[-2:]
                x = noisy_input_b.permute(1, 2, 0).unsqueeze(0)  # C, H, W -> 1, H, W, C
                if attention_mask is not None:
                    attention_mask = attention_mask.reshape(1, 1, -1)
                if transformer.visual_cond:
                    visual_cond = torch.zeros_like(x)
                    visual_cond_mask = torch.zeros((*x.shape[:-1], 1), device=accelerator.device, dtype=x.dtype)
                    x = torch.cat([x, visual_cond, visual_cond_mask], dim=-1)

            sparse_params = self._build_sparse_params(x, x.device)

            visual_rope_pos = [
                torch.arange(duration, device=accelerator.device),
                torch.arange(height // patch_size[1], device=accelerator.device),
                torch.arange(width // patch_size[2], device=accelerator.device),
            ]
            text_rope_pos = torch.arange(text_embed.shape[0], device=accelerator.device)

            t_b = timesteps[b]
            if t_b.dim() > 0:
                t_b = t_b.flatten()[0]
            t_b = t_b.to(accelerator.device, dtype=network_dtype).unsqueeze(0)

            with accelerator.autocast():
                model_pred = transformer(
                    x,
                    text_embed,
                    pooled_embed,
                    t_b,
                    visual_rope_pos,
                    text_rope_pos,
                    scale_factor=tuple(self.task_conf.scale_factor),
                    sparse_params=sparse_params,
                    attention_mask=attention_mask,
                )

            # transformer outputs [duration, H, W, C]; align to [duration, C, H, W]
            model_pred = model_pred.permute(0, 3, 1, 2)
            target_d = noise_b - latent_b
            if target_d.dim() == 4:
                # C, F, H, W -> F, C, H, W to match duration
                target_d = target_d.permute(1, 0, 2, 3)
            else:
                target_d = target_d.unsqueeze(0)
            preds.append(model_pred)
            targets.append(target_d)

        model_pred = torch.stack(preds, dim=0)  # B, F, C, H, W
        target = torch.stack(targets, dim=0)  # B, F, C, H, W
        return model_pred, target

    # endregion model specific


def kandinsky5_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--task", type=str, required=True, help="Kandinsky5 task key (see configs.TASK_CONFIGS)")
    parser.add_argument("--override_dit", type=str, default=None, help="JSON dict to override DiT params")
    # fp8 and block swap flags are defined in setup_parser_common; reuse them to avoid conflicts
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--text_encoder_qwen", type=str, default=None, help="Override Qwen text encoder path")
    parser.add_argument("--text_encoder_clip", type=str, default=None, help="Override CLIP text encoder path")
    parser.add_argument("--offload_dit_during_sampling", action="store_true", help="Offload DiT during sampling")
    parser.add_argument("--no_vae_load", action="store_true", help="Skip loading VAE; use cached scaling factor only")
    parser.add_argument("--scheduler_scale", type=float, default=None, help="Override scheduler scale (default from task config)")
    parser.add_argument(
        "--i2v_mode",
        type=str,
        default="first",
        choices=["first", "first_last"],
        help="I2V conditioning mode: first frame only (default) or first+last frame.",
    )
    parser.add_argument(
        "--force_nabla_attention", action="store_true", help="Force nabla attention for training regardless of task default"
    )
    parser.add_argument("--nabla_P", type=float, default=0.9, help="CDF threshold P for nabla attention (default 0.9)")
    parser.add_argument("--nabla_wT", type=int, default=11, help="Temporal STA window for nabla attention (default 11)")
    parser.add_argument("--nabla_wH", type=int, default=3, help="Height STA window for nabla attention (default 3)")
    parser.add_argument("--nabla_wW", type=int, default=3, help="Width STA window for nabla attention (default 3)")
    parser.add_argument("--nabla_method", type=str, default="topcdf", help="Nabla map binarization method (default topcdf)")
    parser.add_argument(
        "--nabla_add_sta",
        dest="nabla_add_sta",
        action="store_true",
        default=True,
        help="Include STA prior when forcing nabla attention (default: True)",
    )
    parser.add_argument(
        "--no_nabla_add_sta", dest="nabla_add_sta", action="store_false", help="Disable STA prior when forcing nabla attention"
    )

    return parser


def main():
    parser = setup_parser_common()
    parser = kandinsky5_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    # Propagate compile flag to Kandinsky modules (defaults to disabled).
    k5_attention.set_compile_enabled(bool(getattr(args, "compile", False)))
    k5_nn.set_compile_enabled(bool(getattr(args, "compile", False)))

    # defaults for fp8 flags (not defined in common parser)
    if not hasattr(args, "fp8_base"):
        args.fp8_base = False
    if not hasattr(args, "fp8_scaled"):
        args.fp8_scaled = False
    if not hasattr(args, "fp8_fast"):
        args.fp8_fast = False
    if not hasattr(args, "blocks_to_swap"):
        args.blocks_to_swap = 0
    if not hasattr(args, "dit_dtype"):
        args.dit_dtype = None
    # Avoid casting the entire DiT to float8 during sampling/training when fp8_base is used.
    # Setting fp8_scaled=True keeps the loader's fp8 quantization path but prevents the downstream float8 cast in hv_train_network.
    if args.fp8_base and not args.fp8_scaled:
        args.fp8_scaled = True

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Kandinsky5NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
