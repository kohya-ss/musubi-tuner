"""FramePack Image Pipeline for programmatic use.

This module provides a Pipeline class for FramePack one-frame image generation,
similar in spirit to Diffusers pipelines but without depending on the Diffusers library.

Usage:
    from musubi_tuner.frame_pack.framepack_image_pipeline import FramePackImagePipeline

    pipeline = FramePackImagePipeline.from_pretrained(
        dit="path/to/dit",
        vae="path/to/vae",
        text_encoder1="path/to/llm",
        text_encoder2="path/to/clip_text",
        image_encoder="path/to/clip_vision",
    )
    results = pipeline(
        inputs=[ImageInput(prompt="a cat sitting on a couch")],
        batch_size=4,
    )
    image = results[0].image
"""

import gc
import random
import logging
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any

import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import tqdm

from musubi_tuner.frame_pack.utils import crop_or_pad_yield_mask
from musubi_tuner.frame_pack import hunyuan
from musubi_tuner.frame_pack.hunyuan_video_packed import load_packed_model
from musubi_tuner.frame_pack.hunyuan_video_packed_inference import HunyuanVideoTransformer3DModelPackedInference
from musubi_tuner.frame_pack.clip_vision import hf_clip_vision_encode
from musubi_tuner.frame_pack.k_diffusion_hunyuan import sample_hunyuan
from musubi_tuner.frame_pack.framepack_utils import load_vae, load_text_encoder1, load_text_encoder2, load_image_encoders
from musubi_tuner.fpack_generate_video import convert_lora_for_framepack
from musubi_tuner.dataset import image_video_dataset
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.lora_utils import filter_lora_state_dict
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)


@dataclass
class OneFrameSettings:
    """Settings for one-frame inference mode.

    Attributes:
        no_2x: Disable 2x clean latents.
        no_4x: Disable 4x clean latents.
        no_post: Disable post clean latents (zero latents appended after control images).
        target_index: Index for the target latent. Defaults to latent_window_size.
        control_indices: List of indices for control latent frames. If None, uses defaults.
    """

    no_2x: bool = False
    no_4x: bool = False
    no_post: bool = False
    target_index: Optional[int] = None
    control_indices: Optional[List[int]] = None


@dataclass
class ImageInput:
    """A single generation request.

    Attributes:
        prompt: Text prompt for generation.
        negative_prompt: Negative prompt. Defaults to empty string.
        image: Optional PIL Image for CLIP embedding (I2V). If None, a black placeholder is used.
        control_images: Optional list of PIL Images as control/reference images.
        control_masks: Optional list of PIL Images (mode 'L') as masks for control images.
        image_size: (height, width) tuple. Defaults to (256, 256).
        seed: Random seed. If None, a random seed is generated.
        infer_steps: Number of inference steps. Defaults to 25.
        guidance_scale: Classifier-free guidance scale. Defaults to 1.0 (no guidance).
        embedded_cfg_scale: Distilled CFG scale. Defaults to 10.0.
        guidance_rescale: CFG rescale factor. Defaults to 0.0.
        flow_shift: Flow matching shift factor. If None, uses FramePack default.
        one_frame_settings: Settings for one-frame inference. If None, uses default settings.
        custom_system_prompt: Custom system prompt for LLM. If None, uses default.
    """

    prompt: str = ""
    negative_prompt: str = ""
    image_path: Optional[str] = None
    image: Optional[Image.Image] = None
    control_images: Optional[List[Image.Image]] = None
    control_masks: Optional[List[Image.Image]] = None
    image_size: tuple = (256, 256)
    seed: Optional[int] = None
    infer_steps: int = 25
    strength: Optional[float] = None
    guidance_scale: float = 1.0
    embedded_cfg_scale: float = 10.0
    guidance_rescale: float = 0.0
    flow_shift: Optional[float] = None
    one_frame_settings: Optional[OneFrameSettings] = None
    custom_system_prompt: Optional[str] = None


@dataclass
class PipelineOutput:
    """Output of the pipeline for a single generation.

    Attributes:
        image: Decoded PIL Image (None if output_type is "latent").
        latent: Raw latent tensor (None if output_type is "images").
        seed: The seed used for generation.
    """

    image: Optional[Image.Image] = None
    latent: Optional[torch.Tensor] = None
    seed: int = 0


def _make_args_namespace(**kwargs):
    """Create a simple namespace object to pass to loader functions that expect args."""
    import argparse

    ns = argparse.Namespace()
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


class FramePackImagePipeline:
    """Pipeline for FramePack one-frame image generation.

    This class manages model loading, text/image encoding, latent generation,
    and VAE decoding with automatic GPU offloading between phases.
    """

    def __init__(
        self,
        dit_model: HunyuanVideoTransformer3DModelPackedInference,
        vae,
        tokenizer1,
        text_encoder1,
        tokenizer2,
        text_encoder2,
        feature_extractor,
        image_encoder,
        device: torch.device,
        blocks_to_swap: int = 0,
        use_pinned_memory_for_block_swap: bool = False,
        latent_window_size: int = 9,
        sample_solver: str = "unipc",
        vae_chunk_size: Optional[int] = None,
        vae_spatial_tile_sample_min_size: Optional[int] = None,
        vae_tiling: bool = False,
    ):
        self.dit_model = dit_model
        self.vae = vae
        self.tokenizer1 = tokenizer1
        self.text_encoder1 = text_encoder1
        self.tokenizer2 = tokenizer2
        self.text_encoder2 = text_encoder2
        self.feature_extractor = feature_extractor
        self.image_encoder = image_encoder
        self.device = device
        self.blocks_to_swap = blocks_to_swap
        self.use_pinned_memory_for_block_swap = use_pinned_memory_for_block_swap
        self.latent_window_size = latent_window_size
        self.sample_solver = sample_solver

    @classmethod
    def from_pretrained(
        cls,
        dit: str,
        vae: str,
        text_encoder1: str,
        text_encoder2: str,
        image_encoder: str,
        device: Optional[Union[str, torch.device]] = None,
        attn_mode: str = "torch",
        split_attn: bool = False,
        fp8: bool = False,
        fp8_scaled: bool = False,
        fp8_fast_quantization_mode: Optional[str] = None,
        fp8_block_size: int = 64,
        fp8_llm: bool = False,
        nvfp4: bool = False,
        nvfp4_compile: bool = False,
        blocks_to_swap: int = 0,
        use_pinned_memory_for_block_swap: bool = False,
        lora_weights: Optional[List[str]] = None,
        lora_multipliers: Optional[List[float]] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        latent_window_size: int = 9,
        sample_solver: str = "unipc",
        rope_scaling_factor: float = 0.5,
        rope_scaling_timestep_threshold: Optional[int] = None,
        disable_numpy_memmap: bool = False,
        compile: bool = False,
        compile_args: Optional[dict] = None,
        vae_chunk_size: Optional[int] = None,
        vae_spatial_tile_sample_min_size: Optional[int] = None,
        vae_tiling: bool = False,
    ) -> "FramePackImagePipeline":
        """Load all models and create a pipeline instance.

        Args:
            dit: Path to DiT model weights.
            vae: Path to VAE model weights.
            text_encoder1: Path to LLM text encoder.
            text_encoder2: Path to CLIP text encoder.
            image_encoder: Path to CLIP vision encoder.
            device: Device for inference. Defaults to CUDA if available.
            attn_mode: Attention mode ("torch", "flash", "sageattn", "xformers", "sdpa").
            split_attn: Whether to split attention for memory efficiency.
            fp8: Use fp8 for DiT model.
            fp8_scaled: Use scaled fp8 for DiT.
            fp8_fast_quantization_mode: Quantization mode for fp8 ("tensor", "block", "channel", or None).
            fp8_block_size: Block size for fp8 optimization.
            fp8_llm: Use fp8 for Text Encoder 1 (LLM).
            nvfp4: Use NVFP4 for DiT model.
            nvfp4_compile: Enable torch.compile for NVFP4.
            blocks_to_swap: Number of DiT blocks to swap to CPU.
            use_pinned_memory_for_block_swap: Use pinned memory for block swapping.
            lora_weights: List of paths to LoRA weight files.
            lora_multipliers: List of LoRA multipliers.
            include_patterns: LoRA module include patterns.
            exclude_patterns: LoRA module exclude patterns.
            latent_window_size: Latent window size (default 9).
            sample_solver: Sampling solver ("unipc", "dpm++", "vanilla").
            rope_scaling_factor: RoPE scaling factor.
            rope_scaling_timestep_threshold: RoPE scaling timestep threshold.
            disable_numpy_memmap: Disable numpy memmap for safetensors.
            compile: Whether to use torch.compile on the model.
            compile_args: Additional arguments for torch.compile (passed as args namespace attributes).
            vae_chunk_size: Chunk size for CausalConv3d in VAE.
            vae_spatial_tile_sample_min_size: Spatial tile sample min size for VAE.
            vae_tiling: Enable spatial tiling for VAE.

        Returns:
            FramePackImagePipeline instance.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        # --- Load DiT ---
        loading_device = "cpu" if blocks_to_swap > 0 else device

        lora_weights_list = None
        if lora_weights:
            lora_weights_list = []
            for lw_path in lora_weights:
                logger.info(f"Loading LoRA weight from: {lw_path}")
                lora_sd = load_file(lw_path)
                lora_sd = convert_lora_for_framepack(lora_sd)
                lora_sd = filter_lora_state_dict(lora_sd, include_patterns, exclude_patterns)
                lora_weights_list.append(lora_sd)

        if lora_multipliers is None and lora_weights_list is not None:
            lora_multipliers = [1.0] * len(lora_weights_list)

        logger.info(f"Loading DiT model from: {dit}")
        dit_model: HunyuanVideoTransformer3DModelPackedInference = load_packed_model(
            device,
            dit,
            attn_mode,
            loading_device,
            fp8_scaled,
            nvfp4,
            split_attn=split_attn,
            for_inference=True,
            lora_weights_list=lora_weights_list,
            lora_multipliers=lora_multipliers,
            disable_numpy_memmap=disable_numpy_memmap,
            fp8_fast_quantization_mode=fp8_fast_quantization_mode if fp8_scaled else None,
            nvfp4_use_torch_compile=nvfp4_compile,
            block_size=fp8_block_size,
        )

        if rope_scaling_timestep_threshold is not None:
            logger.info(f"Applying RoPE scaling factor {rope_scaling_factor} for timesteps >= {rope_scaling_timestep_threshold}")
            dit_model.enable_rope_scaling(rope_scaling_timestep_threshold, rope_scaling_factor)

        if not fp8_scaled and not nvfp4:
            target_dtype = torch.float8e4m3fn if fp8 else None
            if target_dtype is not None and blocks_to_swap == 0:
                dit_model.to(device, target_dtype)

        if blocks_to_swap > 0:
            logger.info(f"Enable swap {blocks_to_swap} blocks to CPU from device: {device}")
            dit_model.enable_block_swap(
                blocks_to_swap, device, supports_backward=False, use_pinned_memory=use_pinned_memory_for_block_swap
            )
            dit_model.move_to_device_except_swap_blocks(device)
            dit_model.prepare_block_swap_before_forward()
        else:
            dit_model.to(device)

        if compile:
            compile_ns = _make_args_namespace(compile=True, **(compile_args or {}))
            dit_model = model_utils.compile_transformer(
                compile_ns,
                dit_model,
                [dit_model.transformer_blocks, dit_model.single_transformer_blocks],
                disable_linear=blocks_to_swap > 0,
            )

        dit_model.eval().requires_grad_(False)

        # --- Load VAE (to CPU) ---
        logger.info(f"Loading VAE from: {vae}")
        vae_model = load_vae(vae, vae_chunk_size, vae_spatial_tile_sample_min_size, vae_tiling, "cpu")

        # --- Load Text Encoders (to CPU) ---
        logger.info(f"Loading Text Encoder 1 from: {text_encoder1}")
        te1_args = _make_args_namespace(text_encoder1=text_encoder1)
        tok1, te1_model = load_text_encoder1(te1_args, fp8_llm, "cpu")

        logger.info(f"Loading Text Encoder 2 from: {text_encoder2}")
        te2_args = _make_args_namespace(text_encoder2=text_encoder2)
        tok2, te2_model = load_text_encoder2(te2_args)

        # --- Load Image Encoder (to CPU) ---
        logger.info(f"Loading Image Encoder from: {image_encoder}")
        ie_args = _make_args_namespace(image_encoder=image_encoder)
        feat_ext, ie_model = load_image_encoders(ie_args)

        clean_memory_on_device(device)

        return cls(
            dit_model=dit_model,
            vae=vae_model,
            tokenizer1=tok1,
            text_encoder1=te1_model,
            tokenizer2=tok2,
            text_encoder2=te2_model,
            feature_extractor=feat_ext,
            image_encoder=ie_model,
            device=device,
            blocks_to_swap=blocks_to_swap,
            use_pinned_memory_for_block_swap=use_pinned_memory_for_block_swap,
            latent_window_size=latent_window_size,
            sample_solver=sample_solver,
            vae_chunk_size=vae_chunk_size,
            vae_spatial_tile_sample_min_size=vae_spatial_tile_sample_min_size,
            vae_tiling=vae_tiling,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def __call__(
        self,
        inputs: Union[ImageInput, List[ImageInput]],
        batch_size: int = 1,
        output_type: str = "pil",
        generation_callback: Optional[callable] = None,
    ) -> List[PipelineOutput]:
        """Generate images from the given inputs.

        Args:
            inputs: A single ImageInput or a list of ImageInput objects.
            batch_size: Number of images to generate per DiT forward batch.
                All text/image encoding and VAE decoding are done for all inputs at once,
                while DiT generation is batched by this size.
            output_type: "pil" for PIL Images, "latent" for raw latents, "both" for both.
            generation_callback: Optional callback function called after each batch of generation with signature
                (batch_index: int, latents: List[torch.Tensor], seeds: List[int]) -> None

        Returns:
            List of PipelineOutput, one per input.
        """
        if isinstance(inputs, ImageInput):
            inputs = [inputs]

        if not inputs:
            return []

        device = self.device
        n = len(inputs)

        # --- Phase 1: Text encoding (TE on GPU, DiT off GPU) ---
        logger.info(f"Phase 1: Encoding text for {n} inputs...")
        self._offload_dit()
        all_text_data = self._encode_texts(inputs)

        # --- Phase 2: Image encoding (VAE + Image Encoder on GPU) ---
        logger.info(f"Phase 2: Encoding images for {n} inputs...")
        all_image_data = self._encode_images(inputs)

        # --- Phase 3: DiT generation (DiT on GPU, others off) ---
        logger.info(f"Phase 3: Generating latents for {n} inputs (batch_size={batch_size})...")
        self._load_dit_to_device()
        all_latents = []
        all_seeds = []

        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_inputs = inputs[batch_start:batch_end]
            batch_text = all_text_data[batch_start:batch_end]
            batch_image = all_image_data[batch_start:batch_end]

            latents, seeds = self._generate_batch(batch_inputs, batch_text, batch_image)
            all_latents.extend(latents)
            all_seeds.extend(seeds)

            if generation_callback is not None:
                generation_callback(batch_start // batch_size, latents, seeds)

        # --- Phase 4: VAE decode (VAE on GPU, DiT off) ---
        results = []
        if output_type in ("pil", "both"):
            logger.info(f"Phase 4: Decoding {n} latents with VAE...")
            self._offload_dit()
            clean_memory_on_device(device)
            decoded_images = self._decode_latents(all_latents)

            for i in range(n):
                results.append(
                    PipelineOutput(
                        image=decoded_images[i],
                        latent=all_latents[i] if output_type == "both" else None,
                        seed=all_seeds[i],
                    )
                )
        else:
            for i in range(n):
                results.append(
                    PipelineOutput(
                        image=None,
                        latent=all_latents[i],
                        seed=all_seeds[i],
                    )
                )

        clean_memory_on_device(device)
        return results

    # -------------------------------------------------------------------------
    # Internal: Text encoding
    # -------------------------------------------------------------------------

    def _encode_texts(self, inputs: List[ImageInput]) -> List[Dict[str, Any]]:
        """Encode all prompts using text encoders. Returns list of dicts with context/context_null."""
        device = self.device
        conds_cache: Dict[str, tuple] = {}

        self.text_encoder1.to(device)
        self.text_encoder2.to(device)

        results = []
        try:
            for inp in inputs:
                prompt = inp.prompt or ""
                n_prompt = inp.negative_prompt or ""

                with torch.autocast(device_type=device.type, dtype=self.text_encoder1.dtype), torch.no_grad():
                    # Positive prompt
                    if prompt in conds_cache:
                        llama_vec, clip_l_pooler = conds_cache[prompt]
                    else:
                        llama_vec, clip_l_pooler = hunyuan.encode_prompt_conds(
                            prompt,
                            self.text_encoder1,
                            self.text_encoder2,
                            self.tokenizer1,
                            self.tokenizer2,
                            custom_system_prompt=inp.custom_system_prompt,
                        )
                        llama_vec = llama_vec.cpu()
                        clip_l_pooler = clip_l_pooler.cpu()
                        conds_cache[prompt] = (llama_vec, clip_l_pooler)

                    llama_vec_padded, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)

                    # Negative prompt
                    if inp.guidance_scale == 1.0:
                        llama_vec_n = torch.zeros_like(llama_vec_padded)
                        clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
                        llama_attention_mask_n = torch.zeros_like(llama_attention_mask)
                    elif n_prompt in conds_cache:
                        llama_vec_n, clip_l_pooler_n = conds_cache[n_prompt]
                        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
                    else:
                        llama_vec_n, clip_l_pooler_n = hunyuan.encode_prompt_conds(
                            n_prompt,
                            self.text_encoder1,
                            self.text_encoder2,
                            self.tokenizer1,
                            self.tokenizer2,
                            custom_system_prompt=inp.custom_system_prompt,
                        )
                        llama_vec_n = llama_vec_n.cpu()
                        clip_l_pooler_n = clip_l_pooler_n.cpu()
                        conds_cache[n_prompt] = (llama_vec_n, clip_l_pooler_n)
                        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

                results.append(
                    {
                        "context": {
                            "llama_vec": llama_vec_padded,
                            "llama_attention_mask": llama_attention_mask,
                            "clip_l_pooler": clip_l_pooler,
                        },
                        "context_null": {
                            "llama_vec": llama_vec_n,
                            "llama_attention_mask": llama_attention_mask_n,
                            "clip_l_pooler": clip_l_pooler_n,
                        },
                    }
                )
        finally:
            self.text_encoder1.to("cpu")
            self.text_encoder2.to("cpu")
            gc.collect()
            clean_memory_on_device(device)

        return results

    # -------------------------------------------------------------------------
    # Internal: Image encoding
    # -------------------------------------------------------------------------

    def _preprocess_pil_image(self, image: Image.Image, height: int, width: int) -> tuple:
        """Convert PIL Image to tensor and numpy array for pipeline use."""
        if image.mode == "RGBA":
            alpha = image.split()[-1]
        else:
            alpha = None
        image_rgb = image.convert("RGB")
        image_np = np.array(image_rgb)
        image_np = image_video_dataset.resize_image_to_bucket(image_np, (width, height))
        image_tensor = torch.from_numpy(image_np).float() / 127.5 - 1.0
        image_tensor = image_tensor.permute(2, 0, 1)[None, :, None]  # NCFHW
        return image_tensor, image_np, alpha

    def _encode_images(self, inputs: List[ImageInput]) -> List[Dict[str, Any]]:
        """Encode all images using VAE and image encoder."""
        device = self.device

        self.vae.to(device)
        self.image_encoder.to(device)

        results = []
        try:
            for inp in tqdm.tqdm(inputs, desc="Preprocessing images", unit="image"):
                height, width = inp.image_size

                if inp.image is not None:
                    img_tensor, img_np, _ = self._preprocess_pil_image(inp.image, height, width)
                else:
                    img_tensor = torch.zeros(1, 3, 1, height, width)
                    img_np = np.zeros((height, width, 3), dtype=np.uint8)

                # CLIP Vision encode
                with torch.no_grad():
                    image_encoder_output = hf_clip_vision_encode(img_np, self.feature_extractor, self.image_encoder)
                image_encoder_last_hidden_state = image_encoder_output.last_hidden_state.cpu()

                # # VAE encode
                # start_latent = hunyuan.vae_encode(img_tensor.to(device), self.vae).cpu()

                # Control images
                control_tensors = None
                control_mask_images = None
                if inp.control_images:
                    # control_latents = []
                    control_tensors = []
                    control_mask_images = []
                    for ctrl_img in inp.control_images:
                        ctrl_tensor, _, ctrl_alpha = self._preprocess_pil_image(ctrl_img, height, width)
                        # ctrl_latent = hunyuan.vae_encode(ctrl_tensor.to(device), self.vae).cpu()
                        # control_latents.append(ctrl_latent)
                        control_tensors.append(ctrl_tensor)
                        control_mask_images.append(ctrl_alpha)

                results.append(
                    {
                        "height": height,
                        "width": width,
                        "image_encoder_last_hidden_state": image_encoder_last_hidden_state,
                        "img_tensor": img_tensor,
                        "control_tensors": control_tensors,
                        "control_mask_images": control_mask_images,
                    }
                )

            # Encode with VAE
            for res in tqdm.tqdm(results, desc="Encoding with VAE", unit="image"):
                img_tensor = res["img_tensor"]
                with torch.no_grad():
                    start_latent = hunyuan.vae_encode(img_tensor.to(device), self.vae).cpu()
                res["start_latent"] = start_latent
                res.pop("img_tensor")  # free memory

                control_tensors = res["control_tensors"]
                if control_tensors is not None:
                    control_latents = []
                    for ctrl_tensor in control_tensors:
                        # If ctrl_tensor is same as img_tensor, reuse start_latent to save memory and computation
                        if torch.equal(ctrl_tensor, img_tensor):
                            ctrl_latent = start_latent
                        else:
                            ctrl_tensor = ctrl_tensor.to(device)
                            with torch.no_grad():
                                ctrl_latent = hunyuan.vae_encode(ctrl_tensor, self.vae).cpu()
                        control_latents.append(ctrl_latent)
                    res["control_latents"] = control_latents
                res.pop("control_tensors")  # free memory

        finally:
            self.vae.to("cpu")
            self.image_encoder.to("cpu")
            clean_memory_on_device(device)

        return results

    # -------------------------------------------------------------------------
    # Internal: DiT generation
    # -------------------------------------------------------------------------

    def _generate_batch(
        self,
        inputs: List[ImageInput],
        text_data: List[Dict[str, Any]],
        image_data: List[Dict[str, Any]],
    ) -> tuple:
        """Generate latents for a batch of inputs using true batched inference.

        All inputs in a batch must share the same height, width, infer_steps,
        guidance_scale, embedded_cfg_scale, guidance_rescale, flow_shift,
        and one_frame_settings. If they differ, a ValueError is raised.

        Returns (latents_list, seeds_list) where each latent has batch_size=1.
        """
        device = self.device
        batch_size = len(inputs)

        if batch_size == 0:
            return [], []

        # --- Validate batch uniformity ---
        ref = inputs[0]
        ref_imd = image_data[0]
        ref_one_frame = ref.one_frame_settings or OneFrameSettings()
        ref_height, ref_width = ref_imd["height"], ref_imd["width"]

        for i, (inp, imd) in enumerate(zip(inputs[1:], image_data[1:]), start=1):
            h, w = imd["height"], imd["width"]
            if (h, w) != (ref_height, ref_width):
                raise ValueError(
                    f"Batch item {i} has image_size ({h}, {w}) but item 0 has ({ref_height}, {ref_width}). "
                    "All items in a batch must share the same image_size."
                )
            if inp.infer_steps != ref.infer_steps:
                raise ValueError(f"Batch item {i} has infer_steps={inp.infer_steps} but item 0 has {ref.infer_steps}.")
            if inp.strength != ref.strength:
                raise ValueError(f"Batch item {i} has strength={inp.strength} but item 0 has {ref.strength}.")
            if inp.guidance_scale != ref.guidance_scale:
                raise ValueError(f"Batch item {i} has guidance_scale={inp.guidance_scale} but item 0 has {ref.guidance_scale}.")
            if inp.embedded_cfg_scale != ref.embedded_cfg_scale:
                raise ValueError(
                    f"Batch item {i} has embedded_cfg_scale={inp.embedded_cfg_scale} but item 0 has {ref.embedded_cfg_scale}."
                )
            if inp.guidance_rescale != ref.guidance_rescale:
                raise ValueError(
                    f"Batch item {i} has guidance_rescale={inp.guidance_rescale} but item 0 has {ref.guidance_rescale}."
                )
            if inp.flow_shift != ref.flow_shift:
                raise ValueError(f"Batch item {i} has flow_shift={inp.flow_shift} but item 0 has {ref.flow_shift}.")
            item_one_frame = inp.one_frame_settings or OneFrameSettings()
            if item_one_frame != ref_one_frame:
                raise ValueError(f"Batch item {i} has different one_frame_settings than item 0.")

        height, width = ref_height, ref_width
        latent_window_size = self.latent_window_size
        one_frame = ref_one_frame

        # --- Pre-generate noise per item for seed reproducibility ---
        seeds = []
        noise_list = []
        latent_h, latent_w = height // 8, width // 8
        frames = 1
        latent_t = (frames + 3) // 4

        for inp in inputs:
            seed = inp.seed if inp.seed is not None else random.randint(0, 2**32 - 1)
            seeds.append(seed)
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            noise = torch.randn((1, 16, latent_t, latent_h, latent_w), generator=g, device="cpu")
            noise_list.append(noise)

        batched_noise = torch.cat(noise_list, dim=0)  # (B, 16, T, H, W)

        # --- Build one-frame latent structure (shared across batch) ---
        per_item_clean_latents = []
        for i, (inp, imd) in enumerate(zip(inputs, image_data)):
            control_lats = list(imd["control_latents"]) if imd["control_latents"] else []
            control_masks = imd["control_mask_images"]

            if len(control_lats) == 0:
                control_lats = [torch.zeros(1, 16, 1, latent_h, latent_w, dtype=torch.float32)]

            if not one_frame.no_post:
                control_lats.append(torch.zeros(1, 16, 1, latent_h, latent_w, dtype=torch.float32))

            item_clean = torch.cat(control_lats, dim=2)  # (1, 16, num_ctrl, H, W)

            # Apply control masks
            if inp.control_masks:
                for j, mask_img in enumerate(inp.control_masks):
                    if mask_img is not None and j < item_clean.shape[2]:
                        mask_tensor = self._get_latent_mask(mask_img, height, width)
                        item_clean[:, :, j : j + 1, :, :] = item_clean[:, :, j : j + 1, :, :] * mask_tensor
            elif control_masks:
                for j, alpha in enumerate(control_masks):
                    if alpha is not None and j < item_clean.shape[2]:
                        mask_tensor = self._get_latent_mask(alpha, height, width)
                        item_clean[:, :, j : j + 1, :, :] = item_clean[:, :, j : j + 1, :, :] * mask_tensor

            per_item_clean_latents.append(item_clean)

        # Verify all items have the same number of control frames
        num_ctrl_frames = per_item_clean_latents[0].shape[2]
        for i, cl in enumerate(per_item_clean_latents[1:], start=1):
            if cl.shape[2] != num_ctrl_frames:
                raise ValueError(
                    f"Batch item {i} has {cl.shape[2]} control frames but item 0 has {num_ctrl_frames}. "
                    "All items in a batch must have the same number of control images."
                )

        clean_latents = torch.cat(per_item_clean_latents, dim=0)  # (B, 16, num_ctrl, H, W)

        # Clean latent indices (same for all items in batch)
        num_control_lats = num_ctrl_frames
        clean_latent_indices = torch.zeros((1, num_control_lats), dtype=torch.int64)
        if not one_frame.no_post:
            clean_latent_indices[:, -1] = 1 + latent_window_size

        if one_frame.control_indices is not None:
            for i, idx in enumerate(one_frame.control_indices):
                if i < clean_latent_indices.shape[1]:
                    clean_latent_indices[:, i] = idx

        clean_latent_indices = clean_latent_indices.expand(batch_size, -1)  # (B, num_ctrl)

        # Target latent indices
        target_idx = one_frame.target_index if one_frame.target_index is not None else latent_window_size
        latent_indices = torch.full((batch_size, 1), target_idx, dtype=torch.int64)

        # 2x / 4x clean latents
        if one_frame.no_2x:
            clean_latents_2x = None
            clean_latent_2x_indices = None
        else:
            clean_latents_2x = torch.zeros((batch_size, 16, 2, latent_h, latent_w), dtype=torch.float32)
            index = 1 + latent_window_size + 1
            clean_latent_2x_indices = torch.arange(index, index + 2).unsqueeze(0).expand(batch_size, -1)

        if one_frame.no_4x:
            clean_latents_4x = None
            clean_latent_4x_indices = None
        else:
            clean_latents_4x = torch.zeros((batch_size, 16, 16, latent_h, latent_w), dtype=torch.float32)
            index = 1 + latent_window_size + 1 + 2
            clean_latent_4x_indices = torch.arange(index, index + 16).unsqueeze(0).expand(batch_size, -1)

        # --- Batch conditioning tensors ---
        llama_vecs = torch.cat([td["context"]["llama_vec"] for td in text_data], dim=0).to(device, dtype=torch.bfloat16)
        llama_masks = torch.cat([td["context"]["llama_attention_mask"] for td in text_data], dim=0).to(device)
        clip_poolers = torch.cat([td["context"]["clip_l_pooler"] for td in text_data], dim=0).to(device, dtype=torch.bfloat16)
        img_embeds = torch.cat([imd["image_encoder_last_hidden_state"] for imd in image_data], dim=0).to(
            device, dtype=torch.bfloat16
        )

        llama_vecs_n = torch.cat([td["context_null"]["llama_vec"] for td in text_data], dim=0).to(device, dtype=torch.bfloat16)
        llama_masks_n = torch.cat([td["context_null"]["llama_attention_mask"] for td in text_data], dim=0).to(device)
        clip_poolers_n = torch.cat([td["context_null"]["clip_l_pooler"] for td in text_data], dim=0).to(
            device, dtype=torch.bfloat16
        )

        # --- Start latents for I2I
        start_latents = torch.cat([imd["start_latent"] for imd in image_data], dim=0).to(device)

        # --- Single batched call to sample_hunyuan ---
        generated = sample_hunyuan(
            transformer=self.dit_model,
            sampler=self.sample_solver,
            initial_latent=start_latents,
            strength=1.0 if ref.strength is None else ref.strength,
            width=width,
            height=height,
            frames=frames,
            real_guidance_scale=ref.guidance_scale,
            distilled_guidance_scale=ref.embedded_cfg_scale,
            guidance_rescale=ref.guidance_rescale,
            shift=ref.flow_shift,
            num_inference_steps=ref.infer_steps,
            batch_size=batch_size,
            noise=batched_noise,
            prompt_embeds=llama_vecs,
            prompt_embeds_mask=llama_masks,
            prompt_poolers=clip_poolers,
            negative_prompt_embeds=llama_vecs_n,
            negative_prompt_embeds_mask=llama_masks_n,
            negative_prompt_poolers=clip_poolers_n,
            device=device,
            dtype=torch.bfloat16,
            image_embeddings=img_embeds,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
        )

        # Split batch results into per-item latents
        result_latents = [generated[i : i + 1].to(torch.float32).cpu() for i in range(batch_size)]

        return result_latents, seeds

    @staticmethod
    def _get_latent_mask(mask_image: Image.Image, height: int, width: int) -> torch.Tensor:
        """Convert a mask image to a latent-space mask tensor."""
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        mask_image = mask_image.resize((width // 8, height // 8), Image.LANCZOS)
        mask_np = np.array(mask_image)
        mask_tensor = torch.from_numpy(mask_np).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # HW -> 111HW (BCFHW)
        return mask_tensor

    # -------------------------------------------------------------------------
    # Internal: VAE decoding
    # -------------------------------------------------------------------------

    def _decode_latents(self, latents: List[torch.Tensor]) -> List[Image.Image]:
        """Decode latents to PIL Images using VAE."""
        device = self.device
        self.vae.to(device)

        images = []
        try:
            for latent in tqdm.tqdm(latents, desc="Decoding latents", unit="latent"):
                if latent.ndim == 4:
                    latent = latent.unsqueeze(0)

                # Decode each frame separately (one-frame inference produces 1 frame typically)
                pixels_list = []
                for i in range(latent.shape[2]):
                    frame_pixels = hunyuan.vae_decode(latent[:, :, i : i + 1, :, :].to(device), self.vae).cpu()
                    pixels_list.append(frame_pixels)
                pixels = torch.cat(pixels_list, dim=2)

                # Convert to PIL: pixels is NCFHW, take first batch and first frame
                frame = pixels[0, :, 0]  # CHW
                frame = (frame.clamp(-1, 1) + 1) / 2 * 255
                frame = frame.permute(1, 2, 0).to(torch.uint8).numpy()
                images.append(Image.fromarray(frame))
        finally:
            self.vae.to("cpu")
            clean_memory_on_device(device)

        return images

    # -------------------------------------------------------------------------
    # Internal: Model offloading
    # -------------------------------------------------------------------------

    def _offload_dit(self):
        """Move DiT model to CPU to free GPU memory."""
        if self.blocks_to_swap > 0:
            import time

            time.sleep(5)  # wait for block swap to finish
        self.dit_model.to("cpu")
        clean_memory_on_device(self.device)

    def _load_dit_to_device(self):
        """Move DiT model to device for generation."""
        if self.blocks_to_swap > 0:
            self.dit_model.move_to_device_except_swap_blocks(self.device)
            self.dit_model.prepare_block_swap_before_forward()
        else:
            self.dit_model.to(self.device)
