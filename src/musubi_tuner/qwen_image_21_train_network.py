"""Qwen-Image 2.1 LoRA training."""

import argparse
import gc

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.nn import functional as F

from musubi_tuner.dataset.architectures import ARCHITECTURE_QWEN_IMAGE_21, ARCHITECTURE_QWEN_IMAGE_21_FULL
from musubi_tuner.dataset.bucket import BucketSelector
from musubi_tuner.dataset.media_utils import resize_image_to_bucket
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.qwen_image_21 import qwen_image_21_model
from musubi_tuner.qwen_image_21 import qwen_image_21_utils
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.device_utils import clean_memory_on_device


def prepare_conditioning(batch: dict, device: torch.device, dtype: torch.dtype):
    """Check reference cache consistency, pad text features, and pack reference latents."""
    embeds = batch["vl_embed"]
    lengths = [e.shape[0] for e in embeds]
    if not lengths or min(lengths) < 1 or any(e.ndim != 2 or e.shape[-1] != 4096 for e in embeds):
        raise ValueError("Expected Qwen3-VL 4096-dimensional text features; rebuild 2.1 text caches")
    text = torch.stack([F.pad(e, (0, 0, 0, max(lengths) - e.shape[0])) for e in embeds]).to(device=device, dtype=dtype)
    refs = []
    shapes = []
    control_keys = sorted((k for k in batch if k.startswith("latents_control_")), key=lambda k: int(k.rsplit("_", 1)[1]))
    if control_keys != [f"latents_control_{i}" for i in range(len(control_keys))]:
        raise ValueError("Reference cache indices must be contiguous")
    for key in control_keys:
        latent = batch[key]
        refs.append(qwen_image_21_utils.pack_latents(latent).to(device=device, dtype=dtype))
        shapes.append((1, latent.shape[-2], latent.shape[-1]))
    if "image_slots" not in batch or "reference_grids" not in batch:
        raise ValueError("Missing 2.1 reference metadata; rebuild text caches even for text-to-image")
    slots, grids = batch["image_slots"], batch["reference_grids"]
    if len(slots) != len(embeds) or len(grids) != len(embeds):
        raise ValueError("Reference metadata batch size mismatch")
    expected = [list(shape[1:]) for shape in shapes]
    for row_slots, row_grids in zip(slots, grids):
        if len(row_slots) != len(refs) or row_grids.tolist() != expected:
            raise ValueError("Reference VLM and VAE cache layouts disagree; rebuild both caches with the same dataset config")
    if refs:
        if "reference_hashes" not in batch or "vlm_reference_hashes" not in batch:
            raise ValueError("Missing reference content hashes; rebuild both Qwen-Image 2.1 caches")
        vae_hashes, vlm_hashes = batch["reference_hashes"], batch["vlm_reference_hashes"]
        if len(vae_hashes) != len(embeds) or len(vlm_hashes) != len(embeds):
            raise ValueError("Reference fingerprint batch size mismatch")
        for left, right in zip(vae_hashes, vlm_hashes):
            if left.shape != (len(refs), 32) or not torch.equal(left.cpu(), right.cpu()):
                raise ValueError("Reference contents/order differ between VAE and VLM caches; rebuild both caches")
    return text, lengths, refs, shapes, [s.tolist() for s in slots]


class QwenImage21NetworkTrainer(NetworkTrainer):
    """Trainer for Qwen-Image 2.1."""

    def __init__(self):
        super().__init__()
        self.vae_frame_stride = 1

    @property
    def architecture(self):
        return ARCHITECTURE_QWEN_IMAGE_21

    @property
    def architecture_full_name(self):
        return ARCHITECTURE_QWEN_IMAGE_21_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.default_discrete_flow_shift = None
        if args.sample_prompts and (not args.text_encoder or not args.vae):
            raise ValueError("Sampling requires --text_encoder and --vae")

    def load_transformer(self, accelerator, args, dit_path, attn_mode, split_attn, loading_device, dit_weight_dtype):
        model = qwen_image_21_model.load_model(
            dit_path,
            accelerator.device,
            loading_device,
            dit_weight_dtype,
            args.fp8_scaled,
            args.dit_config,
            args.disable_numpy_memmap,
            attn_mode=attn_mode,
            lora_weights_list=[load_file(path) for path in args.base_weights] if args.base_weights else None,
            lora_multipliers=args.base_weights_multiplier,
            num_layers=args.num_layers,
        )
        model.img_in_txt_in_offloading = args.img_in_txt_in_offloading
        return model

    def merge_base_weights(self, args, accelerator, transformer, network_module, weight_dtype):
        # load_model merges base weights before FP8 quantization.
        pass

    def compile_transformer(self, args, transformer):
        return model_utils.compile_transformer(
            args, transformer, [transformer.transformer_blocks], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # Latents are normalized during caching.
        return latents

    def latent_tokens_for_timestep_sampling(self, latents: torch.Tensor) -> int:
        return latents.shape[-2] * latents.shape[-1]

    def load_vae(self, args, vae_dtype, vae_path):
        return qwen_image_21_utils.load_vae(vae_path, "cpu", vae_dtype, args.vae_tiling)

    def call_dit(
        self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
    ):
        target = qwen_image_21_utils.pack_latents(noisy_model_input).to(device=accelerator.device, dtype=network_dtype)
        text, lengths, refs, shapes, slots = prepare_conditioning(batch, accelerator.device, network_dtype)
        if args.gradient_checkpointing:
            target.requires_grad_(True)
            text.requires_grad_(True)
        shapes = [shapes + [(1, latents.shape[-2], latents.shape[-1])]]
        with accelerator.autocast():
            pred = transformer(
                hidden_states=target,
                encoder_hidden_states=text,
                timestep=timesteps / 1000.0,
                img_shapes=shapes,
                txt_seq_lens=lengths,
                reference_latents=refs,
                image_slots=slots,
            )
        pred = qwen_image_21_utils.unpack_latents(pred, latents.shape[-2], latents.shape[-1])
        return DiTOutput(pred=pred, target=(noise - latents).to(device=accelerator.device, dtype=network_dtype))

    def process_sample_prompts(self, args, accelerator, sample_prompts):
        processor, encoder = qwen_image_21_utils.load_text_encoder(args.text_encoder, device=accelerator.device, fp8_vl=args.fp8_vl)
        prompts = load_prompts(sample_prompts)
        for prompt in prompts:
            images = []
            for path in prompt.get("control_image_path", []):
                with Image.open(path) as source:
                    source = source.convert("RGBA")
                    size = BucketSelector.calculate_bucket_resolution(source.size, (1024, 1024), architecture=self.architecture)
                    images.append(resize_image_to_bucket(source, size))
            prompt["reference_images"] = images
            captions = [("positive", prompt.get("prompt", ""))]
            if prompt.get("negative_prompt") is not None and (prompt.get("cfg_scale") or 1.0) > 1.0:
                captions.append(("negative", prompt["negative_prompt"]))
            for name, caption in captions:
                prompt[name] = tuple(t.cpu() for t in qwen_image_21_utils.encode_prompt(processor, encoder, caption, images))
        del processor, encoder
        gc.collect()
        clean_memory_on_device(accelerator.device)
        return prompts

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
        width, height = max(32, width // 32 * 32), max(32, height // 32 * 32)
        device = accelerator.device
        vae.to(device)
        refs = [qwen_image_21_utils.encode_image(vae, image) for image in sample_parameter["reference_images"]]
        vae.to("cpu")
        shapes = [[(1, ref.shape[-2], ref.shape[-1]) for ref in refs] + [(1, height // 16, width // 16)]]
        refs = [qwen_image_21_utils.pack_latents(ref).to(device=device, dtype=dit_dtype) for ref in refs]
        latents = torch.randn((1, height // 16 * (width // 16), 64), generator=generator, device=device, dtype=dit_dtype)
        # Qwen-Image 2.1 uses the same scheduler configuration as Qwen-Image.
        scheduler = qwen_image_utils.get_scheduler(discrete_flow_shift)
        sigmas = np.linspace(1.0, 1 / sample_steps, sample_steps)
        mu = qwen_image_utils.calculate_shift_qwen_image(latents.shape[1])
        scheduler.set_timesteps(sample_steps, device=device, sigmas=sigmas, mu=mu)
        scheduler.set_begin_index(0)
        cfg_scale = 1.0 if cfg_scale is None else cfg_scale
        do_cfg = do_classifier_free_guidance and cfg_scale > 1.0

        def predict(name, timestep):
            embed, slots, _ = sample_parameter[name]
            return transformer(
                hidden_states=latents,
                encoder_hidden_states=embed[None].to(device=device, dtype=dit_dtype),
                timestep=timestep.expand(1).to(dit_dtype) / 1000,
                img_shapes=shapes,
                reference_latents=refs,
                image_slots=[slots.tolist()],
            )

        with torch.no_grad(), accelerator.autocast():
            for timestep in scheduler.timesteps:
                transformer.prepare_block_swap_before_forward()
                pred = predict("positive", timestep)
                if do_cfg:
                    transformer.prepare_block_swap_before_forward()
                    negative = predict("negative", timestep)
                    pred = negative + cfg_scale * (pred - negative)
                latents = scheduler.step(pred, timestep, latents, return_dict=False)[0]
            vae.to(device)
            pixels = qwen_image_21_utils.decode_latents(vae, qwen_image_21_utils.unpack_latents(latents, height // 16, width // 16))
        # The sample writer expects RGB [B, C, F, H, W]. Composite RGBA over white.
        if pixels.shape[1] == 4:
            pixels = pixels[:, :3] * pixels[:, 3:4] + 1 - pixels[:, 3:4]
        return pixels.float().cpu().unsqueeze(2)


def setup_parser() -> argparse.ArgumentParser:
    parser = setup_parser_common()
    parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled FP8 for DiT weights")
    parser.add_argument("--fp8_vl", action="store_true", help="Store sampling VLM language block linears in FP8")
    parser.add_argument("--num_layers", type=int, default=None, help="Override DiT depth; checkpoint must match")
    parser.add_argument("--dit_config", help="Optional Diffusers transformer config.json")
    parser.add_argument("--text_encoder", help="Qwen3-VL-8B directory or BF16/FP32 safetensors file, needed only for sampling")
    parser.add_argument("--vae_tiling", action="store_true", help="Enable spatial tiling for the Qwen-Image 2.1 VAE")
    parser.set_defaults(network_module="networks.lora_qwen_image_21", dit_dtype="bfloat16", vae_dtype="bfloat16", split_attn=True)
    return parser


def main():
    parser = setup_parser()
    args = read_config_from_file(parser.parse_args(), parser)
    QwenImage21NetworkTrainer().train(args)


if __name__ == "__main__":
    main()
