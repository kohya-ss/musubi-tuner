import argparse
import os
import logging
from typing import Optional

import torch
from accelerate import Accelerator

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_IDEOGRAM4,
    ARCHITECTURE_IDEOGRAM4_FULL,
)
from musubi_tuner.hv_train_network import (
    DiTOutput,
    NetworkTrainer,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.ideogram4.ideogram4_utils import (
    load_ideogram4_transformer,
    load_ideogram4_vae,
)
from musubi_tuner.ideogram4.pipeline import predict_velocity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Ideogram4NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

        # Ideogram4 is image-only for this first integration.
        self.vae_frame_stride = 1
        self.default_guidance_scale = 1.0
        self._i2v_training = False
        self._control_training = False

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_IDEOGRAM4

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_IDEOGRAM4_FULL

    def handle_model_specific_args(self, args):
        # Ideogram4 transformer and VAE are expected to run in bf16 for now.
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0

        # Training uses cached Qwen3-VL text features from the dataset cache.
        # We still validate/store --text_encoder for Musubi-native parity,
        # metadata, and future sample prompt support.
        if getattr(args, "text_encoder", None) is not None:
            if not os.path.exists(args.text_encoder):
                raise FileNotFoundError(f"Ideogram4 text encoder path does not exist: {args.text_encoder}")
            logger.info(f"Ideogram4 text encoder path registered: {args.text_encoder}")
        else:
            logger.warning("No --text_encoder provided. Training can still run from cached text features, but metadata will not record a text encoder.")

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
        logger.info(f"Loading Ideogram4 transformer from {dit_path}")

        dtype = dit_weight_dtype or torch.bfloat16

        transformer = load_ideogram4_transformer(
            dit_path,
            dtype=dtype,
            device=accelerator.device,
            low_vram_dequant=True,
        )

        return transformer

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        # For Ideogram4, --vae should usually point to the same folder as --dit:
        # /workspace/models/ideogram-4-fp8
        vae_path = args.vae
        logger.info(f"Loading Ideogram4 VAE from {vae_path}")

        vae = load_ideogram4_vae(
            vae_path,
            dtype=vae_dtype,
            device="cpu",
        )
        vae.eval()
        return vae

    def scale_shift_latents(self, latents):
        # Ideogram4 latent cache already stores:
        # VAE mean -> patchify -> Ideogram latent normalization.
        # So the base trainer should not apply extra scaling/shifting.
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
        **kwargs,
    ) -> DiTOutput:
        """
        Musubi-native Ideogram4 forward call.

        Inputs from NetworkTrainer:
          latents: clean cached Ideogram latents
          noise: noise tensor
          noisy_model_input: flow-interpolated latent
          timesteps: usually 0..1000-style training timesteps
          batch["vl_embed"]: list of variable-length Qwen3-VL feature tensors

        Ideogram predict_velocity expects:
          latents/noisy_model_input: (B, 128, H/16, W/16)
          t: [0,1], where 0 = clean and 1 = noise
          llm_features: padded (B, L, 53248)
          text_mask: (B, L)
        """

        bsize = latents.shape[0]

        # Qwen-style dataset cache provides variable-length embeddings as a list.
        vl_embed = batch["vl_embed"]  # list of (L, D)

        txt_seq_lens = [x.shape[0] for x in vl_embed]
        max_len = max(txt_seq_lens)

        padded = []
        for x in vl_embed:
            x = x.to(device=accelerator.device, dtype=network_dtype)
            if x.shape[0] < max_len:
                pad_len = max_len - x.shape[0]
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
            padded.append(x)

        llm_features = torch.stack(padded, dim=0)  # (B, L, D)

        text_mask = torch.zeros(bsize, max_len, dtype=torch.long, device=accelerator.device)
        for i, n in enumerate(txt_seq_lens):
            text_mask[i, :n] = 1

        # Musubi image cache may load latents as 5D with a frame dimension:
        #   (B, C, F, H, W)
        # Ideogram4 predict_velocity expects:
        #   (B, C, H, W)
        # For image training F should be 1, so squeeze it.
        if latents.dim() == 5:
            if latents.shape[2] != 1:
                raise ValueError(f"Ideogram4 image training expects F=1 latents, got shape {latents.shape}")
            latents = latents[:, :, 0, :, :]

        if noise.dim() == 5:
            if noise.shape[2] != 1:
                raise ValueError(f"Ideogram4 image training expects F=1 noise, got shape {noise.shape}")
            noise = noise[:, :, 0, :, :]

        if noisy_model_input.dim() == 5:
            if noisy_model_input.shape[2] != 1:
                raise ValueError(f"Ideogram4 image training expects F=1 noisy_model_input, got shape {noisy_model_input.shape}")
            noisy_model_input = noisy_model_input[:, :, 0, :, :]

        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)

        # NetworkTrainer timesteps are usually 0..1000. Ideogram helper expects [0,1].
        # Clamp to be safe in case a scheduler returns already-normalized values.
        if timesteps.max() > 1.0:
            t = timesteps.float() / 1000.0
        else:
            t = timesteps.float()
        t = t.to(device=accelerator.device)

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            llm_features.requires_grad_(True)

        with accelerator.autocast():
            model_pred = predict_velocity(
                transformer,
                noisy_model_input,
                t,
                llm_features,
                text_mask,
            )

        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noise = noise.to(device=accelerator.device, dtype=network_dtype)

        target = noise - latents

        return DiTOutput(pred=model_pred, target=target)

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        metadata = {
            "ss_base_model_version": "ideogram4",
            "ss_architecture": ARCHITECTURE_IDEOGRAM4_FULL,
        }

        if getattr(args, "text_encoder", None) is not None:
            text_encoder_name = args.text_encoder
            if os.path.exists(text_encoder_name):
                text_encoder_name = os.path.basename(os.path.normpath(text_encoder_name))
            metadata["ss_text_encoder_name"] = text_encoder_name

        return metadata


def ideogram4_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Ideogram4 specific parser setup.

    Some flags are expected by the shared NetworkTrainer validation path.
    We expose them here so Ideogram4 can use the native trainer base cleanly.
    """
    parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled fp8 for DiT if supported")
    parser.add_argument("--num_layers", type=int, default=None, help="Reserved for future partial-layer loading")
    parser.add_argument("--text_encoder", type=str, default=None, help="Qwen3-VL text encoder path, used for sampling/cache workflows")
    return parser


def setup_parser() -> argparse.ArgumentParser:
    parser = setup_parser_common()
    parser = ideogram4_setup_parser(parser)
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Ideogram4NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
