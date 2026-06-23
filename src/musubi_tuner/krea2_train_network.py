"""LoRA training for Krea 2 (K2). Phase 4: minimal trainable loop.

Implements the architecture-specific hooks of the shared NetworkTrainer for K2:
build/load the single-stream MMDiT, reuse the Qwen-Image VAE, and run flow-matching
in K2's token space (replicating sampling.prepare batched, with varlen text padding).

Wired: bf16 + gradient checkpointing, sample generation during training (text-to-image,
optional CFG), dynamic scaled fp8 for the DiT (--fp8_base --fp8_scaled), block swap
(--blocks_to_swap, CPU offloading of the main blocks), and torch.compile of the main
blocks (--compile).
"""

import argparse
import gc
import itertools
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from einops import rearrange, repeat

from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2, ARCHITECTURE_KREA2_FULL
from musubi_tuner.hv_train_network import (
    DiTOutput,
    NetworkTrainer,
    clean_memory_on_device,
    load_prompts,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.krea2 import krea2_utils
from musubi_tuner.krea2 import krea2_sampling
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.utils import model_utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Krea2NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_frame_stride = 1  # single image
        # M1 (resident) weight stashes for RAW-train / Turbo-sample. Both None when unused or
        # in M2 (per-validation streaming) mode; built lazily on the first sample step.
        # _turbo_stash holds the (fp8-quantized) Turbo weights; _raw_stash a one-time snapshot
        # of the RAW base weights to restore after sampling (the fp8 base is frozen for LoRA,
        # so a single snapshot stays valid for the whole run).
        self._turbo_stash = None
        self._raw_stash = None

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_KREA2

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_KREA2_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0  # K2 t2i, not used at train time
        # self.blocks_to_swap is set by the base trainer (handle_model_specific_args runs first).
        # K2 fp8 supports only the scaled (dynamic) path; plain --fp8_base alone would cast the
        # whole DiT (incl. norms) to fp8, which breaks. Require --fp8_scaled with --fp8_base.
        if args.fp8_base and not args.fp8_scaled:
            raise ValueError("Krea 2 fp8 supports only scaled fp8: pass --fp8_scaled together with --fp8_base.")
        # RAW-train / Turbo-sample: the recommended K2 LoRA workflow is to train on the RAW
        # checkpoint and run inference on the distilled Turbo. --turbo_dit makes sample
        # generation during training swap the base weights to Turbo (LoRA, hooked on the live
        # Linears, applies on top automatically) and use the Turbo sampling schedule.
        if args.turbo_dit_cache and not args.turbo_dit:
            raise ValueError("--turbo_dit_cache (M1, resident Turbo weights) requires --turbo_dit.")
        # Turbo sample generation swaps the base weights from outside the model, which is unsafe
        # under block swap: the offloader (esp. --block_swap_h2d_only's LoRAStreamOffloader) keeps
        # its own CPU master and streams it to the GPU, so an external weight swap does NOT reach
        # the weights the forward actually uses -> RAW/Turbo mix (bf16: loss drift; fp8: pure noise
        # from RAW weight x Turbo scale_weight). Restrict Turbo sampling to the block-swap-disabled
        # case (it is an optional, VRAM-permitting convenience); with block swap, sample on RAW.
        if args.turbo_dit and args.blocks_to_swap:
            raise ValueError(
                "--turbo_dit (Turbo sample generation) is not supported together with --blocks_to_swap: "
                "the block-swap offloader manages the base weights and an external swap would mix RAW/Turbo. "
                "Use Turbo sampling without block swap (VRAM permitting), or omit --turbo_dit to sample on RAW."
            )
        if args.turbo_dit and not args.sample_prompts:
            logger.warning("--turbo_dit is set but --sample_prompts is not; Turbo is only used for sample generation.")

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        """Encode the sample prompts with Qwen3-VL up front, cache the embeds, free the encoder.

        Kept deliberately simple (text-to-image only): for each prompt and its optional
        negative prompt we store the varlen selected-layer hidden stack (valid tokens only),
        matching the training cache format, then drop the 4B encoder before training resumes.
        """
        device = accelerator.device

        assert args.text_encoder is not None, "--text_encoder is required for sample generation during training"
        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        encoder = krea2_utils.load_krea2_text_encoder(args.text_encoder, dtype=torch.bfloat16, device=device)

        logger.info("Encoding sample prompts with Qwen3-VL")
        te_outputs = {}  # prompt str -> (valid_len, num_layers, hidden) on cpu
        with torch.no_grad():
            for prompt_dict in prompts:
                for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", None)]:
                    if p is None or p in te_outputs:
                        continue
                    hiddens, mask = krea2_utils.get_krea2_prompt_embeds(encoder, [p])  # (1, seq, L, D), (1, seq)
                    embed = hiddens[0][mask[0]]  # gather valid tokens -> (valid_len, L, D), drops padding
                    te_outputs[p] = embed.to("cpu")

        del encoder
        gc.collect()
        clean_memory_on_device(device)

        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            prompt_dict_copy["krea2_vl_embed"] = te_outputs[prompt_dict.get("prompt", "")]
            negative_prompt = prompt_dict.get("negative_prompt", None)
            if negative_prompt is not None:
                prompt_dict_copy["negative_krea2_vl_embed"] = te_outputs[negative_prompt]
            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(device)
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
        """Architecture-dependent inference: K2 flow-matching Euler sampler with optional CFG.

        Replicates krea2_sampling.sample's core loop but with embeds taken from the cached
        sample parameters (no encoder) and the trainer's DiT + Qwen-Image VAE. CFG (standard
        uncond + scale*(cond-uncond)) is enabled when a negative prompt is present and cfg_scale > 1
        (musubi convention).
        Resolution-aware mu time-shift uses the K2 raw defaults (y1=0.5, y2=1.15); the distilled
        (turbo) fixed-mu schedule is not wired here.
        """
        model = transformer  # SingleStreamDiT
        device = accelerator.device
        patch = model.config.patch
        compression = qwen_image_utils.VAE_SCALE_FACTOR  # Qwen-Image VAE: 8x

        # Standard CFG (uncond + scale*(cond-uncond)), enabled when cfg_scale > 1 — matches the
        # rest of musubi. The official Krea 2 "guidance" value maps as cfg_scale = guidance + 1
        # (official default guidance 4.5 -> cfg_scale 5.5).
        cfg = cfg_scale if cfg_scale is not None else 5.5
        do_cfg = do_classifier_free_guidance and cfg > 1.0

        # The latent grid is patchified in `patch`-sized blocks, so spatial dims must be
        # multiples of compression * patch. The base flow already rounds to 8; align to 16.
        align = compression * patch
        width = krea2_sampling.roundup(width, align, "width")
        height = krea2_sampling.roundup(height, align, "height")
        lat_h, lat_w = height // compression, width // compression

        def build_branch(embed):
            embed = embed.to(device=device, dtype=torch.bfloat16).unsqueeze(0)  # (1, seq, L, D)
            txtmask = torch.ones(1, embed.shape[1], device=device, dtype=torch.bool)
            return embed, txtmask

        txt, txtmask = build_branch(sample_parameter["krea2_vl_embed"])
        if do_cfg:
            untxt, untxtmask = build_branch(sample_parameter["negative_krea2_vl_embed"])

        # Seeded gaussian latent noise (generator already seeded by the base sampler).
        noise = torch.randn(1, model.config.channels, lat_h, lat_w, device=device, dtype=torch.bfloat16, generator=generator)

        img, pos, mask = krea2_sampling.prepare(noise, txt.shape[1], patch, txtmask)
        if do_cfg:
            _, unpos, unmask = krea2_sampling.prepare(noise, untxt.shape[1], patch, untxtmask)

        # mu interpolation endpoints (krea2 sample defaults minres=256, maxres=1280).
        x1 = (256 // align) ** 2
        x2 = (1280 // align) ** 2
        # The distilled Turbo checkpoint was trained at a fixed mu=1.15; the RAW checkpoint
        # uses resolution-aware mu interpolation. When sampling on Turbo (--turbo_dit), pin mu.
        turbo_mu = 1.15 if args.turbo_dit else None
        ts = krea2_sampling.timesteps(img.shape[1], sample_steps, x1, x2, y1=0.5, y2=1.15, mu=turbo_mu)

        for tcurr, tprev in tqdm(zip(ts[:-1], ts[1:]), total=len(ts) - 1, desc="Denoising steps"):
            t = torch.full((1,), tcurr, dtype=img.dtype, device=device)
            cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
            if do_cfg:
                uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
                v = uncond + cfg * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v

        # Unpatchify token sequence back to a latent (1, C, 1, H, W) for the VAE.
        latent = rearrange(
            img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=lat_h // patch, w=lat_w // patch
        )
        latent = latent.unsqueeze(2)  # (1, C, 1, H, W)

        vae.to(device)
        vae.eval()
        with torch.no_grad():
            pixels = vae.decode_to_pixels(latent.to(vae.dtype))  # (1, C, H, W) in [0, 1]
        vae.to("cpu")
        clean_memory_on_device(device)

        pixels = pixels.unsqueeze(2).to(torch.float32).cpu()  # (1, C, 1, H, W) for the grid saver
        return pixels

    # region RAW-train / Turbo-sample (base-weight swap)

    @staticmethod
    def _named_live_tensors(model):
        """Map checkpoint-style key -> live parameter/buffer tensor of the (unwrapped) DiT.

        Includes the fp8 ``scale_weight`` buffers; excludes LoRA (its modules live on the
        network, not as children of the transformer, so they are never touched here).

        Keys are normalized by stripping the ``._orig_mod`` segment that ``torch.compile``
        inserts when --compile wraps each block (``blocks.0._orig_mod.attn...`` ->
        ``blocks.0.attn...``), so they line up with the on-disk checkpoint names. We iterate
        this LIVE mapping and index the source state dict (which is a superset of these keys),
        never the reverse — that way any extra keys the source might carry are simply ignored,
        and a missing source key surfaces immediately as a clear KeyError.
        """
        live = {}
        for k, t in itertools.chain(model.named_parameters(), model.named_buffers()):
            live[k.replace("._orig_mod.", ".")] = t
        return live

    def _snapshot_weights(self, model) -> dict:
        """Take an independent CPU snapshot of the model's current base weights (key by key).

        Used once to capture RAW before the first Turbo swap-in. Each entry is a fresh CPU
        tensor (``copy=True``), so later in-place writes to the live weights never touch it.
        """
        return {k: t.detach().to("cpu", copy=True) for k, t in self._named_live_tensors(model).items()}

    def _overwrite_weights(self, model, src: dict):
        """Copy ``src`` into the model's base weights in place (used by BOTH M1 and M2).

        Critically this writes through ``copy_`` and never reassigns ``weight.data``: the live
        tensor keeps its exact storage object and device, so it composes with the block-swap
        offloader (which itself recycles the ``weight.data`` storages) and with fp8 (where the
        streamed ``weight`` and the resident ``scale_weight`` must stay consistent). Reassigning
        ``.data`` instead — the old M1 ping-pong — corrupts that recycling and produced pure
        noise specifically in the M1 x fp8 x block-swap case. ``copy_`` casts dtype and crosses
        devices, so block-swapped weights that currently live on CPU stay on CPU.
        """
        for k, t in self._named_live_tensors(model).items():
            t.data.copy_(src[k])

    def _free_base_weights(self, model):
        """Release the live base weight/buffer storages (set ``.data`` to a 0-size tensor).

        Used by the M2 GPU-direct swap to free the current weights *before* loading the
        replacement to the same device, so the GPU never holds both copies at once (~1x peak
        instead of ~2x). Each tensor keeps its dtype/device. This reassigns ``.data`` and is
        therefore only safe without the block-swap offloader — which ``--turbo_dit`` forbids
        via ``--blocks_to_swap`` (see handle_model_specific_args), so there is no offloader to
        collide with here (cf. the copy_-only requirement of _overwrite_weights under swap).
        """
        for t in self._named_live_tensors(model).values():
            t.data = t.data.new_empty((0,))

    def _assign_weights(self, model, src: dict):
        """Attach ``src`` tensors as the model's live base weights by reassigning ``.data``.

        Counterpart to _free_base_weights: the old storage is gone, so we reassign rather than
        copy_. The Parameter/buffer objects are unchanged, so the LoRA hooks (which read
        ``module.weight`` at call time) pick up the new weights automatically. Same offloader
        caveat as _free_base_weights — safe because --turbo_dit excludes block swap.
        """
        for k, t in self._named_live_tensors(model).items():
            t.data = src[k]

    def on_before_sample_images(self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype):
        # Swap RAW -> Turbo base weights for sample generation (LoRA stays hooked and applies on top).
        if not args.turbo_dit:
            return
        model = accelerator.unwrap_model(transformer)
        if args.turbo_dit_cache:
            # M1: build the resident (fp8-quantized at startup) Turbo stash once and snapshot the
            # RAW base once, then copy Turbo into the live weights (storage preserved, see
            # _overwrite_weights). The snapshot must be taken before the first overwrite.
            if self._turbo_stash is None:
                logger.info(f"Krea 2: caching Turbo weights for sampling (M1) from {args.turbo_dit}")
                self._turbo_stash = krea2_utils.load_krea2_dit_state_dict(
                    args.turbo_dit, fp8_scaled=args.fp8_scaled, calc_device=accelerator.device, result_device="cpu"
                )
            if self._raw_stash is None:
                logger.info("Krea 2: snapshotting RAW weights for restore (M1)")
                self._raw_stash = self._snapshot_weights(model)
            logger.info("Krea 2: swapping in Turbo weights for sampling (M1)")
            self._overwrite_weights(model, self._turbo_stash)
        else:
            # M2: free the RAW base weights, then load the Turbo weights (re-quantized if fp8)
            # straight onto the GPU and reassign them. Freeing first keeps the GPU at ~1x the
            # model size; loading directly to the GPU keeps the CPU peak at ~1 tensor (no full
            # intermediate CPU dict). Safe to reassign here because --turbo_dit forbids block swap.
            logger.info(f"Krea 2: loading Turbo weights for sampling (M2, GPU-direct) from {args.turbo_dit}")
            self._free_base_weights(model)
            clean_memory_on_device(accelerator.device)
            turbo_sd = krea2_utils.load_krea2_dit_state_dict(
                args.turbo_dit, fp8_scaled=args.fp8_scaled, calc_device=accelerator.device, result_device=accelerator.device
            )
            self._assign_weights(model, turbo_sd)
            del turbo_sd
            gc.collect()
            clean_memory_on_device(accelerator.device)

    def on_after_sample_images(self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype):
        # Restore RAW base weights after sampling.
        if not args.turbo_dit:
            return
        model = accelerator.unwrap_model(transformer)
        if args.turbo_dit_cache:
            logger.info("Krea 2: restoring RAW weights after sampling (M1)")
            self._overwrite_weights(model, self._raw_stash)  # copy RAW back in place (storage preserved)
        else:
            # M2: free the Turbo base weights, then reload RAW (re-quantized if fp8) straight
            # onto the GPU and reassign — same GPU-direct, CPU-peak-~1-tensor path as swap-in.
            # RAW is frozen during training, so reloading it from disk reproduces it exactly.
            logger.info(f"Krea 2: restoring RAW weights after sampling (M2, GPU-direct) from {args.dit}")
            self._free_base_weights(model)
            clean_memory_on_device(accelerator.device)
            raw_sd = krea2_utils.load_krea2_dit_state_dict(
                args.dit, fp8_scaled=args.fp8_scaled, calc_device=accelerator.device, result_device=accelerator.device
            )
            self._assign_weights(model, raw_sd)
            del raw_sd
            gc.collect()
            clean_memory_on_device(accelerator.device)
        logger.info("Krea 2: RAW weights restored")

    # endregion RAW-train / Turbo-sample

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        logger.info(f"Loading VAE model from {args.vae}")
        vae = qwen_image_utils.load_vae(args.vae, input_channels=3, device="cpu", disable_mmap=True)
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
        # For fp8_scaled, dit_weight_dtype is None (the base trainer skips the post-load cast);
        # the fp8 path ignores dtype and keeps non-target weights in their checkpoint dtype.
        dtype = dit_weight_dtype if dit_weight_dtype is not None else torch.bfloat16
        model = krea2_utils.load_krea2_dit(
            dit_path,
            device=loading_device,
            dtype=dtype,
            fp8_scaled=args.fp8_scaled,
            loading_device=loading_device,
            attn_mode=attn_mode,
            split_attn=split_attn,
        )
        return model

    def compile_transformer(self, args, transformer):
        model = transformer  # SingleStreamDiT
        # Compile the per-block SingleStreamBlocks (the heavy, repeated compute). The forward
        # already pads the combined sequence to a multiple of 256 to keep kernel shapes stable.
        # When block swap is on, exclude the swap blocks' Linears from compile (cf. zimage/qwen_image).
        return model_utils.compile_transformer(
            args, model, [model.blocks], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
        # K2 latents are already normalized by the Qwen-Image VAE caching ((raw-mean)/std).
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
        model = transformer  # SingleStreamDiT
        device = accelerator.device
        patch = model.config.patch

        latents = batch["latents"]  # (B, C, 1, H, W)
        bsize = latents.shape[0]
        assert latents.shape[2] == 1, f"K2 expects single-frame latents (B,C,1,H,W), got {latents.shape}"

        # --- image tokens / pos / mask (replicates krea2 sampling.prepare) ---
        nmi = noisy_model_input.squeeze(2)  # (B, C, H, W)
        _, _, lat_h, lat_w = nmi.shape
        h_, w_ = lat_h // patch, lat_w // patch

        img_tokens = rearrange(nmi, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

        imgids = torch.zeros((h_, w_, 3), device=device)
        imgids[..., 1] = torch.arange(h_, device=device)[:, None]
        imgids[..., 2] = torch.arange(w_, device=device)[None, :]
        imgpos = repeat(imgids, "h w three -> b (h w) three", b=bsize, three=3)
        imgmask = torch.ones(bsize, h_ * w_, device=device, dtype=torch.bool)

        # --- text tokens / pos / mask (varlen -> padded batch) ---
        vl_embed = batch["krea2_vl_embed"]  # list of (valid_len, num_layers, hidden)
        txt_seq_lens = [x.shape[0] for x in vl_embed]
        max_len = max(txt_seq_lens)
        # pad along the sequence axis (dim 0): F.pad pads last dim first, so (0,0)x2 then (0, pad)
        vl_embed = [F.pad(x, (0, 0, 0, 0, 0, max_len - x.shape[0])) for x in vl_embed]
        context = torch.stack(vl_embed, dim=0).to(device=device, dtype=network_dtype)  # (B, max_len, L, D)

        txtmask = torch.zeros(bsize, max_len, device=device, dtype=torch.bool)
        for i, n in enumerate(txt_seq_lens):
            txtmask[i, :n] = True
        txtpos = torch.zeros(bsize, max_len, 3, device=device)

        # --- combine (image-first: valid tokens form a contiguous prefix per sample) ---
        mask = torch.cat((imgmask, txtmask), dim=1)
        pos = torch.cat((imgpos, txtpos), dim=1)

        img_tokens = img_tokens.to(device=device, dtype=network_dtype)
        t = (timesteps / 1000.0).to(device=device)

        if args.gradient_checkpointing:
            img_tokens.requires_grad_(True)
            context.requires_grad_(True)

        with accelerator.autocast():
            model_pred = model(img=img_tokens, context=context, t=t, pos=pos, mask=mask)  # (B, h*w, c*ph*pw)

        # unpatchify to latent space (B, C, 1, H, W)
        model_pred = rearrange(
            model_pred, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=h_, w=w_
        )
        model_pred = model_pred.unsqueeze(2)  # (B, C, 1, H, W)

        # flow matching target (velocity): noise - data
        latents = latents.to(device=device, dtype=network_dtype)
        target = noise - latents
        return DiTOutput(pred=model_pred, target=target)

    # endregion model specific


def krea2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="use dynamic scaled fp8 for the DiT (requires --fp8_base). Quantizes per-block Linears at load time.",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="Qwen3-VL-4B text encoder safetensors path (only needed for sample generation during training)",
    )
    parser.add_argument(
        "--turbo_dit",
        type=str,
        default=None,
        help="Distilled Turbo DiT safetensors path. Recommended K2 LoRA workflow: train on RAW (--dit), "
        "generate samples on Turbo. When set, sample generation swaps the base weights to Turbo (the LoRA "
        "applies on top automatically) and uses the Turbo schedule (fixed mu=1.15; set CFG off (--l 1) and a "
        "low step count (--s 8) in the sample prompt). Restored to RAW after each sample step. "
        "Fully optional: omit to sample on RAW.",
    )
    parser.add_argument(
        "--turbo_dit_cache",
        action="store_true",
        help="M1 memory mode for --turbo_dit: keep the (fp8-quantized at startup) Turbo weights resident in "
        "CPU RAM and ping-pong-swap them in (~1x extra CPU, faster). Default (M2) streams Turbo from disk each "
        "sample step (re-quantizing if fp8) for ~0x steady CPU at the cost of per-validation load time.",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Krea2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
