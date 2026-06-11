"""Self-Flow training entry point for FLUX.2.

Skeleton for Self-Supervised Flow Matching (Self-Flow) on the FLUX.2 backbone.
Each base-class extension seam is overridden here as a stub showing how the
algorithm hooks in. Actual algorithmic logic is intentionally left as
``NotImplementedError`` / ``# TODO`` markers — to be filled in by porting
rockerBOO's PR #913 implementation onto these seams.

Reference: https://github.com/kohya-ss/musubi-tuner/pull/913

This file is the proposed extension surface, not a runnable trainer. Do not
use it to start a training run yet.

Internal extension point — no API stability guarantees. Subclasses live in
this repo; if you fork, expect breakage on updates.
"""

import argparse
import logging
from typing import Optional

import torch
from accelerate import Accelerator

from musubi_tuner.flux_2.flux2_models import timestep_embedding
from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.hv_train_network import (
    DiTOutput,
    setup_parser_common,
    read_config_from_file,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# region self-flow math helpers


def assign_teacher_student_timesteps(timesteps_a: torch.Tensor, timesteps_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample teacher/student split (paper §3.3): teacher = min (cleaner), student = max."""
    t_a = timesteps_a.float()
    t_b = timesteps_b.float()
    timesteps_teacher = torch.min(t_a, t_b).to(timesteps_a.dtype)
    timesteps_student = torch.max(t_a, t_b).to(timesteps_a.dtype)
    return timesteps_teacher, timesteps_student


def reconstruct_noisy_input(latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """Flow-matching interpolation (1-t)*latents + t*noise; timesteps in [1, 1001]."""
    t = (timesteps.float() - 1.0) / 1000.0
    if latents.ndim == 5:
        t_exp = t.view(-1, 1, 1, 1, 1)
    else:
        t_exp = t.view(-1, 1, 1, 1)
    return (1 - t_exp) * latents + t_exp * noise


def apply_per_token_mask(
    noisy_input_student: torch.Tensor,
    noisy_input_teacher: torch.Tensor,
    mask_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token masking (paper Eq. 4-5): masked tokens take the teacher (cleaner) values.

    Returns (masked_input, mask_flat) with mask_flat (B, N) bool, True = masked.
    N indexes latent pixels, which map 1:1 to packed FLUX.2 image tokens (prc_img
    does no patchify). Independent per-token draw; locality modes are a follow-up.
    """
    B = noisy_input_student.shape[0]
    if noisy_input_student.ndim == 5:
        _, _, T, H, W = noisy_input_student.shape
        mask_flat = torch.rand(B, T * H * W, device=device) < mask_ratio
        mask_spatial = mask_flat.view(B, 1, T, H, W).expand_as(noisy_input_student)
    else:
        _, _, H, W = noisy_input_student.shape
        mask_flat = torch.rand(B, H * W, device=device) < mask_ratio
        mask_spatial = mask_flat.view(B, 1, H, W).expand_as(noisy_input_student)
    masked_input = torch.where(mask_spatial, noisy_input_teacher, noisy_input_student)
    return masked_input, mask_flat


def build_per_token_timestep_map(
    timesteps_teacher: torch.Tensor,
    timesteps_student: torch.Tensor,
    mask_flat: torch.Tensor,
    mismatch_prob: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token timestep map for dual-timestep conditioning (paper §3.3).

    Unmasked tokens get the student timestep. Masked tokens get the teacher
    timestep, except with probability ``mismatch_prob`` they get the student
    timestep (deliberate mismatch — experimental, 0.0 = paper behaviour).
    Returns (per_token_timesteps (B, N), mismatch_mask (B, N) bool).
    """
    B, N = mask_flat.shape
    t_student = timesteps_student.unsqueeze(1).expand(B, N)
    t_teacher = timesteps_teacher.unsqueeze(1).expand(B, N)

    if mismatch_prob <= 0.0:
        per_token_t = torch.where(mask_flat, t_teacher, t_student)
        return per_token_t, torch.zeros_like(mask_flat)

    coin = torch.rand(B, N, device=mask_flat.device)
    mismatch_mask = mask_flat & (coin < mismatch_prob)
    per_token_t = torch.where(mask_flat & ~mismatch_mask, t_teacher, t_student)
    return per_token_t, mismatch_mask


def update_ema_weights(
    ema_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    decay: float,
) -> None:
    """In-place EMA update: ema = decay * ema + (1 - decay) * current."""
    with torch.no_grad():
        for k, v in current_state.items():
            if v.is_floating_point():
                ema_state[k].lerp_(v, 1 - decay)
            else:
                ema_state[k].copy_(v)


def compute_ema_weight_drift(
    ema_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Mean L2 distance between EMA and current weights (floating-point only)."""
    with torch.no_grad():
        dists = [
            torch.linalg.vector_norm(current_state[k].float() - v.float()) for k, v in ema_state.items() if v.is_floating_point()
        ]
        if not dists:
            return torch.tensor(0.0)
        return torch.stack(dists).mean()


def compute_representation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    rep_proj: torch.nn.Module,
) -> torch.Tensor:
    """L_rep (paper Eq. 6): negative mean cosine similarity of projected student vs teacher."""
    student_proj = rep_proj(student_features)
    cos_sim = torch.nn.functional.cosine_similarity(student_proj, teacher_features, dim=-1)
    return -cos_sim.mean()


def effective_gamma(gamma: float, global_step: int, warmup_steps: int) -> float:
    """Linear warmup of the L_rep weight: 0 -> gamma over warmup_steps, then constant."""
    if warmup_steps <= 0:
        return gamma
    return gamma * min(1.0, global_step / warmup_steps)


# endregion self-flow math helpers


class PerTokenModulationController:
    """Reroutes Flux2's modulation path to per-token timesteps via forward hooks.

    When staged with a per-token map tau (B, N_img) in model scale [0, 1], the
    hooks turn the modulation vec from (B, D) into (B, N_img, D); Modulation and
    LastLayer broadcast 3D vecs natively, so no model code changes. When not
    staged, every hook is a pass-through and the model is bit-identical to
    vanilla. Install on the raw (unwrapped) model before accelerator.prepare.
    """

    REQUIRED_MODULES = ("time_in", "double_stream_modulation_txt", "single_stream_modulation")

    def __init__(self) -> None:
        self._handles: list = []
        self._tau: Optional[torch.Tensor] = None
        self._num_txt_tokens: Optional[int] = None

    def install(self, model) -> None:
        for name in self.REQUIRED_MODULES:
            if not hasattr(model, name):
                raise AttributeError(
                    f"Flux2 model has no module '{name}' — upstream may have renamed it; "
                    "the Self-Flow per-token hooks need updating."
                )
        self._handles.append(model.time_in.register_forward_hook(self._time_in_hook))
        if model.use_guidance_embed:
            self._handles.append(model.guidance_in.register_forward_hook(self._guidance_in_hook))
        self._handles.append(model.double_stream_modulation_txt.register_forward_pre_hook(self._txt_modulation_pre_hook))
        self._handles.append(model.single_stream_modulation.register_forward_pre_hook(self._single_modulation_pre_hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def stage(self, per_token_timesteps: torch.Tensor, num_txt_tokens: int) -> None:
        """Arm the hooks for the next forward. tau in model scale [0, 1], shape (B, N_img)."""
        self._tau = per_token_timesteps
        self._num_txt_tokens = num_txt_tokens

    def clear(self) -> None:
        self._tau = None
        self._num_txt_tokens = None

    def _time_in_hook(self, module, inputs, output):
        if self._tau is None:
            return None
        B, N = self._tau.shape
        emb = timestep_embedding(self._tau.reshape(-1), 256)
        # module.forward (not module.__call__): bypasses hooks, so no recursion
        return module.forward(emb).reshape(B, N, -1)

    def _guidance_in_hook(self, module, inputs, output):
        if self._tau is None:
            return None
        return output.unsqueeze(1)  # (B, D) -> (B, 1, D): broadcasts in `vec + guidance`

    def _txt_modulation_pre_hook(self, module, args):
        if self._tau is None:
            return None
        (vec,) = args  # (B, N_img, D) after the time_in hook (+ guidance)
        return (vec.mean(dim=1),)  # text tokens get the global (mean) vec

    def _single_modulation_pre_hook(self, module, args):
        if self._tau is None:
            return None
        (vec,) = args  # (B, N_img, D)
        vec_global = vec.mean(dim=1)
        vec_txt = vec_global.unsqueeze(1).expand(-1, self._num_txt_tokens, -1)
        return (torch.cat([vec_txt, vec], dim=1),)


class BlockFeatureExtractor:
    """Captures hidden states from Flux2 blocks via forward hooks.

    Layer indices are global, matching the old branch and the CLI help:
    0..len(double_blocks)-1 address double blocks (img stream is captured),
    len(double_blocks).. address single blocks (text tokens are sliced off).
    Blocks self-checkpoint internally, so hook outputs are differentiable
    even with gradient checkpointing enabled.
    """

    def __init__(self) -> None:
        self._handles: list = []
        self._armed_layer: Optional[int] = None
        self._num_txt_tokens: Optional[int] = None
        self._features: Optional[torch.Tensor] = None

    def install(self, model, layer_indices: list[int]) -> None:
        num_double = len(model.double_blocks)
        num_total = num_double + len(model.single_blocks)
        for layer in sorted(set(layer_indices)):
            if not 0 <= layer < num_total:
                raise ValueError(f"feature layer {layer} out of range (model has {num_total} blocks)")
            if layer < num_double:
                handle = model.double_blocks[layer].register_forward_hook(self._make_double_hook(layer))
            else:
                handle = model.single_blocks[layer - num_double].register_forward_hook(self._make_single_hook(layer))
            self._handles.append(handle)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def arm(self, layer: int, num_txt_tokens: int) -> None:
        self._armed_layer = layer
        self._num_txt_tokens = num_txt_tokens
        self._features = None

    def drain(self) -> Optional[torch.Tensor]:
        features = self._features
        self._features = None
        self._armed_layer = None
        self._num_txt_tokens = None
        return features

    def _make_double_hook(self, layer: int):
        def hook(module, inputs, output):
            if self._armed_layer == layer:
                img, _txt = output
                self._features = img

        return hook

    def _make_single_hook(self, layer: int):
        def hook(module, inputs, output):
            if self._armed_layer == layer:
                self._features = output[:, self._num_txt_tokens :, ...]

        return hook


class Flux2SelfFlowNetworkTrainer(Flux2NetworkTrainer):
    """FLUX.2 + Self-Flow.

    Owned state (set during the relevant lifecycle seams, used across steps):
    - ``self.rep_proj``: representation projection head (paper §3.3, Eq. 6).
      Constructed in ``extra_trainable_params``, prepared in ``on_train_start``.
    - ``self.ema_lora_state``: EMA snapshot of LoRA weights (the "teacher").
      Initialised in ``on_train_start`` after ``accelerator.prepare`` so it
      sits on-device. Updated by ``on_post_optimizer_step``.
    - ``self._coupling_scheduler``: dummy LR scheduler whose "lr" is the
      teacher coupling probability (decays over training).
    - ``self._feature_extractor``: holds DiT layer outputs captured via
      ``register_forward_hook`` (no DiT model edit required). Hooks are
      registered in ``on_transformer_loaded``.
    - ``self._self_flow_logs``: per-step state metrics dict drained by
      ``extra_step_logs`` (e.g. coupling-scheduler value). Loss decomposition
      itself (``loss/gen``, ``loss/rep``) flows through ``process_batch``'s
      ``loss_metrics`` return, not through here.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rep_proj: Optional[torch.nn.Module] = None
        self.ema_lora_state: Optional[dict] = None
        self._coupling_scheduler = None
        self._feature_extractor: Optional[BlockFeatureExtractor] = None
        self._modulation_controller: Optional[PerTokenModulationController] = None
        self._self_flow_logs: dict = {}
        self._saved_student_state: Optional[dict] = None

    # region argument validation (existing extension point — handle_model_specific_args)

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        super().handle_model_specific_args(args)
        # Self-Flow CLI sanity. Paper requires student feature layer l <
        # teacher layer k, and mask ratio R_M <= 0.5.
        if args.self_flow:
            if (
                args.student_feature_layer is not None
                and args.teacher_feature_layer is not None
                and args.student_feature_layer >= args.teacher_feature_layer
            ):
                raise ValueError(
                    f"--student_feature_layer ({args.student_feature_layer}) must be less than "
                    f"--teacher_feature_layer ({args.teacher_feature_layer})."
                )
            if args.mask_ratio > 0.5:
                raise ValueError(f"--mask_ratio ({args.mask_ratio}) must be <= 0.5 (paper constraint R_M <= 0.5)")

    # endregion

    # region extension seam overrides

    def extra_trainable_params(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        trainable_params: list,
    ) -> list:
        """Build the projection head and merge it into the optimizer's first group.

        Self-Flow's L_rep is computed against ``rep_proj(student_features)``;
        the projection head's parameters share the network LR schedule because
        Prodigy and friends don't support per-group LRs.
        """
        if not args.self_flow:
            return trainable_params

        # TODO: build Sequential(Linear, GELU, Linear) sized from transformer.hidden_size.
        #       See PR #913 hv_train_network.py around line 2395.
        raise NotImplementedError("Self-Flow rep_proj construction — see PR #913")

    def on_transformer_loaded(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
    ) -> None:
        """Register feature-extraction forward hooks on the raw transformer.

        Done here (rather than in ``on_train_start``) so the hooks attach
        before ``accelerator.prepare`` / block-swap rewrap, which keeps the
        captured tensors aligned with the unwrapped block indices the user
        supplied via ``--student_feature_layer`` / ``--teacher_feature_layer``.
        """
        super().on_transformer_loaded(args, accelerator, transformer)
        if not args.self_flow:
            return

        num_blocks = len(transformer.double_blocks) + len(transformer.single_blocks)
        if args.student_feature_layer is None:
            args.student_feature_layer = max(0, int(num_blocks * 0.3))
        if args.teacher_feature_layer is None:
            args.teacher_feature_layer = min(num_blocks - 1, int(num_blocks * 0.7))
        if args.student_feature_layer >= args.teacher_feature_layer:
            raise ValueError(
                f"--student_feature_layer ({args.student_feature_layer}) must be less than "
                f"--teacher_feature_layer ({args.teacher_feature_layer})."
            )

        self._modulation_controller = PerTokenModulationController()
        self._modulation_controller.install(transformer)
        self._feature_extractor = BlockFeatureExtractor()
        self._feature_extractor.install(transformer, [args.student_feature_layer, args.teacher_feature_layer])
        logger.info(
            f"Self-Flow hooks installed: student_layer={args.student_feature_layer}, "
            f"teacher_layer={args.teacher_feature_layer} (of {num_blocks} blocks)"
        )

    def on_train_start(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        optimizer,
    ) -> None:
        """Finish Self-Flow setup once everything is on-device.

        Steps to perform here:
        1. Snapshot ``network.state_dict()`` into ``self.ema_lora_state``.
        2. If ``--network_weights_ema`` is given, load it into the network,
           re-snapshot, then restore student weights from ``--network_weights``.
        3. ``self.rep_proj = accelerator.prepare(self.rep_proj)``.
        4. Optionally load ``--network_weights_proj`` into ``self.rep_proj``.
        5. Build ``self._coupling_scheduler`` (constant / cosine / linear / rex).

        Forward-hook registration for feature extraction lives in
        ``on_transformer_loaded`` (runs earlier, before accelerator.prepare).
        """
        if not args.self_flow:
            return
        # TODO: see PR #913 hv_train_network.py around lines 2490-2522.
        raise NotImplementedError("Self-Flow on_train_start — see PR #913")

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
        """Extends Flux2NetworkTrainer.call_dit.

        Recognised kwargs:
        - ``hidden_features`` (bool): when True, return captured features in
          ``DiTOutput.extra["features"]`` (drained from ``self._feature_extractor``).
        - ``feature_layer`` (int): which registered layer's output to return.
        - ``per_token_timesteps`` (Tensor): per-token timestep map for
          dual-timestep conditioning (paper §A.3). Staged into
          ``self._modulation_controller`` before the forward and cleared after.
        """
        hidden_features = kwargs.pop("hidden_features", False)
        feature_layer = kwargs.pop("feature_layer", None)
        per_token_timesteps = kwargs.pop("per_token_timesteps", None)

        if not hidden_features and per_token_timesteps is None:
            return super().call_dit(
                args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
            )

        if any(k.startswith("latents_control_") for k in batch):
            raise NotImplementedError("Self-Flow does not support control images yet")

        num_txt_tokens = batch["ctx_vec"].shape[1]
        if per_token_timesteps is not None:
            # model scale: base call_dit divides 1D timesteps by 1000; mirror that here
            tau = per_token_timesteps.to(device=accelerator.device) / 1000.0
            self._modulation_controller.stage(tau, num_txt_tokens)
        if hidden_features:
            self._feature_extractor.arm(feature_layer, num_txt_tokens)
        try:
            output = super().call_dit(
                args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
            )
        finally:
            self._modulation_controller.clear()
        if hidden_features:
            output.extra["features"] = self._feature_extractor.drain()
        return output

    def process_batch(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        network,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor,
        noise: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        vae,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Self-Flow step replacing vanilla flow matching.

        Outline (mirrors PR #913's ``_self_flow_step``):
          1. Sample two timesteps; per-sample ``teacher = min(t_a, t_b)``,
             ``student = max(t_a, t_b)``.
          2. Reconstruct two noisy inputs (teacher / student) via flow
             matching ``(1-t)*latents + t*noise``.
          3. Apply per-token mask of cleaner (teacher) noise into student
             input; build per-token timestep map with optional mismatch.
          4. Teacher forward (no_grad, EMA weights swapped in).
          5. Student forward (gradients flow); collect features.
          6. ``L_gen`` (MSE w/ flow weighting) + ``gamma * L_rep`` (negative
             cosine similarity through ``self.rep_proj``).
          7. Return ``(L_gen + gamma * L_rep, {"loss/gen": ..., "loss/rep": ...,
             "self_flow/gamma": ...})`` so the loss decomposition lands in the
             per-step logs.

        Vanilla path (when ``--self_flow`` is off) falls through to the base
        implementation so this trainer can be used for non-Self-Flow runs too.
        """
        if not args.self_flow:
            return super().process_batch(
                args,
                accelerator,
                transformer,
                network,
                batch,
                latents,
                noise,
                noise_scheduler,
                dit_dtype,
                network_dtype,
                vae,
                global_step,
            )
        # TODO: port PR #913's _self_flow_step body here, calling
        #       ``self.call_dit(..., hidden_features=True, feature_layer=...)``
        #       and ``self.call_dit(..., per_token_timesteps=...)`` for
        #       teacher and student forwards respectively.
        raise NotImplementedError("Self-Flow process_batch — see PR #913 _self_flow_step")

    def on_post_optimizer_step(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        sync_gradients: bool,
        global_step: int,
    ) -> None:
        """EMA update of LoRA weights only on real optimizer steps.

        ema_lora_state[k].lerp_(network.state_dict()[k], 1 - args.ema_decay)
        """
        if not args.self_flow or not sync_gradients:
            return
        if self.ema_lora_state is None:
            return
        # TODO: in-place lerp_ across floating-point params; copy_ otherwise.
        raise NotImplementedError("Self-Flow EMA update — see PR #913 _update_ema_weights")

    def on_before_sample_images(
        self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype
    ) -> None:
        """Swap to EMA (teacher) weights before sampling when Self-Flow is active."""
        if not args.self_flow or self.ema_lora_state is None:
            return
        network = accelerator.unwrap_model(network)
        self._saved_student_state = {k: v.clone() for k, v in network.state_dict().items()}
        network.load_state_dict(self.ema_lora_state)

    def on_after_sample_images(
        self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype
    ) -> None:
        """Restore student weights after EMA sampling."""
        if not args.self_flow or self.ema_lora_state is None:
            return
        if self._saved_student_state is None:
            return
        network = accelerator.unwrap_model(network)
        network.load_state_dict(self._saved_student_state)
        self._saved_student_state = None

    def on_post_save(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        ckpt_name: str,
        save_dtype,
        metadata: dict,
        force_sync_upload: bool,
    ) -> None:
        """Save EMA (teacher) and projection-head companion files.

        - ``<ckpt_name stem>-ema.safetensors``  (LoRA-format teacher weights)
        - ``<ckpt_name stem>-proj.safetensors`` (raw rep_proj state_dict)
        Each follows the main checkpoint's HuggingFace upload toggle.
        """
        if not args.self_flow:
            return
        # TODO: see PR #913 hv_train_network.py around lines 2750-2774.
        raise NotImplementedError("Self-Flow companion file saving — see PR #913")

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        """Return ``ss_self_flow_*`` keys for embedding into safetensors metadata."""
        if not args.self_flow:
            return {}
        return {
            "ss_self_flow": True,
            "ss_self_flow_gamma": args.self_flow_gamma,
            "ss_self_flow_gamma_warmup_steps": args.self_flow_gamma_warmup_steps,
            "ss_self_flow_mask_ratio": args.mask_ratio,
            "ss_self_flow_ema_decay": args.ema_decay,
            "ss_self_flow_student_layer": args.student_feature_layer,
            "ss_self_flow_teacher_layer": args.teacher_feature_layer,
            "ss_self_flow_teacher_coupling_prob": args.self_flow_teacher_coupling_prob,
            "ss_self_flow_teacher_coupling_decay": args.self_flow_teacher_coupling_decay,
            "ss_self_flow_teacher_mismatch_ratio": args.self_flow_teacher_mismatch_ratio,
        }

    def extra_step_logs(self, args: argparse.Namespace, logs: dict) -> dict:
        """Drain per-step Self-Flow metrics into the trainer's log payload."""
        if not args.self_flow:
            return {}
        # ``self._self_flow_logs`` is populated inside process_batch.
        return dict(self._self_flow_logs)

    # endregion


def self_flow_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Self-Flow-specific CLI arguments. Mirrors PR #913's additions."""
    parser.add_argument(
        "--self_flow",
        action="store_true",
        help="Enable Self-Flow training (dual-timestep scheduling + representation alignment).",
    )
    parser.add_argument(
        "--self_flow_gamma",
        type=float,
        default=0.8,
        help="Weight for representation alignment loss L_rep (paper Eq. 7). 0 disables L_rep.",
    )
    parser.add_argument(
        "--self_flow_gamma_warmup_steps",
        type=int,
        default=0,
        help="Linearly ramp gamma from 0 to --self_flow_gamma over this many steps. 0 disables warmup.",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.25,
        help="Token mask ratio for dual-timestep scheduling (paper Eq. 4, R_M <= 0.5).",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay for the Self-Flow teacher LoRA weights.",
    )
    parser.add_argument(
        "--student_feature_layer",
        type=int,
        default=None,
        help="Global block index for student feature extraction (l in paper, must be < teacher_feature_layer). Recommended ~0.3 * num_blocks.",
    )
    parser.add_argument(
        "--teacher_feature_layer",
        type=int,
        default=None,
        help="Global block index for teacher feature extraction (k in paper, must be > student_feature_layer). Recommended ~0.7 * num_blocks.",
    )
    parser.add_argument(
        "--self_flow_teacher_coupling_prob",
        type=float,
        default=0.0,
        help="Per-step gate probability for applying timestep mismatch on masked patches. 0 disables mismatch.",
    )
    parser.add_argument(
        "--self_flow_teacher_coupling_decay",
        type=str,
        default="constant",
        choices=["constant", "cosine", "linear", "rex"],
        help="Decay schedule for --self_flow_teacher_coupling_prob.",
    )
    parser.add_argument(
        "--self_flow_teacher_coupling_decay_steps",
        type=int,
        default=None,
        help="Steps over which to decay coupling prob to 0. Defaults to max_train_steps.",
    )
    parser.add_argument(
        "--self_flow_teacher_mismatch_ratio",
        type=float,
        default=1.0,
        help="When the coupling gate fires, fraction of masked patches receiving the timestep mismatch.",
    )
    parser.add_argument(
        "--network_weights_ema",
        type=str,
        default=None,
        help="Pretrained EMA (teacher) weights for resumption. Requires --network_weights.",
    )
    parser.add_argument(
        "--network_weights_proj",
        type=str,
        default=None,
        help="Pretrained projection head weights for resumption.",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)
    parser = self_flow_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    trainer = Flux2SelfFlowNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
