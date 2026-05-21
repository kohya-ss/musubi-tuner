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

from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.hv_train_network import (
    DiTOutput,
    setup_parser_common,
    read_config_from_file,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
        self._feature_extractor = None
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
        if not args.self_flow:
            return
        # TODO: register forward_hook on transformer.double_blocks[student_layer]
        #       and transformer.double_blocks[teacher_layer], stashing outputs
        #       into self._feature_extractor for call_dit to drain.
        raise NotImplementedError("Self-Flow forward-hook registration — see PR #913")

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
          dual-timestep conditioning (paper §A.3). Requires per-token
          timestep support inside the FLUX.2 transformer — out of scope of
          this skeleton, see open question (3) below.

        For the skeleton we delegate to the parent and ignore kwargs; once
        the FLUX.2 model side is wired up, this override forwards
        per_token_timesteps and surfaces captured features.
        """
        return super().call_dit(
            args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
        )

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
