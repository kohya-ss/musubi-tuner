"""Matched-condition output anchoring to the initial, frozen NR base."""

from contextlib import nullcontext
from copy import deepcopy

import torch
from torch import nn

from musubi_tuner.dlssnr.fp8 import canonical_tensor_sha256, iter_canonical_tensors
from musubi_tuner.dlssnr.training_step import supervised_outputs
from musubi_tuner.training.dlssnr_services import evaluation_mode


class NRBaseAnchor(nn.Module):
    def __init__(self, model, network=None):
        super().__init__()
        if network is not None and any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("LoRA base anchoring requires a completely frozen base")
        # An injected LoRA forward closes over its adapters: never deepcopy that
        # model. Reuse its frozen base with adapters bypassed for this pass only.
        self.reference = deepcopy(model).requires_grad_(False).eval() if network is None else None
        self.reference_identity = {
            "schema": "dlssnr_frozen_reference_v1",
            "reference_kind": "initial_model" if network is None else "adapter_disabled_effective_base",
            "base_parameters_sha256": canonical_tensor_sha256(iter_canonical_tensors(model)),
            "runtime_policy": deepcopy(getattr(model, "runtime_policy", None)),
            "conditioning": "same_inputs_controls_seeds",
            "history": "independent_reference_rollout",
            "burn_in": "excluded",
        }
        self.identity = {
            **deepcopy(self.reference_identity),
            "schema": "dlssnr_base_anchor_v1",
            "loss": "masked_charbonnier_rendered_rgb",
            "mask": "supervised_loss_mask",
        }
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, model, network, batch, seeds, burn_in):
        return self.predict(model, network, batch, seeds, burn_in)["rendered_proxy"]

    def predict(self, model, network, batch, seeds, burn_in):
        reference = self.reference if self.reference is not None else model
        adapters = network.disable_adapters() if network is not None else nullcontext()
        with evaluation_mode(reference), adapters, torch.autocast(batch["source"].device.type, enabled=False):
            outputs = supervised_outputs(reference, batch, seeds, burn_in)
            for frame in outputs:
                for name in ("raw_head", "neural_preclamp", "rendered_proxy", "blend_weight"):
                    if not torch.isfinite(frame[name]).all():
                        raise RuntimeError(f"non-finite base anchor reference: {name}")
            return {
                name: torch.cat([frame[name] for frame in outputs], dim=0).detach().float()
                for name in ("neural_preclamp", "rendered_proxy")
            }
