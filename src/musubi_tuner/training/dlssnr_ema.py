"""FP32 EMA of trainable parameters, separate from live optimizer weights."""

from contextlib import contextmanager
import math

import torch


class NRParameterEMA:
    def __init__(self, module, decay):
        if type(decay) not in (int, float) or not math.isfinite(decay) or not 0 < decay < 1:
            raise ValueError("EMA decay must be finite and strictly between 0 and 1")
        self.parameters = {name: value for name, value in module.named_parameters() if value.requires_grad}
        if not self.parameters or any(value.dtype != torch.float32 for value in self.parameters.values()):
            raise ValueError("EMA requires nonempty FP32 trainable parameters")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {name: value.detach().clone() for name, value in self.parameters.items()}
        self._active = False

    def _check_inactive(self):
        if self._active:
            raise RuntimeError("EMA weights are active; updates, restores and nested swaps are forbidden")

    @torch.no_grad()
    def update(self):
        self._check_inactive()
        for name, parameter in self.parameters.items():
            self.shadow[name].lerp_(parameter.detach(), 1 - self.decay)
        self.num_updates += 1

    def state_dict(self):
        return {
            "schema": "dlssnr_parameter_ema_v1",
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: value.detach().cpu().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state, *, expected_updates=None):
        self._check_inactive()
        if (
            not isinstance(state, dict)
            or set(state) != {"schema", "decay", "num_updates", "shadow"}
            or state["schema"] != "dlssnr_parameter_ema_v1"
            or state["decay"] != self.decay
        ):
            raise ValueError("EMA state schema or decay does not match")
        updates = state["num_updates"]
        if type(updates) is not int or updates < 0 or expected_updates is not None and updates != expected_updates:
            raise ValueError("EMA update count does not match the training state")
        shadows = state["shadow"]
        if not isinstance(shadows, dict) or shadows.keys() != self.parameters.keys():
            raise ValueError("EMA parameter names do not match the trainable parameters")
        for name, parameter in self.parameters.items():
            value = shadows[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or value.dtype != torch.float32
                or value.layout != torch.strided
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"EMA parameter shape, dtype or values are invalid: {name}")
        self.shadow = {name: shadows[name].detach().to(value.device).clone() for name, value in self.parameters.items()}
        self.num_updates = updates

    @contextmanager
    def average_parameters(self):
        self._check_inactive()
        originals = {name: value.detach().clone() for name, value in self.parameters.items()}
        self._active = True
        try:
            with torch.no_grad():
                for name, parameter in self.parameters.items():
                    parameter.copy_(self.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self.parameters.items():
                    parameter.copy_(originals[name])
            self._active = False
