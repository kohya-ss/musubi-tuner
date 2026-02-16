from __future__ import annotations

import warnings
from typing import Any, Dict, List

import torch

MUON_AVAILABLE = False
try:
    from muon import SingleDeviceMuonWithAuxAdam as OfficialSingleDeviceMuonWithAuxAdam

    MUON_AVAILABLE = True
except Exception:
    OfficialSingleDeviceMuonWithAuxAdam = None


def zeropower_via_newtonschulz5(G: torch.Tensor, *, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration used by Muon to (approximately) orthogonalize 2D updates.

    Coefficients follow the official Muon reference implementation.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, *, beta: float = 0.95, ns_steps: int = 5, nesterov: bool = True) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def _adam_update(grad: torch.Tensor, buf1: torch.Tensor, buf2: torch.Tensor, step: int, betas: tuple[float, float], eps: float) -> torch.Tensor:
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class FallbackSingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    A minimal SingleDeviceMuonWithAuxAdam fallback when the official `muon` package isn't installed.

    The param-group schema matches the official implementation so that higher-level code
    can behave identically.
    """

    def __init__(self, param_groups: List[Dict[str, Any]]):
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each Muon param group must include `use_muon`.")

            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)

        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon"):
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = _adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


def create_muon_optimizer(param_groups: List[Dict[str, Any]], *, prefer_official: bool = True) -> torch.optim.Optimizer:
    """
    Create a Muon optimizer instance (SingleDeviceMuonWithAuxAdam style).

    If the official `muon` package is available, it will be used by default.
    Otherwise, a built-in fallback is returned (with a warning).
    """
    if prefer_official and MUON_AVAILABLE:
        return OfficialSingleDeviceMuonWithAuxAdam(param_groups)

    if prefer_official and not MUON_AVAILABLE:
        warnings.warn(
            "Official Muon package not found. Using built-in fallback. "
            "Install with: pip install git+https://github.com/KellerJordan/Muon"
        )

    return FallbackSingleDeviceMuonWithAuxAdam(param_groups)
