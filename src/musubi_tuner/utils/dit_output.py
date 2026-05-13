from __future__ import annotations
from dataclasses import dataclass, field
import torch


@dataclass
class DitOutput:
    """Return value of NetworkTrainer.call_dit().

    pred:     model prediction tensor (B, N, C)
    target:   regression target tensor (B, N, C)
    features: intermediate hidden features for representation loss, or None
    """
    pred: torch.Tensor
    target: torch.Tensor
    features: torch.Tensor | None = field(default=None)
