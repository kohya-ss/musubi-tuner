# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

from typing import List, Optional
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class HypoNetActivationConfig:
    type: str = "relu"
    siren_w0: Optional[float] = 30.0


@dataclass
class HypoNetInitConfig:
    weight_init_type: Optional[str] = "kaiming_uniform"
    bias_init_type: Optional[str] = "zero"


@dataclass
class HypoNetConfig:
    type: str = "mlp"
    n_layer: int = 5
    hidden_dim: List[int] = MISSING
    use_bias: bool = True
    input_dim: int = 2
    output_dim: int = 3
    output_bias: float = 0.5
    activation: HypoNetActivationConfig = field(default_factory=HypoNetActivationConfig)
    initialization: HypoNetInitConfig = field(default_factory=HypoNetInitConfig)

    normalize_weight: bool = True
    linear_interpo: bool = False


@dataclass
class CoordSamplerConfig:
    data_type: str = "image"
    t_coord_only: bool = False
    coord_range: List[float] = MISSING
    time_range: List[float] = MISSING
    train_strategy: Optional[str] = MISSING
    val_strategy: Optional[str] = MISSING
    patch_size: Optional[int] = MISSING
