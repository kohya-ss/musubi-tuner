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

from omegaconf import OmegaConf, MISSING
from .modules.module_config import HypoNetConfig


@dataclass
class GIMMConfig:
    type: str = "gimm"
    ema: Optional[bool] = None
    ema_value: Optional[float] = None
    fwarp_type: str = "linear"
    hyponet: HypoNetConfig = field(default_factory=HypoNetConfig)
    coord_range: List[float] = MISSING
    modulated_layer_idxs: Optional[List[int]] = None

    @classmethod
    def create(cls, config):
        # We need to specify the type of the default DataEncoderConfig.
        # Otherwise, data_encoder will be initialized & structured as "unfold" type (which is default value)
        # hence merging with the config with other type would cause config error.
        defaults = OmegaConf.structured(cls(ema=False))
        config = OmegaConf.merge(defaults, config)
        return config


@dataclass
class GIMMVFIConfig:
    type: str = "gimmvfi"
    ema: Optional[bool] = None
    ema_value: Optional[float] = None
    fwarp_type: str = "linear"
    rec_weight: float = 0.1
    hyponet: HypoNetConfig = field(default_factory=HypoNetConfig)
    raft_iter: int = 20
    coord_range: List[float] = MISSING
    modulated_layer_idxs: Optional[List[int]] = None

    @classmethod
    def create(cls, config):
        # We need to specify the type of the default DataEncoderConfig.
        # Otherwise, data_encoder will be initialized & structured as "unfold" type (which is default value)
        # hence merging with the config with other type would cause config error.
        defaults = OmegaConf.structured(cls(ema=False))
        config = OmegaConf.merge(defaults, config)
        return config
