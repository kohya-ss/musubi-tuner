# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

import math
import torch
import torch.nn as nn

from .layers import Sine, Damping


def convert_int_to_list(size, len_list=2):
    if isinstance(size, int):
        return [size] * len_list
    else:
        assert len(size) == len_list
        return size


def initialize_params(params, init_type, **kwargs):
    fan_in, fan_out = params.shape[0], params.shape[1]
    if init_type is None or init_type == "normal":
        nn.init.normal_(params)
    elif init_type == "kaiming_uniform":
        nn.init.kaiming_uniform_(params, a=math.sqrt(5))
    elif init_type == "uniform_fan_in":
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(params, -bound, bound)
    elif init_type == "zero":
        nn.init.zeros_(params)
    elif "siren" == init_type:
        assert "siren_w0" in kwargs.keys() and "is_first" in kwargs.keys()
        w0 = kwargs["siren_w0"]
        if kwargs["is_first"]:
            w_std = 1 / fan_in
        else:
            w_std = math.sqrt(6.0 / fan_in) / w0
        nn.init.uniform_(params, -w_std, w_std)
    else:
        raise NotImplementedError


def create_params_with_init(
    shape, init_type="normal", include_bias=False, bias_init_type="zero", **kwargs
):
    if not include_bias:
        params = torch.empty([shape[0], shape[1]])
        initialize_params(params, init_type, **kwargs)
        return params
    else:
        params = torch.empty([shape[0] - 1, shape[1]])
        bias = torch.empty([1, shape[1]])

        initialize_params(params, init_type, **kwargs)
        initialize_params(bias, bias_init_type, **kwargs)
        return torch.cat([params, bias], dim=0)


def create_activation(config):
    if config.type == "relu":
        activation = nn.ReLU()
    elif config.type == "siren":
        activation = Sine(config.siren_w0)
    elif config.type == "silu":
        activation = nn.SiLU()
    elif config.type == "damp":
        activation = Damping(config.siren_w0)
    else:
        raise NotImplementedError
    return activation
