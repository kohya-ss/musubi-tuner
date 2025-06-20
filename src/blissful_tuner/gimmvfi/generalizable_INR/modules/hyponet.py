# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ..configs import HypoNetConfig
from .utils import create_params_with_init, create_activation


class HypoNet(nn.Module):
    r"""
    The Hyponetwork with a coordinate-based MLP to be modulated.
    """

    def __init__(self, config: HypoNetConfig, add_coord_dim=32):
        super().__init__()
        self.config = config
        self.use_bias = config.use_bias
        self.init_config = config.initialization
        self.num_layer = config.n_layer
        self.hidden_dims = config.hidden_dim
        self.add_coord_dim = add_coord_dim

        if len(self.hidden_dims) == 1:
            self.hidden_dims = OmegaConf.to_object(self.hidden_dims) * (
                self.num_layer - 1
            )  # exclude output layer
        else:
            assert len(self.hidden_dims) == self.num_layer - 1

        if self.config.activation.type == "siren":
            assert self.init_config.weight_init_type == "siren"
            assert self.init_config.bias_init_type == "siren"

        # after computes the shape of trainable parameters, initialize them
        self.params_dict = None
        self.params_shape_dict = self.compute_params_shape()
        self.activation = create_activation(self.config.activation)
        self.build_base_params_dict(self.config.initialization)
        self.output_bias = config.output_bias

        self.normalize_weight = config.normalize_weight

        self.ignore_base_param_dict = {name: False for name in self.params_dict}

    @staticmethod
    def subsample_coords(coords, subcoord_idx=None):
        if subcoord_idx is None:
            return coords

        batch_size = coords.shape[0]
        sub_coords = []
        coords = coords.view(batch_size, -1, coords.shape[-1])
        for idx in range(batch_size):
            sub_coords.append(coords[idx : idx + 1, subcoord_idx[idx]])
        sub_coords = torch.cat(sub_coords, dim=0)
        return sub_coords

    def forward(self, coord, modulation_params_dict=None, pixel_latent=None):
        sub_idx = None
        if isinstance(coord, tuple):
            coord, sub_idx = coord[0], coord[1]

        if modulation_params_dict is not None:
            self.check_valid_param_keys(modulation_params_dict)

        batch_size, coord_shape, input_dim = (
            coord.shape[0],
            coord.shape[1:-1],
            coord.shape[-1],
        )
        coord = coord.view(batch_size, -1, input_dim)  # flatten the coordinates
        assert pixel_latent is not None
        pixel_latent = F.interpolate(
            pixel_latent.permute(0, 3, 1, 2),
            size=(coord_shape[1], coord_shape[2]),
            mode="bilinear",
        ).permute(0, 2, 3, 1)
        pixel_latent_dim = pixel_latent.shape[-1]
        pixel_latent = pixel_latent.view(batch_size, -1, pixel_latent_dim)
        hidden = coord

        hidden = torch.cat([pixel_latent, hidden], dim=-1)

        hidden = self.subsample_coords(hidden, sub_idx)

        for idx in range(self.config.n_layer):
            param_key = f"linear_wb{idx}"
            base_param = einops.repeat(
                self.params_dict[param_key], "n m -> b n m", b=batch_size
            )

            if (modulation_params_dict is not None) and (
                param_key in modulation_params_dict.keys()
            ):
                modulation_param = modulation_params_dict[param_key]
            else:
                if self.config.use_bias:
                    modulation_param = torch.ones_like(base_param[:, :-1])
                else:
                    modulation_param = torch.ones_like(base_param)

            if self.config.use_bias:
                ones = torch.ones(*hidden.shape[:-1], 1, device=hidden.device)
                hidden = torch.cat([hidden, ones], dim=-1)

                base_param_w, base_param_b = (
                    base_param[:, :-1, :],
                    base_param[:, -1:, :],
                )

                if self.ignore_base_param_dict[param_key]:
                    base_param_w = 1.0
                param_w = base_param_w * modulation_param
                if self.normalize_weight:
                    param_w = F.normalize(param_w, dim=1)
                modulated_param = torch.cat([param_w, base_param_b], dim=1)
            else:
                if self.ignore_base_param_dict[param_key]:
                    base_param = 1.0
                if self.normalize_weight:
                    modulated_param = F.normalize(base_param * modulation_param, dim=1)
                else:
                    modulated_param = base_param * modulation_param
            # print([param_key,hidden.shape,modulated_param.shape])
            hidden = torch.bmm(hidden, modulated_param)

            if idx < (self.config.n_layer - 1):
                hidden = self.activation(hidden)

        outputs = hidden + self.output_bias
        if sub_idx is None:
            outputs = outputs.view(batch_size, *coord_shape, -1)
        return outputs

    def compute_params_shape(self):
        """
        Computes the shape of MLP parameters.
        The computed shapes are used to build the initial weights by `build_base_params_dict`.
        """
        config = self.config
        use_bias = self.use_bias

        param_shape_dict = dict()

        fan_in = config.input_dim
        add_dim = self.add_coord_dim
        fan_in = fan_in + add_dim
        fan_in = fan_in + 1 if use_bias else fan_in

        for i in range(config.n_layer - 1):
            fan_out = self.hidden_dims[i]
            param_shape_dict[f"linear_wb{i}"] = (fan_in, fan_out)
            fan_in = fan_out + 1 if use_bias else fan_out

        param_shape_dict[f"linear_wb{config.n_layer-1}"] = (fan_in, config.output_dim)
        return param_shape_dict

    def build_base_params_dict(self, init_config):
        assert self.params_shape_dict
        params_dict = nn.ParameterDict()
        for idx, (name, shape) in enumerate(self.params_shape_dict.items()):
            is_first = idx == 0
            params = create_params_with_init(
                shape,
                init_type=init_config.weight_init_type,
                include_bias=self.use_bias,
                bias_init_type=init_config.bias_init_type,
                is_first=is_first,
                siren_w0=self.config.activation.siren_w0,  # valid only for siren
            )
            params = nn.Parameter(params)
            params_dict[name] = params
        self.set_params_dict(params_dict)

    def check_valid_param_keys(self, params_dict):
        predefined_params_keys = self.params_shape_dict.keys()
        for param_key in params_dict.keys():
            if param_key in predefined_params_keys:
                continue
            else:
                raise KeyError

    def set_params_dict(self, params_dict):
        self.check_valid_param_keys(params_dict)
        self.params_dict = params_dict
