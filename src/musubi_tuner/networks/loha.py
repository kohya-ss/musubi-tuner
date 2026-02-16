# NOTE: LoHa uses its own LoHaNetwork class (not LoRANetwork reuse).
# LoKr (lokr.py) reuses LoRANetwork via module_class/module_kwargs injection.
# This asymmetry exists because LoHa was implemented first with Conv2d support
# and a standalone network class. Convergence to LoRANetwork reuse is tracked
# as a future follow-up (Slice 8).

import ast
import math
import os
import re
from typing import Dict, List, Optional, Type, Union
from transformers import CLIPTextModel
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)
# Note: logging.basicConfig removed to avoid conflicts with BlissfulLogger - configure at entry points

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_HUNYUAN_VIDEO
from musubi_tuner.networks.network_arch import get_arch_config


class LoHaModule(torch.nn.Module):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.org_module = org_module
        self.enabled = True

        self.is_conv2d = org_module.__class__.__name__ == "Conv2d"
        self.is_linear = org_module.__class__.__name__ == "Linear"
        if not (self.is_conv2d or self.is_linear):
            raise ValueError(f"LoHa only supports Linear or Conv2d, got {org_module.__class__.__name__}")

        if self.is_conv2d:
            self.in_dim = org_module.in_channels
            self.out_dim = org_module.out_channels
            self.kernel_size = org_module.kernel_size
            self.stride = org_module.stride
            self.padding = org_module.padding
            self.dilation = org_module.dilation
            self.groups = org_module.groups
            self.weight_shape = (self.out_dim, self.in_dim, *self.kernel_size)
            self.flatten_in_dim = self.in_dim * math.prod(self.kernel_size)
        else:
            self.in_dim = org_module.in_features
            self.out_dim = org_module.out_features
            self.kernel_size = None
            self.stride = None
            self.padding = None
            self.dilation = None
            self.groups = 1
            self.weight_shape = (self.out_dim, self.in_dim)
            self.flatten_in_dim = self.in_dim

        self.hada_w1_a = nn.Parameter(torch.empty(self.out_dim, self.lora_dim))
        self.hada_w1_b = nn.Parameter(torch.empty(self.lora_dim, self.flatten_in_dim))
        self.hada_w2_a = nn.Parameter(torch.empty(self.out_dim, self.lora_dim))
        self.hada_w2_b = nn.Parameter(torch.empty(self.lora_dim, self.flatten_in_dim))

        # weight initialization follows LyCORIS reference
        torch.nn.init.normal_(self.hada_w1_b, std=1.0)
        torch.nn.init.zeros_(self.hada_w1_a)
        torch.nn.init.normal_(self.hada_w2_b, std=1.0)
        torch.nn.init.normal_(self.hada_w2_a, std=0.1)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _compute_diff_weight(self, apply_rank_dropout: bool = True):
        w1_a = self.hada_w1_a
        w1_b = self.hada_w1_b
        w2_a = self.hada_w2_a
        w2_b = self.hada_w2_b

        scale = self.scale
        if self.rank_dropout is not None and self.training and apply_rank_dropout:
            mask = (torch.rand((1, self.lora_dim), device=w1_a.device) > self.rank_dropout).float()
            w1_a = w1_a * mask
            w2_a = w2_a * mask
            scale = scale * (1.0 / (1.0 - self.rank_dropout))

        diff_weight = (w1_a @ w1_b) * (w2_a @ w2_b)
        if self.is_conv2d:
            diff_weight = diff_weight.view(self.weight_shape)
        return diff_weight, scale

    def default_forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        x_for_delta = x
        if self.dropout is not None and self.training:
            x_for_delta = F.dropout(x_for_delta, p=self.dropout)

        diff_weight, scale = self._compute_diff_weight(apply_rank_dropout=True)
        if self.is_conv2d:
            delta = F.conv2d(
                x_for_delta,
                diff_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            delta = F.linear(x_for_delta, diff_weight)

        return org_forwarded + delta * self.multiplier * scale

    def forward(self, x):
        return self.default_forward(x)

    # return weight for merge
    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        diff_weight, _ = self._compute_diff_weight(apply_rank_dropout=False)
        diff_weight = diff_weight * self.scale * multiplier
        return diff_weight

    def merge_to(self, sd, dtype, device, non_blocking=False):
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        merge_device = org_device if device is None else device
        merge_dtype = torch.float if dtype is None else dtype

        w1_a = sd["hada_w1_a"].to(device=merge_device, dtype=torch.float, non_blocking=non_blocking)
        w1_b = sd["hada_w1_b"].to(device=merge_device, dtype=torch.float, non_blocking=non_blocking)
        w2_a = sd["hada_w2_a"].to(device=merge_device, dtype=torch.float, non_blocking=non_blocking)
        w2_b = sd["hada_w2_b"].to(device=merge_device, dtype=torch.float, non_blocking=non_blocking)
        alpha = sd.get("alpha", torch.tensor(self.lora_dim)).to(device=merge_device, dtype=torch.float)

        scale = alpha / w1_b.shape[0]
        diff_weight = (w1_a @ w1_b) * (w2_a @ w2_b) * scale * self.multiplier
        if self.is_conv2d:
            diff_weight = diff_weight.view(weight.shape)

        weight = weight.to(device=merge_device, dtype=torch.float, non_blocking=non_blocking) + diff_weight
        org_sd["weight"] = weight.to(device=org_device, dtype=org_dtype)
        self.org_module.load_state_dict(org_sd)


class LoHaInfModule(LoHaModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, dropout=None, rank_dropout=None, module_dropout=None)
        self.org_module_ref = [org_module]
        self.enabled = True
        self.network: "LoHaNetwork" = None

    def set_network(self, network):
        self.network = network

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)

    def precompute_weight(self, device=None, dtype=None, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        diff_weight = self.get_weight(multiplier=multiplier).to(device=device, dtype=dtype or torch.float)
        return diff_weight

    def merge_to(self, sd, dtype, device, non_blocking=False):
        # use base implementation but keep org_module reference intact
        super().merge_to(sd, dtype, device, non_blocking)


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    architecture = kwargs.get("architecture", ARCHITECTURE_HUNYUAN_VIDEO)
    config = get_arch_config(architecture)

    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    exclude_patterns.extend(config["exclude_patterns"])

    # Qwen exclude_mod support (parity with lora_qwen_image.py)
    if "exclude_mod_patterns" in config:
        exclude_mod = kwargs.get("exclude_mod", True)
        if isinstance(exclude_mod, str):
            exclude_mod = ast.literal_eval(exclude_mod)
        if exclude_mod:
            exclude_patterns.extend(config["exclude_mod_patterns"])

    kwargs["exclude_patterns"] = exclude_patterns

    # Kandinsky include_patterns support
    if "include_patterns" in config and "include_patterns" not in kwargs:
        kwargs["include_patterns"] = config["include_patterns"]

    return create_network(
        config["target_modules"],
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_network(
    target_replace_modules: List[str],
    prefix: str,
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    """architecture independent network creation"""
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # TODO generic rank/dim setting with regular expression

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # verbose
    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    # regular expression for module selection: exclude and include
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None and isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None and isinstance(include_patterns, str):
        include_patterns = ast.literal_eval(include_patterns)

    # too many arguments ( ^ω^)･･･
    network = LoHaNetwork(
        target_replace_modules,
        prefix,
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        verbose=verbose,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    if loraplus_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)

    return network


class LoHaNetwork(torch.nn.Module):
    # only supports U-Net (DiT), Text Encoders are not supported

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: Union[List[CLIPTextModel], CLIPTextModel],
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Type[object] = LoHaModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.target_replace_modules = target_replace_modules
        self.prefix = prefix

        self.loraplus_lr_ratio = None

        if modules_dim is not None:
            logger.info("create LoHa network from weights")
        else:
            logger.info(f"create LoHa network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )

        # compile regular expression if specified
        exclude_re_patterns = []
        if exclude_patterns is not None:
            for pattern in exclude_patterns:
                try:
                    re_pattern = re.compile(pattern)
                except re.error as e:
                    logger.error(f"Invalid exclude pattern '{pattern}': {e}")
                    continue
                exclude_re_patterns.append(re_pattern)

        include_re_patterns = []
        if include_patterns is not None:
            for pattern in include_patterns:
                try:
                    re_pattern = re.compile(pattern)
                except re.error as e:
                    logger.error(f"Invalid include pattern '{pattern}': {e}")
                    continue
                include_re_patterns.append(re_pattern)

        # create module instances
        def create_modules(
            is_unet: bool,
            pfx: str,
            root_module: torch.nn.Module,
            target_replace_mods: Optional[List[str]] = None,
            filter: Optional[str] = None,
            default_dim: Optional[int] = None,
        ) -> List[LoHaModule]:
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if target_replace_mods is None or module.__class__.__name__ in target_replace_mods:
                    if target_replace_mods is None:  # dirty hack for all modules
                        module = root_module  # search all modules

                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            original_name = (name + "." if name else "") + child_name
                            lora_name = f"{pfx}.{original_name}".replace(".", "_")

                            # exclude/include filter
                            excluded = False
                            for pattern in exclude_re_patterns:
                                if pattern.fullmatch(original_name):
                                    excluded = True
                                    break
                            included = False
                            for pattern in include_re_patterns:
                                if pattern.fullmatch(original_name):
                                    included = True
                                    break
                            if excluded and not included:
                                if verbose:
                                    logger.info(f"exclude: {original_name}")
                                continue

                            # filter by name (not used in the current implementation)
                            if filter is not None and filter not in lora_name:
                                continue

                            dim = None
                            alpha = None

                            if modules_dim is not None:
                                # モジュール指定あり
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            else:
                                # 通常、すべて対象とする
                                if is_linear or is_conv2d_1x1:
                                    dim = default_dim if default_dim is not None else self.lora_dim
                                    alpha = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha

                            if dim is None or dim == 0:
                                # skipした情報を出力
                                if is_linear or is_conv2d_1x1 or (self.conv_lora_dim is not None):
                                    skipped.append(lora_name)
                                continue

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                            )
                            loras.append(lora)

                if target_replace_mods is None:
                    break  # all modules are searched
            return loras, skipped

        # create LoHa for Text Encoders (not used currently)
        self.text_encoder_loras: List[Union[LoHaModule]] = []

        # create LoHa for U-Net
        self.unet_loras: List[Union[LoHaModule]]
        self.unet_loras, skipped_un = create_modules(True, prefix, unet, target_replace_modules)

        logger.info(f"create LoHa for U-Net/DiT: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:50} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(
                f"because dim (rank) is 0, {len(skipped)} LoHa modules are skipped / dim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
            )
            for name in skipped:
                logger.info(f"\t{name}")

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def prepare_network(self, args):
        """
        called after the network is created
        """
        pass

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    # マージできるかどうかを返す
    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None, non_blocking=False):
        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            if len(sd_for_lora) == 0:
                logger.info(f"no weight for {lora.lora_name}")
                continue

            lora.merge_to(sd_for_lora, dtype, device, non_blocking)

        logger.info("weights are merged")

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu", weights_only=True)

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(
        self,
        text_encoders: Optional[nn.Module],
        unet: Optional[nn.Module],
        apply_text_encoder: bool = True,
        apply_unet: bool = True,
    ):
        if apply_text_encoder:
            logger.info(f"enable LoHa for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable LoHa for U-Net: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        if len(self.text_encoder_loras) == 0 and len(self.unet_loras) == 0:
            logger.error(
                "No LoHa modules. Please check `--network_module` and `--network_args`"
                " / LoHaモジュールがありません。`--network_module`と`--network_args`を確認してください"
            )
            raise RuntimeError("No LoHa modules found")

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio):  # , loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio

        logger.info(f"LoHa+ UNet LR Ratio: {self.loraplus_lr_ratio}")
        # logger.info(f"LoHa+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def prepare_optimizer_params(self, unet_lr: float = 1e-4, **kwargs):
        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            for lora in loras:
                for name, param in lora.named_parameters():
                    is_plus_weight = "lora_up" in name or "hada_w2" in name
                    if loraplus_ratio is not None and is_plus_weight:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}

                if len(param_data["params"]) == 0:
                    continue

                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    else:
                        param_data["lr"] = lr

                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    logger.info("NO LR skipping!")
                    continue

                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")

            return params, descriptions

        if self.unet_loras:
            params, descriptions = assemble_params(self.unet_loras, unet_lr, self.loraplus_lr_ratio)
            all_params.extend(params)
            lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, unet):
        self.train()

    def on_step_start(self):
        pass

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from musubi_tuner.utils import model_utils

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def apply_max_norm_regularization(self, max_norm_value, device):
        loras = self.text_encoder_loras + self.unet_loras
        if len(loras) == 0:
            return 0, 0.0, 0.0

        keys_scaled = 0
        norms = []

        for lora in loras:
            diff_weight = lora.get_weight(multiplier=1.0).to(device)
            norm = diff_weight.norm()
            norm = torch.clamp(norm, min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = (desired / norm).item()

            if ratio != 1.0:
                keys_scaled += 1
                scale_factor = ratio**0.25  # distribute scaling across four factorized matrices
                lora.hada_w1_a.data *= scale_factor
                lora.hada_w1_b.data *= scale_factor
                lora.hada_w2_a.data *= scale_factor
                lora.hada_w2_b.data *= scale_factor

            scaled_norm = diff_weight.norm() * ratio
            norms.append(scaled_norm.item())

        mean_norm = sum(norms) / len(norms)
        maximum_norm = max(norms)
        return keys_scaled, mean_norm, maximum_norm


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoHaNetwork:
    architecture = kwargs.get("architecture", ARCHITECTURE_HUNYUAN_VIDEO)
    config = get_arch_config(architecture)

    return create_network_from_weights(
        config["target_modules"], multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )


# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoHaNetwork:
    # get dim/alpha mapping
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "hada_w1_b" in key:
            dim = value.shape[0]
            modules_dim[lora_name] = dim
            # logger.info(lora_name, value.size(), dim)

    module_class = LoHaInfModule if for_inference else LoHaModule

    network = LoHaNetwork(
        target_replace_modules,
        "lora_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
    )
    return network


def merge_weights_to_tensor(
    model_weight: torch.Tensor,
    lora_name: str,
    lora_sd: Dict[str, torch.Tensor],
    lora_weight_keys: set,
    multiplier: float,
    calc_device: torch.device,
) -> torch.Tensor:
    """Merge LoHa weights directly into a model weight tensor.

    Supports Linear and Conv2d. Consumed keys are removed from lora_weight_keys.
    Returns model_weight unchanged if no matching LoHa keys found.
    """
    w1a_key = lora_name + ".hada_w1_a"
    w1b_key = lora_name + ".hada_w1_b"
    w2a_key = lora_name + ".hada_w2_a"
    w2b_key = lora_name + ".hada_w2_b"
    alpha_key = lora_name + ".alpha"

    if w1a_key not in lora_weight_keys:
        return model_weight

    w1a = lora_sd[w1a_key].to(calc_device)
    w1b = lora_sd[w1b_key].to(calc_device)
    w2a = lora_sd[w2a_key].to(calc_device)
    w2b = lora_sd[w2b_key].to(calc_device)

    dim = w1b.shape[0]
    alpha = lora_sd.get(alpha_key, torch.tensor(dim))
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    scale = alpha / dim

    org_device = model_weight.device
    original_dtype = model_weight.dtype
    compute_dtype = torch.float16 if original_dtype.itemsize == 1 else torch.float32
    model_weight = model_weight.to(calc_device, dtype=compute_dtype)
    w1a, w1b = w1a.to(compute_dtype), w1b.to(compute_dtype)
    w2a, w2b = w2a.to(compute_dtype), w2b.to(compute_dtype)

    # ΔW = ((w1a @ w1b) * (w2a @ w2b)) * scale
    diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale

    # Reshape for Conv2d if needed (diff is always 2D from matmul)
    if model_weight.dim() == 4 and diff_weight.dim() == 2:
        diff_weight = diff_weight.view(model_weight.shape)

    model_weight = model_weight + multiplier * diff_weight

    model_weight = model_weight.to(device=org_device, dtype=original_dtype)

    # Remove consumed keys
    for key in [w1a_key, w1b_key, w2a_key, w2b_key, alpha_key]:
        lora_weight_keys.discard(key)

    return model_weight
