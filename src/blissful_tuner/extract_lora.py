#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic cript for extracting the difference between two diffusion models as a LoRA for Blissful Tuner
Originally based on kohya-ss sdscripts flux_extract_lora.py which is originally based upon the repo in the comment below!
Created on Mon Jun 23 18:08:45 2025

@author: blyss
"""
# extract approximating LoRA by svd from two FLUX models
# The code is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo!

import argparse
import json
import torch
from safetensors.torch import save_file
from safetensors import safe_open
from tqdm import tqdm
from rich_argparse import RichHelpFormatter
from rich.traceback import install as install_rich_tracebacks
from blissful_tuner.utils import str_to_dtype
from blissful_tuner.blissful_core import get_current_version
from blissful_tuner.blissful_logger import BlissfulLogger

install_rich_tracebacks()
logger = BlissfulLogger(__name__, "#8e00ed")


def save_to_file(file_name, state_dict, metadata, dtype):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].to(dtype)

    save_file(state_dict, file_name, metadata=metadata)


def svd(
    model_org=None,
    model_tuned=None,
    save_to=None,
    dim=4,
    device=None,
    save_precision=None,
    clamp_quantile=0.99,
    no_metadata=False,
    mem_eff_safe_open=False,
    prefix="lora_unet",
    target_keys=["blocks"],
    exclude_keys=["bias", "norm", "modulation"],
):
    calc_dtype = torch.float
    save_dtype = str_to_dtype(save_precision) if save_precision is not None else None
    store_device = "cpu"

    # open models
    lora_weights = {}
    with safe_open(model_org, framework="pt") as f_org:
        # filter keys
        keys = []
        for key in f_org.keys():
            is_target = target_keys is None or any(pattern in key for pattern in target_keys)
            is_excluded = exclude_keys is not None and any(pattern in key for pattern in exclude_keys)
            is_target = is_target and not is_excluded
            if not is_target:
                continue
            keys.append(key)

        with safe_open(model_tuned, framework="pt") as f_tuned:
            for key in tqdm(keys, desc="Extracting LoRA"):
                key2 = key if key in f_tuned.keys() else "model.diffusion_model." + key

                if key2 not in f_tuned.keys():
                    logger.warning(f"{key} not found")
                    continue

                # get tensors and calculate difference
                value_o = f_org.get_tensor(key)
                value_t = f_tuned.get_tensor(key2)
                mat = value_t.to(calc_dtype) - value_o.to(calc_dtype)
                del value_o, value_t

                # extract LoRA weights
                if device:
                    mat = mat.to(device)
                out_dim, in_dim = mat.size()[0:2]
                rank = min(dim, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

                mat = mat.squeeze()

                U, S, Vh = torch.linalg.svd(mat)

                U = U[:, :rank]
                S = S[:rank]
                U = U @ torch.diag(S)

                Vh = Vh[:rank, :]

                dist = torch.cat([U.flatten(), Vh.flatten()])
                hi_val = torch.quantile(dist, clamp_quantile)
                low_val = -hi_val

                U = U.clamp(low_val, hi_val)
                Vh = Vh.clamp(low_val, hi_val)

                U = U.to(store_device, dtype=save_dtype).contiguous()
                Vh = Vh.to(store_device, dtype=save_dtype).contiguous()

                # print(f"key: {key}, U: {U.size()}, Vh: {Vh.size()}")
                lora_weights[key] = (U, Vh)
                del mat, U, S, Vh

    # make state dict for LoRA
    lora_sd = {}
    for key, (up_weight, down_weight) in lora_weights.items():
        lora_name = key.replace(".weight", "").replace(".", "_")
        lora_name = prefix + "_" + lora_name
        lora_sd[lora_name + ".lora_up.weight"] = up_weight
        lora_sd[lora_name + ".lora_down.weight"] = down_weight
        lora_sd[lora_name + ".alpha"] = torch.tensor(down_weight.size()[0])  # same as rank

    # minimum metadata
    net_kwargs = {}
    metadata = {
        "ss_v2": str(False),
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(dim),
        "ss_network_alpha": str(float(dim)),
        "ss_network_args": json.dumps(net_kwargs),
        "bt_tunerver": get_current_version(),
        "bt_target_keys": ", ".join(target_keys),
        "bt_exclude_keys": ", ".join(exclude_keys),
    }

    save_to_file(save_to, lora_sd, metadata, save_dtype)

    logger.info(f"LoRA weights saved to {save_to}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generic script for extracting LoRA from the difference between two diffusion models of the same arch",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Precision to save the LoRA. Default is float32 / 保存時に精度を変更して保存する、省略時はfloat",
    )
    parser.add_argument(
        "--model_org",
        type=str,
        default=None,
        required=True,
        help="Original model: safetensors file / 元モデル、safetensors",
    )
    parser.add_argument(
        "--model_tuned",
        type=str,
        default=None,
        required=True,
        help="Tuned model, LoRA is difference of `original to tuned`: safetensors file / 派生モデル（生成されるLoRAは元→派生の差分になります）、ckptまたはsafetensors",
    )

    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        required=True,
        help="destination file name: safetensors file / 保存先のファイル名、safetensors",
    )
    parser.add_argument(
        "--dim", type=int, default=4, help="dimension (rank) of LoRA (default 4) / LoRAの次元数（rank）（デフォルト4）"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う"
    )
    parser.add_argument(
        "--clamp_quantile",
        type=float,
        default=0.99,
        help="Quantile clamping value, float, (0-1). Default = 0.99 / 値をクランプするための分位点、float、(0-1)。デフォルトは0.99",
    )
    parser.add_argument(
        "--target_keys",
        nargs="*",
        type=str,
        default=["blocks"],
        help="Keys to target for LoRA, default is any key containing 'blocks'",
    )
    parser.add_argument(
        "--exclude_keys",
        nargs="*",
        type=str,
        default=["norm", "bias", "modulation"],
        help="Keys to exclude for LoRA, default is any key containing 'norm', 'bias', or 'modulation'",
    )
    parser.add_argument("--prefix", type=str, default="lora_unet", help="Prefix for LoRA modules, default is 'lora_unet'")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    svd(**vars(args))
