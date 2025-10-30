import argparse
import logging
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _str_to_dtype(p):
    if p == "float":
        return torch.float
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    return None


def _maybe_strip_prefix(key: str) -> str:
    if key.startswith("model.diffusion_model."):
        return key[22:]
    return key


def _select_key(unified_key: str) -> bool:
    if not unified_key.endswith(".weight"):
        return False
    base = unified_key[:-7]
    if base.startswith("transformer_blocks."):
        return True
    if base.startswith("img_in") or base.startswith("txt_in"):
        return True
    if base.startswith("proj_out"):
        return True
    if base.startswith("norm_out.linear"):
        return True
    if base.startswith("time_text_embed.timestep_embedder.linear_"):
        return True
    return False


def _loar_name_from_key(unified_key: str) -> str:
    base = unified_key.replace(".weight", "")
    return ("lora_unet_" + base.replace(".", "_")).strip()


def save_to_file(file_name, state_dict, metadata, dtype):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].to(dtype)
    save_file(state_dict, file_name, metadata=metadata)


def svd(
    model_org: str,
    model_tuned: str,
    save_to: str,
    dim: int = 4,
    device: str | None = None,
    save_precision: str | None = None,
    clamp_quantile: float = 0.99,
    mem_eff_safe_open: bool = False,
    no_metadata: bool = False,
):
    calc_dtype = torch.float
    save_dtype = _str_to_dtype(save_precision)
    store_device = "cpu"

    if not mem_eff_safe_open:
        open_fn = lambda fn: safe_open(fn, framework="pt")
    else:
        logger.info("Using memory efficient safe_open")
        open_fn = lambda fn: MemoryEfficientSafeOpen(fn)

    with open_fn(model_org) as f_org, open_fn(model_tuned) as f_tuned:
        org_keys_map = {}
        tuned_keys_map = {}
        for k in f_org.keys():
            org_keys_map[_maybe_strip_prefix(k)] = k
        for k in f_tuned.keys():
            tuned_keys_map[_maybe_strip_prefix(k)] = k

        unified_keys = sorted(set(org_keys_map.keys()) & set(tuned_keys_map.keys()))
        selected_keys = [k for k in unified_keys if _select_key(k)]
        logger.info(f"candidate linear weight keys: {len(selected_keys)}")

        lora_weights = {}
        for unified_key in tqdm(selected_keys):
            ko = org_keys_map[unified_key]
            kt = tuned_keys_map[unified_key]
            v_o = f_org.get_tensor(ko)
            v_t = f_tuned.get_tensor(kt)
            if v_o.ndim != 2 or v_t.ndim != 2:
                continue

            mat = v_t.to(calc_dtype) - v_o.to(calc_dtype)
            if device:
                mat = mat.to(device)

            out_dim, in_dim = mat.shape
            rank = min(dim, in_dim, out_dim)
            if rank == 0:
                continue

            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
            U = U @ torch.diag(S)

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi = torch.quantile(dist, clamp_quantile)
            lo = -hi
            U = U.clamp(lo, hi)
            Vh = Vh.clamp(lo, hi)

            U = U.to(store_device, dtype=save_dtype).contiguous()
            Vh = Vh.to(store_device, dtype=save_dtype).contiguous()

            lora_weights[unified_key] = (U, Vh)
            del v_o, v_t, mat, U, S, Vh

    lora_sd = {}
    for unified_key, (up_weight, down_weight) in lora_weights.items():
        lora_name = _loar_name_from_key(unified_key)
        lora_sd[lora_name + ".lora_up.weight"] = up_weight
        lora_sd[lora_name + ".lora_down.weight"] = down_weight
        lora_sd[lora_name + ".alpha"] = torch.tensor(down_weight.size(0))

    metadata = {}
    if not no_metadata:
        title = os.path.splitext(os.path.basename(save_to))[0]
        metadata = {"title": title, "created_at": str(int(time.time()))}

    save_to_file(save_to, lora_sd, metadata, save_dtype)
    logger.info(f"LoRA weights saved to {save_to}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_org", type=str, required=True)
    parser.add_argument("--model_tuned", type=str, required=True)
    parser.add_argument("--save_to", type=str, required=True)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--clamp_quantile", type=float, default=0.99)
    parser.add_argument("--mem_eff_safe_open", action="store_true")
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"])
    parser.add_argument("--no_metadata", action="store_true")
    return parser


def main():
    args = setup_parser().parse_args()
    svd(**vars(args))


if __name__ == "__main__":
    main()
