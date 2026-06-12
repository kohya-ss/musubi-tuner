import json
import os
from typing import Dict

import torch
from safetensors.torch import load_file

from musubi_tuner.ideogram4.transformer import Ideogram4Config, Ideogram4Transformer2DModel
from musubi_tuner.ideogram4.vae import AutoEncoder, AutoEncoderParams, convert_diffusers_state_dict


FP8_SCALE_SUFFIX = ".weight_scale"


def load_component_state_dict(base_path: str, subfolder: str, basename: str = "diffusion_pytorch_model") -> Dict[str, torch.Tensor]:
    """
    Load an Ideogram component state dict.

    Supports:
    - parent model folder:
      /workspace/models/ideogram-4-fp8

    - component folder:
      /workspace/models/ideogram-4-fp8/vae
      /workspace/models/ideogram-4-fp8/transformer

    - direct safetensors file:
      /workspace/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors

    - direct sharded index:
      /workspace/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors.index.json
    """
    base_path = os.path.normpath(base_path)

    # Direct file path support.
    if os.path.isfile(base_path):
        if base_path.endswith(".safetensors.index.json"):
            component_dir = os.path.dirname(base_path)
            index_path = base_path
            print(f"Loading sharded {subfolder} state dict from index: {index_path}")
            with open(index_path, "r") as f:
                index = json.load(f)

            shard_files = sorted(set(index["weight_map"].values()))
            state_dict = {}
            for i, shard in enumerate(shard_files):
                shard_path = os.path.join(component_dir, shard)
                print(f"  shard {i + 1}/{len(shard_files)}: {shard_path}")
                state_dict.update(load_file(shard_path))
            return state_dict

        if base_path.endswith(".safetensors"):
            print(f"Loading single {subfolder} state dict: {base_path}")
            return load_file(base_path)

        raise FileNotFoundError(f"Unsupported {subfolder} file path: {base_path}")

    # Folder support. First try the folder itself, then <folder>/<subfolder>.
    candidate_dirs = [base_path, os.path.join(base_path, subfolder)]

    looked_for = []
    for component_dir in candidate_dirs:
        index_path = os.path.join(component_dir, f"{basename}.safetensors.index.json")
        single_path = os.path.join(component_dir, f"{basename}.safetensors")
        looked_for.extend([index_path, single_path])

        if os.path.exists(index_path):
            print(f"Loading sharded {subfolder} state dict from index: {index_path}")
            with open(index_path, "r") as f:
                index = json.load(f)

            shard_files = sorted(set(index["weight_map"].values()))
            state_dict = {}
            for i, shard in enumerate(shard_files):
                shard_path = os.path.join(component_dir, shard)
                print(f"  shard {i + 1}/{len(shard_files)}: {shard_path}")
                state_dict.update(load_file(shard_path))
            return state_dict

        if os.path.exists(single_path):
            print(f"Loading single {subfolder} state dict: {single_path}")
            return load_file(single_path)

    raise FileNotFoundError(
        f"Could not find {subfolder} weights. Looked for:\n  " + "\n  ".join(looked_for)
    )


def dequantize_fp8_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.bfloat16,
    work_device: str | torch.device = "cuda",
    low_vram: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convert Ideogram FP8 weight-only tensors into normal bf16/fp16 tensors.

    Ideogram FP8 stores many linear weights as:
      layer.weight
      layer.weight_scale

    The usable weight is:
      layer.weight.float32 * layer.weight_scale[:, None]

    If low_vram=True, reconstructed tensors are moved back to CPU immediately.
    """
    work_device = torch.device(work_device)

    num_scale = sum(1 for key in state_dict if key.endswith(FP8_SCALE_SUFFIX))
    print(f"FP8 scale tensors found: {num_scale}")

    out = {}

    for key, tensor in state_dict.items():
        if key.endswith(FP8_SCALE_SUFFIX):
            continue

        scale_key = key + "_scale"

        if key.endswith(".weight") and scale_key in state_dict:
            w = tensor.to(work_device, torch.float32)
            scale = state_dict[scale_key].to(work_device, torch.float32)
            rebuilt = (w * scale.unsqueeze(1)).to(dtype)

            if low_vram:
                rebuilt = rebuilt.to("cpu")

            out[key] = rebuilt
            del w, scale, rebuilt

        elif tensor.is_floating_point():
            out[key] = tensor.to(dtype)

        else:
            out[key] = tensor

    return out


def rebuild_rotary_buffer(transformer: Ideogram4Transformer2DModel, config: Ideogram4Config) -> None:
    """
    Rebuild non-persistent rotary embedding buffer.

    This buffer is not stored in the checkpoint. If the model is created on
    the meta device, it must be recreated before moving the model to CUDA.
    """
    head_dim = config.emb_dim // config.num_heads
    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    transformer.rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)


def load_ideogram4_transformer(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    low_vram_dequant: bool = True,
) -> Ideogram4Transformer2DModel:
    """
    Load Ideogram4 transformer from official FP8 folder.

    Creates the model on meta device, dequantizes FP8 weights, loads state dict,
    rebuilds missing rotary buffer, then moves model to target device.
    """
    config = Ideogram4Config()

    print("Creating Ideogram4 transformer on meta device...")
    with torch.device("meta"):
        transformer = Ideogram4Transformer2DModel(config)

    print("Loading transformer state dict...")
    state_dict = load_component_state_dict(model_path, "transformer")

    print("Dequantizing/casting transformer state dict...")
    state_dict = dequantize_fp8_state_dict(
        state_dict,
        dtype=dtype,
        work_device=device,
        low_vram=low_vram_dequant,
    )

    print("Loading transformer weights...")
    missing, unexpected = transformer.load_state_dict(state_dict, assign=True, strict=False)

    print(f"Transformer missing keys: {len(missing)}")
    print(f"Transformer unexpected keys: {len(unexpected)}")

    if missing:
        for key in missing[:20]:
            print("  MISSING:", key)
    if unexpected:
        for key in unexpected[:20]:
            print("  UNEXPECTED:", key)

    rebuild_rotary_buffer(transformer, config)

    print(f"Moving transformer to {device} with dtype {dtype}...")
    transformer.to(device, dtype=dtype)
    transformer.eval()

    return transformer


def load_ideogram4_vae(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> AutoEncoder:
    """
    Load Ideogram4 VAE from official folder.
    """
    print("Loading VAE state dict...")
    vae_sd = load_component_state_dict(model_path, "vae")

    print("Converting VAE state dict...")
    vae_sd = convert_diffusers_state_dict(vae_sd)

    print("Creating VAE...")
    vae = AutoEncoder(AutoEncoderParams())

    print("Loading VAE weights...")
    missing, unexpected = vae.load_state_dict(vae_sd, strict=False)

    print(f"VAE missing keys: {len(missing)}")
    print(f"VAE unexpected keys: {len(unexpected)}")

    if missing:
        for key in missing[:20]:
            print("  MISSING:", key)
    if unexpected:
        for key in unexpected[:20]:
            print("  UNEXPECTED:", key)

    print(f"Moving VAE to {device} with dtype {dtype}...")
    vae.to(device, dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)

    return vae
