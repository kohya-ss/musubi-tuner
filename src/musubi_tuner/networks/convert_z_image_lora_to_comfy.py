import argparse
from safetensors.torch import save_file
from safetensors import safe_open

import logging

import torch

from musubi_tuner.utils.model_utils import precalculate_safetensors_hashes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    # load source safetensors
    logger.info(f"Loading source file {args.src_path}")
    state_dict = {}
    with safe_open(args.src_path, framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    logger.info("Converting...")

    # Key mapping tables: (sd-scripts format, ComfyUI format)
    blocks_mappings = [
        ("attention_to_out_0", "attention_out"),
        ("attention_norm_k", "attention_k_norm"),
        ("attention_norm_q", "attention_q_norm"),
    ]

    keys = list(state_dict.keys())
    count = 0

    for key in keys:
        new_k = key

        if "layers" in key:
            mappings = blocks_mappings
        else:
            continue

        # Apply mappings based on conversion direction
        for src_key, dst_key in mappings:
            if args.reverse:
                # ComfyUI to sd-scripts: swap src and dst
                new_k = new_k.replace(dst_key, src_key)
            else:
                # sd-scripts to ComfyUI: use as-is
                new_k = new_k.replace(src_key, dst_key)

        if new_k != key:
            state_dict[new_k] = state_dict.pop(key)
            count += 1
            # print(f"Renamed {k} to {new_k}")

    # concat or split LoRA/LoHa/LoKr for QKV layers
    qkv_count = 0
    if args.reverse:
        # ComfyUI to sd-scripts: split QKV (LoRA only)
        keys = list(state_dict.keys())
        for key in keys:
            if "attention_qkv" in key and "lora_down" in key:
                # get LoRA base name. e.g., "lora_unet_blocks_0_attn1_to_qkv.lora_down.weight" -> "lora_unet_blocks_0_attn1_to_qkv"
                lora_name = key.split(".", 1)[0]
                down_weight = state_dict.pop(f"{lora_name}.lora_down.weight")
                up_weight = state_dict.pop(f"{lora_name}.lora_up.weight")
                alpha = state_dict.pop(f"{lora_name}.alpha")
                split_dims = [down_weight.size(0) // 3] * 3  # assume equal split for Q, K, V

                lora_name_prefix = lora_name.replace("qkv", "")

                # dense weight (rank*3, in_dim)
                split_weights = torch.chunk(down_weight, len(split_dims), dim=0)
                for i, split_w in enumerate(split_weights):
                    suffix = ["to_q", "to_k", "to_v"][i]
                    state_dict[f"{lora_name_prefix}{suffix}.lora_down.weight"] = split_w
                    state_dict[f"{lora_name_prefix}{suffix}.alpha"] = alpha / 3  # adjust alpha because rank is 3x larger

                # sparse weight (out_dim=sum(split_dims), rank*3)
                split_dims = [up_weight.size(0) // 3] * 3  # assume equal split for Q, K, V
                rank = up_weight.size(1) // len(split_dims)
                weight_index = 0
                for i in range(len(split_dims)):
                    suffix = ["to_q", "to_k", "to_v"][i]
                    split_up_weight = up_weight[weight_index : weight_index + split_dims[i], i * rank : (i + 1) * rank]
                    split_up_weight = split_up_weight.contiguous()  # this solves an error in saving safetensors
                    state_dict[f"{lora_name_prefix}{suffix}.lora_up.weight"] = split_up_weight
                    state_dict[f"{lora_name_prefix}{suffix}.alpha"] = alpha / 3  # adjust alpha because rank is 3x larger
                    weight_index += split_dims[i]

                qkv_count += 1
    else:
        # sd-scripts to ComfyUI: concat QKV

        # LoRA QKV merge
        keys = list(state_dict.keys())
        for key in keys:
            if key not in state_dict:
                continue
            if "attention" in key and ("to_q" in key or "to_k" in key or "to_v" in key):
                if "to_q" not in key or "lora_up" not in key:  # ensure we process only once per QKV set
                    continue

                lora_name = key.split(".", 1)[0]  # get LoRA base name
                split_dims = [state_dict[key].size(0)] * 3  # assume equal split for Q, K, V

                lora_name_prefix = lora_name.replace("to_q", "")
                down_weights = []  # (rank, in_dim) * 3
                up_weights = []  # (split dim, rank) * 3
                for weight_index in range(len(split_dims)):
                    if weight_index == 0:
                        suffix = "to_q"
                    elif weight_index == 1:
                        suffix = "to_k"
                    else:
                        suffix = "to_v"
                    down_weights.append(state_dict.pop(f"{lora_name_prefix}{suffix}.lora_down.weight"))
                    up_weights.append(state_dict.pop(f"{lora_name_prefix}{suffix}.lora_up.weight"))

                alpha = state_dict.pop(f"{lora_name}.alpha")
                state_dict.pop(f"{lora_name_prefix}to_k.alpha")
                state_dict.pop(f"{lora_name_prefix}to_v.alpha")

                # merge down weight
                down_weight = torch.cat(down_weights, dim=0)  # (rank, split_dim) * 3 -> (rank*3, sum of split_dim)

                # merge up weight (sum of split_dim, rank*3), dense to sparse
                rank = up_weights[0].size(1)
                up_weight = torch.zeros((sum(split_dims), down_weight.size(0)), device=down_weight.device, dtype=down_weight.dtype)
                weight_index = 0
                for i in range(len(split_dims)):
                    up_weight[weight_index : weight_index + split_dims[i], i * rank : (i + 1) * rank] = up_weights[i]
                    weight_index += split_dims[i]

                new_lora_name = lora_name_prefix + "qkv"
                state_dict[f"{new_lora_name}.lora_down.weight"] = down_weight
                state_dict[f"{new_lora_name}.lora_up.weight"] = up_weight

                # adjust alpha because rank is 3x larger. See https://github.com/kohya-ss/sd-scripts/issues/2204
                state_dict[f"{new_lora_name}.alpha"] = alpha * 3

                qkv_count += 1

        # LoHa QKV merge (block-diagonal, lossless)
        # ΔW = (w1a @ w1b) ⊙ (w2a @ w2b) * scale
        # Using block_diag for w1a/w2a and cat for w1b/w2b preserves the exact result:
        #   block_diag(w1a_q, w1a_k, w1a_v) @ cat(w1b_q, w1b_k, w1b_v)
        #   = [w1a_q@w1b_q; w1a_k@w1b_k; w1a_v@w1b_v]
        # Rank becomes 3x, alpha is adjusted accordingly.
        keys = list(state_dict.keys())
        for key in keys:
            if key not in state_dict:
                continue
            if "attention" in key and ("to_q" in key or "to_k" in key or "to_v" in key):
                if "to_q" not in key or "hada_w1_a" not in key:
                    continue

                lora_name = key.split(".", 1)[0]
                lora_name_prefix = lora_name.replace("to_q", "")

                w1a_list, w1b_list, w2a_list, w2b_list = [], [], [], []
                for suffix in ["to_q", "to_k", "to_v"]:
                    name = f"{lora_name_prefix}{suffix}"
                    w1a_list.append(state_dict.pop(f"{name}.hada_w1_a"))
                    w1b_list.append(state_dict.pop(f"{name}.hada_w1_b"))
                    w2a_list.append(state_dict.pop(f"{name}.hada_w2_a"))
                    w2b_list.append(state_dict.pop(f"{name}.hada_w2_b"))

                alpha = state_dict.pop(f"{lora_name}.alpha")
                state_dict.pop(f"{lora_name_prefix}to_k.alpha", None)
                state_dict.pop(f"{lora_name_prefix}to_v.alpha", None)

                w1a_qkv = torch.block_diag(*w1a_list)  # (3*out, 3r)
                w1b_qkv = torch.cat(w1b_list, dim=0)  # (3r, in)
                w2a_qkv = torch.block_diag(*w2a_list)  # (3*out, 3r)
                w2b_qkv = torch.cat(w2b_list, dim=0)  # (3r, in)

                new_lora_name = lora_name_prefix + "qkv"
                state_dict[f"{new_lora_name}.hada_w1_a"] = w1a_qkv
                state_dict[f"{new_lora_name}.hada_w1_b"] = w1b_qkv
                state_dict[f"{new_lora_name}.hada_w2_a"] = w2a_qkv
                state_dict[f"{new_lora_name}.hada_w2_b"] = w2b_qkv
                state_dict[f"{new_lora_name}.alpha"] = alpha * 3  # rank is 3x, so alpha * 3 keeps scale unchanged

                qkv_count += 1

        # LoKr QKV merge: materialize weight deltas via Kronecker product, concatenate, convert to LoRA via SVD
        # Kronecker product structure cannot be preserved across QKV concatenation,
        # so we convert QKV layers to LoRA format. Non-QKV layers remain as LoKr.
        keys = list(state_dict.keys())
        for key in keys:
            if key not in state_dict:
                continue
            if "attention" in key and ("to_q" in key or "to_k" in key or "to_v" in key):
                if "to_q" not in key or "lokr_w1" not in key:
                    continue

                lora_name = key.split(".", 1)[0]
                lora_name_prefix = lora_name.replace("to_q", "")

                delta_weights = []
                original_dtype = None
                for suffix in ["to_q", "to_k", "to_v"]:
                    name = f"{lora_name_prefix}{suffix}"
                    w1 = state_dict.pop(f"{name}.lokr_w1")
                    if original_dtype is None:
                        original_dtype = w1.dtype

                    w2a_key = f"{name}.lokr_w2_a"
                    w2b_key = f"{name}.lokr_w2_b"
                    w2_key = f"{name}.lokr_w2"

                    if w2a_key in state_dict:
                        # low-rank mode: w2 = w2_a @ w2_b
                        w2a = state_dict.pop(w2a_key)
                        w2b = state_dict.pop(w2b_key)
                        dim = w2a.shape[1]
                        w2 = w2a.float() @ w2b.float()
                    elif w2_key in state_dict:
                        # full matrix mode
                        w2 = state_dict.pop(w2_key).float()
                        dim = None
                    else:
                        raise ValueError(f"Missing lokr_w2 weights for {name}")

                    alpha_i = state_dict.pop(f"{name}.alpha", None)

                    # Compute scale: low-rank uses alpha/dim, full matrix uses 1.0
                    if dim is not None:
                        if alpha_i is not None:
                            alpha_val = alpha_i.item() if isinstance(alpha_i, torch.Tensor) else alpha_i
                        else:
                            alpha_val = dim
                        scale = alpha_val / dim
                    else:
                        scale = 1.0

                    delta_w = torch.kron(w1.float(), w2) * scale
                    delta_weights.append(delta_w)

                # Concatenate QKV deltas
                delta_qkv = torch.cat(delta_weights, dim=0)  # (3*out, in)

                # SVD decomposition
                U, S, Vt = torch.linalg.svd(delta_qkv, full_matrices=False)

                # Determine rank
                if args.lokr_rank is not None:
                    svd_rank = min(args.lokr_rank, S.shape[0])
                else:
                    # Auto: keep singular values contributing to 99.99% of total energy (Frobenius norm squared)
                    total_energy = (S**2).sum()
                    cumulative_energy = (S**2).cumsum(dim=0)
                    threshold_idx = (cumulative_energy >= total_energy * 0.9999).nonzero(as_tuple=True)[0]
                    if len(threshold_idx) > 0:
                        svd_rank = threshold_idx[0].item() + 1
                    else:
                        svd_rank = S.shape[0]

                # Log reconstruction quality
                reconstructed = (U[:, :svd_rank] * S[:svd_rank].unsqueeze(0)) @ Vt[:svd_rank, :]
                rel_error = (delta_qkv - reconstructed).norm() / delta_qkv.norm()
                logger.info(f"  LoKr->LoRA QKV {lora_name_prefix}qkv: rank={svd_rank}/{S.shape[0]}, relative error={rel_error:.6f}")

                # Construct LoRA weights: ΔW ≈ lora_up @ lora_down, with alpha = rank (scale = 1)
                sqrt_S = torch.sqrt(S[:svd_rank])
                lora_up = (U[:, :svd_rank] * sqrt_S.unsqueeze(0)).to(original_dtype)  # (3*out, rank)
                lora_down = (sqrt_S.unsqueeze(1) * Vt[:svd_rank, :]).to(original_dtype)  # (rank, in)

                new_lora_name = lora_name_prefix + "qkv"
                state_dict[f"{new_lora_name}.lora_up.weight"] = lora_up
                state_dict[f"{new_lora_name}.lora_down.weight"] = lora_down
                state_dict[f"{new_lora_name}.alpha"] = torch.tensor(float(svd_rank))

                qkv_count += 1

    logger.info(f"Direct key renames applied: {count}")
    logger.info(f"QKV layers processed: {qkv_count}")

    # Calculate hash
    if metadata is not None:
        logger.info("Calculating hashes and creating metadata...")
        model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

    # save destination safetensors
    logger.info(f"Saving destination file {args.dst_path}")
    save_file(state_dict, args.dst_path, metadata=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LoRA/LoHa/LoKr format for Z-Image ComfyUI")
    parser.add_argument("src_path", type=str, default=None, help="source path, sd-scripts format")
    parser.add_argument("dst_path", type=str, default=None, help="destination path, ComfyUI format")
    parser.add_argument("--reverse", action="store_true", help="reverse conversion direction (LoRA only)")
    parser.add_argument(
        "--lokr_rank", type=int, default=None, help="max rank for LoKr to LoRA QKV conversion (auto if not specified)"
    )
    args = parser.parse_args()
    main(args)
