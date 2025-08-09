import os
import re
from typing import Dict, List, Optional, Union
import torch

import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


from musubi_tuner.modules.fp8_optimization_utils import load_safetensors_with_fp8_optimization, optimize_state_dict_with_fp8
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, load_safetensors


def extract_loraid_from_lora_metadata(lora_sd: Dict[str, torch.Tensor], lora_path: str = None) -> Optional[int]:
    """
    Extract LORAID from LoRA state dict metadata.
    
    Args:
        lora_sd: LoRA state dictionary
        lora_path: Optional path to LoRA file (for loading metadata directly)
        
    Returns:
        LORAID if found, None otherwise
    """
    # Try to get metadata from file if path provided
    if lora_path and os.path.exists(lora_path):
        try:
            from safetensors import safe_open
            with safe_open(lora_path, framework="pt") as f:
                metadata = f.metadata()
                if metadata and "ss_loraid" in metadata:
                    return int(metadata["ss_loraid"])
        except Exception as e:
            logger.warning(f"Failed to extract LORAID from {lora_path}: {e}")
    
    return None


def validate_loraid_compatibility(lora_weights_list: List[Dict[str, torch.Tensor]], lora_paths: List[str] = None) -> bool:
    """
    Validate LoRA compatibility and log LORAIDs. For single LoRA: logs LORAID. For multiple LoRAs: checks for conflicts.
    
    Args:
        lora_weights_list: List of LoRA state dictionaries
        lora_paths: Optional list of LoRA file paths
        
    Returns:
        True if compatible, False otherwise
    """
    if not lora_weights_list:
        return True
        
    loraids = []
    paths = lora_paths or [None] * len(lora_weights_list)
    logger.info(f"DEBUG: Validating {len(lora_weights_list)} LoRAs with paths: {paths}")
    
    for i, (lora_sd, path) in enumerate(zip(lora_weights_list, paths)):
        logger.info(f"DEBUG: Processing LoRA {i}: path={path}")
        loraid = extract_loraid_from_lora_metadata(lora_sd, path)
        logger.info(f"DEBUG: Extracted LORAID: {loraid}")
        if loraid is not None:
            loraids.append(loraid)
            
    # Check for duplicate LORAIDs (which would indicate conflicting parameter spaces)
    unique_loraids = set(loraids)
    if len(loraids) != len(unique_loraids):
        logger.error(f"Duplicate LORAIDs detected: {loraids}. This would cause parameter conflicts!")
        return False
        
    logger.info(f"LORAID compatibility check passed. LORAIDs: {loraids}")
    return True


def filter_lora_state_dict(
    weights_sd: Dict[str, torch.Tensor],
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    # apply include/exclude patterns
    original_key_count = len(weights_sd.keys())
    if include_pattern is not None:
        regex_include = re.compile(include_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
        logger.info(f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}")

    if exclude_pattern is not None:
        original_key_count_ex = len(weights_sd.keys())
        regex_exclude = re.compile(exclude_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if not regex_exclude.search(k)}
        logger.info(f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}")

    if len(weights_sd) != original_key_count:
        remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys()]))
        remaining_keys.sort()
        logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
        if len(weights_sd) == 0:
            logger.warning(f"No keys left after filtering.")

    return weights_sd


def load_safetensors_with_lora_and_fp8(
    model_files: Union[str, List[str]],
    lora_weights_list: Optional[Dict[str, torch.Tensor]],
    lora_multipliers: Optional[List[float]],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    lora_file_paths: Optional[List[str]] = None,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model with fp8 optimization if needed.

    Args:
        model_files (Union[str, List[str]]): Path to the model file or list of paths. If the path matches a pattern like `00001-of-00004`, it will load all files with the same prefix.
        lora_weights_list (Optional[Dict[str, torch.Tensor]]): Dictionary of LoRA weight tensors to load.
        lora_multipliers (Optional[List[float]]): List of multipliers for LoRA weights.
        fp8_optimization (bool): Whether to apply FP8 optimization.
        calc_device (torch.device): Device to calculate on.
        move_to_device (bool): Whether to move tensors to the calculation device after loading.
        target_keys (Optional[List[str]]): Keys to target for optimization.
        exclude_keys (Optional[List[str]]): Keys to exclude from optimization.
        lora_file_paths (Optional[List[str]]): Original LoRA file paths for metadata extraction.
    """

    # if the file name ends with 00001-of-00004 etc, we need to load the files with the same prefix
    if isinstance(model_files, str):
        model_files = [model_files]

    extended_model_files = []
    for model_file in model_files:
        basename = os.path.basename(model_file)
        match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
        if match:
            prefix = basename[: match.start(2)]
            count = int(match.group(3))
            state_dict = {}
            for i in range(count):
                filename = f"{prefix}{i+1:05d}-of-{count:05d}.safetensors"
                filepath = os.path.join(os.path.dirname(model_file), filename)
                if os.path.exists(filepath):
                    extended_model_files.append(filepath)
                else:
                    raise FileNotFoundError(f"File {filepath} not found")
        else:
            extended_model_files.append(model_file)
    model_files = extended_model_files
    logger.info(f"Loading model files: {model_files}")

    # load LoRA weights
    weight_hook = None
    if lora_weights_list is None or len(lora_weights_list) == 0:
        lora_weights_list = []
        lora_multipliers = []
        list_of_lora_weight_keys = []
    else:
        # Validate LORAID compatibility and log LORAIDs (works for single and multi-LoRA loading)
        logger.info(f"DEBUG: About to validate {len(lora_weights_list)} LoRAs with file paths: {lora_file_paths}")
        
        # DEBUG: Check what blocks each LoRA actually contains and detect overlaps
        all_lora_blocks = {}
        overlapping_blocks = set()
        
        for i, lora_sd in enumerate(lora_weights_list):
            blocks = set()
            for key in lora_sd.keys():
                if "lora_unet_blocks_" in key:
                    # Extract block number from key like "lora_unet_blocks_5_self_attn_q.lora_down.weight"
                    try:
                        block_part = key.split("lora_unet_blocks_")[1].split("_")[0]
                        blocks.add(int(block_part))
                    except (IndexError, ValueError):
                        pass
            all_lora_blocks[i] = blocks
            logger.info(f"DEBUG: LoRA {i} contains parameters for blocks: {sorted(blocks)}")
        
        # Check for overlapping blocks between LoRAs
        for i in range(len(all_lora_blocks)):
            for j in range(i + 1, len(all_lora_blocks)):
                overlap = all_lora_blocks[i].intersection(all_lora_blocks[j])
                if overlap:
                    logger.error(f"CRITICAL: LoRA {i} and LoRA {j} have overlapping blocks: {sorted(overlap)}!")
                    overlapping_blocks.update(overlap)
        
        if overlapping_blocks:
            logger.error(f"CRITICAL: Total overlapping blocks detected: {sorted(overlapping_blocks)}. This WILL cause blending!")
        else:
            logger.info("DEBUG: No block overlaps detected - LoRAs should be independent")
        
        if not validate_loraid_compatibility(lora_weights_list, lora_file_paths):
            raise ValueError("LORAID compatibility check failed. Cannot load LoRAs with conflicting parameter spaces.")
            
        list_of_lora_weight_keys = []
        for lora_sd in lora_weights_list:
            lora_weight_keys = set(lora_sd.keys())
            list_of_lora_weight_keys.append(lora_weight_keys)

        if lora_multipliers is None:
            lora_multipliers = [1.0] * len(lora_weights_list)
        while len(lora_multipliers) < len(lora_weights_list):
            lora_multipliers.append(1.0)
        if len(lora_multipliers) > len(lora_weights_list):
            lora_multipliers = lora_multipliers[: len(lora_weights_list)]

        # Merge LoRA weights into the state dict
        logger.info(f"Merging LoRA weights into state dict. multipliers: {lora_multipliers}")

        # make hook for LoRA merging
        def weight_hook_func(model_weight_key, model_weight):
            nonlocal list_of_lora_weight_keys, lora_weights_list, lora_multipliers, calc_device

            if not model_weight_key.endswith(".weight"):
                return model_weight

            original_device = model_weight.device
            if original_device != calc_device:
                model_weight = model_weight.to(calc_device)  # to make calculation faster

            for lora_idx, (lora_weight_keys, lora_sd, multiplier) in enumerate(zip(list_of_lora_weight_keys, lora_weights_list, lora_multipliers)):
                # check if this weight has LoRA weights
                lora_name = model_weight_key.rsplit(".", 1)[0]  # remove trailing ".weight"
                lora_name = "lora_unet_" + lora_name.replace(".", "_")
                down_key = lora_name + ".lora_down.weight"
                up_key = lora_name + ".lora_up.weight"
                alpha_key = lora_name + ".alpha"
                if down_key not in lora_weight_keys or up_key not in lora_weight_keys:
                    continue
                    
                # DEBUG: Log when a LoRA is being applied
                if "blocks_0_" in lora_name or "blocks_12_" in lora_name:
                    logger.info(f"DEBUG: Applying LoRA {lora_idx} to {lora_name} (multiplier: {multiplier})")
                
                # CRITICAL DEBUG: Check for potential issues (commented out - too spammy)
                # if len(lora_weights_list) > 1:
                #     # Log the before/after weight changes for multi-LoRA scenarios
                #     original_weight_norm = model_weight.norm().item()
                #     logger.info(f"DEBUG: Before applying LoRA {lora_idx} to {lora_name}: weight norm = {original_weight_norm:.6f}")

                # get LoRA weights
                down_weight = lora_sd[down_key]
                up_weight = lora_sd[up_key]

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim
                
                # DEBUG: Log alpha values for multi-LoRA scenarios (commented out - too spammy)
                # if len(lora_weights_list) > 1 and ("blocks_0_" in lora_name or "blocks_12_" in lora_name):
                #     logger.info(f"DEBUG: LoRA {lora_idx} {lora_name}: dim={dim}, alpha={alpha}, scale={scale:.4f}")

                down_weight = down_weight.to(calc_device)
                up_weight = up_weight.to(calc_device)

                # W <- W + U * D
                if len(model_weight.size()) == 2:
                    # linear
                    if len(up_weight.size()) == 4:  # use linear projection mismatch
                        up_weight = up_weight.squeeze(3).squeeze(2)
                        down_weight = down_weight.squeeze(3).squeeze(2)
                    lora_delta = multiplier * (up_weight @ down_weight) * scale
                    model_weight = model_weight + lora_delta
                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    lora_delta = (
                        multiplier
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                    model_weight = model_weight + lora_delta
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    lora_delta = multiplier * conved * scale
                    model_weight = model_weight + lora_delta
                
                # CRITICAL DEBUG: Log the delta magnitude for multi-LoRA scenarios (commented out - too spammy)
                # if len(lora_weights_list) > 1 and ("blocks_0_" in lora_name or "blocks_12_" in lora_name):
                #     delta_norm = lora_delta.norm().item()
                #     final_weight_norm = model_weight.norm().item()
                #     logger.info(f"DEBUG: After applying LoRA {lora_idx}: delta norm = {delta_norm:.6f}, final weight norm = {final_weight_norm:.6f}")

                # remove LoRA keys from set
                lora_weight_keys.remove(down_key)
                lora_weight_keys.remove(up_key)
                if alpha_key in lora_weight_keys:
                    lora_weight_keys.remove(alpha_key)

            model_weight = model_weight.to(original_device)  # move back to original device
            return model_weight

        weight_hook = weight_hook_func

    state_dict = load_safetensors_with_fp8_optimization_and_hook(
        model_files,
        fp8_optimization,
        calc_device,
        move_to_device,
        dit_weight_dtype,
        target_keys,
        exclude_keys,
        weight_hook=weight_hook,
    )

    for lora_weight_keys in list_of_lora_weight_keys:
        # check if all LoRA keys are used
        if len(lora_weight_keys) > 0:
            # if there are still LoRA keys left, it means they are not used in the model
            # this is a warning, not an error
            logger.warning(f"Warning: not all LoRA keys are used: {', '.join(lora_weight_keys)}")

    return state_dict


def load_safetensors_with_fp8_optimization_and_hook(
    model_files: list[str],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    weight_hook: callable = None,
) -> dict[str, torch.Tensor]:
    """
    Load state dict from safetensors files and merge LoRA weights into the state dict with fp8 optimization if needed.
    """
    if fp8_optimization:
        logger.info(f"Loading state dict with FP8 optimization. Hook enabled: {weight_hook is not None}")
        # dit_weight_dtype is not used because we use fp8 optimization
        state_dict = load_safetensors_with_fp8_optimization(
            model_files, calc_device, target_keys, exclude_keys, move_to_device=move_to_device, weight_hook=weight_hook
        )
    else:
        logger.info(f"Loading state dict without FP8 optimization. Hook enabled: {weight_hook is not None}")
        state_dict = {}
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file) as f:
                for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", leave=False):
                    value = f.get_tensor(key)
                    if weight_hook is not None:
                        value = weight_hook(key, value)
                    if move_to_device:
                        if dit_weight_dtype is None:
                            value = value.to(calc_device)
                        else:
                            value = value.to(calc_device, dtype=dit_weight_dtype)
                    elif dit_weight_dtype is not None:
                        value = value.to(dit_weight_dtype)

                    state_dict[key] = value

    return state_dict
