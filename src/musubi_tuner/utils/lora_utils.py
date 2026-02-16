import os
import re
from typing import Dict, Iterable, List, Optional, Union
import torch
from tqdm import tqdm

from musubi_tuner.utils.device_utils import synchronize_device
from musubi_tuner.utils.safetensors_utils import (
    MemoryEfficientSafeOpen,
    TensorWeightAdapter,
    WeightTransformHooks,
    get_split_weight_filenames,
)
from musubi_tuner.modules.fp8_optimization_utils import load_safetensors_with_fp8_optimization
from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "green")


UNKNOWN_NETWORK_FORMAT_HINT = (
    "Some scripts support --prefer_lycoris for non-native formats (IA3, DyLoRA, etc.). Otherwise, convert to a supported format."
)


def format_unknown_network_type_error(lora_path: str) -> str:
    """Build a consistent error message for unsupported/unknown LoRA weight formats."""
    return f"Unrecognized weight format in {lora_path}. {UNKNOWN_NETWORK_FORMAT_HINT}"


_DIFFUSERS_PREFIXES = frozenset(("diffusion_model", "transformer"))


def convert_diffusers_if_needed(lora_sd: Dict[str, torch.Tensor], prefix: str = "lora_unet_") -> Dict[str, torch.Tensor]:
    """Convert Diffusers-format keys to default format, preserving non-Diffusers keys.

    Splits the state dict into Diffusers-prefixed keys and passthrough keys,
    converts only the Diffusers subset, and merges back. This avoids data loss
    when a state dict contains both Diffusers and already-normalized keys.
    """
    from musubi_tuner.convert_lora import convert_from_diffusers

    diffusers_sd = {}
    passthrough_sd = {}
    for key, value in lora_sd.items():
        if "." in key and key.split(".", 1)[0] in _DIFFUSERS_PREFIXES:
            diffusers_sd[key] = value
        else:
            passthrough_sd[key] = value

    if not diffusers_sd:
        return lora_sd  # nothing to convert

    logger.info("Converting LoRA from foreign key naming format")
    converted = convert_from_diffusers(prefix, diffusers_sd)
    # Merge: passthrough first, converted on top (converted keys take priority on collision)
    passthrough_sd.update(converted)
    return passthrough_sd


_MODEL_KEY_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def _make_lora_name_from_model_key(model_weight_key: str) -> str:
    """Convert a model state-dict key (e.g. 'model.diffusion_model.blocks.0.attn.q.weight')
    to the corresponding LoRA module name (e.g. 'lora_unet_blocks_0_attn_q').

    Strips the trailing '.weight' suffix and any model-level prefixes that don't
    appear in LoRA key naming (e.g. 'model.diffusion_model.' from WAN checkpoints).
    """
    lora_name = model_weight_key.rsplit(".", 1)[0]  # remove trailing ".weight"
    for pfx in _MODEL_KEY_PREFIXES:
        if lora_name.startswith(pfx):
            lora_name = lora_name[len(pfx):]
            break
    return "lora_unet_" + lora_name.replace(".", "_")


def detect_network_type(lora_sd_or_keys: Union[Dict[str, torch.Tensor], Iterable[str]]) -> str:
    """Detect network type from state dict keys.

    Returns 'lora', 'loha', 'lokr', 'hybrid', or 'unknown'.
    'hybrid' means multiple key families coexist (e.g. after QKV conversion).
    Accepts a state dict or an iterable of key strings.
    """
    keys = lora_sd_or_keys.keys() if isinstance(lora_sd_or_keys, dict) else lora_sd_or_keys
    found_types = set()
    for key in keys:
        # Standard LoRA keys (lora_down/lora_up) AND Diffusers-format keys (lora_A/lora_B)
        if "lora_down" in key or "lora_up" in key or "lora_A" in key or "lora_B" in key:
            found_types.add("lora")
        elif "hada_w1_a" in key or "hada_w2_a" in key:
            found_types.add("loha")
        elif "lokr_w1" in key or "lokr_w2" in key or "lokr_w2_a" in key:
            found_types.add("lokr")
    if len(found_types) > 1:
        return "hybrid"
    if len(found_types) == 1:
        return found_types.pop()
    return "unknown"


def filter_lora_state_dict(
    weights_sd: Dict[str, torch.Tensor],
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    # apply include/exclude patterns
    original_key_count = len(weights_sd.keys())
    if include_pattern is not None:
        regex_include = re.compile(include_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if "." not in k or regex_include.search(k)}
        logger.info(f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}")

    if exclude_pattern is not None:
        original_key_count_ex = len(weights_sd.keys())
        regex_exclude = re.compile(exclude_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if "." not in k or not regex_exclude.search(k)}
        logger.info(f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}")

    if len(weights_sd) != original_key_count:
        remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys() if "." in k]))
        remaining_keys.sort()
        logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
        if len(weights_sd) == 0:
            logger.warning("No keys left after filtering.")

    return weights_sd


def load_safetensors_with_lora_and_fp8(
    model_files: Union[str, List[str]],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]],
    lora_multipliers: Optional[List[float]],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    quantization_mode: str = "block",
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model with fp8 optimization if needed.

    Args:
        model_files (Union[str, List[str]]): Path to the model file or list of paths. If the path matches a pattern like `00001-of-00004`, it will load all files with the same prefix.
        lora_weights_list (Optional[List[Dict[str, torch.Tensor]]]): List of dictionaries of LoRA weight tensors to load.
        lora_multipliers (Optional[List[float]]): List of multipliers for LoRA weights.
        fp8_optimization (bool): Whether to apply FP8 optimization.
        calc_device (torch.device): Device to calculate on.
        move_to_device (bool): Whether to move tensors to the calculation device after loading.
        target_keys (Optional[List[str]]): Keys to target for optimization.
        exclude_keys (Optional[List[str]]): Keys to exclude from optimization.
        disable_numpy_memmap (bool): Whether to disable numpy memmap when loading safetensors.
        weight_transform_hooks (Optional[WeightTransformHooks]): Hooks for transforming weights during loading.
    """

    # if the file name ends with 00001-of-00004 etc, we need to load the files with the same prefix
    if isinstance(model_files, str):
        model_files = [model_files]

    extended_model_files = []
    for model_file in model_files:
        split_filenames = get_split_weight_filenames(model_file)
        if split_filenames is not None:
            extended_model_files.extend(split_filenames)
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

        # Detect network types for summary logging (actual dispatch is per-key-family)
        lora_network_types = [detect_network_type(lora_sd) for lora_sd in lora_weights_list]
        logger.info(f"Merging LoRA weights into state dict. multipliers: {lora_multipliers}, types: {lora_network_types}")

        # Import merge functions once (deferred to avoid circular imports at module level)
        from musubi_tuner.networks.loha import merge_weights_to_tensor as loha_merge
        from musubi_tuner.networks.lokr import merge_weights_to_tensor as lokr_merge

        # make hook for LoRA merging
        def weight_hook_func(model_weight_key, model_weight: torch.Tensor, keep_on_calc_device=False):
            nonlocal list_of_lora_weight_keys, lora_weights_list, lora_multipliers, calc_device

            if not model_weight_key.endswith(".weight"):
                return model_weight

            original_device = model_weight.device
            original_dtype = model_weight.dtype
            if original_device != calc_device:
                model_weight = model_weight.to(calc_device)  # to make calculation faster

            for lora_weight_keys, lora_sd, multiplier in zip(list_of_lora_weight_keys, lora_weights_list, lora_multipliers):
                lora_name = _make_lora_name_from_model_key(model_weight_key)

                # Per-key-family dispatch: try each family in deterministic order.
                # Each merge function is a no-op if no matching keys found.
                # This handles hybrid dicts (lokr_* + lora_* after QKV conversion).
                model_weight = loha_merge(model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device)
                model_weight = lokr_merge(model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device)

                # Standard LoRA path (delegates to shared merge function for dtype safety)
                model_weight = lora_merge_weights_to_tensor(
                    model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device
                )

            if not keep_on_calc_device and original_device != calc_device:
                model_weight = model_weight.to(original_device, original_dtype)  # move back to original device

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
        quantization_mode=quantization_mode,
        disable_numpy_memmap=disable_numpy_memmap,
        weight_transform_hooks=weight_transform_hooks,
    )

    for lora_weight_keys in list_of_lora_weight_keys:
        # Exclude non-dotted keys (network-level metadata like lokr_factor, use_rslora_flag)
        remaining = {k for k in lora_weight_keys if "." in k}
        if len(remaining) > 0:
            logger.warning(f"Warning: not all LoRA keys are used: {', '.join(sorted(remaining))}")

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
    quantization_mode: str = "block",
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """
    Load state dict from safetensors files and merge LoRA weights into the state dict with fp8 optimization if needed.
    """
    if fp8_optimization:
        logger.info(
            f"Loading state dict with FP8 optimization. Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}"
        )
        # dit_weight_dtype is not used because we use fp8 optimization
        state_dict = load_safetensors_with_fp8_optimization(
            model_files,
            calc_device,
            target_keys,
            exclude_keys,
            move_to_device=move_to_device,
            weight_hook=weight_hook,
            quantization_mode=quantization_mode,
            disable_numpy_memmap=disable_numpy_memmap,
            weight_transform_hooks=weight_transform_hooks,
        )
    else:
        logger.info(
            f"Loading state dict without FP8 optimization. Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}"
        )
        state_dict = {}
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
                f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f
                for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", leave=False):
                    if weight_hook is None and move_to_device:
                        value = f.get_tensor(key, device=calc_device, dtype=dit_weight_dtype)
                    else:
                        value = f.get_tensor(key)  # we cannot directly load to device because get_tensor does non-blocking transfer
                        if weight_hook is not None:
                            value = weight_hook(key, value, keep_on_calc_device=move_to_device)
                        if move_to_device:
                            value = value.to(calc_device, dtype=dit_weight_dtype, non_blocking=True)
                        elif dit_weight_dtype is not None:
                            value = value.to(dit_weight_dtype)

                    state_dict[key] = value
        if move_to_device:
            synchronize_device(calc_device)

    return state_dict


def lora_merge_weights_to_tensor(
    model_weight: torch.Tensor,
    lora_name: str,
    lora_sd: Dict[str, torch.Tensor],
    lora_weight_keys: set,
    multiplier: float,
    calc_device: torch.device,
) -> torch.Tensor:
    """Merge standard LoRA weights directly into a model weight tensor.

    Supports Linear and Conv2d (1x1 and 3x3). Consumed keys are removed from lora_weight_keys.
    Returns model_weight unchanged if no matching LoRA keys found.
    """
    down_key = lora_name + ".lora_down.weight"
    up_key = lora_name + ".lora_up.weight"
    alpha_key = lora_name + ".alpha"

    if down_key not in lora_weight_keys or up_key not in lora_weight_keys:
        return model_weight

    down_weight = lora_sd[down_key].to(calc_device)
    up_weight = lora_sd[up_key].to(calc_device)

    dim = down_weight.size()[0]
    alpha = lora_sd.get(alpha_key, dim)
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    scale = alpha / dim

    org_device = model_weight.device
    original_dtype = model_weight.dtype
    compute_dtype = torch.float16 if original_dtype.itemsize == 1 else torch.float32
    model_weight = model_weight.to(calc_device, dtype=compute_dtype)
    down_weight = down_weight.to(compute_dtype)
    up_weight = up_weight.to(compute_dtype)

    if len(model_weight.size()) == 2:
        # linear
        if len(up_weight.size()) == 4:  # use linear projection mismatch
            up_weight = up_weight.squeeze(3).squeeze(2)
            down_weight = down_weight.squeeze(3).squeeze(2)
        model_weight = model_weight + multiplier * (up_weight @ down_weight) * scale
    elif down_weight.size()[2:4] == (1, 1):
        # conv2d 1x1
        model_weight = (
            model_weight
            + multiplier * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * scale
        )
    else:
        # conv2d 3x3
        conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
        model_weight = model_weight + multiplier * conved * scale

    model_weight = model_weight.to(device=org_device, dtype=original_dtype)

    # Remove consumed keys
    for key in [down_key, up_key, alpha_key]:
        lora_weight_keys.discard(key)

    return model_weight


def merge_nonlora_to_model(
    model: torch.nn.Module,
    weights_sd: Dict[str, torch.Tensor],
    multiplier: float,
    device: torch.device,
) -> int:
    """Merge LoHa/LoKr/LoRA weights directly into model parameters via per-key-family dispatch.

    Iterates model named_parameters, constructs lora_name from each param name,
    and tries each merge family in order (each is a no-op if no matching keys).
    Handles hybrid dicts (e.g. lokr_* + lora_* after QKV conversion).
    Returns number of consumed keys.
    """
    from musubi_tuner.networks.loha import merge_weights_to_tensor as loha_merge
    from musubi_tuner.networks.lokr import merge_weights_to_tensor as lokr_merge

    lora_weight_keys = set(weights_sd.keys())
    initial_key_count = len(lora_weight_keys)

    for param_name, param in model.named_parameters():
        if not param_name.endswith(".weight"):
            continue

        lora_name = "lora_unet_" + param_name.rsplit(".", 1)[0].replace(".", "_")

        # Per-key-family dispatch: LoHa → LoKr → LoRA
        param.data = loha_merge(param.data, lora_name, weights_sd, lora_weight_keys, multiplier, device)
        param.data = lokr_merge(param.data, lora_name, weights_sd, lora_weight_keys, multiplier, device)
        param.data = lora_merge_weights_to_tensor(param.data, lora_name, weights_sd, lora_weight_keys, multiplier, device)

    merged_count = initial_key_count - len(lora_weight_keys)

    # Warn about remaining unmerged keys (exclude non-dotted metadata like lokr_factor)
    remaining = {k for k in lora_weight_keys if "." in k}
    if remaining:
        logger.warning(f"{len(remaining)} LoHa/LoKr/LoRA keys were not matched to model parameters")

    return merged_count
