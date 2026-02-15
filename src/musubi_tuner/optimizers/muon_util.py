from __future__ import annotations

import fnmatch
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# Model-specific default patterns for "hidden" layers (where Muon is typically beneficial).
#
# Note: Blissful-Tuner LoRA module names are underscore-normalized
# (see `.replace(".", "_")` in `src/musubi_tuner/networks/lora.py`), so
# patterns that contain dots will be dual-matched against both the original
# pattern and a dot->underscore normalized variant.
MODEL_LAYER_PATTERNS: Dict[str, List[str]] = {
    # Diffusion Transformers
    "flux": ["transformer_blocks", "single_transformer_blocks", "encoder.block"],
    "sd3": ["transformer_blocks", "encoder.block"],
    "wan": ["blocks", "transformer"],
    "hunyuan": ["transformer_blocks", "single_transformer_blocks"],
    "framepack": ["transformer_blocks"],
    # Vision Models
    # Qwen-Image backbone uses `transformer_blocks` (see `src/musubi_tuner/qwen_image/qwen_image_model.py:1175`).
    "qwen_image": ["transformer_blocks"],
    # Generic fallback (matches many transformer implementations)
    "default": ["transformer_blocks", "encoder.block", "blocks", "layers"],
}


@dataclass(frozen=True)
class MuonParamStats:
    muon_count: int
    adam_count: int
    muon_params_total: int
    adam_params_total: int
    matched_patterns: List[str]
    unmatched_patterns: List[str]


class LayerFilter:
    """Substring/fnmatch filter supporting dot + underscore variants."""

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pattern_underscore = pattern.replace(".", "_")
        self._compiled = f"*{self.pattern}*"
        self._compiled_underscore = f"*{self.pattern_underscore}*"

    def matches(self, param_name: str) -> bool:
        return (
            self.pattern in param_name
            or self.pattern_underscore in param_name
            or fnmatch.fnmatch(param_name, self._compiled)
            or fnmatch.fnmatch(param_name, self._compiled_underscore)
        )


def get_default_patterns(model_type: str) -> List[str]:
    model_type_lower = (model_type or "default").lower()

    if model_type_lower in MODEL_LAYER_PATTERNS:
        return MODEL_LAYER_PATTERNS[model_type_lower]

    for key, patterns in MODEL_LAYER_PATTERNS.items():
        if key in model_type_lower or model_type_lower in key:
            return patterns

    return MODEL_LAYER_PATTERNS["default"]


def _classify_parameter(param_name: str, param: torch.nn.Parameter, layer_filters: Sequence[LayerFilter]) -> str:
    # 1D params (bias, norm scales, DoRA magnitudes, etc.) always go to Adam.
    if param.ndim <= 1:
        return "adam"

    for layer_filter in layer_filters:
        if layer_filter.matches(param_name):
            return "muon"

    return "adam"


def split_param_groups_for_muon(
    trainable_params: Sequence[Dict[str, Any]],
    param_name_map: Dict[int, str],
    *,
    hidden_layer_patterns: Optional[Sequence[str]] = None,
    model_type: str = "default",
    muon_lr: Optional[float] = None,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
    adam_lr: float = 3e-4,
    adam_betas: Tuple[float, float] = (0.9, 0.95),
    adam_eps: float = 1e-8,
    adam_weight_decay: float = 0.0,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], MuonParamStats]:
    """
    Split *each* incoming param group into Muon and Adam subgroups.

    This preserves the original group structure (e.g., LoRA+ multiple groups),
    while allowing a Muon/Adam split inside each group.

    Returned groups are compatible with the official Muon optimizer classes:
    - Muon group keys:  {"params","lr","momentum","weight_decay","use_muon"}
    - Adam group keys:  {"params","lr","betas","eps","weight_decay","use_muon"}
    """
    user_supplied_patterns = hidden_layer_patterns is not None
    patterns = list(hidden_layer_patterns) if hidden_layer_patterns is not None else get_default_patterns(model_type)
    patterns = [p for p in (p.strip() for p in patterns) if p]

    filters = [LayerFilter(p) for p in patterns]

    all_muon_params: List[torch.nn.Parameter] = []
    all_adam_params: List[torch.nn.Parameter] = []
    matched_patterns = set()
    result_groups: List[Dict[str, Any]] = []

    for group_index, group in enumerate(trainable_params):
        if not isinstance(group, dict) or "params" not in group:
            raise TypeError(
                f"Expected trainable_params to be a sequence of param-group dicts. Got element {group_index}: {type(group)}"
            )

        group_params: Iterable[torch.nn.Parameter] = group["params"]
        base_lr = group.get("lr")

        muon_params: List[torch.nn.Parameter] = []
        adam_params: List[torch.nn.Parameter] = []

        for param in group_params:
            if not param.requires_grad:
                continue

            param_name = param_name_map.get(id(param), "")
            classification = _classify_parameter(param_name, param, filters)

            if classification == "muon":
                muon_params.append(param)
                all_muon_params.append(param)
                for layer_filter in filters:
                    if layer_filter.matches(param_name):
                        matched_patterns.add(layer_filter.pattern)
            else:
                adam_params.append(param)
                all_adam_params.append(param)

        if muon_params:
            result_groups.append(
                {
                    "params": muon_params,
                    "use_muon": True,
                    # If muon_lr is provided, treat it as a global override. Otherwise preserve the group's LR (LoRA+).
                    "lr": muon_lr if muon_lr is not None else (base_lr if base_lr is not None else 0.02),
                    "momentum": muon_momentum,
                    "weight_decay": muon_weight_decay,
                }
            )

        if adam_params:
            result_groups.append(
                {
                    "params": adam_params,
                    "use_muon": False,
                    # Adam LR must be in Adam units (NOT the Muon group's LR units).
                    "lr": adam_lr,
                    "betas": adam_betas,
                    "eps": adam_eps,
                    "weight_decay": adam_weight_decay,
                }
            )

    unmatched_patterns = [p for p in patterns if p not in matched_patterns]

    stats = MuonParamStats(
        muon_count=len(all_muon_params),
        adam_count=len(all_adam_params),
        muon_params_total=sum(p.numel() for p in all_muon_params),
        adam_params_total=sum(p.numel() for p in all_adam_params),
        matched_patterns=sorted(matched_patterns),
        unmatched_patterns=unmatched_patterns,
    )

    if verbose and user_supplied_patterns and unmatched_patterns:
        warnings.warn(f"Muon hidden-layer patterns did not match any parameters: {unmatched_patterns}")

    if verbose:
        total_tensors = stats.muon_count + stats.adam_count
        if total_tensors > 0 and stats.muon_count == 0:
            warnings.warn(
                "0% of trainable tensors were assigned to Muon. All tensors will use Adam. "
                "Check `--muon_hidden_layers` / `--muon_model_type`."
            )

    return result_groups, stats


def print_muon_summary(stats: MuonParamStats) -> None:
    total_params = stats.muon_params_total + stats.adam_params_total
    muon_pct = 100 * stats.muon_params_total / total_params if total_params > 0 else 0.0

    print("\n" + "=" * 60)
    print("Muon Parameter Assignment Summary")
    print("=" * 60)
    print(f"  Muon parameters:  {stats.muon_count:,} tensors ({stats.muon_params_total:,} params, {muon_pct:.1f}%)")
    print(f"  Adam parameters:  {stats.adam_count:,} tensors ({stats.adam_params_total:,} params, {100 - muon_pct:.1f}%)")
    if stats.matched_patterns:
        print(f"  Matched patterns: {stats.matched_patterns}")
    if stats.unmatched_patterns:
        print(f"  Unmatched patterns: {stats.unmatched_patterns}")
    print("=" * 60 + "\n")

