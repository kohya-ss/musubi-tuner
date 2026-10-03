"""LoRA for DLSS-NR canonical linears.

Adapters live on this module, not on the frozen base. QKV stays one head-major matrix.
`vit_only` covers blocks 31-38 expand, contract, QKV and attention projection.
`multiscale` covers every other FFN and attention linear, with rank chosen by block width.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from musubi_tuner.dlssnr.model import NRModel, ChannelLinear
from musubi_tuner.dlssnr.profiles import PROFILE_ID, block_channels

VIT_ONLY_ELEMENTS = 2_097_152
RANK_BY_WIDTH = {32: 2, 64: 4, 128: 8, 256: 8, 512: 16, 1024: 16}
LORA_SCHEMA = "dlssnr_lora_v2"
LEGACY_LORA_SCHEMA = "dlssnr_lora_v1"
LORA_FORWARD = "canonical_weight_plus_delta_v1"


class LoRADelta(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dropout: float, target: str) -> None:
        super().__init__()
        if rank < 1 or rank > min(in_features, out_features):
            raise ValueError(f"{target}: rank {rank} is outside 1..min({in_features}, {out_features})")
        if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError(f"{target}: invalid alpha or dropout")
        self.target = target
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        self.dropout = float(dropout)
        self.lora_down = nn.Parameter(torch.empty(rank, in_features))
        self.lora_up = nn.Parameter(torch.empty(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            hidden = F.conv2d(x, self.lora_down.view(self.rank, -1, 1, 1))
            if self.training and self.dropout > 0:
                hidden = F.dropout(hidden, self.dropout)
            return F.conv2d(hidden, self.lora_up.view(-1, self.rank, 1, 1)) * self.scale
        hidden = F.linear(x, self.lora_down)
        if self.training and self.dropout > 0:
            hidden = F.dropout(hidden, self.dropout)
        return F.linear(hidden, self.lora_up) * self.scale

    def delta_weight(self) -> torch.Tensor:
        with torch.autocast(self.lora_up.device.type, enabled=False):
            return self.scale * (self.lora_up @ self.lora_down)


class DLSSNRLoRA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapters = nn.ModuleList()
        self.target_names: list[str] = []
        self.profile = ""
        self.report: dict = {}

    @property
    def elements(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def add(self, target: str, module: ChannelLinear, rank: int, alpha: float, dropout: float) -> None:
        if getattr(module, "_dlssnr_lora_target", None):
            raise RuntimeError(f"{target} already has a LoRA adapter")
        out_features, in_features = module.weight.shape
        adapter = LoRADelta(in_features, out_features, rank, alpha, dropout, target).to(
            device=module.weight.device, dtype=torch.float32
        )
        self.adapters.append(adapter)
        self.target_names.append(target)
        module._dlssnr_lora_target = target
        module.weight.requires_grad_(False)

        def forward(x: torch.Tensor, module: ChannelLinear = module, adapter: LoRADelta = adapter) -> torch.Tensor:
            # Real NR weights amplify split-GEMM rounding through cosine attention.
            # Use the same effective weight and projection as a merged checkpoint.
            result = module.project(module.materialized_weight() + adapter.delta_weight(), x)
            if adapter.training and adapter.dropout > 0:
                hidden = module.project(adapter.lora_down, x)
                dropped = F.dropout(hidden, adapter.dropout) - hidden
                result = result + module.project(adapter.lora_up, dropped) * adapter.scale
            return result

        module.forward = forward  # type: ignore[method-assign]


def inject(model: nn.Module, table: dict) -> DLSSNRLoRA:
    profile = table["profile"]
    dropout = float(table.get("dropout", 0.0))
    if table.get("qkv_mode", "fused_head_major") != "fused_head_major":
        raise ValueError("only qkv_mode = fused_head_major is implemented")
    selected = _select_targets(model, profile, table)
    if not selected:
        raise ValueError(f"LoRA profile {profile} matched no linears")
    network = DLSSNRLoRA()
    network.profile = profile
    for name, module, rank, alpha in selected:
        network.add(name, module, rank, alpha, dropout)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    counted = sum(rank * (module.weight.shape[0] + module.weight.shape[1]) for _, module, rank, _ in selected)
    if counted != network.elements:
        raise RuntimeError(f"LoRA element recount {counted} != parameter count {network.elements}")
    expected = VIT_ONLY_ELEMENTS // 16 * int(table.get("rank", 16))
    if profile == "vit_only" and (len(selected) != 32 or network.elements != expected):
        raise RuntimeError(f"vit_only has {len(selected)} targets / {network.elements} elements, expected 32 / {expected}")
    network.report = {
        "schema": "dlssnr_lora_report_v1",
        "profile": profile,
        "qkv_mode": "fused_head_major",
        "forward_mode": LORA_FORWARD,
        "elements": network.elements,
        "excluded": ["input_adapter", "head", "prior", "temperature", "skip_scale", "blend_scale", "transition"],
        "targets": [
            {
                "name": name,
                "out": int(module.weight.shape[0]),
                "in": int(module.weight.shape[1]),
                "rank": rank,
                "alpha": alpha,
                "elements": rank * (int(module.weight.shape[0]) + int(module.weight.shape[1])),
            }
            for name, module, rank, alpha in selected
        ],
    }
    return network


def _select_targets(model: nn.Module, profile: str, table: dict) -> list[tuple[str, ChannelLinear, int, float]]:
    found = []
    for module_name, module in model.named_modules():
        if not isinstance(module, ChannelLinear):
            continue
        weight_name = f"{module_name}.weight"
        choice = _rank_for(weight_name, module, profile, table)
        if choice is None:
            continue
        rank, alpha = choice
        found.append((weight_name, module, rank, alpha))
    return found


def _rank_for(weight_name: str, module: ChannelLinear, profile: str, table: dict) -> tuple[int, float] | None:
    if profile == "vit_only":
        if "rank_by_width" in table:
            raise ValueError("vit_only cannot also set rank_by_width")
        if not _is_vit_only_target(weight_name):
            return None
        return int(table["rank"]), float(table["alpha"])
    if profile == "multiscale":
        if "rank" in table or "alpha" in table:
            raise ValueError("multiscale uses the width table, not a single rank/alpha")
        if _is_excluded(weight_name):
            return None
        block = int(weight_name.split(".")[1])
        channels = block_channels(block)
        ranks = {int(key): int(value) for key, value in table.get("rank_by_width", RANK_BY_WIDTH).items()}
        alphas = {
            int(key): float(value) for key, value in table.get("alpha_by_width", table.get("rank_by_width", RANK_BY_WIDTH)).items()
        }
        if channels not in ranks or channels not in alphas:
            raise ValueError(f"{weight_name} has width {channels}, which is not in the rank map")
        return ranks[channels], alphas[channels]
    raise ValueError(f"unknown LoRA profile {profile}")


def _is_vit_only_target(weight_name: str) -> bool:
    parts = weight_name.split(".")
    if len(parts) < 4 or parts[0] != "blocks":
        return False
    block = int(parts[1])
    if block < 31 or block > 38:
        return False
    tail = ".".join(parts[2:])
    return tail in {"ffn.fc1.weight", "ffn.fc2.weight", "attn.qkv.weight", "attn.proj.weight"}


def _is_excluded(weight_name: str) -> bool:
    if "input_adapter" in weight_name or ".head." in weight_name:
        return True
    if any(token in weight_name for token in (".down.", ".to_vit.", ".up.")):
        return True
    return weight_name.startswith("blocks.39.")


def merge_adapter(base: dict[str, torch.Tensor], network: DLSSNRLoRA) -> dict[str, torch.Tensor]:
    from musubi_tuner.dlssnr.fp8 import canonical_tensor_sha256

    quantization = getattr(network, "base_quantization", None)
    if quantization is not None:
        if any(value.dtype == torch.float8_e4m3fn for value in base.values()):
            raise ValueError("materialize the quantized effective base before merging an FP8 adapter")
        identity = canonical_tensor_sha256((name, base[name]) for name in sorted(base) if ".opaque." not in name)
        if identity != quantization["effective_base_sha256"]:
            raise ValueError("FP8 adapter effective quantized base identity does not match")
    merged = {key: value.detach().clone() for key, value in base.items()}
    for adapter in network.adapters:
        if adapter.target not in merged:
            raise KeyError(f"base is missing LoRA target {adapter.target}")
        merged[adapter.target] = merged[adapter.target] + adapter.delta_weight().detach().to(merged[adapter.target])
    return merged


def merge_to_directory(base_dir: str | Path, adapter_path: str | Path, output_dir: str | Path) -> None:
    """Write a canonical directory whose weights are base + adapter. The base file is not modified."""
    from safetensors import safe_open
    from musubi_tuner.dlssnr.artifacts import inspect_canonical, save_canonical
    from musubi_tuner.dlssnr.fp8 import materialize_state_dict, quantize_frozen_base

    base = Path(base_dir)
    output = Path(output_dir)
    if output.resolve() == base.resolve():
        raise ValueError("refusing to overwrite the source canonical directory")
    inspect_canonical(base, development_smoke=True)
    model = NRModel().to(dtype=torch.float32)
    model.load_canonical(str(base / "model.safetensors"))
    with safe_open(str(adapter_path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    policy, quantization = _read_adapter_runtime(metadata)
    report = json.loads(metadata.get("report") or "{}")
    network = _network_from_report(model, report)
    identity = base_target_sha256(model, network.target_names)
    network.runtime_policy = policy
    if quantization is not None:
        model.requires_grad_(False)
        network.base_quantization = quantize_frozen_base(model, scaled=quantization["scaled"])
    load_adapter(network, adapter_path, identity)
    merged = merge_adapter(materialize_state_dict(model), network)
    materialized = NRModel().to(dtype=torch.float32)
    materialized.load_state_dict(merged, strict=True)
    output_policy = {**policy, "fp8_base": False, "fp8_scaled": False} if policy is not None else None
    save_canonical(
        materialized,
        output,
        source_dir=base,
        metadata={
            "experimental_surrogate": True,
            "base_weight_sha256": identity,
            "adapter": str(adapter_path),
            "runtime_policy": output_policy,
            "training_runtime_policy": policy,
            "base_quantization": quantization,
            "fp8_materialized": quantization is not None,
        },
    )
    (output / "merge_report.json").write_text(
        json.dumps(
            {
                "base_profile": PROFILE_ID,
                "base_weight_sha256": identity,
                "adapter": str(adapter_path),
                "targets": network.target_names,
                "runtime_policy": output_policy,
                "base_quantization": quantization,
                "fp8_materialized": quantization is not None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _network_from_report(model: nn.Module, report: dict) -> DLSSNRLoRA:
    modules = {f"{name}.weight": module for name, module in model.named_modules() if isinstance(module, ChannelLinear)}
    network = DLSSNRLoRA()
    network.profile = report.get("profile", "")
    network.report = report
    for target in report.get("targets", []):
        module = modules.get(target["name"])
        if module is None:
            raise KeyError(f"base has no linear {target['name']}")
        network.add(target["name"], module, int(target["rank"]), float(target["alpha"]), 0.0)
    return network


def save_adapter(network: DLSSNRLoRA, path: str | Path, base_sha256: str) -> None:
    from safetensors.torch import save_file
    from musubi_tuner.dlssnr.runtime import default_runtime_policy

    policy = getattr(network, "runtime_policy", None) or default_runtime_policy()
    quantization = getattr(network, "base_quantization", None)
    _validate_adapter_runtime(policy, quantization, base_sha256)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.detach().to(dtype=torch.float32).contiguous().cpu() for key, value in network.state_dict().items()}
    save_file(
        tensors,
        str(path),
        metadata={
            "schema": LORA_SCHEMA,
            "profile": network.profile,
            "base_profile": PROFILE_ID,
            "base_weight_sha256": base_sha256,
            "base_identity_scope": "all_canonical_parameters",
            "forward_mode": LORA_FORWARD,
            "adapter_config": json.dumps(
                [{"target": item.target, "rank": item.rank, "alpha": item.alpha} for item in network.adapters]
            ),
            "qkv_mode": "fused_head_major",
            "targets": json.dumps(network.target_names),
            "report": json.dumps(network.report),
            "base_quantization": json.dumps(quantization),
            "runtime_policy": json.dumps(policy),
        },
    )


def load_adapter(network: DLSSNRLoRA, path: str | Path, expected_base_sha256: str | None = None) -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file
    from musubi_tuner.dlssnr.runtime import default_runtime_policy

    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    policy, quantization = _read_adapter_runtime(metadata)
    if metadata.get("base_profile") != PROFILE_ID:
        raise ValueError(f"LoRA base profile {metadata.get('base_profile')} != {PROFILE_ID}")
    if metadata.get("forward_mode") != LORA_FORWARD:
        raise ValueError("LoRA forward mode does not match the canonical-weight implementation")
    if json.loads(metadata.get("targets", "[]")) != network.target_names:
        raise ValueError("LoRA target map does not match this network")
    settings = [{"target": item.target, "rank": item.rank, "alpha": item.alpha} for item in network.adapters]
    if json.loads(metadata.get("adapter_config", "null")) != settings:
        raise ValueError("LoRA rank/alpha configuration does not match this network")
    if expected_base_sha256 is not None and metadata.get("base_identity_scope") != "all_canonical_parameters":
        raise ValueError("LoRA base identity does not cover the complete model")
    if expected_base_sha256 is not None and metadata.get("base_weight_sha256") != expected_base_sha256:
        raise ValueError("LoRA base identity does not match the loaded checkpoint")
    if quantization != getattr(network, "base_quantization", None):
        raise ValueError("LoRA effective base quantization does not match the loaded base")
    if policy != (getattr(network, "runtime_policy", None) or default_runtime_policy()):
        raise ValueError("LoRA runtime policy does not match this network")
    tensors = load_file(str(path))
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("adapter weights must be finite FP32")
    network.load_state_dict(tensors, strict=True)


def _validate_adapter_runtime(policy, quantization, base_sha256):
    from musubi_tuner.dlssnr.runtime import validate_runtime_policy

    validate_runtime_policy(policy)
    if policy["fp8_base"] != (quantization is not None):
        raise ValueError("adapter runtime policy and base quantization disagree")
    if quantization is not None:
        if (
            not isinstance(quantization, dict)
            or quantization.get("schema") != "dlssnr_fp8_base_v1"
            or quantization.get("dtype") != "float8_e4m3fn"
            or type(quantization.get("scaled")) is not bool
            or quantization["scaled"] != policy["fp8_scaled"]
            or quantization.get("source_base_sha256") != base_sha256
        ):
            raise ValueError("invalid adapter base quantization recipe or source identity")


def _read_adapter_runtime(metadata):
    from musubi_tuner.dlssnr.runtime import default_runtime_policy

    schema = metadata.get("schema")
    if schema == LEGACY_LORA_SCHEMA:
        if (
            json.loads(metadata.get("runtime_policy", "null")) is not None
            or json.loads(metadata.get("base_quantization", "null")) is not None
        ):
            raise ValueError("legacy LoRA schema cannot encode an experimental runtime or quantization recipe")
        return default_runtime_policy(), None
    if schema != LORA_SCHEMA:
        raise ValueError(f"unsupported LoRA schema {schema}")
    if "runtime_policy" not in metadata or "base_quantization" not in metadata:
        raise ValueError("v2 adapter is missing required runtime policy or base quantization metadata")
    policy, quantization = json.loads(metadata["runtime_policy"]), json.loads(metadata["base_quantization"])
    _validate_adapter_runtime(policy, quantization, metadata.get("base_weight_sha256"))
    return policy, quantization


def base_target_sha256(model: nn.Module, target_names: list[str]) -> str:
    """Identify the complete base, including frozen layers outside the target map."""
    from musubi_tuner.dlssnr.fp8 import canonical_tensor_sha256, iter_canonical_tensors

    names = set(dict(model.named_parameters())) | {
        f"{name}.weight" for name, module in model.named_modules() if isinstance(module, ChannelLinear)
    }
    if missing := set(target_names) - names:
        raise ValueError(f"unknown base targets: {sorted(missing)}")
    return canonical_tensor_sha256(iter_canonical_tensors(model))
