"""Select HT/CC canonical weights or CC packed records through the DLL CLIs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from musubi_tuner.dlssnr.cc_dll_io import MANIFEST_SCHEMA, pack_cc_dll, unpack_cc_dll
from musubi_tuner.dlssnr.dll import has_weight_resource
from musubi_tuner.dlssnr.dll_io import _require_new, _sha256, pack_dll, unpack_dll
from musubi_tuner.dlssnr.identity import file_sha256
from musubi_tuner.dlssnr.profiles import PROFILE_ID, CC_PROFILE_ID


RESOURCES = {"HT": "WEIGHTS_HT", "CC": "WEIGHTS_CC_SMALL_FP8"}
BUNDLE_SCHEMA = "dlssnr_weight_bundle_v1"


def _unpack_selected(source: Path, output: Path, weights: str, cc_format: str = "records", save_precision="native") -> dict:
    if weights == "CC":
        if cc_format == "canonical":
            return unpack_dll(source, output, profile=CC_PROFILE_ID, save_precision=save_precision)
        return unpack_cc_dll(source, output)
    return unpack_dll(source, output, save_precision=save_precision)


def unpack_models(
    source_dll: str | Path, output_dir: str | Path, *, weights: str = "all", cc_format: str = "records", save_precision="native"
) -> dict:
    """A single selection keeps the legacy output root; multiple selections get named children."""
    source, output = Path(source_dll), Path(output_dir)
    _require_new(output)
    if cc_format not in ("records", "canonical"):
        raise ValueError("cc_format must be records or canonical")
    if weights in RESOURCES:
        return _unpack_selected(source, output, weights, cc_format, save_precision)
    if weights != "all":
        raise ValueError("weights must be all, HT or CC")
    data = source.read_bytes()
    selected = [name for name, resource in RESOURCES.items() if has_weight_resource(data, resource)]
    source_hash = _sha256(data)
    del data
    if not selected:
        raise ValueError("DLL has no supported HT or CC weight resource")
    if len(selected) == 1:
        return _unpack_selected(source, output, selected[0], cc_format, save_precision)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = {}
    with tempfile.TemporaryDirectory(prefix=".dlssnr-models-", dir=output.parent) as temporary:
        staged = Path(temporary) / "models"
        staged.mkdir()
        for name in selected:
            report = _unpack_selected(source, staged / name, name, cc_format, save_precision)
            report["output_dir"] = str((output / name).resolve())
            if name == "HT" or cc_format == "canonical":
                (staged / name / "conversion_report.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
                )
            reports[name] = report
        index = {
            "schema": BUNDLE_SCHEMA,
            "source_dll": {"path": str(source.resolve()), "sha256": source_hash},
            "models": {
                name: {
                    "directory": name,
                    "resource_name": RESOURCES[name],
                    "storage": f"canonical_{save_precision}"
                    if name == "HT" or cc_format == "canonical"
                    else "opaque_packed_records",
                }
                for name in selected
            },
        }
        (staged / "models.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        if file_sha256(source) != source_hash:
            raise ValueError("source DLL changed during multi-model unpack")
        _require_new(output)
        staged.rename(output)
    return {"output_dir": str(output.resolve()), "models": reports, "roundtrip": "byte_identical"}


def _select_directory(model: Path, weights: str | None) -> tuple[Path, str]:
    if weights is not None and weights not in RESOURCES:
        raise ValueError("weights must be HT or CC")
    index_path = model / "models.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(index, dict) or index.get("schema") != BUNDLE_SCHEMA:
            raise ValueError("unsupported weight bundle manifest")
        if weights is None:
            raise ValueError("multiple weight outputs: select --weights HT or --weights CC, or pass a model subdirectory")
        models = index.get("models")
        entry = models.get(weights) if isinstance(models, dict) else None
        if not isinstance(entry, dict) or entry.get("directory") != weights:
            raise ValueError(f"weight bundle does not contain a valid {weights} directory")
        selected = (model / weights).resolve()
        if not selected.is_relative_to(model.resolve()):
            raise ValueError("selected model path escapes the weight bundle")
        model = selected
    detected = None
    manifest_path = model / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and manifest.get("schema") == MANIFEST_SCHEMA:
            detected = "CC"
    if (model / "model_config.json").is_file():
        if detected is not None:
            raise ValueError("ambiguous model directory contains both HT canonical and CC record manifests")
        from musubi_tuner.dlssnr.artifacts import read_canonical_profile

        detected = {PROFILE_ID: "HT", CC_PROFILE_ID: "CC"}[read_canonical_profile(model)]
    if detected is not None and weights is not None and detected != weights:
        raise ValueError(f"selected resource {weights} does not match the {detected} model directory")
    return model, weights or detected or "HT"


def pack_selected_model(
    template_dll: str | Path,
    model_dir: str | Path,
    output_dll: str | Path,
    *,
    weights: str | None = None,
    mix: float = 1.0,
    strength: float = 1.0,
    merge_lora: str | Path | None = None,
    lora_multiplier: float = 1.0,
) -> dict:
    """Infer one resource from its model directory, or select one named bundle child."""
    output = Path(output_dll)
    _require_new(output)
    _require_new(Path(str(output) + ".report.json"))
    model, selected = _select_directory(Path(model_dir), weights)
    if selected == "CC" and not (model / "model_config.json").is_file():
        if mix != 1 or strength != 1 or merge_lora is not None or lora_multiplier != 1:
            raise ValueError("CC packed records do not support mix, strength or LoRA options; edit same-size record bytes instead")
        return pack_cc_dll(template_dll, model, output)
    return pack_dll(template_dll, model, output, mix=mix, strength=strength, merge_lora=merge_lora, lora_multiplier=lora_multiplier)
