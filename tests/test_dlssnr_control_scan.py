"""Control sweeps must compare matched conditions, not different noise draws."""

import csv
import importlib
import json
import random
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from musubi_tuner.dlssnr.runtime import configure_model_runtime, default_runtime_policy
from musubi_tuner.training.dlssnr_services import capture_rng


def scanner():
    return importlib.import_module("musubi_tuner.dlssnr.control_scan")


def config(**kwargs):
    return scanner().build_scan_config(48, 48, **kwargs)


class ControlResponseNR(torch.nn.Module):
    """Cheap conditioning-sensitive math; keep the real preprocessing and I/O."""

    def __init__(self, *, fail_at=None):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.0))
        block = torch.nn.Module()
        block.blend_scale = torch.nn.Parameter(torch.tensor([0.5]))
        self.blocks = torch.nn.ModuleDict({"70": block})
        self.calls = []
        self.fail_at = fail_at

    def forward(self, features, geometry):
        self.calls.append(
            {
                "noise": features[:, :3, :48, :48].clone(),
                "source": features[:, 4:7, :48, :48].clone(),
                "history": features[:, 7:10, :48, :48].clone(),
                "controls": features[0, 10:15, 0, 0].tolist(),
                "grad": torch.is_grad_enabled(),
                "training": self.training,
                "native": getattr(self, "native_weight_qat", False),
            }
        )
        if self.fail_at == len(self.calls):
            torch.rand(1)
            random.random()
            np.random.rand()
            raise RuntimeError("probe failed")
        checker = (torch.arange(features.shape[-1], device=features.device) % 2) * 2 - 1
        raw = 0.08 + 0.2 * features[:, 11:12] + 0.1 * features[:, 14:15] * checker
        raw = raw + 0.01 * features[:, :3]
        return torch.cat([raw * self.gain, torch.zeros_like(raw[:, :1])], dim=1)


def source_manifest(root, *, sample_id="frame", masked=True):
    root.mkdir(parents=True, exist_ok=True)
    source = np.linspace(0.2, 0.7, 3 * 48 * 48, dtype=np.float32).reshape(3, 48, 48)
    np.save(root / "source.npy", source)
    frame = {"frame_index": 17, "input_path": "source.npy", "reset": True}
    if masked:
        mask = np.ones((1, 48, 48), np.float32)
        mask[:, :12] = 0
        mask[:, 12:24] = 0.5
        np.save(root / "mask.npy", mask)
        frame["loss_mask_path"] = "mask.npy"
    manifest = root / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema": "dlssnr_pairs_v1",
                "sample_id": sample_id,
                "sequence_id": "scene",
                "source_encoding": "srgb_proxy",
                "frames": [frame],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_default_grid_encodes_all_nine_controls_without_assuming_zero_is_identity():
    settings = config()
    points = settings["points"]
    assert [(point["requested"]["nr_tone"], point["requested"]["nr_structure"]) for point in points] == [
        (0, 0),
        (0, 0.5),
        (0, 1),
        (0.5, 0),
        (0.5, 0.5),
        (0.5, 1),
        (1, 0),
        (1, 0.5),
        (1, 1),
    ]
    assert points[0]["encoded_lanes"] == [0, 0, 1, 0, 0]
    assert points[-1]["encoded_lanes"] == [0, 1, 1, 1, 1]
    assert len({point["id"] for point in points}) == 9


@pytest.mark.parametrize(
    "auto_mask,skin,expected",
    [
        (True, -1, [0.0234375, 0.5, 1, 1, 1]),
        (True, 0.25, [0.0234375, 0.5, 1, 0.25, 1]),
        (False, -1, [0.0234375, 0.5, 1, -1, -1]),
    ],
)
def test_sweep_uses_the_existing_ui_to_lane_encoding(auto_mask, skin, expected):
    settings = config(tone_values=[0.5], structure_values=[1], style=3, skin=skin, auto_mask=auto_mask)
    assert settings["points"][0]["encoded_lanes"] == expected


@pytest.mark.parametrize(
    "options",
    [
        {"tone_values": []},
        {"structure_values": []},
        {"tone_values": [-0.1]},
        {"structure_values": [1.1]},
        {"tone_values": [float("nan")]},
        {"structure_values": [float("inf")]},
        {"tone_values": [False]},
        {"tone_values": [0.5, 0.5]},
        {"tone_values": [0.5, 0.50001]},
        {"structure_values": [0.5, 0.50001]},
        {"style": -1},
        {"style": 1.5},
        {"skin": -0.5},
        {"auto_mask": "false"},
        {"seed": -1},
        {"seed": True},
        {"lowpass_sigma": 0},
        {"lowpass_sigma": float("nan")},
        {"lowpass_sigma": 33},
        {"compare_native": 1},
    ],
)
def test_invalid_or_encoding_duplicate_scan_settings_fail_early(options):
    with pytest.raises(ValueError):
        config(**options)


@pytest.mark.parametrize("size", [(0, 48), (48, 16), (True, 48)])
def test_unsupported_dimensions_are_rejected(size):
    with pytest.raises(ValueError):
        scanner().build_scan_config(*size)


def outputs(rendered, preclamp=None):
    return {"rendered_proxy": rendered, "neural_preclamp": rendered if preclamp is None else preclamp}


def test_metrics_are_mask_weighted_and_flat_input_ratios_are_null():
    source = torch.full((1, 3, 7, 9), 0.25)
    image = source + 0.125
    alternate = image + 0.0625
    mask = torch.ones(1, 1, 7, 9)
    mask[..., :3] = 0
    mask[..., 3:5] = 0.25
    image[mask.expand_as(image) == 0] = 0.9
    alternate[mask.expand_as(image) == 0] = 0.01
    measured = scanner().response_metrics(source, outputs(image), outputs(alternate), mask, sigma=1)
    assert measured["valid_rgb_values"] == 94.5
    assert measured["rgb_mae_vs_input"] == pytest.approx(0.125)
    assert measured["lowpass_delta_rms"] == pytest.approx(0.125)
    assert measured["highpass_delta_rms"] < 1e-7
    assert measured["noise_rgb_mae"] == pytest.approx(0.0625)
    for sigma in (1, 4):
        assert measured[f"hf_output_to_input_sigma_{sigma}px"] is None
        assert measured[f"noise_hf_rms_sigma_{sigma}px"] < 1e-7


def test_frequency_response_matches_an_independent_two_dimensional_filter():
    source = torch.zeros(1, 3, 9, 13)
    delta = (torch.arange(13) % 2).float()[None, None, None, :].expand_as(source) * 0.2
    mask = torch.ones(1, 1, 9, 13)
    mask[..., 2:5] = 0
    coordinates = torch.arange(-3, 4).float()
    kernel = torch.exp(-0.5 * coordinates.square())
    kernel /= kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None]
    numerator = F.conv2d(F.pad(delta * mask, (3, 3, 3, 3), mode="replicate"), kernel.expand(3, 1, 7, 7), groups=3)
    mass = F.conv2d(F.pad(mask, (3, 3, 3, 3), mode="replicate"), kernel)
    lowpass = numerator / mass
    expected_low = float(((lowpass.square() * mask).sum() / (3 * mask.sum())).sqrt())
    expected_high = float((((delta - lowpass).square() * mask).sum() / (3 * mask.sum())).sqrt())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        measured = scanner().response_metrics(source, outputs(delta), outputs(delta), mask, sigma=1)
    assert measured["lowpass_delta_rms"] == pytest.approx(expected_low, rel=1e-6)
    assert measured["highpass_delta_rms"] == pytest.approx(expected_high, rel=1e-6)
    assert measured["noise_rgb_mae"] == 0


def test_metrics_reject_empty_or_broadcastable_but_wrong_masks():
    image = torch.zeros(1, 3, 7, 9)
    with pytest.raises(ValueError, match="mask"):
        scanner().response_metrics(image, outputs(image), outputs(image), torch.zeros(1, 1, 7, 9), sigma=1)
    with pytest.raises(ValueError, match="mask"):
        scanner().response_metrics(image, outputs(image), outputs(image), torch.ones(1, 1, 1, 9), sigma=1)


@pytest.mark.parametrize("compare_native", [False, True])
def test_scan_writes_source_only_reports_and_reuses_noise_across_all_conditions(tmp_path, compare_native):
    model = ControlResponseNR()
    manifest = source_manifest(tmp_path / "data")
    settings = config(compare_native=compare_native, seed=23)
    before = deepcopy(model.state_dict())
    folder = tmp_path / "scan"
    report = scanner().scan_controls(model, manifest, folder, settings)
    assert json.loads((folder / "scan_report.json").read_text()) == report
    assert report["schema"] == "dlssnr_control_scan_v1"
    assert report["native_equivalent"] is False
    assert len(report["measurements"]) == (18 if compare_native else 9)
    assert len(list(folder.rglob("*.png"))) == (19 if compare_native else 10)
    assert len({row["frame_seed"] for row in report["measurements"]}) == 1
    assert len({row["alternate_seed"] for row in report["measurements"]}) == 1
    assert {row["runtime"] for row in report["measurements"]} == ({"configured", "native"} if compare_native else {"configured"})
    for row in report["measurements"]:
        assert row["valid_rgb_values"] == 4320
        assert row["noise_rgb_mae"] > 0
        assert (folder / row["image"]).is_file()
        assert "target" not in row
    zero = report["measurements"][0]
    assert zero["tone"] == zero["structure"] == 0
    assert zero["rgb_mae_vs_input"] > 0
    for index, call in enumerate(model.calls):
        torch.testing.assert_close(call["noise"], model.calls[index % 2]["noise"], rtol=0, atol=0)
        torch.testing.assert_close(call["source"], model.calls[0]["source"], rtol=0, atol=0)
        torch.testing.assert_close(call["history"], call["source"], rtol=0, atol=0)
        assert not call["training"] and not call["grad"]
        assert call["native"] == (compare_native and index >= 18)
        assert call["controls"] == settings["points"][(index // 2) % 9]["encoded_lanes"]
    assert not torch.equal(model.calls[0]["noise"], model.calls[1]["noise"])
    assert model.training
    assert not hasattr(model, "native_weight_qat")
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    with (folder / "scan_metrics.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(report["measurements"])
    assert float(csv_rows[0]["rgb_mae_vs_input"]) == pytest.approx(zero["rgb_mae_vs_input"])
    assert len(report["data_identity"]) == len(report["model_parameters_sha256"]) == 64
    assert "control_scan.py" in report["implementation"]


def test_native_scan_restores_runtime_and_all_rng_after_forward_failure(tmp_path):
    model = ControlResponseNR(fail_at=3)
    model.blocks["70"].eval()
    policy = {**default_runtime_policy(), "numerics_profile": "train_experimental", "attention_backend": "sdpa"}
    configure_model_runtime(model, policy)
    manifest = source_manifest(tmp_path / "data")
    settings = config(tone_values=[0], structure_values=[0], compare_native=True)
    before = capture_rng()
    modes = [child.training for child in model.modules()]
    with pytest.raises(RuntimeError, match="probe failed"):
        scanner().scan_controls(model, manifest, tmp_path / "scan", settings)
    assert model.runtime_policy == policy
    assert model.native_weight_qat is False
    assert [child.training for child in model.modules()] == modes
    after = capture_rng()
    torch.testing.assert_close(after.pop("torch"), before.pop("torch"), rtol=0, atol=0)
    torch.testing.assert_close(after.pop("cuda"), before.pop("cuda"), rtol=0, atol=0)
    assert after == before
    assert not (tmp_path / "scan/scan_report.json").exists()


def test_existing_output_and_invalid_last_sample_fail_before_any_forward(tmp_path):
    manifest = source_manifest(tmp_path / "data")
    output = tmp_path / "scan"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("unchanged")
    model = ControlResponseNR()
    with pytest.raises(FileExistsError):
        scanner().scan_controls(model, manifest, output, config())
    assert sentinel.read_text() == "unchanged" and not model.calls
    row = json.loads(manifest.read_text())
    second = deepcopy(row)
    second["sample_id"] = "bad-frame"
    np.save(manifest.parent / "empty.npy", np.zeros((1, 48, 48), np.float32))
    second["frames"][0]["loss_mask_path"] = "empty.npy"
    manifest.write_text("\n".join(json.dumps(value) for value in (row, second)))
    with pytest.raises(ValueError, match="mask"):
        scanner().scan_controls(model, manifest, tmp_path / "invalid", config())
    assert not model.calls and not (tmp_path / "invalid").exists()


def test_scan_is_repeatable_and_csv_does_not_turn_sample_ids_into_formulas(tmp_path):
    model = ControlResponseNR()
    manifest = source_manifest(tmp_path / "data", sample_id="=SUM(1,2)")
    settings = config(tone_values=[0.5], structure_values=[1], seed=9)
    first = scanner().scan_controls(model, manifest, tmp_path / "first", settings)
    second = scanner().scan_controls(model, manifest, tmp_path / "second", settings)
    assert first == second
    assert first["measurements"][0]["sample_id"] == "=SUM(1,2)"
    with (tmp_path / "first/scan_metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert next(csv.DictReader(handle))["sample_id"] == "'=SUM(1,2)"
    for path in (tmp_path / "first").rglob("*.png"):
        assert path.read_bytes() == (tmp_path / "second" / path.relative_to(tmp_path / "first")).read_bytes()


def test_scan_overrides_manifest_control_files_and_detects_changed_input_content(tmp_path):
    manifest = source_manifest(tmp_path / "data", masked=False)
    row = json.loads(manifest.read_text())
    row["frames"][0]["controls_path"] = "not-needed.npy"
    manifest.write_text(json.dumps(row))
    settings = config(tone_values=[0.5], structure_values=[0.5])
    first = scanner().scan_controls(ControlResponseNR(), manifest, tmp_path / "first", settings)
    source = np.load(manifest.parent / "source.npy")
    np.save(manifest.parent / "source.npy", source * 0.5)
    second = scanner().scan_controls(ControlResponseNR(), manifest, tmp_path / "second", settings)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["data_identity"] != second["data_identity"]
    assert first["model_parameters_sha256"] == second["model_parameters_sha256"]


def test_scan_rejects_unmerged_lora_and_nonfinite_raw_outputs_before_completion(tmp_path):
    from test_dlssnr_training import SmallNR, small_inject

    model = SmallNR()
    small_inject(model, {})
    manifest = source_manifest(tmp_path / "data")
    with pytest.raises(ValueError, match="merge"):
        scanner().scan_controls(model, manifest, tmp_path / "unmerged", config())
    model = ControlResponseNR()
    model.gain.data.fill_(float("inf"))
    with pytest.raises(RuntimeError, match="non-finite"):
        scanner().scan_controls(model, manifest, tmp_path / "nonfinite", config())
    assert not (tmp_path / "nonfinite/scan_report.json").exists()
    assert model.training


def test_multi_frame_manifest_is_not_silently_treated_as_independent_stills(tmp_path):
    manifest = source_manifest(tmp_path / "data")
    row = json.loads(manifest.read_text())
    row["frames"].append({**row["frames"][0], "frame_index": 18})
    manifest.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="sequence_length"):
        scanner().scan_controls(ControlResponseNR(), manifest, tmp_path / "scan", config())
    assert not (tmp_path / "scan").exists()


def test_modified_inputs_cannot_publish_a_report_with_a_stale_fingerprint(tmp_path):
    manifest = source_manifest(tmp_path / "data")

    class ChangedInput(ControlResponseNR):
        def forward(self, features, geometry):
            result = super().forward(features, geometry)
            if len(self.calls) == 1:
                path = manifest.parent / "source.npy"
                np.save(path, np.load(path) * 0.5)
            return result

    with pytest.raises(RuntimeError, match="input.*changed"):
        scanner().scan_controls(ChangedInput(), manifest, tmp_path / "scan", config(tone_values=[0], structure_values=[0]))
    assert not (tmp_path / "scan/scan_report.json").exists()
