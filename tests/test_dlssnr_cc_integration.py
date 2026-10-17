"""Opt-in real-weight CC training, export, and CLI inference checks."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
CC_PROFILE = "dlss_nr_310_9_16_cc"


def run_cli(script, *args):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("lora", [False, True], ids=["full-single-frame", "lora-temporal"])
def test_real_cc_training_merge_export_and_image_video_inference(tmp_path, lora):
    model_dir = os.environ.get("DLSSNR_CC_MODEL_DIR")
    dll = os.environ.get("DLSSNR_CC_DLL_PATH")
    if not model_dir or not dll or not torch.cuda.is_available():
        pytest.skip("requires CUDA, DLSSNR_CC_MODEL_DIR and DLSSNR_CC_DLL_PATH")
    from musubi_tuner.dlssnr.dll import read_weights
    from musubi_tuner.dlssnr.identity import file_sha256

    original_hash = file_sha256(dll)
    Image.fromarray(np.full((48, 48, 3), 128, np.uint8)).save(tmp_path / "source.png")
    Image.fromarray(np.full((48, 48, 3), 115, np.uint8)).save(tmp_path / "target.png")
    controls = np.full((5, 48, 48), 0.5, np.float32)
    controls[0] = 1 / 128
    np.save(tmp_path / "controls.npy", controls)
    np.save(tmp_path / "motion.npy", np.zeros((2, 48, 48), np.float32))
    np.save(tmp_path / "valid.npy", np.ones((1, 48, 48), np.float32))
    frames = [
        {
            "frame_index": index,
            "input_path": "source.png",
            "target_path": "target.png",
            "controls_path": "controls.npy",
            "motion_path": "motion.npy",
            "history_valid_path": "valid.npy",
            "temporal_valid_path": "valid.npy",
            "reset": index == 0,
        }
        for index in range(3)
    ]
    row = {
        "schema": "dlssnr_pairs_v1",
        "sample_id": "cc",
        "sequence_id": "cc",
        "source_encoding": "srgb_proxy",
        "target_encoding": "srgb_proxy",
        "controls_encoding": "dlssnr_lanes_10_14_v1",
        "motion_layout": "chw",
        "frames": frames,
    }
    (tmp_path / "sequence.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "single.jsonl").write_text(json.dumps({**row, "frames": frames[:1]}) + "\n")
    manifest = "sequence.jsonl" if lora else "single.jsonl"
    dataset = tmp_path / "dataset.toml"
    dataset.write_text(f'[general]\nresolution = [48, 48]\n[[datasets]]\ntrain_manifest = "{manifest}"\n')
    arguments = [
        "--model_dir",
        model_dir,
        "--dataset_config",
        dataset,
        "--device",
        "cuda",
        "--max_train_steps",
        "1",
        "--optimizer_type",
        "SGD",
        "--learning_rate",
        "1e-5",
        "--max_grad_norm",
        "1",
        "--loss_edge",
        "0",
        "--native_weight_qat",
        "--ema_decay",
        "0.99",
        "--gradient_checkpointing",
        "--save_state",
        "--save_every_n_steps",
        "0",
        "--output_dir",
        tmp_path / "trained",
        "--output_name",
        "cc",
    ]
    if lora:
        arguments += [
            "--network_dim",
            "2",
            "--network_alpha",
            "2",
            "--training_mode",
            "temporal",
            "--sequence_length",
            "3",
            "--burn_in",
            "1",
            "--tbptt_length",
            "2",
            "--loss_temporal",
            "0.1",
        ]
    run_cli("dlssnr_train_network.py" if lora else "dlssnr_train.py", *arguments)
    run = tmp_path / "trained/cc"
    final = run / "final"
    metadata = json.loads((final / "training_metadata.json").read_text())
    assert metadata["profile"] == CC_PROFILE
    assert metadata["global_update"] == 1
    assert metadata["native_export_validated"] is False
    assert json.loads((final / "run_config.json").read_text())["model"]["profile"] == CC_PROFILE
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 1 and np.isfinite(metrics[0]["loss"])
    if lora:
        adapter = final / "adapter.safetensors"
        with safe_open(str(adapter), framework="pt") as handle:
            assert handle.metadata()["base_profile"] == CC_PROFILE
            targets = json.loads(handle.metadata()["targets"])
            assert len(targets) == 32
            assert all(11 <= int(name.split(".")[1]) <= 18 for name in targets)
            assert any(torch.count_nonzero(handle.get_tensor(name)) for name in handle.keys() if name.endswith("lora_up"))
            assert all(handle.get_tensor(name).dtype == torch.float32 for name in handle.keys())
        final = tmp_path / "merged"
        run_cli("dlssnr_merge_lora.py", "--base_model_dir", model_dir, "--adapter", adapter, "--output_dir", final)
    else:
        for variant in ("", "ema"):
            with safe_open(str(run / "state-step000001" / variant / "model.safetensors"), framework="pt") as handle:
                assert handle.get_tensor("blocks.11.ffn.fc1.weight").dtype == torch.float32
            with safe_open(str(final / variant / "model.safetensors"), framework="pt") as handle:
                assert handle.get_tensor("blocks.11.ffn.fc1.weight").dtype == torch.float8_e4m3fn
    config = json.loads((final / "model_config.json").read_text())
    assert config["profile"] == CC_PROFILE
    assert config["weight_storage"]["precision"] == "native"
    exported = tmp_path / "export.dll"
    run_cli("dlssnr_pack_model.py", "--template_dll", dll, "--model_dir", final, "--output_dll", exported)
    report = json.loads(Path(str(exported) + ".report.json").read_text())
    assert report["profile"] == CC_PROFILE
    assert report["resource_name"] == "WEIGHTS_CC_SMALL_FP8"
    assert report["native_export_validated"] is False
    original, changed = Path(dll).read_bytes(), exported.read_bytes()
    resource = read_weights(original, resource_name="WEIGHTS_CC_SMALL_FP8")
    assert changed[: resource.offset] == original[: resource.offset]
    assert changed[resource.offset + resource.size :] == original[resource.offset + resource.size :]
    for script, option, manifest, subfolder in (
        ("dlssnr_generate_image.py", "--sample_manifest", "single.jsonl", "image"),
        ("dlssnr_generate_video.py", "--sequence_manifest", "sequence.jsonl", "video"),
    ):
        output = tmp_path / subfolder
        run_cli(
            script,
            "--model_dir",
            final,
            option,
            tmp_path / manifest,
            "--output_dir",
            output,
            "--bucket_width",
            "48",
            "--bucket_height",
            "48",
            "--device",
            "cuda",
        )
        images = list(output.rglob("*.png"))
        assert len(images) == (1 if subfolder == "image" else 3)
        assert all(Image.open(path).size == (48, 48) for path in images)
        info = json.loads(next(output.rglob("inference_metadata.json")).read_text())
        assert info["profile"] == CC_PROFILE
        assert info["native_equivalent"] is False
    assert file_sha256(dll) == original_hash
