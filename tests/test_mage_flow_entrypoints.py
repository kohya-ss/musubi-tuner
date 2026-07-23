from pathlib import Path
import subprocess
import sys

import pytest
from safetensors.torch import save_file
import torch

import musubi_tuner.mage_flow_generate_image as generate_module


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "mage_flow_cache_latents.py",
        "mage_flow_cache_text_encoder_outputs.py",
        "mage_flow_train_network.py",
        "mage_flow_generate_image.py",
    ],
)
def test_help_does_not_allocate_models(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_generation_rejects_invalid_steps_before_model_loading(monkeypatch):
    parser = generate_module.setup_parser()
    args = parser.parse_args(
        [
            "--dit",
            "dit.safetensors",
            "--vae",
            "vae.safetensors",
            "--text_encoder",
            "text.safetensors",
            "--prompt",
            "test",
            "--steps",
            "0",
        ]
    )
    monkeypatch.setattr(
        generate_module,
        "load_mage_flow_text_encoder",
        lambda *_args, **_kwargs: pytest.fail("allocated text encoder before argument validation"),
    )

    with pytest.raises(ValueError, match="steps"):
        generate_module.generate(args, parser)


def test_generation_checks_lora_mode_before_model_loading(tmp_path, monkeypatch):
    adapter = tmp_path / "edit.safetensors"
    save_file({"weight": torch.zeros(1)}, adapter, metadata={"ss_base_model_version": "mage_flow_edit"})
    parser = generate_module.setup_parser()
    args = parser.parse_args(
        [
            "--dit",
            "dit.safetensors",
            "--vae",
            "vae.safetensors",
            "--text_encoder",
            "text.safetensors",
            "--prompt",
            "test",
            "--lora_weight",
            str(adapter),
        ]
    )
    monkeypatch.setattr(
        generate_module,
        "load_mage_flow_text_encoder",
        lambda *_args, **_kwargs: pytest.fail("allocated text encoder before LoRA metadata validation"),
    )

    with pytest.raises(ValueError, match="architecture mismatch"):
        generate_module.generate(args, parser)
