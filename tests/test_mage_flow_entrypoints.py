from pathlib import Path
import subprocess
import sys

import pytest


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
