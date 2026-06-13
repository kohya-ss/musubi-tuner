from argparse import Namespace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.ideogram4.sampling_policy import should_use_unconditional_dit_for_lora_sampling


def _args(**overrides):
    values = {
        "unconditional_dit": "uncond.safetensors",
        "use_unconditional_dit_for_lora_sampling": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_lora_sampling_ignores_unconditional_dit_by_default():
    assert should_use_unconditional_dit_for_lora_sampling(_args()) is False


def test_lora_sampling_can_opt_into_unconditional_dit():
    assert should_use_unconditional_dit_for_lora_sampling(_args(use_unconditional_dit_for_lora_sampling=True)) is True


def test_lora_sampling_does_not_use_missing_unconditional_dit():
    assert should_use_unconditional_dit_for_lora_sampling(_args(unconditional_dit=None)) is False
