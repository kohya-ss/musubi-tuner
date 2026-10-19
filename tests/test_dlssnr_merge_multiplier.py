import pytest
import torch

from musubi_tuner.dlssnr.model import ChannelLinear
from musubi_tuner.networks.lora_dlssnr import DLSSNRLoRA, merge_adapter


@pytest.mark.parametrize("multiplier,expected", [(0, 3.0), (0.5, 4.0), (1, 5.0), (-1, 1.0)])
def test_merge_multiplier_scales_only_lora_delta(multiplier, expected):
    layer = ChannelLinear(2, 2)
    network = DLSSNRLoRA()
    network.add("weight", layer, 1, 1, 0)
    with torch.no_grad():
        network.adapters[0].lora_down.fill_(1)
        network.adapters[0].lora_up.fill_(2)
    base = {"weight": torch.full((2, 2), 3.0), "untouched": torch.tensor([-0.0])}
    result = merge_adapter(base, network, multiplier=multiplier)
    assert torch.all(result["weight"] == expected)
    assert torch.all(base["weight"] == 3)
    assert torch.signbit(result["untouched"]).all()


@pytest.mark.parametrize("multiplier", [float("nan"), float("inf")])
def test_merge_rejects_nonfinite_multiplier(multiplier):
    with pytest.raises(ValueError, match="finite"):
        merge_adapter({}, DLSSNRLoRA(), multiplier=multiplier)
