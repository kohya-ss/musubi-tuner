import pytest
from safetensors.torch import save_file

from musubi_tuner.mage_flow.model import MageFlow
from musubi_tuner.mage_flow.utils import MageFlowConfig
from musubi_tuner.networks import lora_mage_flow


def _tiny_model(depth: int = 2) -> MageFlow:
    config = MageFlowConfig(
        in_channels=4,
        out_channels=4,
        context_in_dim=6,
        hidden_size=16,
        depth=depth,
        num_heads=2,
        axes_dim=(2, 2, 4),
        text_max_length=16,
    )
    return MageFlow(config)


def test_lora_targets_only_attention_and_image_text_mlps():
    network = lora_mage_flow.create_arch_network(1.0, 2, 1.0, None, [], _tiny_model())
    target_names = [module.lora_name.removeprefix("lora_unet_") for module in network.unet_loras]

    assert len(target_names) == 24
    assert all(
        name.startswith(("transformer_blocks_0_attn_", "transformer_blocks_0_img_mlp_", "transformer_blocks_0_txt_mlp_"))
        or name.startswith(("transformer_blocks_1_attn_", "transformer_blocks_1_img_mlp_", "transformer_blocks_1_txt_mlp_"))
        for name in target_names
    )
    assert not any("_mod_" in name for name in target_names)
    assert not any(name.startswith(("img_in", "txt_in", "time_text_embed", "proj_out")) for name in target_names)


def test_user_patterns_cannot_expand_mage_flow_lora_scope():
    network = lora_mage_flow.create_arch_network(
        1.0,
        2,
        1.0,
        None,
        [],
        _tiny_model(depth=1),
        include_patterns="['.*']",
        exclude_patterns="[]",
    )

    assert len(network.unet_loras) == 12
    assert all("transformer_blocks_0_" in module.lora_name for module in network.unet_loras)


def test_loading_rejects_lora_weights_outside_fixed_scope():
    weights = {
        "lora_unet_img_in.lora_down.weight": pytest.importorskip("torch").zeros(2, 4),
        "lora_unet_img_in.lora_up.weight": pytest.importorskip("torch").zeros(16, 2),
        "lora_unet_img_in.alpha": pytest.importorskip("torch").tensor(2.0),
    }

    with pytest.raises(ValueError, match="outside the supported Mage-Flow scope"):
        lora_mage_flow.create_arch_network_from_weights(1.0, weights, unet=_tiny_model())


def test_loading_rejects_unknown_module_even_when_name_looks_in_scope():
    torch = pytest.importorskip("torch")
    weights = {
        "lora_unet_transformer_blocks_99_attn_to_q.lora_down.weight": torch.zeros(2, 16),
        "lora_unet_transformer_blocks_99_attn_to_q.lora_up.weight": torch.zeros(16, 2),
        "lora_unet_transformer_blocks_99_attn_to_q.alpha": torch.tensor(2.0),
    }

    with pytest.raises(ValueError, match="do not map"):
        lora_mage_flow.create_arch_network_from_weights(1.0, weights, unet=_tiny_model())


def test_adapter_architecture_metadata_requires_explicit_cross_mode_override(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "edit.safetensors"
    save_file({"weight": torch.zeros(1)}, path, metadata={"ss_base_model_version": "mage_flow_edit"})

    with pytest.raises(ValueError, match="architecture mismatch"):
        lora_mage_flow.validate_adapter_architecture(path, expected="mage_flow", allow_mismatch=False)

    lora_mage_flow.validate_adapter_architecture(path, expected="mage_flow_edit", allow_mismatch=False)
    lora_mage_flow.validate_adapter_architecture(path, expected="mage_flow", allow_mismatch=True)
