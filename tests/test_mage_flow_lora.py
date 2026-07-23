import pytest
from safetensors.torch import load_file
from safetensors.torch import save_file
import torch

from musubi_tuner.mage_flow.model import MageFlow
from musubi_tuner.mage_flow.utils import MageFlowConfig, pack_training_batch
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
    path = tmp_path / "edit.safetensors"
    save_file({"weight": torch.zeros(1)}, path, metadata={"ss_base_model_version": "mage_flow_edit"})

    with pytest.raises(ValueError, match="architecture mismatch"):
        lora_mage_flow.validate_adapter_architecture(path, expected="mage_flow", allow_mismatch=False)

    lora_mage_flow.validate_adapter_architecture(path, expected="mage_flow_edit", allow_mismatch=False)
    lora_mage_flow.validate_adapter_architecture(path, expected="mage_flow", allow_mismatch=True)


def test_one_lora_step_and_safetensors_round_trip(tmp_path):
    torch.manual_seed(123)
    model = _tiny_model(depth=1).eval().requires_grad_(False)
    base_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    packed = pack_training_batch(
        torch.randn(1, 4, 2, 2),
        [torch.randn(3, 6)],
        torch.tensor([0.4]),
    )
    network = lora_mage_flow.create_arch_network(1.0, 2, 2.0, None, [], model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.5)

    before = model(packed).detach()
    loss = (model(packed) - torch.ones_like(before)).square().mean()
    loss.backward()
    optimizer.step()
    after = model(packed).detach()

    assert not torch.equal(before, after)
    path = tmp_path / "adapter.safetensors"
    network.save_weights(path, torch.float32, {"ss_base_model_version": "mage_flow"})
    weights = load_file(path)

    reloaded_model = _tiny_model(depth=1).eval()
    reloaded_model.load_state_dict(base_state, strict=True)
    reloaded_model.requires_grad_(False)
    reloaded_network = lora_mage_flow.create_arch_network_from_weights(
        1.0,
        weights,
        unet=reloaded_model,
    )
    reloaded_network.apply_to(None, reloaded_model, apply_text_encoder=False, apply_unet=True)
    info = reloaded_network.load_state_dict(weights, strict=True)

    assert not info.missing_keys
    assert not info.unexpected_keys
    torch.testing.assert_close(reloaded_model(packed), after)
