import pytest
import torch

from musubi_tuner.dlssnr import profiles
from musubi_tuner.dlssnr.convert import canonical_config
from musubi_tuner.dlssnr.model import NRModel


CC_PROFILE = "dlss_nr_310_9_16_cc"


def test_cc_layout_covers_the_audited_record_sizes_and_block_order():
    records = profiles.build_records(CC_PROFILE)
    assert len(records) == 77
    assert sorted({record.block for record in records}) == list(range(31))
    assert sum(record.nbytes for record in records) == 113_887_490
    sizes = {record.name: record.nbytes for record in records}
    expected_single = {
        0: 21696,
        1: 20672,
        2: 22720,
        3: 61760,
        4: 69936,
        5: 197184,
        6: 229936,
        7: 689232,
        8: 820288,
        19: 525312,
        22: 820784,
        23: 689232,
        24: 230176,
        25: 197184,
        26: 70048,
        27: 61760,
        28: 22784,
        29: 20672,
        30: 21808,
    }
    for block, size in expected_single.items():
        assert sizes[f"block{block}.layer0.layer"] == size
    for block in (9, 10, 20, 21):
        for layer, size in enumerate((524288, 263168, 917568, 263168)):
            assert sizes[f"block{block}.layer{layer}.layer"] == size
    assert sizes["block10.layer4.layer"] == 524304
    for block in range(11, 19):
        for layer, size in enumerate((4194320, 4196352, 3145856, 2, 1050624)):
            assert sizes[f"block{block}.layer{layer}.layer"] == size
    assert sizes["block30.layer0.blend_scale"] == 2


def test_cc_canonical_layout_uses_cc_stages_phases_and_tensor_names():
    config = canonical_config(profile=CC_PROFILE)
    assert config["profile"] == CC_PROFILE
    blocks = config["blocks"]
    assert [block["phase"] for block in blocks] == [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] + [None] * 9 + [2, 3] * 5 + [1]
    assert blocks[10]["stage"] == "enc512"
    assert blocks[11]["kind"] == "vit"
    assert blocks[19]["kind"] == "transition"
    assert blocks[22]["stage"] == "dec256"
    assert blocks[30]["channels"] == 32
    assert config["opaque_vit_layer3"] == [f"blocks.{block}.opaque.layer3" for block in range(11, 19)]
    tensors = {row["name"]: row for row in config["tensors"]}
    assert tensors["blocks.2.down.weight"]["shape"] == [64, 32]
    assert tensors["blocks.10.to_vit.weight"]["shape"] == [1024, 512]
    assert tensors["blocks.11.attn.qkv.weight"]["shape"] == [3072, 1024]
    assert tensors["blocks.19.proj.weight"]["shape"] == [512, 1024]
    assert tensors["blocks.30.head.rgb.weight"]["shape"] == [3, 32]
    assert "blocks.70.head.rgb.weight" not in tensors


def test_cc_model_constructs_the_canonical_parameter_map_without_ht_blocks():
    with torch.device("meta"):
        model = NRModel(profile=CC_PROFILE)
    assert model.profile == CC_PROFILE
    assert len(model.blocks) == 31
    expected = {row["name"]: tuple(row["shape"]) for row in canonical_config(profile=CC_PROFILE)["tensors"]}
    assert {name: tuple(value.shape) for name, value in model.state_dict().items()} == expected
    assert model.blocks["20"].attn.phase == 2
    model.freeze_single_frame()
    assert not model.blocks["30"].head.logit.weight.requires_grad
    assert not model.blocks["30"].blend_scale.requires_grad
    assert model.blocks["30"].head.rgb.weight.requires_grad


def test_unknown_model_profile_is_rejected_instead_of_using_ht():
    with pytest.raises(ValueError, match="profile"):
        profiles.build_records("unknown")


def test_cc_forward_and_backward_use_the_cc_head_and_all_skip_stages():
    from musubi_tuner.dlssnr.pipeline import forward_frame

    torch.manual_seed(7)
    model = NRModel(profile=CC_PROFILE)
    source = torch.full((1, 3, 48, 48), 0.4)
    controls = torch.zeros(1, 5, 48, 48)
    result = forward_frame(model, source, controls, 42)
    assert result["raw_head"].shape == (1, 4, 48, 48)
    assert torch.isfinite(result["raw_head"]).all()
    result["neural_preclamp"].mean().backward()
    assert model.blocks["30"].head.rgb.weight.grad.abs().sum() > 0
    assert all(parameter.grad is not None for name, parameter in model.named_parameters() if not name.endswith("blend_scale"))
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        temporal = forward_frame(
            model,
            source,
            controls,
            43,
            history=result["next_history"].detach(),
            motion=torch.zeros(1, 2, 48, 48),
        )
    assert temporal["blend_weight"].max() > 0
    assert torch.isfinite(temporal["rendered_proxy"]).all()


@pytest.mark.parametrize("lora_profile", ["vit_only", "multiscale"])
def test_cc_lora_selects_by_cc_block_role_and_width(lora_profile):
    from musubi_tuner.networks.lora_dlssnr import inject

    with torch.device("meta"):
        model = NRModel(profile=CC_PROFILE)
        table = {"profile": lora_profile, "dropout": 0.0}
        if lora_profile == "vit_only":
            table.update(rank=8, alpha=4)
        network = inject(model, table)
    assert network.base_profile == CC_PROFILE
    targets = {item.target: item for item in network.adapters}
    assert "blocks.11.ffn.fc1.weight" in targets
    assert "blocks.18.attn.proj.weight" in targets
    assert not any(name.startswith("blocks.19.") or ".head." in name or ".down." in name for name in targets)
    if lora_profile == "vit_only":
        assert len(targets) == 32
        assert network.elements == 1_048_576
        assert all(11 <= int(name.split(".")[1]) <= 18 for name in targets)
    else:
        assert targets["blocks.3.attn.qkv.weight"].rank == 4
        assert targets["blocks.9.ffn.contract.weight"].rank == 16
        assert targets["blocks.11.ffn.fc1.weight"].rank == 16


def test_cc_qat_uses_cc_native_storage_map():
    from musubi_tuner.dlssnr.runtime import configure_model_runtime, default_runtime_policy, with_native_weight_qat
    from musubi_tuner.dlssnr.weight_quantization import native_storage_kinds

    kinds = native_storage_kinds(CC_PROFILE)
    with torch.device("meta"):
        model = NRModel(profile=CC_PROFILE)
    configure_model_runtime(model, with_native_weight_qat(default_runtime_policy(), True), training=True)
    assert set(kinds) == set(model.state_dict())
    assert model.blocks["11"].ffn.fc1.native_weight_kind == "e4"
    assert model.blocks["30"].head.rgb.native_weight_kind == "f16frag"
    assert kinds["blocks.30.blend_scale"] == "f16"


def test_cc_temporal_optimizer_unfreezes_the_cc_logit_and_blend_parameters():
    from musubi_tuner.training.dlssnr_trainer import build_optimizer

    with torch.device("meta"):
        model = NRModel(profile=CC_PROFILE)
    model.freeze_single_frame()
    optimizer = build_optimizer(model, 1e-4, {}, 0, include_temporal=True)
    groups = {group["name"]: {id(value) for value in group["params"]} for group in optimizer.param_groups}
    assert groups["temporal_head"] == {id(model.blocks["30"].head.logit.weight)}
    assert groups["temporal_blend"] == {id(model.blocks["30"].blend_scale)}
    assert model.blocks["30"].blend_scale.requires_grad
