import copy
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from musubi_tuner.dlssnr.config import build_train_config, config_sha256, validate_lora
from musubi_tuner.dlssnr.model import ChannelLinear, NRModel
from musubi_tuner.dlssnr.profiles import CC_PROFILE_ID, PROFILE_ID, block_channels
from musubi_tuner.networks.lora_dlssnr import base_target_sha256, inject, load_adapter, save_adapter
from test_dlssnr_config import make_args


WIDTHS = ("32", "64", "128", "256", "512", "1024")


def oversized_table(*, alpha=64):
    table = {"profile": "multiscale", "rank_by_width": {width: 128 for width in WIDTHS}}
    if alpha is not None:
        table["alpha_by_width"] = {width: alpha for width in WIDTHS}
    return table


def test_cli_clamps_multiscale_ranks_and_records_effective_scaling(tmp_path, caplog):
    table = oversized_table()
    args = make_args(
        tmp_path,
        ["--network_args", *[f"{key}={value!r}" for key, value in table.items()]],
        lora=True,
    )
    config = build_train_config(args, lora=True)
    effective = config["lora"]
    assert effective["rank_by_width"] == {"32": 32, "64": 32, "128": 32, "256": 32, "512": 64, "1024": 128}
    assert effective["alpha_by_width"] == {"32": 16, "64": 16, "128": 16, "256": 16, "512": 32, "1024": 64}
    assert len(caplog.records) == 5
    assert all("rank" in record.message and "alpha" in record.message for record in caplog.records)
    before = copy.deepcopy(effective)
    caplog.clear()
    validate_lora(effective)
    assert effective == before
    assert not caplog.records


def test_omitted_alpha_follows_clamped_rank_without_changing_scale():
    table = oversized_table(alpha=None)
    validate_lora(table)
    assert table["rank_by_width"] == {"32": 32, "64": 32, "128": 32, "256": 32, "512": 64, "1024": 128}
    assert table["alpha_by_width"] == {"32": 32, "64": 32, "128": 32, "256": 32, "512": 64, "1024": 128}


@pytest.mark.parametrize("alpha_type", [int, float])
def test_clamped_config_has_same_resume_identity_as_explicit_effective_values(tmp_path, alpha_type):
    requested = oversized_table(alpha=alpha_type(64))
    options = ["--network_args", *[f"{key}={value!r}" for key, value in requested.items()]]
    automatic = build_train_config(make_args(tmp_path, options, lora=True), lora=True)
    effective = {
        "profile": "multiscale",
        "rank_by_width": {"32": 32, "64": 32, "128": 32, "256": 32, "512": 64, "1024": 128},
        "alpha_by_width": {width: alpha_type(value) for width, value in zip(WIDTHS, (16, 16, 16, 16, 32, 64))},
    }
    options = ["--network_args", *[f"{key}={value!r}" for key, value in effective.items()]]
    explicit = build_train_config(make_args(tmp_path, options, lora=True), lora=True)
    assert config_sha256(automatic) == config_sha256(explicit)


@pytest.mark.parametrize(
    ("requested_rank", "effective_ranks"),
    [(49, (32, 32, 32, 32, 49, 49)), (98, (32, 32, 32, 32, 64, 98)), (2050, (32, 32, 32, 32, 64, 1024))],
)
@pytest.mark.parametrize("alpha_factor", [None, 1, 0.5, 2**60 + 1])
def test_non_power_of_two_rank_has_exact_effective_config_identity(tmp_path, requested_rank, effective_ranks, alpha_factor):
    requested = {"profile": "multiscale", "rank_by_width": dict.fromkeys(WIDTHS, requested_rank)}
    effective = {"profile": "multiscale", "rank_by_width": dict(zip(WIDTHS, effective_ranks))}
    if alpha_factor is not None:
        requested["alpha_by_width"] = dict.fromkeys(WIDTHS, requested_rank * alpha_factor)
        effective["alpha_by_width"] = {width: rank * alpha_factor for width, rank in zip(WIDTHS, effective_ranks)}
    options = ["--network_args", *[f"{key}={value!r}" for key, value in requested.items()]]
    automatic = build_train_config(make_args(tmp_path, options, lora=True), lora=True)
    options = ["--network_args", *[f"{key}={value!r}" for key, value in effective.items()]]
    explicit = build_train_config(make_args(tmp_path, options, lora=True), lora=True)
    assert automatic["lora"] == explicit["lora"]
    assert config_sha256(automatic) == config_sha256(explicit)


@pytest.mark.parametrize(("rank", "alpha", "expected"), [(33, 5e-324, 5e-324), (128, 1e308, 2.5e307)])
def test_alpha_clamping_avoids_intermediate_underflow_and_overflow(rank, alpha, expected):
    table = oversized_table(alpha=alpha)
    table["rank_by_width"] = dict.fromkeys(WIDTHS, rank)
    validate_lora(table)
    assert table["alpha_by_width"]["32"] == expected


def test_alpha_clamping_rejects_unrepresentable_positive_result():
    table = oversized_table(alpha=5e-324)
    with pytest.raises(ValueError, match="alpha.*positive"):
        validate_lora(table)


def test_clamping_preserves_legal_groups_and_handles_the_vit_group_limit():
    table = {
        "profile": "multiscale",
        "rank_by_width": {"32": 8, "64": 32, "128": 128, "256": 16, "512": 128, "1024": 2048},
        "alpha_by_width": {"32": 3, "64": 16, "128": 64, "256": 5, "512": 32, "1024": 512},
    }
    validate_lora(table)
    assert table["rank_by_width"] == {"32": 8, "64": 32, "128": 32, "256": 16, "512": 64, "1024": 1024}
    assert table["alpha_by_width"] == {"32": 3, "64": 16, "128": 16, "256": 5, "512": 16, "1024": 256}


@pytest.mark.parametrize("rank", [0, -1, 1.5, True])
def test_invalid_multiscale_ranks_are_not_repaired_by_clamping(rank):
    table = oversized_table()
    table["rank_by_width"]["32"] = rank
    with pytest.raises(ValueError, match="integer"):
        validate_lora(table)


@pytest.mark.parametrize("alpha", [0, -1, float("inf"), float("nan")])
def test_invalid_multiscale_alpha_is_rejected_before_clamping(alpha):
    table = oversized_table(alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        validate_lora(table)


@pytest.mark.parametrize("profile", [PROFILE_ID, CC_PROFILE_ID])
@pytest.mark.parametrize("integer_keys", [False, True])
def test_direct_injection_uses_group_limits_on_ht_and_cc_without_mutating_requests(profile, integer_keys):
    table = oversized_table()
    if integer_keys:
        for field in ("rank_by_width", "alpha_by_width"):
            table[field] = {int(key): value for key, value in table[field].items()}
    requested = copy.deepcopy(table)
    with torch.device("meta"):
        model = NRModel(profile=profile)
        network = inject(model, table)
    expected = {32: 32, 64: 32, 128: 32, 256: 32, 512: 64, 1024: 128}
    assert table == requested
    assert len(network.adapters) == (312 if profile == CC_PROFILE_ID else 996)
    for adapter, target in zip(network.adapters, network.report["targets"]):
        width = block_channels(int(adapter.target.split(".")[1]), profile)
        assert adapter.rank == target["rank"] == expected[width]
        assert adapter.alpha == target["alpha"] == expected[width] / 2
        assert adapter.scale == 0.5
        assert adapter.lora_down.shape == (expected[width], target["in"])
        assert adapter.lora_up.shape == (target["out"], expected[width])
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_clamped_adapter_trains_and_roundtrips_with_effective_rank_and_alpha(tmp_path):
    def make_model():
        model = torch.nn.Module()
        block = torch.nn.Module()
        block.ffn = torch.nn.Module()
        block.ffn.fc1 = ChannelLinear(128, 32)
        torch.nn.init.normal_(block.ffn.fc1.weight, std=0.02)
        model.blocks = torch.nn.ModuleDict({"0": block})
        return model

    model = make_model()
    base = {name: value.clone() for name, value in model.state_dict().items()}
    network = inject(model, oversized_table())
    adapter = network.adapters[0]
    source = torch.randn(2, 32)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
    model.blocks["0"].ffn.fc1(source).square().mean().backward()
    assert torch.isfinite(adapter.lora_up.grad).all()
    assert torch.count_nonzero(adapter.lora_up.grad)
    optimizer.step()
    torch.testing.assert_close(model.state_dict(), base, rtol=0, atol=0)
    assert torch.count_nonzero(adapter.lora_up)
    path = tmp_path / "adapter.safetensors"
    identity = base_target_sha256(model, network.target_names)
    save_adapter(network, path, identity)
    with safe_open(str(path), framework="pt") as handle:
        target = json.loads(handle.metadata()["adapter_config"])[0]
        assert target["rank"] == 32 and target["alpha"] == 16
    fresh = make_model()
    fresh.load_state_dict(base)
    restored = inject(fresh, oversized_table())
    load_adapter(restored, path, identity)
    torch.testing.assert_close(fresh.blocks["0"].ffn.fc1(source), model.blocks["0"].ffn.fc1(source), rtol=0, atol=0)
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
    configuration = json.loads(metadata["adapter_config"])
    configuration[0].update(rank=128, alpha=64)
    metadata["adapter_config"] = json.dumps(configuration)
    corrupt = tmp_path / "wrong-config.safetensors"
    save_file(load_file(path), str(corrupt), metadata=metadata)
    with pytest.raises(ValueError, match="configuration"):
        load_adapter(restored, corrupt, identity)


def test_oversized_multiscale_request_resumes_exactly_with_effective_config(tmp_path, monkeypatch):
    from accelerate.state import AcceleratorState
    from test_dlssnr_training import SmallNR, make_args as make_train_args
    from musubi_tuner.training import dlssnr_trainer as trainer

    class SmallMultiscaleNR(SmallNR):
        def __init__(self, profile=PROFILE_ID):
            super().__init__(profile)
            self.blocks["0"].ffn = torch.nn.Module()
            self.blocks["0"].ffn.fc1 = ChannelLinear(32, 32)
            torch.nn.init.normal_(self.blocks["0"].ffn.fc1.weight, std=0.05)

        def forward(self, features, geometry):
            hidden = self.blocks["0"].input_adapter(features)
            hidden = self.blocks["0"].ffn.fc1(hidden)
            head = self.blocks["70"].head
            return torch.cat((head.rgb(hidden), head.logit(hidden)), dim=1)

    AcceleratorState._reset_state(reset_partial_state=True)
    monkeypatch.setattr(trainer, "NRModel", SmallMultiscaleNR)
    try:
        args = make_train_args(tmp_path, lora=True)
        args.network_dim = args.network_alpha = None
        args.network_args = [f"{key}={value!r}" for key, value in oversized_table().items()]
        trainer.train_lora_from_args(args)
        folder = args.output_dir / args.output_name
        path = folder / "final/adapter.safetensors"
        expected = {name: value.clone() for name, value in load_file(path).items()}
        assert any(torch.count_nonzero(value) for name, value in expected.items() if name.endswith("lora_up"))
        config = json.loads((folder / "final/run_config.json").read_text())
        assert config["lora"]["rank_by_width"]["32"] == 32
        assert config["lora"]["alpha_by_width"]["32"] == 16
        args.resume = folder / "state-step000001"
        trainer.train_lora_from_args(args)
        torch.testing.assert_close(load_file(path), expected, rtol=0, atol=0)
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)


def test_vit_only_keeps_its_strict_rank_validation():
    table = {"profile": "vit_only", "rank": 2048, "alpha": 1024}
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_lora(table)
    with torch.device("meta"):
        with pytest.raises(ValueError, match="outside"):
            inject(NRModel(), table)
