from pathlib import Path

import pytest
import toml

from musubi_tuner.dlssnr.config import build_train_config, config_sha256, load_dataset_config
from musubi_tuner.training.dlssnr_parser import setup_parser


def make_args(tmp_path, options=(), *, lora=False, dataset=None):
    path = tmp_path / "dataset.toml"
    if dataset is None:
        dataset = {"general": {"resolution": [48, 48]}, "datasets": [{"train_manifest": "data.jsonl"}]}
    path.write_text(toml.dumps(dataset), encoding="utf-8")
    return setup_parser(lora=lora).parse_args(
        [
            "--dataset_config",
            str(path),
            "--output_dir",
            str(tmp_path / "out"),
            "--output_name",
            "test",
            "--development_smoke",
            *options,
        ]
    )


@pytest.mark.parametrize(
    "options",
    [
        ["--profile", "other"],
        ["--numerics_profile", "native_reference"],
        ["--fp8_base"],
        ["--learning_raet", "0.001"],
        ["--mixed_precision", "bf16"],
        ["--learning_rate", "nan"],
        ["--gradient_accumulation_steps", "0"],
        ["--max_train_steps", "-1"],
        ["--tbptt_length", "2"],
        ["--loss_temporal", "0.1"],
        ["--loss_pre", "-1"],
        ["--max_grad_norm", "-1"],
        ["--sample_every_n_steps", "1"],
        ["--lr_scheduler", "not_a_scheduler"],
    ],
)
def test_rejects_invalid_or_unsupported_cli_configuration(tmp_path, options):
    with pytest.raises((ValueError, SystemExit)):
        build_train_config(make_args(tmp_path, options))


@pytest.mark.parametrize(
    "general,dataset",
    [
        ({"batch_size": 0}, {"train_manifest": "data.jsonl"}),
        ({"resolution": [0, 48]}, {"train_manifest": "data.jsonl"}),
        ({"resolution": [48]}, {"train_manifest": "data.jsonl"}),
        ({"learning_rate": 0.1}, {"train_manifest": "data.jsonl"}),
        ({}, {"train_manifest": "data.jsonl", "network_dim": 16}),
        ({}, {"train_manifest": ""}),
    ],
)
def test_dataset_toml_rejects_invalid_or_non_dataset_fields(tmp_path, general, dataset):
    args = make_args(tmp_path, dataset={"general": general, "datasets": [dataset]})
    with pytest.raises(ValueError):
        load_dataset_config(args.dataset_config)


def test_legacy_training_toml_is_rejected_with_migration_guidance(tmp_path):
    args = make_args(tmp_path, dataset={"schema_version": 1, "model": {}, "data": {}, "training": {}})
    with pytest.raises(ValueError, match="command-line"):
        load_dataset_config(args.dataset_config)


def test_multiple_dataset_entries_are_not_silently_ignored(tmp_path):
    args = make_args(tmp_path, dataset={"datasets": [{"train_manifest": "one.jsonl"}, {"train_manifest": "two.jsonl"}]})
    data = load_dataset_config(args.dataset_config)
    assert [Path(entry["train_manifest"]).name for entry in data["datasets"]] == ["one.jsonl", "two.jsonl"]


def test_default_dataset_resolution_and_explicit_override(tmp_path):
    args = make_args(
        tmp_path,
        dataset={
            "datasets": [
                {"train_manifest": "default.jsonl"},
                {"train_manifest": "explicit.jsonl", "resolution": [512, 768]},
            ]
        },
    )
    entries = load_dataset_config(args.dataset_config)["datasets"]
    assert entries[0]["bucket_size"] == [1024, 1024]
    assert entries[1]["bucket_size"] == [512, 768]


@pytest.mark.parametrize("mode", ["single", "temporal"])
def test_dataset_templates_use_1024_bucket_budget(mode):
    path = Path(__file__).resolve().parents[1] / "configs" / f"dlssnr_dataset_{mode}.toml"
    data = load_dataset_config(path)
    assert data["bucket_size"] == [1024, 1024]
    assert data["enable_bucket"] and data["bucket_no_upscale"]


def test_pretrained_weights_are_required_unless_smoke_is_explicit(tmp_path):
    args = make_args(tmp_path)
    args.development_smoke = False
    with pytest.raises(ValueError, match="model_dir"):
        build_train_config(args)
    args.development_smoke = True
    assert build_train_config(args)["training"]["development_smoke"]


def test_explicit_forward_evidence_requires_a_source_model_even_for_smoke(tmp_path):
    args = make_args(tmp_path, ["--forward_validation_report", "forward.json"])
    with pytest.raises(ValueError, match="model_dir"):
        build_train_config(args)


def test_dataset_overrides_general_and_resolves_relative_paths(tmp_path):
    args = make_args(
        tmp_path,
        dataset={
            "general": {"resolution": [512, 512], "batch_size": 1},
            "datasets": [
                {
                    "resolution": 48,
                    "batch_size": 2,
                    "train_manifest": "train.jsonl",
                    "validation_manifest": "validation.jsonl",
                    "sequence_manifest": "sequences.jsonl",
                }
            ],
        },
    )
    config = build_train_config(args)
    assert config["data"]["bucket_size"] == [48, 48]
    assert config["training"]["batch_size"] == 2
    assert Path(config["data"]["train_manifest"]) == tmp_path / "train.jsonl"
    assert Path(config["data"]["validation_manifest"]) == tmp_path / "validation.jsonl"
    assert Path(config["evaluation"]["sequence_manifest"]) == tmp_path / "sequences.jsonl"


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../outside", "nested/run", "nested\\run", "C:run", "CON", "nul.txt", "run.", "run ", "bad\tname"]
)
def test_output_name_must_be_a_portable_directory_name(tmp_path, name):
    with pytest.raises(ValueError, match="output_name"):
        build_train_config(make_args(tmp_path, ["--output_name", name]))


@pytest.mark.parametrize(
    "options",
    [
        ["--network_dropout", "1"],
        ["--network_dim", "0"],
        ["--network_alpha", "inf"],
        ["--network_args", "droput=0.1"],
        ["--network_args", "qkv_mode=split"],
        ["--network_args", "profile=multiscale", "--network_dim", "16"],
    ],
)
def test_lora_rejects_unknown_fields_and_invalid_scaling(tmp_path, options):
    with pytest.raises(ValueError):
        build_train_config(make_args(tmp_path, options, lora=True), lora=True)


def test_multiscale_uses_network_args_width_tables(tmp_path):
    config = build_train_config(make_args(tmp_path, ["--network_args", "profile=multiscale"], lora=True), lora=True)
    assert config["lora"]["rank_by_width"]["32"] == 2
    assert config["lora"]["rank_by_width"]["1024"] == 16
    assert config["lora"]["alpha_by_width"] == config["lora"]["rank_by_width"]


def test_cli_order_and_dataset_comments_do_not_change_effective_identity(tmp_path):
    args = make_args(tmp_path, ["--optimizer_args", "betas=0.9,0.999", "weight_decay=0.0"])
    expected = config_sha256(build_train_config(args))
    args.optimizer_args.reverse()
    args.dataset_config.write_text(args.dataset_config.read_text() + "\n# edited comment\n", encoding="utf-8")
    assert config_sha256(build_train_config(args)) == expected


@pytest.mark.parametrize("options", [["lr=0.1"], ["momentum=not_a_literal"], ["weight_decay=0", "weight_decay=1"]])
def test_invalid_optimizer_args_are_rejected_before_training(tmp_path, options):
    with pytest.raises(ValueError, match="optimizer|learning_rate"):
        build_train_config(make_args(tmp_path, ["--optimizer_args", *options]))
