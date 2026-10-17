"""Epoch permutations are random-access plans, not mutable training RNG state."""

import random

import numpy as np
import pytest
import torch

from musubi_tuner.dlssnr.dataset import NRBatchPlan


class PlanDataset:
    def __init__(self):
        self.bucket_sizes = [(64, 48)] * 9 + [(48, 64)] * 5 + [(64, 48)] * 7
        self.rows = [{"frames": [{}] * (3 if index >= 17 else 1)} for index in range(21)]
        self.group_indices = [0] * 14 + [1] * 7
        self.batch_sizes = [4, 3]

    def __len__(self):
        return len(self.rows)


def epoch_batches(plan, epoch):
    return [plan.indices(epoch * len(plan) + offset) for offset in range(len(plan))]


def test_shuffle_changes_both_batch_order_and_bucket_membership_between_epochs():
    data = PlanDataset()
    plan = NRBatchPlan(data, 4, shuffle=True, seed=42)
    epochs = [epoch_batches(plan, epoch) for epoch in range(4)]
    assert epochs[0] != epochs[1]
    assert {tuple(sorted(batch)) for batch in epochs[0]} != {tuple(sorted(batch)) for batch in epochs[1]}
    for batches in epochs:
        assert sorted(index for batch in batches for index in batch) == list(range(21))
        assert sorted(map(len, batches)) == [1, 1, 1, 3, 3, 4, 4, 4]
        for batch in batches:
            assert len({(data.group_indices[i], data.bucket_sizes[i], len(data.rows[i]["frames"])) for i in batch}) == 1


def test_shuffle_is_seeded_random_access_without_consuming_global_rng():
    data = PlanDataset()
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    first = NRBatchPlan(data, 4, shuffle=True, seed=42)
    sequential = {epoch: epoch_batches(first, epoch) for epoch in range(5)}
    resumed = NRBatchPlan(data, 4, shuffle=True, seed=42)
    for epoch in (4, 0, 3, 1, 4):
        assert epoch_batches(resumed, epoch) == sequential[epoch]
    other_seed = NRBatchPlan(data, 4, shuffle=True, seed=43)
    assert epoch_batches(other_seed, 0) != sequential[0]
    assert random.getstate() == python_rng
    np.testing.assert_equal(np.random.get_state(), numpy_rng)
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)


def test_shuffled_prefix_counts_follow_partial_batches_in_each_epoch():
    plan = NRBatchPlan(PlanDataset(), 4, shuffle=True, seed=47)
    consumed = 0
    lengths = []
    for index in range(4 * len(plan)):
        assert plan.sample_count(index) == consumed
        length = len(plan.indices(index))
        consumed += length
        lengths.append(length)
    assert plan.sample_count(4 * len(plan)) == 84
    assert lengths[: len(plan)] != lengths[len(plan) : 2 * len(plan)]
    epoch = 10**9
    prefix = sum(map(len, epoch_batches(plan, epoch)[:3]))
    assert plan.sample_count(epoch * len(plan) + 3) == 21 * epoch + prefix


def test_report_is_stable_and_indices_do_not_expose_mutable_cached_batches():
    plan = NRBatchPlan(PlanDataset(), 4, shuffle=True, seed=42)
    report = plan.report()
    original = plan.indices(0)
    altered = plan.indices(0)
    altered.clear()
    assert plan.indices(0) == original
    epoch_batches(plan, 57)
    assert plan.report() == report
    assert report["shuffle_seed"] == 42
    assert report["order_sha256_scope"] == "epoch0"
    assert NRBatchPlan(PlanDataset(), 4, shuffle=True, seed=43).report() != report


def test_disabled_shuffle_retains_the_legacy_batch_plan_and_report():
    data = PlanDataset()
    plan = NRBatchPlan(data, 4)
    disabled = NRBatchPlan(data, 4, shuffle=False, seed=99)
    assert plan.report() == disabled.report()
    assert plan.report()["policy"] == "dlssnr_same_bucket_batches_v1"
    expected = [[9, 10, 11, 12], [13], [0, 1, 2, 3], [4, 5, 6, 7], [8], [14, 15, 16], [17, 18, 19], [20]]
    assert epoch_batches(plan, 0) == expected
    assert epoch_batches(disabled, 2) == expected


@pytest.mark.parametrize("options", [{"shuffle": "false"}, {"seed": -1}, {"seed": True}])
def test_shuffle_rejects_ambiguous_seeds_and_flags(options):
    with pytest.raises(ValueError, match="seed|shuffle"):
        NRBatchPlan(PlanDataset(), 4, **options)


@pytest.mark.parametrize("lora", [False, True])
def test_shuffle_cli_defaults_on_without_extra_parameters(tmp_path, lora):
    from musubi_tuner.dlssnr.config import build_train_config
    from test_dlssnr_config import make_args

    automatic = build_train_config(make_args(tmp_path, lora=lora), lora=lora)
    enabled = build_train_config(make_args(tmp_path, ["--shuffle_dataset"], lora=lora), lora=lora)
    disabled = build_train_config(make_args(tmp_path, ["--shuffle_dataset", "--no-shuffle_dataset"], lora=lora), lora=lora)
    assert automatic["training"]["shuffle_dataset"] is True
    assert automatic == enabled
    assert disabled["training"]["shuffle_dataset"] is False
    assert {key: value for key, value in disabled["training"].items() if key != "shuffle_dataset"} == {
        key: value for key, value in automatic["training"].items() if key != "shuffle_dataset"
    }


def test_shuffled_microbatches_keep_sample_noise_seeds_bound_to_epoch(tmp_path):
    from musubi_tuner.dlssnr.config import build_train_config
    from musubi_tuner.dlssnr.temporal import stable_frame_seed
    from musubi_tuner.training.dlssnr_trainer import _datasets, _microbatch
    from test_dlssnr_training import make_args

    config = build_train_config(make_args(tmp_path, batch=2))
    dataset, _ = _datasets(config)
    plan = NRBatchPlan(dataset, 2, shuffle=True, seed=config["training"]["seed"])
    for index in (0, 2, 1):
        batch, seeds = _microbatch(dataset, index, config, plan)
        assert seeds == [stable_frame_seed(4, index // len(plan), f"sample{sample}", 10, 0) for sample in plan.indices(index)]
        assert len(batch["source"]) == 2


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("mode", ["single_frame", "temporal"])
def test_shuffled_runner_resumes_inside_epoch_with_variable_tails(tmp_path, monkeypatch, lora, mode):
    import json
    import toml
    from accelerate.state import AcceleratorState
    from safetensors.torch import load_file
    from musubi_tuner.dlssnr.config import build_train_config
    from musubi_tuner.networks import lora_dlssnr
    from musubi_tuner.training import dlssnr_trainer as trainer
    from test_dlssnr_buckets import write_pairs
    from test_dlssnr_training import SmallNR, make_args, small_inject

    monkeypatch.setattr(trainer, "NRModel", SmallNR)
    monkeypatch.setattr(lora_dlssnr, "inject", small_inject)
    args = make_args(tmp_path, lora=lora, accum=2, batch=2, mode=mode)
    path = write_pairs(tmp_path, [(64, 48)] * 3 + [(48, 64)] * 2, frames=1 if mode == "single_frame" else 3)
    args.dataset_config.write_text(
        toml.dumps(
            {
                "general": {"resolution": 64, "batch_size": 2, "enable_bucket": True, "bucket_no_upscale": True},
                "datasets": [{"train_manifest": path.name}],
            }
        ),
        encoding="utf-8",
    )
    args.max_train_steps = 4
    config = build_train_config(args, lora=lora)
    data, _ = trainer._datasets(config)
    plan = NRBatchPlan(data, 2, shuffle=True, seed=4)
    expected_order = [plan.indices(index) for index in range(8)]
    seen = []
    original = trainer._microbatch

    def observed(dataset, index, config, plan):
        seen.append(plan.indices(index))
        return original(dataset, index, config, plan)

    monkeypatch.setattr(trainer, "_microbatch", observed)
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        train(args)
        assert seen == expected_order
        folder = args.output_dir / args.output_name
        filename = "adapter.safetensors" if lora else "model.safetensors"
        expected = {name: value.clone() for name, value in load_file(folder / "final" / filename).items()}
        saved = torch.load(folder / "state-step000004/trainer_state.pt", weights_only=True)
        metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        assert [row["consumed_samples"] for row in metrics] == [sum(map(len, expected_order[: 2 * step])) for step in range(1, 5)]
        args.resume = folder / "state-step000001"
        seen.clear()
        train(args)
        assert seen == expected_order[2:]
        torch.testing.assert_close(load_file(folder / "final" / filename), expected, rtol=0, atol=0)
        restored = torch.load(folder / "state-step000004/trainer_state.pt", weights_only=True)
        assert restored["consumed_samples"] == saved["consumed_samples"]
        assert restored["scheduler"] == saved["scheduler"]
        assert restored["optimizer"]["param_groups"] == saved["optimizer"]["param_groups"]
        torch.testing.assert_close(restored["optimizer"]["state"], saved["optimizer"]["state"], rtol=0, atol=0)
        torch.testing.assert_close(
            restored["rank_states"][0]["rng"]["torch"], saved["rank_states"][0]["rng"]["torch"], rtol=0, atol=0
        )
        args.shuffle_dataset = False
        with pytest.raises(ValueError, match="identity"):
            train(args)
        args.shuffle_dataset, args.seed = True, 5
        with pytest.raises(ValueError, match="identity"):
            train(args)
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
