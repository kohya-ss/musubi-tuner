from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.dlssnr.config import build_train_config
from musubi_tuner.training.dlssnr_parser import setup_parser


@pytest.mark.parametrize(
    "name",
    [
        "constant",
        "constant_with_warmup",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "cosine_with_min_lr",
        "polynomial",
        "inverse_sqrt",
        "warmup_stable_decay",
    ],
)
def test_nr_accepts_shared_scheduler_and_parameters(tmp_path, name):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text('[[datasets]]\ntrain_manifest = "train.jsonl"\n', encoding="utf-8")
    args = setup_parser().parse_args(
        [
            "--dataset_config",
            str(dataset),
            "--development_smoke",
            "--output_dir",
            str(tmp_path),
            "--output_name",
            "test",
            "--max_train_steps",
            "10",
            "--lr_scheduler",
            name,
            "--lr_warmup_steps",
            "0" if name == "constant" else "0.2",
            "--lr_decay_steps",
            "0.3",
            "--lr_scheduler_min_lr_ratio",
            "0.1",
            "--lr_scheduler_timescale",
            "2",
        ]
    )
    config = build_train_config(args)
    from musubi_tuner.training.dlssnr_services import create_nr_lr_scheduler, create_nr_optimizer

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = create_nr_optimizer([parameter], config["optimizer"])
    scheduler = create_nr_lr_scheduler(optimizer, config["optimizer"], 10)
    rates = []
    for _ in range(10):
        rates.append(optimizer.param_groups[0]["lr"])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    assert all(0 <= rate <= args.learning_rate for rate in rates)
    assert scheduler.last_epoch == 10
    if name == "constant":
        assert len(set(rates)) == 1
    else:
        assert len(set(rates)) > 1


def test_shared_factory_preserves_process_scaled_diffusion_schedule():
    from musubi_tuner.training.lr_scheduler import create_lr_scheduler

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    args = SimpleNamespace(
        lr_scheduler="linear",
        max_train_steps=4,
        lr_warmup_steps=0.25,
        lr_decay_steps=0,
        lr_scheduler_num_cycles=1,
        lr_scheduler_power=1,
        lr_scheduler_timescale=None,
        lr_scheduler_min_lr_ratio=None,
        lr_scheduler_args=None,
        lr_scheduler_type="",
        learning_rate=1.0,
    )
    scheduler = create_lr_scheduler(args, optimizer, num_processes=2)
    rates = []
    for _ in range(8):
        rates.append(optimizer.param_groups[0]["lr"])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    assert rates == pytest.approx([0, 0.5, 1, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6])


def test_constant_rejects_warmup_ratio_even_when_it_rounds_to_zero(tmp_path):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text('[[datasets]]\ntrain_manifest = "train.jsonl"\n', encoding="utf-8")
    args = setup_parser().parse_args(
        [
            "--dataset_config",
            str(dataset),
            "--development_smoke",
            "--output_dir",
            str(tmp_path),
            "--output_name",
            "test",
            "--max_train_steps",
            "10",
            "--lr_scheduler",
            "constant",
            "--lr_warmup_steps",
            "0.01",
        ]
    )
    with pytest.raises(ValueError, match="constant.*lr_warmup_steps"):
        build_train_config(args)
