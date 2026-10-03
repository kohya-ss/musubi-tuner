import numpy as np
import pytest
import toml
import torch
from PIL import Image

from musubi_tuner.dlssnr.config import build_train_config
from musubi_tuner.dlssnr.dataset import NRBatchPlan
from musubi_tuner.training.dlssnr_parser import setup_parser
from musubi_tuner.training.dlssnr_trainer import _datasets


def paired_directories(root, name="pair", count=2):
    source, target = root / name / "source", root / name / "target"
    source.mkdir(parents=True)
    target.mkdir()
    for index in range(count):
        Image.fromarray(np.full((48, 64, 3), 40 + index, dtype=np.uint8)).save(source / f"frame{index}.png")
        Image.fromarray(np.full((48, 64, 3), 180 + index, dtype=np.uint8)).save(target / f"frame{index}.png")
    return {"image_directory": str(target), "control_directory": str(source)}


def configuration(root, datasets, **general):
    path = root / "dataset.toml"
    path.write_text(
        toml.dumps({"general": {"resolution": [64, 48], "batch_size": 1, **general}, "datasets": datasets}), encoding="utf-8"
    )
    args = setup_parser().parse_args(
        [
            "--dataset_config",
            str(path),
            "--development_smoke",
            "--output_dir",
            str(root / "output"),
            "--output_name",
            "test",
        ]
    )
    return build_train_config(args)


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, [0, 1, 1, 1, 1]),
        (
            {"nr_style": 2, "nr_tone": 0.5, "nr_structure": 0.25, "nr_skin": 0.75, "nr_auto_mask": True},
            [2 / 128, 0.5, 1, 0.75, 0.25],
        ),
        ({"nr_structure": 0.25, "nr_skin": -1, "nr_auto_mask": True}, [0, 1, 1, 0.25, 0.25]),
        ({"nr_style": 1, "nr_tone": 0.5, "nr_structure": 0.25, "nr_auto_mask": False}, [1 / 128, 0.5, 0.25, -1, -1]),
    ],
)
def test_directory_pairs_generate_fixed_five_lane_conditions(tmp_path, settings, expected):
    entry = {**paired_directories(tmp_path), "nr_controls_mode": "fixed", **settings}
    dataset, validation = _datasets(configuration(tmp_path, [entry], caption_extension=".txt"))
    sample = dataset[0]
    assert not validation
    assert sample["source"].mean() == pytest.approx(40 / 255)
    assert sample["target"].mean() == pytest.approx(180 / 255)
    torch.testing.assert_close(sample["controls"], torch.tensor(expected, dtype=torch.float32)[:, None, None].expand(5, 48, 64))
    assert not list(tmp_path.rglob("*.npy"))
    assert not list(tmp_path.rglob("*.jsonl"))


def test_multiple_datasets_keep_conditions_repeats_and_batch_sizes(tmp_path):
    first = {**paired_directories(tmp_path, "a", 3), "batch_size": 2, "nr_tone": 0.25}
    second = {**paired_directories(tmp_path, "b", 2), "batch_size": 1, "num_repeats": 2, "nr_tone": 0.75}
    config = configuration(tmp_path, [first, second])
    dataset, _ = _datasets(config)
    plan = NRBatchPlan(dataset, config["training"]["batch_size"])
    assert len(dataset) == 7
    assert [len(plan.indices(index)) for index in range(len(plan))] == [2, 1, 1, 1, 1, 1]
    assert plan.sample_count(len(plan)) == 7
    assert len({dataset[index]["sample_id"] for index in range(len(dataset))}) == 7
    assert [float(dataset[index]["controls"][1, 0, 0]) for index in range(len(dataset))] == [0.25] * 3 + [0.75] * 4


@pytest.mark.parametrize("problem", ["missing", "ambiguous", "misaligned"])
def test_directory_pairing_rejects_bad_pairs(tmp_path, problem):
    entry = paired_directories(tmp_path, count=1)
    source = tmp_path / "pair/source/frame0.png"
    if problem == "missing":
        source.unlink()
    elif problem == "ambiguous":
        Image.open(source).save(tmp_path / "pair/source/frame0_1.png")
    else:
        Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(source)
    with pytest.raises(ValueError, match="pair|match|control|grid"):
        _datasets(configuration(tmp_path, [entry]))


@pytest.mark.parametrize(
    "settings",
    [{"nr_auto_mask": "false"}, {"nr_tone": -0.1}, {"nr_skin": -0.5}, {"nr_structure": float("nan")}, {"num_repeats": 0}],
)
def test_invalid_dataset_conditions_are_rejected(tmp_path, settings):
    with pytest.raises(ValueError):
        configuration(tmp_path, [{**paired_directories(tmp_path), **settings}])


def test_condition_changes_change_dataset_identity_but_comments_do_not(tmp_path):
    entry = paired_directories(tmp_path)
    first, _ = _datasets(configuration(tmp_path, [entry]))
    fingerprint = first.fingerprint()
    second, _ = _datasets(configuration(tmp_path, [{**entry, "nr_auto_mask": False}]))
    assert second.fingerprint() != fingerprint
    third, _ = _datasets(configuration(tmp_path, [entry]))
    assert third.fingerprint() == fingerprint
