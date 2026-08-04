import os

import pytest
import torch
from PIL import Image

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_MAGE_FLOW,
    ARCHITECTURE_MAGE_FLOW_EDIT,
    ARCHITECTURE_MAGE_FLOW_EDIT_FULL,
    ARCHITECTURE_MAGE_FLOW_FULL,
)
from musubi_tuner.dataset.bucket import BucketSelector
from musubi_tuner.dataset.image_video_dataset import ImageDataset
from musubi_tuner.mage_flow.utils import MageFlowConfig, PackedMageFlowInputs, architecture_for_mode, pack_training_batch


BASE_ARCHITECTURE_STEPS = {
    "hv": 16,
    "wan": 16,
    "fp": 16,
    "fk": 16,
    "f2d": 16,
    "f2k4b": 16,
    "f2k9b": 16,
    "qi": 16,
    "qie": 16,
    "qil": 16,
    "k5": 16,
    "hv15": 16,
    "zi": 16,
    "ho1": 32,
    "i4": 16,
    "kr2": 16,
}


def test_mage_flow_architecture_identity_is_mode_explicit():
    assert (ARCHITECTURE_MAGE_FLOW, ARCHITECTURE_MAGE_FLOW_FULL) == ("mf", "mage_flow")
    assert (ARCHITECTURE_MAGE_FLOW_EDIT, ARCHITECTURE_MAGE_FLOW_EDIT_FULL) == ("mfe", "mage_flow_edit")
    assert architecture_for_mode(False) == ("mf", "mage_flow")
    assert architecture_for_mode(True) == ("mfe", "mage_flow_edit")


def test_bucket_registration_is_additive_and_per_architecture():
    assert {key: BucketSelector.ARCHITECTURE_STEPS_MAP[key] for key in BASE_ARCHITECTURE_STEPS} == BASE_ARCHITECTURE_STEPS
    assert BucketSelector.ARCHITECTURE_STEPS_MAP["mf"] == 16
    assert BucketSelector.ARCHITECTURE_STEPS_MAP["mfe"] == 16
    assert set(BucketSelector.ARCHITECTURE_STEPS_MAP) == set(BASE_ARCHITECTURE_STEPS) | {"mf", "mfe"}


@pytest.mark.parametrize("control_count", [1, 2, 3])
def test_edit_dataset_keeps_one_to_three_ordered_control_images(tmp_path, control_count):
    targets = tmp_path / "targets"
    controls = tmp_path / "controls"
    targets.mkdir()
    controls.mkdir()
    Image.new("RGB", (32, 16)).save(targets / "sample.png")
    (targets / "sample.txt").write_text("caption", encoding="utf-8")
    for index in range(control_count):
        Image.new("RGB", (32, 16), color=(index, 0, 0)).save(controls / f"sample_{index}.png")

    dataset = ImageDataset(
        resolution=(32, 16),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        image_directory=str(targets),
        control_directory=str(controls),
        cache_directory=str(tmp_path / "cache"),
        architecture=ARCHITECTURE_MAGE_FLOW_EDIT,
    )
    assert dataset.datasource.control_count_per_image is None
    assert [os.path.basename(path) for path in next(iter(dataset.datasource.control_paths.values()))] == [
        f"sample_{index}.png" for index in range(control_count)
    ]


def test_released_config_matches_the_pinned_public_architecture():
    config = MageFlowConfig.released()
    assert (
        config.in_channels,
        config.out_channels,
        config.context_in_dim,
        config.hidden_size,
        config.depth,
        config.num_heads,
        config.axes_dim,
        config.patch_size,
        config.text_max_length,
        config.static_shift,
    ) == (128, 128, 2560, 3072, 12, 24, (16, 56, 56), 1, 2048, 6.0)


def test_t2i_pack_accepts_heterogeneous_target_and_text_lengths():
    targets = [
        torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3),
        torch.arange(4 * 1 * 2, dtype=torch.float32).reshape(4, 1, 2),
    ]
    text = [torch.zeros(2, 7), torch.ones(3, 7)]

    packed = pack_training_batch(targets, text, torch.tensor([0.2, 0.8]))

    assert packed.image_tokens.shape == (1, 8, 4)
    assert packed.image_cu_seqlens.dtype == torch.int32
    assert packed.image_cu_seqlens.tolist() == [0, 6, 8]
    assert packed.text_tokens.shape == (1, 5, 7)
    assert packed.text_cu_seqlens.tolist() == [0, 2, 5]
    assert packed.image_shapes == [[(1, 2, 3)], [(1, 1, 2)]]
    assert packed.target_token_mask.tolist() == [True] * 8


def test_edit_pack_keeps_reference_order_and_one_sample_timestep():
    targets = torch.zeros(2, 4, 2, 3)
    references = [
        torch.ones(2, 4, 1, 2),
        torch.full((2, 4, 1, 1), 2.0),
    ]
    text = [torch.zeros(2, 7), torch.zeros(3, 7)]

    packed = pack_training_batch(targets, text, torch.tensor([0.2, 0.8]), controls=references)

    assert packed.image_cu_seqlens.tolist() == [0, 9, 18]
    assert packed.image_shapes == [
        [(1, 2, 3), (1, 1, 2), (1, 1, 1)],
        [(1, 2, 3), (1, 1, 2), (1, 1, 1)],
    ]
    assert packed.target_token_mask.tolist() == [True] * 6 + [False] * 3 + [True] * 6 + [False] * 3
    assert packed.timesteps.tolist() == pytest.approx([0.2, 0.8])
    first_sample = packed.image_tokens[0, :9]
    assert torch.equal(first_sample[6:8], torch.ones(2, 4))
    assert torch.equal(first_sample[8], torch.full((4,), 2.0))


def test_edit_pack_accepts_future_heterogeneous_per_sample_controls():
    targets = [torch.zeros(4, 2, 2), torch.zeros(4, 1, 3)]
    controls = [
        [torch.ones(4, 1, 1)],
        [torch.full((4, 1, 2), 2.0), torch.full((4, 2, 1), 3.0)],
    ]

    packed = pack_training_batch(
        targets,
        [torch.zeros(1, 7), torch.zeros(2, 7)],
        torch.tensor([0.1, 0.9]),
        controls=controls,
    )

    assert packed.image_cu_seqlens.tolist() == [0, 5, 12]
    assert packed.image_shapes == [[(1, 2, 2), (1, 1, 1)], [(1, 1, 3), (1, 1, 2), (1, 2, 1)]]
    assert packed.target_token_mask.tolist() == [True] * 4 + [False] + [True] * 3 + [False] * 4


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("text", [torch.empty(0, 7)], "between 1 and"),
        ("text", [torch.zeros(2, 8)], "text feature"),
        ("timesteps", torch.tensor([0.2, 0.3]), "batch size"),
        ("controls", [[]], "between 1 and 3"),
    ],
)
def test_pack_rejects_contract_mismatches(field, value, match):
    kwargs = {
        "targets": [torch.zeros(4, 2, 2)],
        "text_tokens": [torch.zeros(2, 7)],
        "timesteps": torch.tensor([0.2]),
        "image_dim": 4,
        "text_dim": 7,
    }
    if field == "text":
        kwargs["text_tokens"] = value
    elif field == "timesteps":
        kwargs["timesteps"] = value
    else:
        kwargs["controls"] = value

    with pytest.raises(ValueError, match=match):
        pack_training_batch(**kwargs)


def test_packed_object_rejects_non_finite_values_and_invalid_boundaries():
    packed = PackedMageFlowInputs(
        image_tokens=torch.tensor([[[float("nan"), 0.0]]]),
        image_cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        text_tokens=torch.zeros(1, 1, 3),
        text_cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        image_shapes=[[(1, 1, 1)]],
        timesteps=torch.tensor([0.5]),
        target_token_mask=torch.ones(1, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="finite"):
        packed.validate(image_dim=2, text_dim=3)

    invalid = PackedMageFlowInputs(
        image_tokens=torch.zeros(1, 2, 2),
        image_cu_seqlens=torch.tensor([0, 2, 2], dtype=torch.int32),
        text_tokens=torch.zeros(1, 2, 3),
        text_cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32),
        image_shapes=[[(1, 1, 2)], [(1, 1, 0)]],
        timesteps=torch.tensor([0.5, 0.6]),
        target_token_mask=torch.ones(2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        invalid.validate(image_dim=2, text_dim=3)
