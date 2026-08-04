import pytest
import torch

from musubi_tuner.mage_flow.sampling import (
    predict_target_velocity,
    resolve_output_size,
    scheduler_step_targets_only,
)
from musubi_tuner.mage_flow.utils import pack_training_batch


class _TextDrivenVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_packed = None

    def forward(self, packed):
        self.last_packed = packed
        output = torch.empty_like(packed.image_tokens)
        image_cu = packed.image_cu_seqlens.tolist()
        text_cu = packed.text_cu_seqlens.tolist()
        for index, (image_start, image_end) in enumerate(zip(image_cu, image_cu[1:])):
            value = packed.text_tokens[0, text_cu[index] : text_cu[index + 1]].mean()
            output[0, image_start:image_end] = value
        return output


@pytest.mark.parametrize("reference_count", [1, 3])
def test_scheduler_step_changes_only_edit_target(reference_count):
    target = torch.full((1, 2, 2, 2), 5.0)
    references = [torch.full((1, 2, 1, index + 1), 10.0 + index) for index in range(reference_count)]
    packed = pack_training_batch(target, [torch.zeros(1, 3)], torch.tensor([1.0]), references)
    before = packed.image_tokens.clone()
    velocity = torch.full_like(before, 3.0)

    after = scheduler_step_targets_only(
        before,
        velocity,
        packed.target_token_mask,
        sigma=1.0,
        next_sigma=0.0,
    )

    torch.testing.assert_close(after[0, packed.target_token_mask], before[0, packed.target_token_mask] - 3.0)
    torch.testing.assert_close(after[0, ~packed.target_token_mask], before[0, ~packed.target_token_mask])


def test_cfg_fuses_positive_and_negative_with_independent_boundaries():
    model = _TextDrivenVelocity()
    target = [torch.zeros(2, 2, 2), torch.zeros(2, 1, 3)]
    references = [[torch.ones(2, 1, 1)], [torch.ones(2, 1, 2)]]
    positive = [torch.full((2, 4), 3.0), torch.full((1, 4), 4.0)]
    negative = [torch.full((1, 4), 1.0), torch.full((3, 4), 2.0)]

    velocity = predict_target_velocity(
        model,
        target,
        positive,
        sigma=0.5,
        controls=references,
        negative_text_tokens=negative,
        cfg_scale=2.0,
    )

    assert model.last_packed.batch_size == 4
    assert model.last_packed.image_shapes == [
        [(1, 2, 2), (1, 1, 1)],
        [(1, 1, 3), (1, 1, 2)],
        [(1, 2, 2), (1, 1, 1)],
        [(1, 1, 3), (1, 1, 2)],
    ]
    torch.testing.assert_close(velocity[0], torch.full_like(target[0], 5.0))
    torch.testing.assert_close(velocity[1], torch.full_like(target[1], 6.0))


@pytest.mark.parametrize(
    ("source", "width", "height", "max_size", "expected"),
    [
        ((641, 479), 512, 768, None, (512, 768)),
        ((1200, 600), None, None, 1024, (1024, 512)),
        ((641, 479), None, None, None, (640, 464)),
    ],
)
def test_edit_output_size_precedence_and_alignment(source, width, height, max_size, expected):
    assert resolve_output_size(source, width=width, height=height, max_size=max_size) == expected


def test_output_size_requires_width_and_height_together():
    with pytest.raises(ValueError, match="together"):
        resolve_output_size((640, 480), width=512, height=None, max_size=None)
