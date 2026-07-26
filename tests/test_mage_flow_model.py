import torch

from musubi_tuner.mage_flow.layers import MageFlowEmbedRope, MageFlowTransformerBlock
from musubi_tuner.mage_flow.model import MageFlow
from musubi_tuner.mage_flow.utils import MageFlowConfig, pack_training_batch


def tiny_config(checkpoint=False):
    return MageFlowConfig(
        in_channels=4,
        out_channels=4,
        context_in_dim=7,
        hidden_size=16,
        depth=2,
        num_heads=2,
        axes_dim=(2, 2, 4),
        text_max_length=16,
        checkpoint=checkpoint,
    )


def test_tiny_model_packed_batch_preserves_sample_boundaries():
    torch.manual_seed(123)
    model = MageFlow(tiny_config()).eval()
    targets = [torch.randn(4, 2, 2), torch.randn(4, 1, 3)]
    text = [torch.randn(2, 7), torch.randn(3, 7)]
    timesteps = torch.tensor([0.2, 0.8])
    packed = pack_training_batch(targets, text, timesteps, image_dim=4, text_dim=7, text_max_length=16)
    changed = pack_training_batch(
        [targets[0], torch.randn_like(targets[1])],
        [text[0], torch.randn_like(text[1])],
        timesteps,
        image_dim=4,
        text_dim=7,
        text_max_length=16,
    )

    with torch.no_grad():
        batched = model(packed)
        changed_batched = model(changed)

    assert batched.shape == (1, 7, 4)
    torch.testing.assert_close(batched[:, :4], changed_batched[:, :4], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(batched[:, 4:], changed_batched[:, 4:])


def test_edit_reference_tokens_share_the_sample_timestep_modulation():
    torch.manual_seed(321)
    model = MageFlow(tiny_config()).eval()
    target = torch.randn(1, 4, 2, 2)
    reference = [torch.randn(1, 4, 1, 2)]
    text = [torch.randn(2, 7)]
    low = pack_training_batch(
        target,
        text,
        torch.tensor([0.2]),
        controls=reference,
        image_dim=4,
        text_dim=7,
        text_max_length=16,
    )
    high = pack_training_batch(
        target,
        text,
        torch.tensor([0.8]),
        controls=reference,
        image_dim=4,
        text_dim=7,
        text_max_length=16,
    )

    with torch.no_grad():
        low_out = model(low)
        high_out = model(high)

    assert not torch.allclose(low_out[:, :4], high_out[:, :4])
    assert not torch.allclose(low_out[:, 4:], high_out[:, 4:])
    torch.testing.assert_close(low.image_tokens[:, 4:], high.image_tokens[:, 4:])


def test_rope_frame_coordinates_follow_global_packed_image_order():
    rope = MageFlowEmbedRope(theta=10000, axes_dim=(2, 2, 4), scale_rope=True)
    shapes = [[(1, 2, 2), (1, 1, 2)], [(1, 2, 2), (1, 1, 2)]]

    frequencies = rope(shapes, device=torch.device("cpu"))

    assert frequencies.shape == (12, 4)
    torch.testing.assert_close(frequencies[6:10], rope._compute_video_freqs(1, 2, 2, frame_index=2))
    torch.testing.assert_close(frequencies[10:], rope._compute_video_freqs(1, 1, 2, frame_index=3))


def test_rope_frequency_cache_stays_on_requested_device():
    rope = MageFlowEmbedRope(theta=10000, axes_dim=(2, 2, 4), scale_rope=True)
    shapes = [[(1, 2, 2)]]
    rope(shapes, device=torch.device("cpu"))
    device = torch.device("meta")

    frequencies = rope(shapes, device=device)

    assert frequencies.device == device
    assert {cached.device for cached in rope.video_freq_cache.values()} == {device}


def test_block_modulation_repeats_one_embedding_over_complete_sample_segments():
    block = MageFlowTransformerBlock(dim=4, num_attention_heads=1, attention_head_dim=4)
    x = torch.ones(1, 5, 4)
    mod_params = torch.tensor(
        [
            [1.0] * 4 + [0.0] * 4 + [2.0] * 4,
            [3.0] * 4 + [0.0] * 4 + [4.0] * 4,
        ]
    )

    modulated, gates = block._modulate(x, mod_params, torch.tensor([0, 3, 5], dtype=torch.int32))

    torch.testing.assert_close(modulated[0, :3], torch.full((3, 4), 2.0))
    torch.testing.assert_close(modulated[0, 3:], torch.full((2, 4), 4.0))
    torch.testing.assert_close(gates[:3], torch.full((3, 4), 2.0))
    torch.testing.assert_close(gates[3:], torch.full((2, 4), 4.0))


def test_checkpointed_tiny_model_has_finite_backward():
    torch.manual_seed(456)
    model = MageFlow(tiny_config(checkpoint=True)).train()
    packed = pack_training_batch(
        torch.randn(2, 4, 2, 2),
        [torch.randn(2, 7), torch.randn(3, 7)],
        torch.tensor([0.3, 0.7]),
        image_dim=4,
        text_dim=7,
        text_max_length=16,
    )

    loss = model(packed).square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert model.img_in.weight.grad is not None
    assert torch.isfinite(model.img_in.weight.grad).all()
