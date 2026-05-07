import torch
from unittest.mock import MagicMock


def test_sample_prompt_defaults():
    from musubi_tuner.training.sampling import SamplePrompt
    p = SamplePrompt(prompt="test", width=512, height=512)
    assert p.prompt == "test"
    assert p.width == 512
    assert p.height == 512
    assert p.frame_count == 1
    assert p.sample_steps == 20
    assert p.seed is None
    assert p.ctx_vec is None
    assert p.image_path is None
    assert p.control_video_path is None
    assert p.control_image_path is None


def test_sample_prompt_all_fields():
    from musubi_tuner.training.sampling import SamplePrompt
    tensor = torch.zeros(1, 77, 768)
    p = SamplePrompt(
        prompt="cat",
        width=256,
        height=256,
        frame_count=5,
        sample_steps=10,
        seed=42,
        guidance_scale=3.5,
        discrete_flow_shift=1.0,
        cfg_scale=7.0,
        negative_prompt="ugly",
        enum=1,
        ctx_vec=tensor,
        negative_ctx_vec=tensor,
        image_path="/tmp/img.png",
        control_video_path="/tmp/vid.mp4",
        control_image_path=["/tmp/a.png"],
    )
    assert p.seed == 42
    assert p.frame_count == 5
    assert p.ctx_vec is tensor
    assert p.image_path == "/tmp/img.png"
    assert p.control_image_path == ["/tmp/a.png"]


def test_sampling_context_fields():
    from musubi_tuner.training.sampling import SamplingContext, SamplePrompt
    mock_accel = MagicMock()
    mock_args = MagicMock()
    mock_vae = MagicMock()
    mock_transformer = MagicMock()
    mock_network = MagicMock()
    prompts = [SamplePrompt(prompt="x", width=64, height=64)]

    ctx = SamplingContext(
        accelerator=mock_accel,
        args=mock_args,
        epoch=1,
        steps=100,
        vae=mock_vae,
        transformer=mock_transformer,
        network=mock_network,
        sample_prompts=prompts,
        dit_dtype=torch.bfloat16,
    )
    assert ctx.steps == 100
    assert ctx.epoch == 1
    assert ctx.dit_dtype == torch.bfloat16
    assert len(ctx.sample_prompts) == 1


def test_sampling_context_epoch_none():
    from musubi_tuner.training.sampling import SamplingContext, SamplePrompt
    ctx = SamplingContext(
        accelerator=MagicMock(),
        args=MagicMock(),
        epoch=None,
        steps=0,
        vae=MagicMock(),
        transformer=MagicMock(),
        network=MagicMock(),
        sample_prompts=[],
        dit_dtype=torch.float16,
    )
    assert ctx.epoch is None


def _make_trainer():
    """Import and instantiate a minimal concrete subclass for hook testing."""
    from musubi_tuner.training.trainer_base import NetworkTrainer
    from musubi_tuner.training.sampling import SamplingContext, SamplePrompt

    class ConcreteTrainer(NetworkTrainer):
        def process_sample_prompts(self, args, accelerator, sample_prompts):
            return []
        def do_inference(self, *a, **kw):
            raise NotImplementedError

    return ConcreteTrainer(), SamplingContext(
        accelerator=MagicMock(),
        args=MagicMock(),
        epoch=None,
        steps=0,
        vae=MagicMock(),
        transformer=MagicMock(),
        network=MagicMock(),
        sample_prompts=[],
        dit_dtype=torch.bfloat16,
    )


def test_on_before_sample_images_default_returns_ctx():
    trainer, ctx = _make_trainer()
    returned = trainer.on_before_sample_images(ctx)
    assert returned is ctx


def test_on_after_sample_images_default_is_noop():
    trainer, ctx = _make_trainer()
    trainer.on_after_sample_images(ctx)  # should not raise


def test_on_before_sample_images_can_be_overridden():
    from musubi_tuner.training.trainer_base import NetworkTrainer
    from musubi_tuner.training.sampling import SamplingContext, SamplePrompt

    class PatchingTrainer(NetworkTrainer):
        def process_sample_prompts(self, args, accelerator, sample_prompts):
            return []
        def do_inference(self, *a, **kw):
            raise NotImplementedError
        def on_before_sample_images(self, ctx):
            ctx.steps = 999
            return ctx

    trainer = PatchingTrainer()
    ctx = SamplingContext(
        accelerator=MagicMock(), args=MagicMock(), epoch=None, steps=0,
        vae=MagicMock(), transformer=MagicMock(), network=MagicMock(),
        sample_prompts=[], dit_dtype=torch.bfloat16,
    )
    result = trainer.on_before_sample_images(ctx)
    assert result.steps == 999
