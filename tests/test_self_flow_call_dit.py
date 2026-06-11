import pytest
import torch

from musubi_tuner.flux_2_train_network_self_flow import (
    Flux2SelfFlowNetworkTrainer,
    flux2_setup_parser,
    self_flow_setup_parser,
)
from musubi_tuner.hv_train_network import setup_parser_common


class FakeAccelerator:
    device = torch.device("cpu")

    class _NullCtx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def autocast(self):
        return self._NullCtx()

    def unwrap_model(self, m):
        return m


def make_args(**overrides):
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)
    parser = self_flow_setup_parser(parser)
    args = parser.parse_args([])
    args.self_flow = True
    args.gradient_checkpointing = False
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


@pytest.fixture
def trainer_with_model(tiny_model):
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(student_feature_layer=0, teacher_feature_layer=2)
    trainer.handle_model_specific_args(args)
    trainer.on_transformer_loaded(args, FakeAccelerator(), tiny_model)
    return trainer, args, tiny_model


def make_batch(B=2, H=4, W=4, n_txt=3, ctx_dim=8):
    latents = torch.randn(B, 4, H, W)
    noise = torch.randn_like(latents)
    return (
        {
            "ctx_vec": torch.randn(B, n_txt, ctx_dim),
        },
        latents,
        noise,
    )


def test_call_dit_returns_features_and_pred(trainer_with_model):
    trainer, args, model = trainer_with_model
    batch, latents, noise = make_batch()
    timesteps = torch.tensor([500.0, 800.0])
    noisy = latents  # contents irrelevant for shape test

    output = trainer.call_dit(
        args,
        FakeAccelerator(),
        model,
        latents,
        batch,
        noise,
        noisy,
        timesteps,
        torch.float32,
        hidden_features=True,
        feature_layer=0,
    )
    assert output.pred.shape[0] == 2
    assert output.extra["features"] is not None
    assert output.extra["features"].shape == (2, 16, 16)  # (B, H*W, hidden)


def test_call_dit_per_token_map_staged_and_cleared(trainer_with_model):
    trainer, args, model = trainer_with_model
    batch, latents, noise = make_batch()
    timesteps = torch.tensor([500.0, 800.0])
    tau = torch.full((2, 16), 800.0)

    trainer.call_dit(
        args,
        FakeAccelerator(),
        model,
        latents,
        batch,
        noise,
        latents,
        timesteps,
        torch.float32,
        hidden_features=True,
        feature_layer=0,
        per_token_timesteps=tau,
    )
    assert trainer._modulation_controller._tau is None  # cleared after forward


def test_call_dit_control_images_unsupported(trainer_with_model):
    trainer, args, model = trainer_with_model
    batch, latents, noise = make_batch()
    batch["latents_control_0"] = torch.randn(2, 4, 4, 4)
    with pytest.raises(NotImplementedError, match="control"):
        trainer.call_dit(
            args,
            FakeAccelerator(),
            model,
            latents,
            batch,
            noise,
            latents,
            torch.tensor([500.0, 800.0]),
            torch.float32,
            hidden_features=True,
            feature_layer=0,
        )
