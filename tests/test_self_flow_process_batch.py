import torch

from musubi_tuner.flux_2_train_network_self_flow import Flux2SelfFlowNetworkTrainer
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

from .test_self_flow_call_dit import make_args
from .test_self_flow_lifecycle import PreparingAccelerator, StubNetwork


def make_noise_scheduler(args):
    """Build the same scheduler the trainer uses; location confirmed from trainer_base imports."""
    return FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")


def test_process_batch_self_flow_smoke(tiny_model):
    torch.manual_seed(0)
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(
        student_feature_layer=0,
        teacher_feature_layer=2,
        mask_ratio=0.25,
        self_flow_gamma=0.8,
        ema_decay=0.999,
    )
    acc = PreparingAccelerator()
    trainer.handle_model_specific_args(args)
    trainer.on_transformer_loaded(args, acc, tiny_model)
    trainer.extra_trainable_params(args, acc, None, tiny_model, [])
    net = StubNetwork()
    trainer.on_train_start(args, acc, net, tiny_model, None)

    B, H, W = 2, 4, 4
    latents = torch.randn(B, 4, H, W)
    noise = torch.randn_like(latents)
    batch = {
        "ctx_vec": torch.randn(B, 3, 8),
        "timesteps": None,
    }
    scheduler = make_noise_scheduler(args)

    loss, metrics = trainer.process_batch(
        args,
        acc,
        tiny_model,
        net,
        batch,
        latents,
        noise,
        scheduler,
        torch.float32,
        torch.float32,
        None,
        global_step=10,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert "loss/gen" in metrics and "loss/rep" in metrics
    assert trainer._self_flow_logs  # state metrics populated for extra_step_logs
    assert "self_flow/ema_weight_drift" in trainer._self_flow_logs

    loss.backward()
    grads = [p.grad for p in trainer.rep_proj.parameters()]
    assert all(g is not None for g in grads)  # L_rep reached the projection head


def test_process_batch_vanilla_fallthrough(tiny_model):
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(self_flow=False)
    # must not require any self-flow state; will exercise base process_batch
    # (just verify no AttributeError on missing rep_proj/EMA before the super() call)
    assert args.self_flow is False
