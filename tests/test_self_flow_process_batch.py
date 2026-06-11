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
    # Mismatch metrics must always be present (even when 0) — gappy wandb series
    # where 0.0 and absent are indistinguishable breaks monitoring.
    assert "self_flow/mismatch_patch_frac" in trainer._self_flow_logs, (
        "self_flow/mismatch_patch_frac must be logged unconditionally (0.0 when no mismatch)"
    )
    assert "self_flow/mismatch_patch_count" in trainer._self_flow_logs, (
        "self_flow/mismatch_patch_count must be logged unconditionally (0 when no mismatch)"
    )

    loss.backward()
    grads = [p.grad for p in trainer.rep_proj.parameters()]
    assert all(g is not None for g in grads)  # L_rep reached the projection head


def test_process_batch_restores_student_weights_after_teacher_swap(tiny_model):
    """After the teacher forward, process_batch must restore the student LoRA weights.

    Catches a skipped or misordered ``load_state_dict(student_lora_state)`` restore.
    We pre-load a distinct EMA state (5.0) so the post-call weight value is
    unambiguous: if the student restore is skipped, net.lora_w will read 5.0, not 1.0.
    """
    torch.manual_seed(2)
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

    # Override EMA state to a distinct sentinel so we can tell student from teacher
    trainer.ema_lora_state["lora_w"] = torch.full((4,), 5.0)
    # Record current student value (ones from StubNetwork.__init__)
    expected_student_val = net.lora_w.detach().clone()

    B, H, W = 2, 4, 4
    latents = torch.randn(B, 4, H, W)
    noise = torch.randn_like(latents)
    batch = {
        "ctx_vec": torch.randn(B, 3, 8),
        "timesteps": None,
    }
    scheduler = make_noise_scheduler(args)

    trainer.process_batch(
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
        global_step=5,
    )

    # Student weights must be restored; if the restore is skipped, we'd see 5.0
    assert torch.allclose(net.lora_w.detach(), expected_student_val), (
        f"net.lora_w after process_batch should be {expected_student_val.tolist()} (student), "
        f"got {net.lora_w.detach().tolist()} — teacher-swap restore likely skipped or misordered"
    )


def test_process_batch_vanilla_fallthrough(tiny_model):
    """Base path works without any self-flow state; returns finite loss + empty metrics."""
    torch.manual_seed(1)
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(self_flow=False)
    # NOTE: with self_flow=False, on_transformer_loaded / on_train_start must NOT
    # be called — the base path must not require any self-flow state.
    acc = PreparingAccelerator()
    net = StubNetwork()

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
        global_step=0,
    )
    assert loss.ndim == 0 and torch.isfinite(loss), f"expected finite scalar loss, got {loss}"
    assert metrics == {}, f"base compute_loss should return empty dict, got {metrics}"
