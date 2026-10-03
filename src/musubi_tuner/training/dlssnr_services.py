"""NR runtime services, without diffusion trainer imports."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import DistributedType

from musubi_tuner.training.optimizer_setup import create_optimizer


def create_nr_optimizer(parameters, config):
    args = SimpleNamespace(
        optimizer_type=config["type"],
        optimizer_args=config["args"],
        learning_rate=config["learning_rate"],
        lr_scheduler=config["lr_scheduler"],
        max_grad_norm=config["max_grad_norm"],
    )
    _, _, optimizer, _, _ = create_optimizer(args, parameters)
    if isinstance(optimizer, torch.optim.LBFGS):
        raise ValueError("closure-based optimizers such as LBFGS are not supported by the NR update loop")
    if callable(getattr(optimizer, "train", None)) or callable(getattr(optimizer, "eval", None)):
        raise ValueError("schedule-free optimizer train/eval weight switching is not implemented for NR")
    if args.learning_rate is None:
        raise ValueError("NR requires an explicit learning rate")
    return optimizer


def create_nr_lr_scheduler(optimizer, config, max_train_steps):
    from musubi_tuner.training.lr_scheduler import create_lr_scheduler

    args = SimpleNamespace(learning_rate=config["learning_rate"], max_train_steps=max_train_steps, **config["scheduler"])
    # NR advances once per successful global update, not once per rank or microbatch.
    return create_lr_scheduler(args, optimizer, num_processes=1)


def reject_unsupported_runtime(precision="no"):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        required = {"RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
        if missing := required - os.environ.keys():
            raise RuntimeError(
                f"incomplete multi-GPU/DDP launch environment: missing {sorted(missing)}; use torchrun or accelerate launch"
            )
        if int(os.environ["LOCAL_WORLD_SIZE"]) != world_size:
            raise RuntimeError("NR DDP currently supports one node only")
    for name in ("ACCELERATE_USE_DEEPSPEED", "ACCELERATE_USE_FSDP", "ACCELERATE_USE_MEGATRON_LM"):
        if os.environ.get(name, "false").lower() in ("1", "true"):
            raise RuntimeError("NR supports DDP only, not FSDP/DeepSpeed/Megatron")
    requested = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
    if requested not in ("no", "none", "", precision):
        raise RuntimeError("Accelerate mixed precision conflicts with NR --mixed_precision; pass matching explicit options")


def create_accelerator(training, precision=None):
    mixed_precision = (precision or {}).get("mixed_precision", "no")
    reject_unsupported_runtime(mixed_precision)
    if training["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    handlers = [DistributedDataParallelKwargs(broadcast_buffers=False, gradient_as_bucket_view=True)]
    # A launcher initializes a process group even for a single worker.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or int(os.environ.get("LOCAL_RANK", "-1")) >= 0:
        handlers.append(
            InitProcessGroupKwargs(
                backend="gloo" if os.name == "nt" or training["device"] == "cpu" or not torch.cuda.is_available() else "nccl",
                init_method="env://?use_libuv=False" if os.name == "nt" else None,
                timeout=timedelta(minutes=5),
            )
        )
    accelerator = Accelerator(
        cpu=training["device"] == "cpu",
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        kwargs_handlers=handlers,
    )
    if accelerator.distributed_type not in (DistributedType.NO, DistributedType.MULTI_CPU, DistributedType.MULTI_GPU):
        raise RuntimeError("NR supports DDP only")
    if accelerator.mixed_precision != mixed_precision:
        raise RuntimeError("NR runtime does not match the requested mixed precision")
    if accelerator.device.type not in ("cpu", "cuda"):
        raise RuntimeError(f"unsupported NR device {accelerator.device}")
    if training["device"] != "auto" and accelerator.device.type != training["device"]:
        raise RuntimeError(f"requested {training['device']}, but Accelerate selected {accelerator.device}")
    if mixed_precision != "no" and accelerator.device.type != "cuda":
        raise RuntimeError("NR mixed precision currently requires CUDA")
    return accelerator


def gather_rank_values(accelerator, value):
    if accelerator is None or accelerator.num_processes == 1:
        return [value]
    gathered = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def coordinated_call(accelerator, operation, *, main_only=False):
    """Propagate local/setup or main-process I/O failures before the next collective."""
    result, error = None, None
    if not main_only or accelerator is None or accelerator.is_main_process:
        try:
            result = operation()
        except Exception as caught:
            error = caught
    if accelerator is None or accelerator.num_processes == 1:
        if error is not None:
            raise error
        return result
    errors = gather_rank_values(accelerator, None if error is None else f"{type(error).__name__}: {error}")
    if any(message is not None for message in errors):
        failures = "; ".join(f"rank {rank}: {message}" for rank, message in enumerate(errors) if message is not None)
        raise RuntimeError(f"NR distributed operation failed: {failures}") from error
    return result


def reduce_values(accelerator, values, *, max_keys=()):
    if accelerator.num_processes == 1:
        return dict(values)
    result = {}
    for use_max in (False, True):
        names = sorted(name for name in values if (name in max_keys) == use_max)
        if not names:
            continue
        tensor = torch.tensor([values[name] for name in names], dtype=torch.float64, device=accelerator.device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX if use_max else torch.distributed.ReduceOp.SUM)
        result.update(zip(names, tensor.tolist()))
    return result


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def assert_finite_gradients(module):
    if not gradients_are_finite(module):
        raise RuntimeError("non-finite gradient; optimizer update aborted")


def gradients_are_finite(module):
    parameters = module.parameters() if isinstance(module, torch.nn.Module) else module
    gradients = [parameter.grad for parameter in parameters if parameter.requires_grad and parameter.grad is not None]
    if not gradients:
        raise RuntimeError("no trainable gradients were produced")
    return bool(torch.stack([torch.isfinite(value).all() for value in gradients]).all())


def optimizer_update(accelerator, module, optimizer, max_grad_norm):
    """Return False only for a recoverable FP16 overflow; never step invalid gradients."""
    accelerator.unscale_gradients(optimizer)
    finite = coordinated_call(accelerator, lambda: gradients_are_finite(module))
    if accelerator.num_processes > 1:
        agreed = torch.tensor(int(finite), device=accelerator.device)
        torch.distributed.all_reduce(agreed, op=torch.distributed.ReduceOp.MIN)
        finite = bool(agreed)
    if not finite:
        scaler = accelerator.scaler
        if scaler is None:
            raise RuntimeError("non-finite gradient; optimizer update aborted")
        scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
        state = scaler.state_dict()
        state["_growth_tracker"] = 0
        scaler.load_state_dict(state)
        return False
    if max_grad_norm:
        # Gradients are already unscaled; Accelerate.clip_grad_norm_ would unscale twice.
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    if accelerator.optimizer_step_was_skipped:
        raise RuntimeError("optimizer unexpectedly skipped finite unscaled gradients")
    return True


def assert_finite_parameters(module):
    values = [torch.isfinite(parameter).all() for parameter in module.parameters() if parameter.requires_grad]
    if values and not torch.stack(values).all():
        raise RuntimeError("non-finite parameter update; checkpoint aborted")


def capture_rng(device=None):
    numpy = np.random.get_state()
    active = None
    if torch.cuda.is_initialized() and (device is None or torch.device(device).type == "cuda"):
        active = torch.cuda.current_device() if device is None or torch.device(device).index is None else torch.device(device).index
    return {
        "torch": torch.get_rng_state(),
        "cuda": [torch.cuda.get_rng_state(active)] if active is not None else [],
        "cuda_device": active,
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        active = state.get("cuda_device")
        if not torch.cuda.is_available() or len(state["cuda"]) != 1 or active != torch.cuda.current_device():
            raise ValueError("resume CUDA RNG devices do not match")
        torch.cuda.set_rng_state(state["cuda"][0].cpu(), active)
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4]))


@contextmanager
def evaluation_mode(module):
    modes = [(child, child.training) for child in module.modules()]
    rng = capture_rng()
    try:
        module.eval()
        with torch.no_grad():
            yield
    finally:
        for child, mode in modes:
            child.training = mode
        restore_rng(rng)
