"""Pluggable loss functions for network trainers.

``--loss_fn`` selects the loss the same way ``--optimizer_type`` selects the
optimizer: a built-in name (no dot) or a dotted import path. ``--loss_fn_args``
supplies ``key=value`` construction arguments, mirroring ``--optimizer_args``.
Values must be Python literals (strings must be quoted); non-literals raise at
startup with a helpful error message.

The resolved callable is invoked with a single :class:`LossContext`::

    loss_fn(ctx)  # ctx.args, ctx.output, ctx.timesteps, ctx.noise_scheduler,
                  # ctx.dit_dtype, ctx.network_dtype, ctx.global_step

and returns either a bare scalar loss tensor or ``(loss, metrics_dict)``.
Losses needing more than ``output.pred``/``output.target`` read from
``output.extra``; trainers that support such losses are responsible for
stashing the required tensors there (e.g. a ``call_dit`` override stashing
``noisy_model_input`` for x0 recovery).
"""

import argparse
import ast
import functools
import importlib
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3

logger = logging.getLogger(__name__)


@dataclass
class LossContext:
    """Everything a pluggable ``--loss_fn`` callable receives, as one object.

    Additive by design: new fields may be appended in future versions without
    breaking existing third-party losses (unlike a positional signature).
    ``output`` is the trainer's ``DiTOutput`` (``.pred`` / ``.target`` /
    ``.extra``) — typed ``Any`` because ``trainer_base`` imports this module,
    so importing ``DiTOutput`` here would be circular. ``output.extra`` is the
    trainer→loss escape hatch for tensors only specific trainers provide
    (e.g. ``noisy_model_input`` for x0 recovery); the fields here are values
    every trainer can supply.
    """

    args: argparse.Namespace
    output: Any
    timesteps: torch.Tensor
    noise_scheduler: Any
    dit_dtype: torch.dtype
    network_dtype: torch.dtype
    global_step: int


def mse_loss(ctx: LossContext) -> tuple[torch.Tensor, dict[str, float]]:
    """Default weighted MSE: SD3-style ``args.weighting_scheme``, then mean."""
    weighting = compute_loss_weighting_for_sd3(
        ctx.args.weighting_scheme, ctx.noise_scheduler, ctx.timesteps, ctx.timesteps.device, ctx.dit_dtype
    )
    loss = torch.nn.functional.mse_loss(ctx.output.pred.to(ctx.network_dtype), ctx.output.target, reduction="none")
    if weighting is not None:
        loss = loss * weighting
    return loss.mean(), {}


BUILTIN_LOSS_FNS: dict[str, Callable] = {
    "mse": mse_loss,
}


def parse_loss_fn_args(loss_fn_args: Optional[list[str]]) -> dict[str, Any]:
    """Parse ``key=value`` strings into kwargs via ``ast.literal_eval``.

    Values must be Python literals (numbers, bools, quoted strings,
    dicts/lists/tuples), matching ``--optimizer_args`` semantics. Loss
    functions are third-party code with unknown schemas, so a non-literal
    value raises at startup with a quoting hint instead of silently falling
    back to a raw string (which would mask typos like ``alpha=0.1.``).
    """
    kwargs: dict[str, Any] = {}
    if not loss_fn_args:
        return kwargs
    for arg in loss_fn_args:
        if "=" not in arg:
            raise ValueError(f"--loss_fn_args entry {arg!r} is not in key=value form")
        key, value = arg.split("=", 1)
        try:
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(
                f"--loss_fn_args entry {arg!r}: value {value!r} is not a Python literal."
                f" If you meant a string, quote it: \"{key}='{value}'\""
            ) from None
    return kwargs


def resolve_loss_fn(loss_fn: str, loss_fn_args: Optional[list[str]] = None) -> Callable:
    """Resolve ``--loss_fn`` to a callable invoked as ``loss_fn(ctx: LossContext)``.

    No dot: look up ``BUILTIN_LOSS_FNS``. Dotted path: import the module and
    fetch the attribute (same pattern as ``get_optimizer``). Classes are
    instantiated with the parsed kwargs; plain functions get them bound via
    ``functools.partial``.
    """
    kwargs = parse_loss_fn_args(loss_fn_args)
    logger.info(f"use loss function {loss_fn} | {kwargs}")

    if "." not in loss_fn:
        try:
            fn = BUILTIN_LOSS_FNS[loss_fn]
        except KeyError:
            raise ValueError(
                f"unknown built-in --loss_fn {loss_fn!r}; built-ins: {sorted(BUILTIN_LOSS_FNS)}."
                " Use a dotted import path (e.g. my_pkg.losses.MyLoss) for external losses."
            ) from None
    else:
        module_path, _, attr_name = loss_fn.rpartition(".")
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"--loss_fn {loss_fn!r}: could not import module {module_path!r}: {e}") from e
        try:
            fn = getattr(module, attr_name)
        except AttributeError:
            raise ImportError(f"--loss_fn {loss_fn!r}: module {module_path!r} has no attribute {attr_name!r}") from None

    if inspect.isclass(fn):
        return fn(**kwargs)
    if kwargs:
        return functools.partial(fn, **kwargs)
    return fn


def normalize_loss_output(result) -> tuple[torch.Tensor, dict[str, float]]:
    """Accept a bare loss tensor or ``(loss, metrics)``; return ``(loss, dict)``."""
    if isinstance(result, tuple):
        loss, metrics = result
        return loss, dict(metrics)
    return result, {}
