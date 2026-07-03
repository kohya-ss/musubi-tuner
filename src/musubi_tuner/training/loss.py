"""Pluggable loss functions for network trainers.

``--loss_fn`` selects the loss the same way ``--optimizer_type`` selects the
optimizer: a built-in name (no dot) or a dotted import path. ``--loss_fn_args``
supplies ``key=value`` construction arguments, mirroring ``--optimizer_args``.

The resolved callable has the exact signature of
``NetworkTrainer.compute_loss`` (minus ``self``)::

    loss_fn(args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step)

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
from typing import Any, Callable, Optional

import torch

from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3

logger = logging.getLogger(__name__)


def mse_loss(
    args: argparse.Namespace,
    output,
    timesteps: torch.Tensor,
    noise_scheduler,
    dit_dtype: torch.dtype,
    network_dtype: torch.dtype,
    global_step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Default weighted MSE: SD3-style ``args.weighting_scheme``, then mean."""
    weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, noise_scheduler, timesteps, timesteps.device, dit_dtype)
    loss = torch.nn.functional.mse_loss(output.pred.to(network_dtype), output.target, reduction="none")
    if weighting is not None:
        loss = loss * weighting
    return loss.mean(), {}


BUILTIN_LOSS_FNS: dict[str, Callable] = {
    "mse": mse_loss,
}


def parse_loss_fn_args(loss_fn_args: Optional[list[str]]) -> dict[str, Any]:
    """Parse ``key=value`` strings; literal values via ast, else raw string.

    Unlike ``optimizer_args``, non-literal values (``transform=swt``) are kept
    as strings instead of raising, matching ``network_args`` ergonomics.
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
            kwargs[key] = value
    return kwargs


def resolve_loss_fn(loss_fn: str, loss_fn_args: Optional[list[str]] = None) -> Callable:
    """Resolve ``--loss_fn`` to a callable with the ``compute_loss`` signature.

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
