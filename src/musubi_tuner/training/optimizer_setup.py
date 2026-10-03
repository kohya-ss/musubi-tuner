"""Optimizer construction shared by diffusion and supervised trainers."""

import ast
import importlib
import logging
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)


def create_optimizer(
    args, trainable_params: list[torch.nn.Parameter]
) -> tuple[str, str, torch.optim.Optimizer, Callable, Callable]:
    # adamw, adamw8bit, adafactor

    optimizer_type = args.optimizer_type.lower()

    # split optimizer_type and optimizer_args
    optimizer_kwargs = {}
    if args.optimizer_args is not None and len(args.optimizer_args) > 0:
        for arg in args.optimizer_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value

    lr = args.learning_rate
    optimizer = None
    optimizer_class = None

    if optimizer_type.endswith("8bit".lower()):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです")

        if optimizer_type == "AdamW8bit".lower():
            logger.info(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
            optimizer_class = bnb.optim.AdamW8bit
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Adafactor".lower():
        # Adafactor: check relative_step and warmup_init
        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True  # default
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
            logger.info(
                "set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします"
            )
            optimizer_kwargs["relative_step"] = True
        logger.info(f"use Adafactor optimizer | {optimizer_kwargs}")

        if optimizer_kwargs["relative_step"]:
            logger.info("relative_step is true / relative_stepがtrueです")
            if lr != 0.0:
                logger.warning("learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます")
            args.learning_rate = None

            if args.lr_scheduler != "adafactor":
                logger.info("use adafactor_scheduler / スケジューラにadafactor_schedulerを使用します")
            args.lr_scheduler = f"adafactor:{lr}"  # ちょっと微妙だけど

            lr = None
        else:
            if args.max_grad_norm != 0.0:
                logger.warning(
                    "because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません"
                )
            if args.lr_scheduler != "constant_with_warmup":
                logger.warning("constant_with_warmup will be good / スケジューラはconstant_with_warmupが良いかもしれません")
            if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                logger.warning("clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")

        from transformers.optimization import Adafactor

        optimizer_class = Adafactor
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "AdamW".lower():
        logger.info(f"use AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    if optimizer is None:
        # 任意のoptimizerを使う
        case_sensitive_optimizer_type = args.optimizer_type  # not lower
        logger.info(f"use {case_sensitive_optimizer_type} | {optimizer_kwargs}")

        if "." not in case_sensitive_optimizer_type:  # from torch.optim
            optimizer_module = torch.optim
        else:  # from other library
            values = case_sensitive_optimizer_type.split(".")
            optimizer_module = importlib.import_module(".".join(values[:-1]))
            case_sensitive_optimizer_type = values[-1]

        optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    # for logging
    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    # get train and eval functions
    if hasattr(optimizer, "train") and callable(optimizer.train):
        train_fn = optimizer.train
        eval_fn = optimizer.eval
    else:
        train_fn = lambda: None
        eval_fn = lambda: None

    return optimizer_name, optimizer_args, optimizer, train_fn, eval_fn
