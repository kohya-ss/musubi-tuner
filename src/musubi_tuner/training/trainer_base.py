"""NetworkTrainer base class shared by all architecture-specific training scripts.

Architecture-specific methods (load_vae, load_transformer, call_dit,
process_sample_prompts, do_inference, ...) are declared here as abstract hooks
and implemented by subclasses in each *_train_network.py (see e.g.
HunyuanVideoNetworkTrainer in hv_train_network.py, WanNetworkTrainer in
wan_train_network.py, ...).
"""

import ast
import asyncio
import importlib
import argparse
import math
import os
import sys
import random
import time
import json
from dataclasses import dataclass, field
from multiprocessing import Value
from typing import Any, List, Optional
import accelerate
import numpy as np

import huggingface_hub
import toml

import torch
from tqdm import tqdm
from accelerate.utils import set_seed
from accelerate import Accelerator, PartialState
from safetensors.torch import load_file
import transformers
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

from musubi_tuner.dataset import config_utils
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.modules.lr_schedulers import RexLR
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
import musubi_tuner.networks.lora as lora_module
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.hv_generate_video import save_images_grid, save_videos_grid

import logging

from musubi_tuner.utils import huggingface_utils, model_utils, train_utils, sai_model_spec

# Helpers that used to live alongside NetworkTrainer in hv_train_network.py.
# Imported by name so existing method bodies keep working unchanged.
from musubi_tuner.training.accelerator_setup import (
    clean_memory_on_device,
    collator_class,
    prepare_accelerator,
)
from musubi_tuner.training.sampling_prompts import should_sample_images
from musubi_tuner.training.timesteps import (
    compute_density_for_timestep_sampling,
    compute_ideogram4_shift_timestep,
    compute_loss_weighting_for_sd3,
    get_sigmas,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
]


@dataclass
class DiTOutput:
    """Return type for ``NetworkTrainer.call_dit``.

    Internal extension point — no API stability guarantees. Vanilla flow only
    needs ``pred`` and ``target``; extension subclasses can stash arbitrary
    additional outputs (e.g. hidden features for representation-alignment
    losses) in the ``extra`` dict without breaking the base signature.
    """

    pred: torch.Tensor
    target: torch.Tensor
    extra: dict = field(default_factory=dict)


class NetworkTrainer:
    def __init__(self):
        self.blocks_to_swap = None
        self.timestep_range_pool = []
        self.num_timestep_buckets: Optional[int] = None  # for get_bucketed_timestep()
        self.vae_frame_stride = 4  # all architectures require frames to be divisible by 4, except Qwen-Image-Layered
        self.default_discrete_flow_shift = 14.5  # default value for discrete flow shift for all models TODO may be None is better

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
    ):
        network_train_unet_only = True
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if network_train_unet_only else 1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{i}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            # Check prodigyplusschedulefree first: it is handled via optimizer.param_groups below and does not
            # expose `d` on lr_scheduler.optimizers, so it must not fall into the substring "prodigy" path.
            if args.optimizer_type.lower().endswith("prodigyplusschedulefree") and optimizer is not None:
                # tracking d*lr value of unet.
                logs[f"lr/d*lr/{lr_desc}"] = optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]
                if "effective_lr" in optimizer.param_groups[i]:
                    logs[f"lr/d*eff_lr/{lr_desc}"] = optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["effective_lr"]

            elif args.optimizer_type.lower().startswith("dadapt") or "prodigy" in args.optimizer_type.lower():
                # tracking d*lr value (Prodigy, Prodigy_Adv, Prodigy_Lion_Adv, etc.)
                logs[f"lr/d*lr/{lr_desc}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )

        return logs

    def get_optimizer(self, args, trainable_params: list[torch.nn.Parameter]) -> tuple[str, str, torch.optim.Optimizer]:
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

            optimizer_class = transformers.optimization.Adafactor
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

    def is_schedulefree_optimizer(self, optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> bool:
        return args.optimizer_type.lower().endswith("schedulefree".lower())  # or args.optimizer_schedulefree_wrapper

    def get_dummy_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        # dummy scheduler for schedulefree optimizer. supports only empty step(), get_last_lr() and optimizers.
        # this scheduler is used for logging only.
        # this isn't be wrapped by accelerator because of this class is not a subclass of torch.optim.lr_scheduler._LRScheduler
        class DummyScheduler:
            def __init__(self, optimizer: torch.optim.Optimizer):
                self.optimizer = optimizer

            def step(self):
                pass

            def get_last_lr(self):
                return [group["lr"] for group in self.optimizer.param_groups]

        return DummyScheduler(optimizer)

    def get_lr_scheduler(self, args, optimizer: torch.optim.Optimizer, num_processes: int):
        """
        Unified API to get any scheduler from its name.
        """
        # if schedulefree optimizer, return dummy scheduler
        if self.is_schedulefree_optimizer(optimizer, args):
            return self.get_dummy_scheduler(optimizer)

        name = args.lr_scheduler
        num_training_steps = args.max_train_steps * num_processes  # * args.gradient_accumulation_steps
        num_warmup_steps: Optional[int] = (
            int(args.lr_warmup_steps * num_training_steps) if isinstance(args.lr_warmup_steps, float) else args.lr_warmup_steps
        )
        num_decay_steps: Optional[int] = (
            int(args.lr_decay_steps * num_training_steps) if isinstance(args.lr_decay_steps, float) else args.lr_decay_steps
        )
        num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps
        num_cycles = args.lr_scheduler_num_cycles
        power = args.lr_scheduler_power
        timescale = args.lr_scheduler_timescale
        min_lr_ratio = args.lr_scheduler_min_lr_ratio

        lr_scheduler_kwargs = {}  # get custom lr_scheduler kwargs
        if args.lr_scheduler_args is not None and len(args.lr_scheduler_args) > 0:
            for arg in args.lr_scheduler_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                lr_scheduler_kwargs[key] = value

        def wrap_check_needless_num_warmup_steps(return_vals):
            if num_warmup_steps is not None and num_warmup_steps != 0:
                raise ValueError(f"{name} does not require `num_warmup_steps`. Set None or 0.")
            return return_vals

        # using any lr_scheduler from other library
        if args.lr_scheduler_type:
            lr_scheduler_type = args.lr_scheduler_type
            logger.info(f"use {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler")
            if "." not in lr_scheduler_type:  # default to use torch.optim
                lr_scheduler_module = torch.optim.lr_scheduler
            else:
                values = lr_scheduler_type.split(".")
                lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
                lr_scheduler_type = values[-1]
            lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
            lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
            return lr_scheduler

        if name.startswith("adafactor"):
            assert type(optimizer) == transformers.optimization.Adafactor, (
                "adafactor scheduler must be used with Adafactor optimizer / adafactor schedulerはAdafactorオプティマイザと同時に使ってください"
            )
            initial_lr = float(name.split(":")[1])
            # logger.info(f"adafactor scheduler init lr {initial_lr}")
            return wrap_check_needless_num_warmup_steps(transformers.optimization.AdafactorSchedule(optimizer, initial_lr))

        if name.lower() == "rex":
            return RexLR(
                optimizer,
                max_lr=args.learning_rate,
                min_lr=(  # Will start and end with min_lr, use non-zero min_lr by default
                    args.learning_rate * min_lr_ratio if min_lr_ratio is not None else args.learning_rate * 0.01
                ),
                num_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
                **lr_scheduler_kwargs,
            )

        if name == DiffusersSchedulerType.PIECEWISE_CONSTANT.value:
            name = DiffusersSchedulerType(name)
            schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(optimizer, **lr_scheduler_kwargs)  # step_rules and last_epoch are given as kwargs

        name = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

        if name == SchedulerType.CONSTANT:
            return wrap_check_needless_num_warmup_steps(schedule_func(optimizer, **lr_scheduler_kwargs))

        # All other schedulers require `num_warmup_steps`
        if num_warmup_steps is None:
            raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

        if name == SchedulerType.CONSTANT_WITH_WARMUP:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs)

        if name == SchedulerType.INVERSE_SQRT:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, timescale=timescale, **lr_scheduler_kwargs)

        # All other schedulers require `num_training_steps`
        if num_training_steps is None:
            raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

        if name == SchedulerType.COSINE_WITH_RESTARTS:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
                **lr_scheduler_kwargs,
            )

        if name == SchedulerType.POLYNOMIAL:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=power,
                **lr_scheduler_kwargs,
            )

        if name == SchedulerType.COSINE_WITH_MIN_LR:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles / 2,
                min_lr_rate=min_lr_ratio,
                **lr_scheduler_kwargs,
            )

        # these schedulers do not require `num_decay_steps`
        if name == SchedulerType.LINEAR or name == SchedulerType.COSINE:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **lr_scheduler_kwargs,
            )

        # All other schedulers require `num_decay_steps`
        if num_decay_steps is None:
            raise ValueError(f"{name} requires `num_decay_steps`, please provide that argument.")
        if name == SchedulerType.WARMUP_STABLE_DECAY:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_stable_steps=num_stable_steps,
                num_decay_steps=num_decay_steps,
                num_cycles=num_cycles / 2,
                min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else 0.0,
                **lr_scheduler_kwargs,
            )

        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_decay_steps=num_decay_steps,
            **lr_scheduler_kwargs,
        )

    def resume_from_local_or_hf_if_specified(self, accelerator: Accelerator, args: argparse.Namespace) -> bool:
        if not args.resume:
            return False

        if not args.resume_from_huggingface:
            logger.info(f"resume training from local state: {args.resume}")
            accelerator.load_state(args.resume)
            return True

        logger.info(f"resume training from huggingface state: {args.resume}")
        repo_id = args.resume.split("/")[0] + "/" + args.resume.split("/")[1]
        path_in_repo = "/".join(args.resume.split("/")[2:])
        revision = None
        repo_type = None
        if ":" in path_in_repo:
            divided = path_in_repo.split(":")
            if len(divided) == 2:
                path_in_repo, revision = divided
                repo_type = "model"
            else:
                path_in_repo, revision, repo_type = divided
        logger.info(f"Downloading state from huggingface: {repo_id}/{path_in_repo}@{revision}")

        list_files = huggingface_utils.list_dir(
            repo_id=repo_id,
            subfolder=path_in_repo,
            revision=revision,
            token=args.huggingface_token,
            repo_type=repo_type,
        )

        async def download(filename) -> str:
            def task():
                return huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    repo_type=repo_type,
                    token=args.huggingface_token,
                )

            return await asyncio.get_event_loop().run_in_executor(None, task)

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*[download(filename=filename.rfilename) for filename in list_files]))
        if len(results) == 0:
            raise ValueError(
                "No files found in the specified repo id/path/revision / 指定されたリポジトリID/パス/リビジョンにファイルが見つかりませんでした"
            )
        dirname = os.path.dirname(results[0])
        accelerator.load_state(dirname)

        return True

    def get_bucketed_timestep(self) -> float:
        if self.num_timestep_buckets is None or self.num_timestep_buckets <= 1:
            return random.random()

        if len(self.timestep_range_pool) == 0:
            bucket_size = 1.0 / self.num_timestep_buckets
            for i in range(self.num_timestep_buckets):
                self.timestep_range_pool.append((i * bucket_size, (i + 1) * bucket_size))
            random.shuffle(self.timestep_range_pool)

        # print(f"timestep_range_pool: {self.timestep_range_pool}")
        a, b = self.timestep_range_pool.pop()
        return random.uniform(a, b)

    def get_noisy_model_input_and_timesteps(
        self,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: Optional[List[float]],
        noise_scheduler: FlowMatchDiscreteScheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        batch_size = noise.shape[0]

        if timesteps is not None:
            timesteps = torch.tensor(timesteps, device=device)

        # This function converts uniform distribution samples to logistic distribution samples.
        # The final distribution of the samples after shifting significantly differs from the original normal distribution.
        # So we cannot use this.
        # def uniform_to_normal(t_samples: torch.Tensor) -> torch.Tensor:
        #     # Clip small values to prevent log(0)
        #     eps = 1e-7
        #     t_samples = torch.clamp(t_samples, eps, 1.0 - eps)
        #     # Convert to logit space with inverse function
        #     x_samples = torch.log(t_samples / (1.0 - t_samples))
        #     return x_samples

        def uniform_to_normal_ppF(t_uniform: torch.Tensor) -> torch.Tensor:
            """Use `torch.erfinv` to compute the inverse CDF to generate values from a normal distribution."""
            # Clip small values to prevent inf in erfinv
            eps = 1e-7
            t_uniform = torch.clamp(t_uniform, eps, 1.0 - eps)

            # PPF of standard normal distribution: sqrt(2) * erfinv(2q - 1)
            term = 2.0 * t_uniform - 1.0
            x_normal = math.sqrt(2.0) * torch.erfinv(term)
            return x_normal

        def uniform_to_logsnr_ppF_pytorch(t_uniform: torch.Tensor, mean: float, std: float) -> torch.Tensor:
            """Use erfinv to compute the inverse CDF."""
            # Clip small values to prevent inf in erfinv
            eps = 1e-7
            t_uniform = torch.clamp(t_uniform, eps, 1.0 - eps)

            term = 2.0 * t_uniform - 1.0
            logsnr = mean + std * math.sqrt(2.0) * torch.erfinv(term)
            return logsnr

        if (
            args.timestep_sampling == "uniform"
            or args.timestep_sampling == "sigmoid"
            or args.timestep_sampling == "shift"
            or args.timestep_sampling == "flux_shift"
            or args.timestep_sampling == "qwen_shift"
            or args.timestep_sampling == "krea2_shift"
            or args.timestep_sampling == "ideogram4_shift"
            or args.timestep_sampling == "logsnr"
            or args.timestep_sampling == "qinglong_flux"
            or args.timestep_sampling == "qinglong_qwen"
            or args.timestep_sampling == "flux2_shift"
        ):

            def compute_sampling_timesteps(org_timesteps: Optional[torch.Tensor]) -> torch.Tensor:
                def rand(bs: int, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                    nonlocal device
                    return torch.rand((bs,), device=device) if org_ts is None else org_ts

                def randn(bs: int, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                    nonlocal device
                    return uniform_to_normal_ppF(org_ts) if org_ts is not None else torch.randn((bs,), device=device)

                def rand_logsnr(bs: int, mean: float, std: float, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                    nonlocal device
                    logsnr = (
                        uniform_to_logsnr_ppF_pytorch(org_ts, mean, std)
                        if org_ts is not None
                        else torch.normal(mean=mean, std=std, size=(bs,), device=device)
                    )
                    return logsnr

                if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
                    # Simple random t-based noise sampling
                    if args.timestep_sampling == "sigmoid":
                        t = torch.sigmoid(args.sigmoid_scale * randn(batch_size, org_timesteps))
                    else:
                        t = rand(batch_size, org_timesteps)

                elif args.timestep_sampling == "ideogram4_shift":
                    h, w = latents.shape[-2:]
                    t = compute_ideogram4_shift_timestep(rand(batch_size, org_timesteps), h, w)

                elif args.timestep_sampling.endswith("shift"):
                    if args.timestep_sampling == "shift":
                        shift = args.discrete_flow_shift
                    else:
                        h, w = latents.shape[-2:]
                        # we are pre-packed so must adjust for packed size
                        if args.timestep_sampling == "flux_shift":
                            mu = train_utils.get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
                        elif args.timestep_sampling == "flux2_shift":
                            mu = train_utils.get_lin_function(y1=0.5, y2=1.15)(h * w)
                        elif args.timestep_sampling == "qwen_shift":
                            mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)((h // 2) * (w // 2))
                        elif args.timestep_sampling == "krea2_shift":
                            # Matches krea2_sampling.timesteps at inference defaults (minres=256, maxres=1280):
                            # x1=(256//16)**2=256, x2=(1280//16)**2=6400, y1=0.5, y2=1.15.
                            mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=6400, y2=1.15)((h // 2) * (w // 2))
                        # def time_shift(mu: float, sigma: float, t: torch.Tensor):
                        #     return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma) # sigma=1.0
                        shift = math.exp(mu)

                    logits_norm = randn(batch_size, org_timesteps)
                    logits_norm = logits_norm * args.sigmoid_scale  # larger scale for more uniform sampling
                    t = logits_norm.sigmoid()
                    t = (t * shift) / (1 + (shift - 1) * t)

                elif args.timestep_sampling == "logsnr":
                    # https://arxiv.org/abs/2411.14793v3
                    logsnr = rand_logsnr(batch_size, args.logit_mean, args.logit_std, org_timesteps)
                    t = torch.sigmoid(-logsnr / 2)

                elif args.timestep_sampling.startswith("qinglong"):
                    # Qinglong triple hybrid sampling.
                    # First decide which method to use for each sample independently
                    decision_t = torch.rand((batch_size,), device=device)

                    # Flux uses 79% mid_shift, 11% logsnr, 10% logsnr2. Qwen skips the logsnr middle bucket.
                    mid_mask = decision_t < 0.79 if args.timestep_sampling == "qinglong_flux" else decision_t < 0.95
                    logsnr_mask = (decision_t >= 0.79) & (decision_t < 0.9)
                    logsnr_mask2 = decision_t >= 0.9 if args.timestep_sampling == "qinglong_flux" else decision_t >= 0.95

                    # Initialize output tensor
                    t = torch.zeros((batch_size,), device=device)

                    # Generate mid_shift samples for selected indices.
                    if mid_mask.any():
                        mid_count = mid_mask.sum().item()
                        h, w = latents.shape[-2:]
                        if args.timestep_sampling == "qinglong_flux":
                            mu = train_utils.get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
                        elif args.timestep_sampling == "qinglong_qwen":
                            mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)((h // 2) * (w // 2))
                        shift = math.exp(mu)
                        logits_norm_mid = randn(mid_count, org_timesteps[mid_mask] if org_timesteps is not None else None)
                        logits_norm_mid = logits_norm_mid * args.sigmoid_scale
                        t_mid = logits_norm_mid.sigmoid()
                        t_mid = (t_mid * shift) / (1 + (shift - 1) * t_mid)

                        t[mid_mask] = t_mid

                    # Generate logsnr samples for selected indices.
                    if args.timestep_sampling != "qinglong_qwen" and logsnr_mask.any():
                        logsnr_count = logsnr_mask.sum().item()
                        logsnr = rand_logsnr(
                            logsnr_count,
                            args.logit_mean,
                            args.logit_std,
                            org_timesteps[logsnr_mask] if org_timesteps is not None else None,
                        )
                        t_logsnr = torch.sigmoid(-logsnr / 2)

                        t[logsnr_mask] = t_logsnr

                    # Generate logsnr2 samples with -logit_mean for selected indices.
                    if logsnr_mask2.any():
                        logsnr2_count = logsnr_mask2.sum().item()
                        logsnr2 = rand_logsnr(
                            logsnr2_count, 5.36, 1.0, org_timesteps[logsnr_mask2] if org_timesteps is not None else None
                        )
                        t_logsnr2 = torch.sigmoid(-logsnr2 / 2)

                        t[logsnr_mask2] = t_logsnr2

                return t  # 0 to 1

            t_min = args.min_timestep if args.min_timestep is not None else 0
            t_max = args.max_timestep if args.max_timestep is not None else 1000.0
            t_min /= 1000.0
            t_max /= 1000.0

            if not args.preserve_distribution_shape:
                t = compute_sampling_timesteps(timesteps)
                t = t * (t_max - t_min) + t_min  # scale to [t_min, t_max], default [0, 1]
            else:
                max_loops = 1000
                available_t = []
                for i in range(max_loops):
                    t = None
                    if self.num_timestep_buckets is not None:
                        t = torch.tensor([self.get_bucketed_timestep() for _ in range(batch_size)], device=device)
                    t = compute_sampling_timesteps(t)
                    for t_i in t:
                        if t_min <= t_i <= t_max:
                            available_t.append(t_i)
                        if len(available_t) == batch_size:
                            break
                    if len(available_t) == batch_size:
                        break
                if len(available_t) < batch_size:
                    logger.warning(
                        f"Could not sample {batch_size} valid timesteps in {max_loops} loops / {max_loops}ループで{batch_size}個の有効なタイムステップをサンプリングできませんでした"
                    )
                    available_t = compute_sampling_timesteps(timesteps)
                else:
                    t = torch.stack(available_t, dim=0)  # [batch_size, ]

            timesteps = t * 1000.0
            t = t.view(-1, 1, 1, 1, 1) if latents.ndim == 5 else t.view(-1, 1, 1, 1)
            noisy_model_input = (1 - t) * latents + t * noise

            timesteps += 1  # 1 to 1000
        else:
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=batch_size,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            # indices = (u * noise_scheduler.config.num_train_timesteps).long()
            t_min = args.min_timestep if args.min_timestep is not None else 0
            t_max = args.max_timestep if args.max_timestep is not None else 1000
            indices = (u * (t_max - t_min) + t_min).long()

            timesteps = noise_scheduler.timesteps[indices].to(device=device)  # 1 to 1000

            # Add noise according to flow matching.
            sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
            noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

        # print(f"actual timesteps: {timesteps}")
        return noisy_model_input, timesteps

    def show_timesteps(self, args: argparse.Namespace):
        N_TRY = 100000
        BATCH_SIZE = 1000
        CONSOLE_WIDTH = 64
        N_TIMESTEPS_PER_LINE = 25

        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
        # print(f"Noise scheduler timesteps: {noise_scheduler.timesteps}")

        latents = torch.zeros(BATCH_SIZE, 1, 1, 1024 // 8, 1024 // 8, dtype=torch.float16)
        noise = torch.ones_like(latents)

        # sample timesteps
        sampled_timesteps = [0] * noise_scheduler.config.num_train_timesteps
        for i in tqdm(range(N_TRY // BATCH_SIZE)):
            bucketed_timesteps = None
            if args.num_timestep_buckets is not None and args.num_timestep_buckets > 1:
                self.num_timestep_buckets = args.num_timestep_buckets
                bucketed_timesteps = [self.get_bucketed_timestep() for _ in range(BATCH_SIZE)]

            # we use noise=1, so retured noisy_model_input is same as timestep, because `noisy_model_input = (1 - t) * latents + t * noise`
            actual_timesteps, _ = self.get_noisy_model_input_and_timesteps(
                args, noise, latents, bucketed_timesteps, noise_scheduler, "cpu", torch.float16
            )
            actual_timesteps = actual_timesteps[:, 0, 0, 0, 0] * 1000
            for t in actual_timesteps:
                t = int(t.item())
                sampled_timesteps[t] += 1

        # sample weighting
        sampled_weighting = [0] * noise_scheduler.config.num_train_timesteps
        for i in tqdm(range(len(sampled_weighting))):
            timesteps = torch.tensor([i + 1], device="cpu")
            weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, noise_scheduler, timesteps, "cpu", torch.float16)
            if weighting is None:
                weighting = torch.tensor(1.0, device="cpu")
            elif torch.isinf(weighting).any():
                weighting = torch.tensor(1.0, device="cpu")
            sampled_weighting[i] = weighting.item()

        # show results
        if args.show_timesteps == "image":
            # show timesteps with matplotlib
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.bar(range(len(sampled_timesteps)), sampled_timesteps, width=1.0)
            plt.title("Sampled timesteps")
            plt.xlabel("Timestep")
            plt.ylabel("Count")

            plt.subplot(1, 2, 2)
            plt.bar(range(len(sampled_weighting)), sampled_weighting, width=1.0)
            plt.title("Sampled loss weighting")
            plt.xlabel("Timestep")
            plt.ylabel("Weighting")

            plt.tight_layout()
            plt.show()

        else:
            sampled_timesteps = np.array(sampled_timesteps)
            sampled_weighting = np.array(sampled_weighting)

            # average per line
            sampled_timesteps = sampled_timesteps.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)
            sampled_weighting = sampled_weighting.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)

            max_count = max(sampled_timesteps)
            print(f"Sampled timesteps: max count={max_count}")
            for i, t in enumerate(sampled_timesteps):
                line = f"{(i) * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: "
                line += "#" * int(t / max_count * CONSOLE_WIDTH)
                print(line)

            max_weighting = max(sampled_weighting)
            print(f"Sampled loss weighting: max weighting={max_weighting}")
            for i, w in enumerate(sampled_weighting):
                line = f"{i * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: {w:8.2f} "
                line += "#" * int(w / max_weighting * CONSOLE_WIDTH)
                print(line)

    def sample_images(self, accelerator: Accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype):
        """architecture independent sample images"""
        if not should_sample_images(args, steps, epoch):
            return

        logger.info("")
        logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
        if sample_parameters is None:
            logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
            return

        distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

        # Use the unwrapped model
        transformer = accelerator.unwrap_model(transformer)
        transformer.switch_block_swap_for_inference()

        # Create a directory to save the samples
        save_dir = os.path.join(args.output_dir, "sample")
        os.makedirs(save_dir, exist_ok=True)

        # save random state to restore later
        rng_state = torch.get_rng_state()
        cuda_rng_state = None
        try:
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        except Exception:
            pass

        if distributed_state.num_processes <= 1:
            # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
            with torch.no_grad(), accelerator.autocast():
                for sample_parameter in sample_parameters:
                    self.sample_image_inference(
                        accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps
                    )
                    clean_memory_on_device(accelerator.device)
        else:
            # Creating list with N elements, where each element is a list of prompt_dicts, and N is the number of processes available (number of devices available)
            # prompt_dicts are assigned to lists based on order of processes, to attempt to time the image creation time to match enum order. Probably only works when steps and sampler are identical.
            per_process_params = []  # list of lists
            for i in range(distributed_state.num_processes):
                per_process_params.append(sample_parameters[i :: distributed_state.num_processes])

            with torch.no_grad():
                with distributed_state.split_between_processes(per_process_params) as sample_parameter_lists:
                    for sample_parameter in sample_parameter_lists[0]:
                        self.sample_image_inference(
                            accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps
                        )
                        clean_memory_on_device(accelerator.device)

        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        transformer.switch_block_swap_for_training()
        clean_memory_on_device(accelerator.device)

    def sample_image_inference(self, accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps):
        """architecture independent sample images"""
        sample_steps = sample_parameter.get("sample_steps", 20)
        width = sample_parameter.get("width", 256)  # make smaller for faster and memory saving inference
        height = sample_parameter.get("height", 256)
        frame_count = sample_parameter.get("frame_count", 1)
        guidance_scale = sample_parameter.get("guidance_scale", self.default_guidance_scale)
        discrete_flow_shift = sample_parameter.get("discrete_flow_shift", self.default_discrete_flow_shift)
        seed = sample_parameter.get("seed")
        prompt: str = sample_parameter.get("prompt", "")
        cfg_scale = sample_parameter.get("cfg_scale", None)  # None for architecture default
        negative_prompt = sample_parameter.get("negative_prompt", None)

        # round width and height to multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        # 1, 5, 9, 13, ... For HunyuanVideo and Wan2.1
        frame_count = (frame_count - 1) // self.vae_frame_stride * self.vae_frame_stride + 1

        if self.i2v_training:
            image_path = sample_parameter.get("image_path", None)
            if image_path is None:
                logger.error("No image_path for i2v model / i2vモデルのサンプル画像生成にはimage_pathが必要です")
                return
        else:
            image_path = None

        if self.control_training:
            control_video_path = sample_parameter.get("control_video_path", None)
            if control_video_path is None:
                logger.error(
                    "No control_video_path for control model / controlモデルのサンプル画像生成にはcontrol_video_pathが必要です"
                )
                return
        else:
            control_video_path = None

        device = accelerator.device
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            # True random sample image generation
            torch.seed()
            torch.cuda.seed()
            generator = torch.Generator(device=device).manual_seed(torch.initial_seed())

        logger.info(f"prompt: {prompt}")
        logger.info(f"height: {height}")
        logger.info(f"width: {width}")
        logger.info(f"frame count: {frame_count}")
        logger.info(f"sample steps: {sample_steps}")
        logger.info(f"guidance scale: {guidance_scale}")
        logger.info(f"discrete flow shift: {discrete_flow_shift}")
        if seed is not None:
            logger.info(f"seed: {seed}")

        do_classifier_free_guidance = False
        if negative_prompt is not None:
            do_classifier_free_guidance = True
            logger.info(f"negative prompt: {negative_prompt}")
            logger.info(f"cfg scale: {cfg_scale}")

        if self.i2v_training:
            logger.info(f"image path: {image_path}")
        if self.control_training:
            logger.info(f"control video path: {control_video_path}")

        # inference: architecture dependent
        # Check if transformer has self-referencing _orig_mod (compiled model hack)
        # If so, skip eval/train to avoid infinite recursion
        has_self_ref_orig_mod = getattr(transformer, "_orig_mod", None) is transformer
        was_train = transformer.training if not has_self_ref_orig_mod else True
        if not has_self_ref_orig_mod:
            transformer.eval()

        video = self.do_inference(
            accelerator,
            args,
            sample_parameter,
            vae,
            dit_dtype,
            transformer,
            discrete_flow_shift,
            sample_steps,
            width,
            height,
            frame_count,
            generator,
            do_classifier_free_guidance,
            guidance_scale,
            cfg_scale,
            image_path=image_path,
            control_video_path=control_video_path,
        )

        if not has_self_ref_orig_mod:
            transformer.train(was_train)

        # Save video
        if video is None:
            logger.error("No video generated / 生成された動画がありません")
            return

        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        seed_suffix = "" if seed is None else f"_{seed}"
        prompt_idx = sample_parameter.get("enum", 0)
        save_path = (
            f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{prompt_idx:02d}_{ts_str}{seed_suffix}"
        )

        wandb_tracker = None
        try:
            wandb_tracker = accelerator.get_tracker("wandb")  # raises ValueError if wandb is not initialized
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
        except:  # wandb 無効時
            wandb = None

        if video.shape[2] == 1:
            # In Qwen-Image-Layered, video is (N, C, 1, H, W) where N=Layers, otherwise (1, C, 1, H, W)
            image_paths = save_images_grid(video, save_dir, save_path, n_rows=video.shape[0], create_subdir=False)
            if wandb_tracker is not None and wandb is not None:
                for image_path in image_paths:
                    wandb_tracker.log({f"sample_{prompt_idx}": wandb.Image(image_path)}, step=steps)
        else:
            video_path = os.path.join(save_dir, save_path) + ".mp4"
            save_videos_grid(video, video_path)
            if wandb_tracker is not None and wandb is not None:
                wandb_tracker.log({f"sample_{prompt_idx}": wandb.Video(video_path)}, step=steps)

        # Move models back to initial state
        vae.to("cpu")
        clean_memory_on_device(device)

    # region model specific (abstract hooks — implemented by architecture-specific subclasses)

    @property
    def architecture(self) -> str:
        raise NotImplementedError("subclass must define `architecture`")

    @property
    def architecture_full_name(self) -> str:
        raise NotImplementedError("subclass must define `architecture_full_name`")

    def handle_model_specific_args(self, args: argparse.Namespace):
        # Subclasses must set: self._i2v_training, self._control_training, self.default_guidance_scale.
        # They may also set arch-specific state like self.default_discrete_flow_shift, self.vae_frame_stride.
        raise NotImplementedError("subclass must implement `handle_model_specific_args`")

    @property
    def i2v_training(self) -> bool:
        return self._i2v_training

    @property
    def control_training(self) -> bool:
        return self._control_training

    def convert_weight_keys(self, weights_sd: dict[str, torch.Tensor], network_module: lora_module):
        # Default: assume the saved LoRA is already in this project's native format.
        return weights_sd

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        raise NotImplementedError("subclass must implement `process_sample_prompts`")

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        """Architecture-dependent sample inference used during training."""
        raise NotImplementedError("subclass must implement `do_inference`")

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        raise NotImplementedError("subclass must implement `load_vae`")

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        raise NotImplementedError("subclass must implement `load_transformer`")

    def compile_transformer(self, args, transformer):
        raise NotImplementedError("subclass must implement `compile_transformer`")

    def scale_shift_latents(self, latents):
        raise NotImplementedError("subclass must implement `scale_shift_latents`")

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer_arg,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        """Run the DiT forward and return prediction/target as a ``DiTOutput``.

        ``**kwargs`` is reserved for extension subclasses to pass additional
        conditioning (e.g. ``per_token_timesteps``) or request side outputs
        (e.g. ``hidden_features``). Architecture implementations may ignore
        unknown kwargs.
        """
        raise NotImplementedError("subclass must implement `call_dit`")

    # endregion model specific

    # region extension seams
    # Internal extension points — no API stability guarantees.
    # Subclasses live in this repo; if you fork, expect breakage on updates.

    def process_batch(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        network,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor,
        noise: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        vae,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute scalar loss for one training batch (pre-backward).

        Default implementation: vanilla flow matching, delegating the loss
        formulation itself to ``compute_loss``. Override either method:
        ``process_batch`` to change what gets fed to the model (Self-Flow's
        dual-timestep dance, etc.), or ``compute_loss`` to swap the loss
        formulation while keeping the standard data flow.

        Returns ``(scalar_loss, loss_metrics)`` — ``loss_metrics`` is merged
        into the per-step log dict alongside ``extra_step_logs``.

        ``latents`` is already scale-shifted; ``noise`` is already sampled.
        """
        noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
            args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
        )

        output = self.call_dit(args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype)
        return self.compute_loss(args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step)

    def compute_loss(
        self,
        args: argparse.Namespace,
        output: DiTOutput,
        timesteps: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Reduce a ``DiTOutput`` to a scalar loss + per-step metrics dict.

        Default implementation: weighted MSE between ``output.pred`` and
        ``output.target`` with the SD3-style ``args.weighting_scheme`` applied,
        then ``.mean()``. Override to swap the loss formulation entirely
        (e.g. Self-Flow's L_gen + gamma * L_rep) or to add auxiliary terms
        (e.g. HiDream-O1's step-gated DINO perceptual loss). Subclasses are
        responsible for whatever weighting/reduction they need — this hook owns
        the full loss computation, not just the per-element MSE.

        ``global_step`` is provided for step-gated terms (e.g. computing an
        auxiliary loss only every N steps). ``loss_metrics`` defaults to empty;
        populate with named scalars for loss-decomposition logging
        (e.g. ``{"loss/gen": ..., "loss/rep": ...}``).
        """
        weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, noise_scheduler, timesteps, timesteps.device, dit_dtype)
        loss = torch.nn.functional.mse_loss(output.pred.to(network_dtype), output.target, reduction="none")
        if weighting is not None:
            loss = loss * weighting
        return loss.mean(), {}

    def on_transformer_loaded(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
    ) -> None:
        """Called immediately after ``self.load_transformer(...)`` returns.

        At this point the transformer is on its loading device but not yet wrapped
        by the accelerator and not yet in eval mode. Use this hook for one-time
        post-load setup that needs the raw module (e.g. ``register_forward_hook``
        for feature extraction).
        """

    def on_train_start(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        optimizer,
    ) -> None:
        """Called once after accelerator.prepare and before the training loop starts.

        Use this for initializing extension state that depends on prepared models
        (EMA copies, decay schedulers, register_forward_hook on the transformer, etc.).
        """

    def on_post_optimizer_step(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        sync_gradients: bool,
        global_step: int,
    ) -> None:
        """Called after optimizer.step / lr_scheduler.step / zero_grad each inner step.

        ``sync_gradients`` mirrors ``accelerator.sync_gradients`` and is True only
        on steps where an actual optimizer update occurred (gradient accumulation aware).
        ``transformer`` is the accelerator-wrapped DiT — passed so subclasses doing
        non-network (full fine-tuning) bookkeeping or EMA on transformer weights can
        reach it without stashing a reference in ``on_train_start``.
        Use for EMA updates or any post-step bookkeeping.
        """

    def on_post_save(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        ckpt_name: str,
        save_dtype,
        metadata: dict,
        force_sync_upload: bool,
    ) -> None:
        """Called after the main network checkpoint has been saved.

        ``ckpt_name`` is the basename written to ``args.output_dir``. Use this hook
        to write companion files (EMA weights, projection heads, etc.) alongside.
        ``transformer`` is the accelerator-wrapped DiT — provided so non-network
        (full fine-tuning) subclasses can save companion artifacts derived from it.
        ``force_sync_upload`` mirrors the flag passed to the main HuggingFace upload
        so subclasses uploading companion files can match the same behaviour.
        """

    def on_before_sample_images(
        self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype
    ) -> None:
        """Called just before sample image generation begins, while the transformer is still in training mode.

        The transformer is still wrapped by the accelerator at this point. Use this hook for
        pre-inference setup such as switching auxiliary modules to eval mode or stashing training state.
        """
        pass

    def on_after_sample_images(
        self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype
    ) -> None:
        """Called after sample image generation completes and the transformer has been switched back to training mode.

        Memory has already been cleaned via ``clean_memory_on_device``. Use this hook for
        post-inference cleanup such as restoring auxiliary modules to train mode or updating state
        based on the generated samples.
        """
        pass

    def extra_trainable_params(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        trainable_params: list,
    ) -> list:
        """Optionally augment the param-group list passed to the optimizer.

        Default: pass-through. Override to merge extra modules' parameters
        (e.g. a representation projection head) into ``trainable_params``.
        Subclasses are expected to stash any owned modules on ``self`` so
        ``on_train_start`` and later hooks can use them.
        """
        return trainable_params

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        """Returns extra ``ss_*`` metadata keys to embed in saved safetensors.

        Default: empty dict. Override to add extension-specific metadata.
        """
        return {}

    def extra_step_logs(self, args: argparse.Namespace, logs: dict) -> dict:
        """Returns additional log entries to merge into the per-step log payload.

        Called just before ``accelerator.log`` on logging steps. The returned
        dict is merged into ``logs`` (existing keys are overwritten on collision).
        Default: empty dict.
        """
        return {}

    # endregion extension seams

    def train(self, args):
        if not self._validate_args_and_init(args):
            return

        session_id, training_started_at = self._init_session(args)
        train_dataset_group, collator, current_epoch = self._build_dataset(args)
        accelerator, weight_dtype, dit_dtype, dit_weight_dtype, vae_dtype = self._prepare_accelerator_and_dtypes(args)
        sample_parameters, vae = self._prepare_sampling(args, accelerator, vae_dtype)
        transformer = self._load_dit_and_swap(args, accelerator, dit_weight_dtype)
        network = self._build_network(args, accelerator, transformer, vae, weight_dtype)
        if network is None:
            return
        (
            optimizer,
            optimizer_name,
            optimizer_args,
            optimizer_train_fn,
            optimizer_eval_fn,
            lr_scheduler,
            lr_descriptions,
            train_dataloader,
        ) = self._build_optimizer_and_dataloader(args, accelerator, network, train_dataset_group, collator, transformer)
        (
            transformer,
            network,
            optimizer,
            train_dataloader,
            lr_scheduler,
            training_model,
            network_dtype,
        ) = self._prepare_with_accelerator(
            args,
            accelerator,
            transformer,
            network,
            optimizer,
            train_dataloader,
            lr_scheduler,
            weight_dtype,
            dit_dtype,
            dit_weight_dtype,
        )
        self._register_hooks_and_resume(args, accelerator, network)
        self._run_training_loop(
            args,
            accelerator,
            session_id,
            training_started_at,
            train_dataset_group,
            train_dataloader,
            current_epoch,
            transformer,
            network,
            training_model,
            optimizer,
            optimizer_name,
            optimizer_args,
            optimizer_train_fn,
            optimizer_eval_fn,
            lr_scheduler,
            lr_descriptions,
            vae,
            sample_parameters,
            dit_dtype,
            network_dtype,
        )

    def _validate_args_and_init(self, args) -> bool:
        """Validate required args, configure CUDA flags, handle `--show_timesteps`.

        Returns False if training should stop early (e.g. `--show_timesteps`).
        """
        if torch.cuda.is_available():
            if args.cuda_allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 on CUDA / CUDAでTF32を有効化しました")
            if args.cuda_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark / cuDNNベンチマークを有効化しました")

        # check required arguments
        if args.dataset_config is None:
            raise ValueError("dataset_config is required / dataset_configが必要です")
        if args.dit is None:
            raise ValueError("path to DiT model is required / DiTモデルのパスが必要です")
        assert not args.fp8_scaled or args.fp8_base, "fp8_scaled requires fp8_base / fp8_scaledはfp8_baseが必要です"

        if args.sage_attn:
            raise ValueError(
                "SageAttention doesn't support training currently. Please use `--sdpa` or `--xformers` etc. instead."
                " / SageAttentionは現在学習をサポートしていないようです。`--sdpa`や`--xformers`などの他のオプションを使ってください"
            )

        if args.disable_numpy_memmap:
            logger.info(
                "Disabling numpy memory mapping for model loading (for Wan, FramePack and Qwen-Image). This may lead to higher memory usage but can speed up loading in some cases."
                " / モデル読み込み時のnumpyメモリマッピングを無効にします（Wan、FramePack、Qwen-Imageでのみ有効）。これによりメモリ使用量が増える可能性がありますが、場合によっては読み込みが高速化されることがあります"
            )

        # check model specific arguments
        self.handle_model_specific_args(args)

        # show timesteps for debugging
        if args.show_timesteps:
            self.show_timesteps(args)
            return False

        return True

    def _init_session(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        # setup_logging(args, reset=True)

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)
        return session_id, training_started_at

    def _build_dataset(self, args):
        # Load dataset config
        if args.num_timestep_buckets is not None:
            logger.info(f"Using timestep bucketing. Number of buckets: {args.num_timestep_buckets}")
        self.num_timestep_buckets = args.num_timestep_buckets  # None or int, None makes all the behavior same as before

        current_epoch = Value("i", 0)  # shared between processes

        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_utils.load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args, architecture=self.architecture)
        train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            blueprint.dataset_group, training=True, num_timestep_buckets=self.num_timestep_buckets, shared_epoch=current_epoch
        )

        if train_dataset_group.num_train_items == 0:
            raise ValueError(
                "No training items found in the dataset. Please ensure that the latent/Text Encoder cache has been created beforehand."
                " / データセットに学習データがありません。latent/Text Encoderキャッシュを事前に作成したか確認してください"
            )

        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = collator_class(current_epoch, ds_for_collator)
        return train_dataset_group, collator, current_epoch

    def _prepare_accelerator_and_dtypes(self, args):
        # prepare accelerator
        logger.info("preparing accelerator")
        accelerator = prepare_accelerator(args)
        if args.mixed_precision is None:
            args.mixed_precision = accelerator.mixed_precision
            logger.info(f"mixed precision set to {args.mixed_precision} / mixed precisionを{args.mixed_precision}に設定")

        # prepare dtype
        weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # HunyuanVideo: bfloat16 or float16, Wan2.1: bfloat16
        dit_dtype = torch.bfloat16 if args.dit_dtype is None else model_utils.str_to_dtype(args.dit_dtype)
        dit_weight_dtype = (None if args.fp8_scaled else torch.float8_e4m3fn) if args.fp8_base else dit_dtype
        logger.info(f"DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}")

        vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
        return accelerator, weight_dtype, dit_dtype, dit_weight_dtype, vae_dtype

    def _prepare_sampling(self, args, accelerator, vae_dtype):
        # get embedding for sampling images
        sample_parameters = None
        vae = None
        if args.sample_prompts:
            sample_parameters = self.process_sample_prompts(args, accelerator, args.sample_prompts)

            # Load VAE model for sampling images: VAE is loaded to cpu to save gpu memory
            vae = self.load_vae(args, vae_dtype=vae_dtype, vae_path=args.vae)
            vae.requires_grad_(False)
            vae.eval()
        return sample_parameters, vae

    def _load_dit_and_swap(self, args, accelerator, dit_weight_dtype):
        # load DiT model
        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
        self.blocks_to_swap = blocks_to_swap
        loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device

        logger.info(f"Loading DiT model from {args.dit}")
        if args.sdpa:
            attn_mode = "torch"
        elif args.flash_attn:
            attn_mode = "flash"
        elif args.sage_attn:
            attn_mode = "sageattn"
        elif args.xformers:
            attn_mode = "xformers"
        elif args.flash3:
            attn_mode = "flash3"
        else:
            raise ValueError(
                "either --sdpa, --flash-attn, --flash3, --sage-attn or --xformers must be specified / --sdpa, --flash-attn, --flash3, --sage-attn, --xformersのいずれかを指定してください"
            )
        transformer = self.load_transformer(
            accelerator, args, args.dit, attn_mode, args.split_attn, loading_device, dit_weight_dtype
        )
        self.on_transformer_loaded(args, accelerator, transformer)
        transformer.eval()
        transformer.requires_grad_(False)

        if blocks_to_swap > 0:
            swap_config = BlockSwapConfig.from_args(args, accelerator.device, supports_backward=True)
            logger.info(
                f"enable swap {blocks_to_swap} blocks to CPU from device: {accelerator.device}, "
                f"use pinned memory: {swap_config.use_pinned_memory}, H2D-only: {swap_config.h2d_only}"
            )
            transformer.enable_block_swap(blocks_to_swap, swap_config)
            transformer.move_to_device_except_swap_blocks(accelerator.device)
        return transformer

    def _build_network(self, args, accelerator, transformer, vae, weight_dtype):
        # load network module for differential training
        # Allow short paths like `--network_module networks.lora` by putting the musubi_tuner
        # package dir on sys.path. __file__ is under musubi_tuner/training/, so step up one level.
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        accelerator.print("import network module:", args.network_module)
        network_module: lora_module = importlib.import_module(args.network_module)  # actual module may be different

        if args.base_weights is not None:
            # if base_weights is specified, merge the weights to DiT model
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                weights_sd = load_file(weight_path)
                weights_sd = self.convert_weight_keys(weights_sd, network_module)
                module = network_module.create_arch_network_from_weights(
                    multiplier, weights_sd, unet=transformer, for_inference=True
                )
                module.merge_to(None, transformer, weights_sd, weight_dtype, "cpu")

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        if args.dim_from_weights:
            logger.info(f"Loading network from weights: {args.dim_from_weights}")
            weights_sd = load_file(args.dim_from_weights)
            network, _ = network_module.create_arch_network_from_weights(1, weights_sd, unet=transformer)
        else:
            # We use the name create_arch_network for compatibility with LyCORIS
            if hasattr(network_module, "create_arch_network"):
                network = network_module.create_arch_network(
                    1.0,
                    args.network_dim,
                    args.network_alpha,
                    vae,
                    None,
                    transformer,
                    neuron_dropout=args.network_dropout,
                    **net_kwargs,
                )
            else:
                # LyCORIS compatibility
                network = network_module.create_network(
                    1.0,
                    args.network_dim,
                    args.network_alpha,
                    vae,
                    None,
                    transformer,
                    **net_kwargs,
                )
        if network is None:
            return None

        if hasattr(network_module, "prepare_network"):
            network.prepare_network(args)

        # apply network to DiT
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)

        if args.network_weights is not None:
            # FIXME consider alpha of weights: this assumes that the alpha is not changed
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
            network.enable_gradient_checkpointing()  # may have no effect

        # net_kwargs is reconstructed in the metadata phase from args.network_args
        return network

    def _build_optimizer_and_dataloader(self, args, accelerator, network, train_dataset_group, collator, transformer):
        # prepare optimizer, data loader etc.
        accelerator.print("prepare optimizer, data loader etc.")

        trainable_params, lr_descriptions = network.prepare_optimizer_params(unet_lr=args.learning_rate)
        trainable_params = self.extra_trainable_params(args, accelerator, network, transformer, trainable_params)
        optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn = self.get_optimizer(
            args, trainable_params
        )

        # prepare dataloader

        # num workers for data loader: if 0, persistent_workers is not available
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # calculate max_train_steps
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
            )

        # send max_train_steps to train_dataset_group
        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # prepare lr_scheduler
        lr_scheduler = self.get_lr_scheduler(args, optimizer, accelerator.num_processes)

        return (
            optimizer,
            optimizer_name,
            optimizer_args,
            optimizer_train_fn,
            optimizer_eval_fn,
            lr_scheduler,
            lr_descriptions,
            train_dataloader,
        )

    def _prepare_with_accelerator(
        self,
        args,
        accelerator,
        transformer,
        network,
        optimizer,
        train_dataloader,
        lr_scheduler,
        weight_dtype,
        dit_dtype,
        dit_weight_dtype,
    ):
        # prepare training model. accelerator does some magic here

        # experimental feature: train the model with gradients in fp16/bf16
        network_dtype = torch.float32
        args.full_fp16 = args.full_bf16 = False  # temporary disabled because stochastic rounding is not supported yet
        if args.full_fp16:
            assert args.mixed_precision == "fp16", (
                "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            )
            accelerator.print("enable full fp16 training.")
            network_dtype = weight_dtype
            network.to(network_dtype)
        elif args.full_bf16:
            assert args.mixed_precision == "bf16", (
                "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            )
            accelerator.print("enable full bf16 training.")
            network_dtype = weight_dtype
            network.to(network_dtype)

        if dit_weight_dtype != dit_dtype and dit_weight_dtype is not None:
            logger.info(f"casting model to {dit_weight_dtype}")
            transformer.to(dit_weight_dtype)

        blocks_to_swap = self.blocks_to_swap or 0
        if blocks_to_swap > 0:
            transformer = accelerator.prepare(transformer, device_placement=[not blocks_to_swap > 0])
            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        else:
            transformer = accelerator.prepare(transformer)

        if args.compile:
            transformer = self.compile_transformer(args, transformer)
            transformer.__dict__["_orig_mod"] = transformer  # for annoying accelerator checks

        network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(network, optimizer, train_dataloader, lr_scheduler)
        training_model = network

        if args.gradient_checkpointing:
            transformer.train()
        else:
            transformer.eval()

        accelerator.unwrap_model(network).prepare_grad_etc(transformer)

        if args.full_fp16:
            # patch accelerator for fp16 training
            # def patch_accelerator_for_fp16_training(accelerator):
            org_unscale_grads = accelerator.scaler._unscale_grads_

            def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
                return org_unscale_grads(optimizer, inv_scale, found_inf, True)

            accelerator.scaler._unscale_grads_ = _unscale_grads_replacer

        return transformer, network, optimizer, train_dataloader, lr_scheduler, training_model, network_dtype

    def _register_hooks_and_resume(self, args, accelerator, network):
        # before resuming make hook for saving/loading to save/load the network weights only
        def save_model_hook(models, weights, output_dir):
            # pop weights of other models than network to save only network weights
            # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
            if accelerator.is_main_process:  # or args.deepspeed:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)
                # print(f"save model hook: {len(weights)} weights will be saved")

        def load_model_hook(models, input_dir):
            # remove models except network
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)
            # print(f"load model hook: {len(models)} models will be loaded")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # resume from local or huggingface. accelerator.step is set
        self.resume_from_local_or_hf_if_specified(accelerator, args)  # accelerator.load_state(args.resume)

    def _run_training_loop(
        self,
        args,
        accelerator,
        session_id,
        training_started_at,
        train_dataset_group,
        train_dataloader,
        current_epoch,
        transformer,
        network,
        training_model,
        optimizer,
        optimizer_name,
        optimizer_args,
        optimizer_train_fn,
        optimizer_eval_fn,
        lr_scheduler,
        lr_descriptions,
        vae,
        sample_parameters,
        dit_dtype,
        network_dtype,
    ):
        is_main_process = accelerator.is_main_process

        self.on_train_start(args, accelerator, network, transformer, optimizer)

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # 学習する
        # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train items / 学習画像、動画数: {train_dataset_group.num_train_items}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        # reconstruct net_kwargs for metadata
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_num_train_items": train_dataset_group.num_train_items,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_checkpointing_cpu_offload": args.gradient_checkpointing_cpu_offload,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            SS_METADATA_KEY_BASE_MODEL_VERSION: self.architecture_full_name,
            # "ss_network_module": args.network_module,
            # "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            # "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            SS_METADATA_KEY_NETWORK_MODULE: args.network_module,
            SS_METADATA_KEY_NETWORK_DIM: args.network_dim,
            SS_METADATA_KEY_NETWORK_ALPHA: args.network_alpha,
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_seed": args.seed,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            # "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_fp8_base": bool(args.fp8_base),
            # "ss_fp8_llm": bool(args.fp8_llm), # remove this because this is only for HuanyuanVideo TODO set architecure dependent metadata
            "ss_full_fp16": bool(args.full_fp16),
            "ss_full_bf16": bool(args.full_bf16),
            "ss_weighting_scheme": args.weighting_scheme,
            "ss_logit_mean": args.logit_mean,
            "ss_logit_std": args.logit_std,
            "ss_mode_scale": args.mode_scale,
            "ss_guidance_scale": args.guidance_scale,
            "ss_timestep_sampling": args.timestep_sampling,
            "ss_sigmoid_scale": args.sigmoid_scale,
            "ss_discrete_flow_shift": args.discrete_flow_shift,
        }
        metadata.update(self.extra_metadata(args))

        datasets_metadata = []
        # tag_frequency = {}  # merge tag frequency for metadata editor # TODO support tag frequency
        for dataset in train_dataset_group.datasets:
            dataset_metadata = dataset.get_metadata()
            datasets_metadata.append(dataset_metadata)

        metadata["ss_datasets"] = json.dumps(datasets_metadata)

        # add extra args
        if args.network_args:
            # metadata["ss_network_args"] = json.dumps(net_kwargs)
            metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(net_kwargs)

        # model name and hash
        # calculate hash takes time, so we omit it for now
        if args.dit is not None:
            # logger.info(f"calculate hash for DiT model: {args.dit}")
            logger.info(f"set DiT model name for metadata: {args.dit}")
            sd_model_name = args.dit
            if os.path.exists(sd_model_name):
                # metadata["ss_sd_model_hash"] = model_utils.model_hash(sd_model_name)
                # metadata["ss_new_sd_model_hash"] = model_utils.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            # logger.info(f"calculate hash for VAE model: {args.vae}")
            logger.info(f"set VAE model name for metadata: {args.vae}")
            vae_name = args.vae
            if os.path.exists(vae_name):
                # metadata["ss_vae_hash"] = model_utils.model_hash(vae_name)
                # metadata["ss_new_vae_hash"] = model_utils.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        if accelerator.is_main_process:
            init_kwargs = {}
            if args.wandb_run_name:
                init_kwargs["wandb"] = {"name": args.wandb_run_name}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers(
                "network_train" if args.log_tracker_name is None else args.log_tracker_name,
                config=train_utils.get_sanitized_config_or_none(args),
                init_kwargs=init_kwargs,
            )

        # TODO skip until initial step
        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")

        epoch_to_start = 0
        global_step = 0
        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")

        loss_recorder = train_utils.LossRecorder()
        del train_dataset_group

        # function for saving/removing
        save_dtype = train_utils.resolve_save_dtype(
            args.save_precision, getattr(args, "full_fp16", False), getattr(args, "full_bf16", False)
        )
        logger.info(
            f"network weights will be saved as {save_dtype}"
            + (f" (--save_precision {args.save_precision})" if args.save_precision is not None else " (default)")
        )

        def save_model(ckpt_name: str, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata

            title = args.metadata_title if args.metadata_title is not None else args.output_name
            if args.min_timestep is not None or args.max_timestep is not None:
                min_time_step = args.min_timestep if args.min_timestep is not None else 0
                max_time_step = args.max_timestep if args.max_timestep is not None else 1000
                md_timesteps = (min_time_step, max_time_step)
            else:
                md_timesteps = None

            sai_metadata = sai_model_spec.build_metadata(
                None,
                self.architecture,
                time.time(),
                title,
                args.metadata_reso,
                args.metadata_author,
                args.metadata_description,
                args.metadata_license,
                args.metadata_tags,
                timesteps=md_timesteps,
                custom_arch=args.metadata_arch,
            )

            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_utils.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

            self.on_post_save(args, accelerator, network, transformer, ckpt_name, save_dtype, metadata_to_save, force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        def _do_sample(epoch_arg, steps_arg):
            if not should_sample_images(args, steps_arg, epoch_arg):
                return
            self.on_before_sample_images(
                accelerator, args, epoch_arg, steps_arg, vae, transformer, network, sample_parameters, dit_dtype
            )
            try:
                self.sample_images(accelerator, args, epoch_arg, steps_arg, vae, transformer, sample_parameters, dit_dtype)
            finally:
                self.on_after_sample_images(
                    accelerator, args, epoch_arg, steps_arg, vae, transformer, network, sample_parameters, dit_dtype
                )

        # For --sample_at_first
        if should_sample_images(args, global_step, epoch=0):
            optimizer_eval_fn()
            _do_sample(0, global_step)
            optimizer_train_fn()
        if len(accelerator.trackers) > 0:
            # log empty object to commit the sample images to wandb
            accelerator.log({}, step=0)

        # training loop

        # log device and dtype for each model
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        first_param = next(iter(unwrapped_transformer.parameters()), None)
        logger.info(
            f"DiT dtype: {first_param.dtype if first_param is not None else None}, device: {first_param.device if first_param is not None else accelerator.device}"
        )

        clean_memory_on_device(accelerator.device)

        optimizer_train_fn()  # Set training mode

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            accelerator.unwrap_model(network).on_epoch_start(transformer)

            for step, batch in enumerate(train_dataloader):
                # torch.compiler.cudagraph_mark_step_begin() # for cudagraphs

                latents = batch["latents"]

                with accelerator.accumulate(training_model):
                    accelerator.unwrap_model(network).on_step_start()

                    latents = self.scale_shift_latents(latents)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)

                    loss, loss_metrics = self.process_batch(
                        args,
                        accelerator,
                        transformer,
                        network,
                        batch,
                        latents,
                        noise,
                        noise_scheduler,
                        dit_dtype,
                        network_dtype,
                        vae,
                        global_step,
                    )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        # self.all_reduce_network(accelerator, network)  # sync DDP grad manually
                        state = accelerate.PartialState()
                        if state.distributed_type != accelerate.DistributedType.NO:
                            for param in network.parameters():
                                if param.grad is not None:
                                    param.grad = accelerator.reduce(param.grad, reduction="mean")

                        if args.max_grad_norm != 0.0:
                            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    self.on_post_optimizer_step(args, accelerator, network, transformer, accelerator.sync_gradients, global_step)

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if global_step == 0:
                        progress_bar.reset()  # exclude first step from progress bar, because it may take long due to initializations
                    progress_bar.update(1)
                    global_step += 1

                    # to avoid calling optimizer_eval_fn() too frequently, we call it only when we need to sample images or save the model
                    should_sampling = should_sample_images(args, global_step, epoch=None)
                    should_saving = args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0

                    if should_sampling or should_saving:
                        optimizer_eval_fn()
                        if should_sampling:
                            _do_sample(None, global_step)

                        if should_saving:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process:
                                ckpt_name = train_utils.get_step_ckpt_name(args.output_name, global_step)
                                save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                                if args.save_state:
                                    train_utils.save_and_remove_state_stepwise(args, accelerator, global_step)

                                remove_step_no = train_utils.get_remove_step_no(args, global_step)
                                if remove_step_no is not None:
                                    remove_ckpt_name = train_utils.get_step_ckpt_name(args.output_name, remove_step_no)
                                    remove_model(remove_ckpt_name)
                        optimizer_train_fn()

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if len(accelerator.trackers) > 0:
                    logs = self.generate_step_logs(
                        args, current_loss, avr_loss, lr_scheduler, lr_descriptions, optimizer, keys_scaled, mean_norm, maximum_norm
                    )
                    logs.update(loss_metrics)
                    logs.update(self.extra_step_logs(args, logs))
                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if len(accelerator.trackers) > 0:
                logs = {"loss/epoch": loss_recorder.moving_average}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            # save model at the end of epoch if needed
            optimizer_eval_fn()
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, epoch + 1)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

                    remove_epoch_no = train_utils.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, remove_epoch_no)
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        train_utils.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            _do_sample(epoch + 1, global_step)
            optimizer_train_fn()

            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()
        optimizer_eval_fn()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            train_utils.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_utils.get_last_ckpt_name(args.output_name)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

            logger.info("model saved.")
