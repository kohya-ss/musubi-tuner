import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass

import torch
import toml
from accelerate.utils import DistributedType
from safetensors.torch import save_file
from tqdm import tqdm

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.training.accelerator_setup import clean_memory_on_device, prepare_accelerator
from musubi_tuner.training.sampling_prompts import should_sample_images
from musubi_tuner.utils import huggingface_utils, model_utils, sai_model_spec, train_utils
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file


_LORA_INERT_DEFAULTS = {
    "network_weights": None,
    "network_module": None,
    "network_dim": None,
    "network_alpha": 1,
    "network_dropout": None,
    "network_args": None,
    "dim_from_weights": False,
    "scale_weight_norms": None,
    "base_weights": None,
    "base_weights_multiplier": None,
}

_REPLICA_DISTRIBUTED_TYPES = {
    DistributedType.NO,
    DistributedType.MULTI_CPU,
    DistributedType.MULTI_GPU,
    DistributedType.MULTI_NPU,
    DistributedType.MULTI_MLU,
    DistributedType.MULTI_SDAA,
    DistributedType.MULTI_MUSA,
    DistributedType.MULTI_XPU,
    DistributedType.MULTI_HPU,
}


class FullFineTuneResumeProgressError(NotImplementedError):
    """Raised when a full-finetune state cannot restore its progress cursor."""


@dataclass(eq=True)
class TrainingProgress:
    global_step: int = 0
    epoch: int = 0
    next_batch: int = 0
    sampler_seed: int | None = None

    def state_dict(self) -> dict[str, int | None]:
        return asdict(self)

    def load_state_dict(self, state: dict[str, int | None]) -> None:
        self.global_step = int(state["global_step"])
        self.epoch = int(state["epoch"])
        self.next_batch = int(state["next_batch"])
        sampler_seed = state.get("sampler_seed")
        self.sampler_seed = None if sampler_seed is None else int(sampler_seed)


def add_full_finetune_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--full_bf16", action="store_true", help="enable full bfloat16 training")
    parser.add_argument("--fused_backward_pass", action="store_true", help="use fused backward pass for Adafactor")
    parser.add_argument("--mem_eff_save", action="store_true", help="enable memory-efficient model saving")
    parser.add_argument(
        "--block_swap_optimizer_patch_params",
        action="store_true",
        help="patch optimizer parameters when block swap is enabled",
    )
    return parser


def resolve_trainable_dtype(args: argparse.Namespace) -> torch.dtype:
    return model_utils.str_to_dtype("bf16" if args.full_bf16 else "fp32")


def validate_full_finetune_distributed_type(distributed_type: DistributedType) -> None:
    if distributed_type in _REPLICA_DISTRIBUTED_TYPES:
        return

    backend = getattr(distributed_type, "value", str(distributed_type))
    raise ValueError(
        f"full finetuning does not support distributed backend {backend}; "
        "use ordinary DDP replica training (for example MULTI_GPU or MULTI_CPU) instead"
    )


def validate_full_finetune_args(args: argparse.Namespace, num_processes: int) -> None:
    specified_lora_args = [
        f"--{name}" for name, inert_default in _LORA_INERT_DEFAULTS.items() if getattr(args, name) != inert_default
    ]
    if specified_lora_args:
        raise ValueError("full finetuning does not use LoRA/network arguments: " + ", ".join(specified_lora_args))

    specified_fp8_args = [f"--{name}" for name in ("fp8_base", "fp8_scaled") if getattr(args, name)]
    if specified_fp8_args:
        raise ValueError("full finetuning does not support FP8 options: " + ", ".join(specified_fp8_args))

    if args.block_swap_h2d_only:
        raise ValueError("--block_swap_h2d_only is only supported for frozen-base training")

    blocks_to_swap = 0 if args.blocks_to_swap is None else args.blocks_to_swap
    if blocks_to_swap < 0:
        raise ValueError("--blocks_to_swap must be greater than or equal to 0")
    if blocks_to_swap > 0 and num_processes > 1:
        raise ValueError("--blocks_to_swap is not supported for multi-process full finetuning")

    if args.fused_backward_pass:
        if args.optimizer_type.lower() != "adafactor":
            raise ValueError("--fused_backward_pass requires --optimizer_type Adafactor")
        if num_processes > 1:
            raise ValueError("--fused_backward_pass is not supported for multi-process full finetuning")
        if args.gradient_accumulation_steps != 1:
            raise ValueError("--fused_backward_pass requires --gradient_accumulation_steps 1")
        if args.mixed_precision == "fp16":
            raise ValueError("--fused_backward_pass is incompatible with --mixed_precision fp16")

    if getattr(args, "full_fp16", False):
        raise ValueError("--full_fp16 is not supported for full finetuning; use fp32 or --full_bf16")

    if args.full_bf16 and args.mixed_precision != "bf16":
        raise ValueError("--full_bf16 requires --mixed_precision bf16")

    trainable_dtype = resolve_trainable_dtype(args)
    requested_dit_dtype = getattr(args, "dit_dtype", None)
    if requested_dit_dtype is not None:
        try:
            resolved_dit_dtype = model_utils.str_to_dtype(requested_dit_dtype)
        except ValueError as error:
            raise ValueError(
                f"--dit_dtype must match the full trainable dtype selected by --full_bf16; got {requested_dit_dtype!r}"
            ) from error
        if resolved_dit_dtype != trainable_dtype:
            raise ValueError(
                "--dit_dtype must match the full trainable dtype selected by --full_bf16 "
                f"({trainable_dtype}); got {requested_dit_dtype!r} ({resolved_dit_dtype})"
            )
    save_dtype = train_utils.resolve_save_dtype(args.save_precision, full_bf16=args.full_bf16)
    if save_dtype != trainable_dtype:
        raise ValueError(f"--save_precision must match the trainable dtype ({trainable_dtype}); got {save_dtype}")


def normalize_compiled_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        key = key.removeprefix("_orig_mod.").replace("._orig_mod.", ".")
        if key in normalized:
            raise ValueError(f"compiled state_dict key collision: {key}")
        normalized[key] = value
    return normalized


def _run_main_process_action_all_ranks(accelerator, operation: str, action) -> None:
    main_error = None
    if accelerator.is_main_process:
        try:
            action()
        except Exception as error:
            main_error = error

    failure = torch.tensor(main_error is not None, dtype=torch.int32, device=accelerator.device)
    failure_count = accelerator.reduce(failure, reduction="sum")
    accelerator.wait_for_everyone()

    if main_error is not None:
        raise main_error
    if failure_count.item() != 0:
        raise RuntimeError(f"{operation} failed on the main process; stopping all ranks at this operation")


def save_state_all_ranks(accelerator, args, state_dir, retention_callback) -> None:
    """Save one Accelerate state atomically with respect to main-rank retention."""
    del args
    accelerator.save_state(state_dir)
    accelerator.wait_for_everyone()
    _run_main_process_action_all_ranks(
        accelerator,
        f"state retention for {state_dir}",
        retention_callback,
    )


def _require_resume_training_progress(args: argparse.Namespace) -> None:
    if not args.resume:
        return

    if not getattr(args, "resume_from_huggingface", False):
        progress_path = os.path.join(os.path.expanduser(os.fspath(args.resume)), "custom_checkpoint_0.pkl")
        if os.path.isfile(progress_path):
            return
        raise FullFineTuneResumeProgressError(
            "--resume requires a state directory containing the registered TrainingProgress checkpoint "
            f"custom_checkpoint_0.pkl; not found in {args.resume}"
        )

    resume_parts = args.resume.split("/")
    if len(resume_parts) < 3:
        raise ValueError("--resume_from_huggingface requires --resume in repo_owner/repo_name/path format")
    repo_id = "/".join(resume_parts[:2])
    path_in_repo = "/".join(resume_parts[2:])
    revision = None
    repo_type = None
    if ":" in path_in_repo:
        divided = path_in_repo.split(":")
        if len(divided) == 2:
            path_in_repo, revision = divided
            repo_type = "model"
        elif len(divided) == 3:
            path_in_repo, revision, repo_type = divided
        else:
            raise ValueError("--resume contains too many ':'-separated Hugging Face qualifiers")

    remote_files = huggingface_utils.list_dir(
        repo_id=repo_id,
        subfolder=path_in_repo,
        revision=revision,
        token=getattr(args, "huggingface_token", None),
        repo_type=repo_type,
    )
    progress_path = f"{path_in_repo.rstrip('/')}/custom_checkpoint_0.pkl".lstrip("/")
    if any(getattr(remote_file, "rfilename", "").lstrip("/") == progress_path for remote_file in remote_files):
        return
    raise FullFineTuneResumeProgressError(
        "--resume requires the registered TrainingProgress checkpoint custom_checkpoint_0.pkl; "
        f"not found in Hugging Face path {repo_id}/{path_in_repo}"
    )


class FullFineTuningTrainerMixin:
    """Shared full-transformer lifecycle for image model trainers."""

    def validate_full_finetune_model_args(self, args: argparse.Namespace) -> None:
        del args

    def load_full_finetune_transformer(
        self,
        accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device,
        trainable_dtype: torch.dtype,
    ):
        return self.load_transformer(
            accelerator,
            args,
            dit_path,
            attn_mode,
            split_attn,
            loading_device,
            trainable_dtype,
        )

    def full_finetune_metadata(self, args: argparse.Namespace) -> dict:
        del args
        return {}

    def compute_loss(
        self,
        args,
        output,
        timesteps,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        global_step,
    ):
        output.target = output.target.to(network_dtype)
        return super().compute_loss(
            args,
            output,
            timesteps,
            noise_scheduler,
            dit_dtype,
            network_dtype,
            global_step,
        )

    @staticmethod
    def _attention_mode(args: argparse.Namespace) -> str:
        if args.sdpa:
            return "torch"
        if args.flash_attn:
            return "flash"
        if args.sage_attn:
            return "sageattn"
        if args.xformers:
            return "xformers"
        if args.flash3:
            return "flash3"
        raise ValueError(
            "either --sdpa, --flash-attn, --flash3, --sage-attn or --xformers must be specified / "
            "--sdpa, --flash-attn, --flash3, --sage-attn, --xformersのいずれかを指定してください"
        )

    @staticmethod
    def _metadata_value(value) -> str:
        return str(value)

    def _build_full_finetune_metadata(
        self,
        args,
        session_id,
        training_started_at,
        train_dataset_group,
        train_dataloader,
        num_train_epochs,
        optimizer_name,
        optimizer_args,
    ) -> tuple[dict[str, str], dict[str, str]]:
        metadata = {
            "ss_session_id": session_id,
            "ss_training_started_at": training_started_at,
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
            "ss_base_model_version": self.architecture_full_name,
            "ss_mixed_precision": args.mixed_precision,
            "ss_seed": args.seed,
            "ss_training_comment": args.training_comment,
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if optimizer_args else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_fp8_base": False,
            "ss_full_fp16": False,
            "ss_full_bf16": bool(args.full_bf16),
            "ss_training_type": "full-finetune",
            "ss_full_finetune": True,
            "ss_weighting_scheme": args.weighting_scheme,
            "ss_logit_mean": args.logit_mean,
            "ss_logit_std": args.logit_std,
            "ss_mode_scale": args.mode_scale,
            "ss_guidance_scale": args.guidance_scale,
            "ss_timestep_sampling": args.timestep_sampling,
            "ss_sigmoid_scale": args.sigmoid_scale,
            "ss_discrete_flow_shift": args.discrete_flow_shift,
        }
        model_metadata = dict(self.full_finetune_metadata(args))
        metadata.update(model_metadata)
        metadata.update(self.extra_metadata(args))

        datasets_metadata = [dataset.get_metadata() for dataset in train_dataset_group.datasets]
        metadata["ss_datasets"] = json.dumps(datasets_metadata)

        if args.dit is not None:
            model_name = os.path.basename(args.dit) if os.path.exists(args.dit) else args.dit
            metadata["ss_sd_model_name"] = model_name
        if args.vae is not None:
            vae_name = os.path.basename(args.vae) if os.path.exists(args.vae) else args.vae
            metadata["ss_vae_name"] = vae_name

        metadata = {key: self._metadata_value(value) for key, value in metadata.items()}
        minimum_keys = {
            "ss_base_model_version",
            "ss_fp8_base",
            "ss_full_bf16",
            "ss_full_finetune",
            "ss_training_type",
        }
        minimum_keys.update(model_metadata)
        minimum_metadata = {key: metadata[key] for key in minimum_keys if key in metadata}
        return metadata, minimum_metadata

    def _save_full_finetune_model(
        self,
        accelerator,
        args,
        raw_model,
        forward_model,
        ckpt_name,
        metadata,
        minimum_metadata,
        steps,
        epoch_no,
        save_dtype,
        force_sync_upload=False,
    ) -> None:
        def save_checkpoint():
            self._save_full_finetune_model_on_main(
                accelerator,
                args,
                raw_model,
                forward_model,
                ckpt_name,
                metadata,
                minimum_metadata,
                steps,
                epoch_no,
                save_dtype,
                force_sync_upload,
            )

        _run_main_process_action_all_ranks(
            accelerator,
            f"checkpoint export for {ckpt_name}",
            save_checkpoint,
        )

    def _save_full_finetune_model_on_main(
        self,
        accelerator,
        args,
        raw_model,
        forward_model,
        ckpt_name,
        metadata,
        minimum_metadata,
        steps,
        epoch_no,
        save_dtype,
        force_sync_upload=False,
    ) -> None:

        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")

        checkpoint_metadata = dict(metadata)
        checkpoint_metadata.update(
            {
                "ss_training_finished_at": str(time.time()),
                "ss_steps": str(steps),
                "ss_epoch": str(epoch_no),
            }
        )
        metadata_to_save = dict(minimum_metadata if args.no_metadata else checkpoint_metadata)

        title = args.metadata_title if args.metadata_title is not None else args.output_name
        if args.min_timestep is not None or args.max_timestep is not None:
            min_timestep = args.min_timestep if args.min_timestep is not None else 0
            max_timestep = args.max_timestep if args.max_timestep is not None else 1000
            metadata_timesteps = (min_timestep, max_timestep)
        else:
            metadata_timesteps = None

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
            timesteps=metadata_timesteps,
            is_lora=False,
            custom_arch=args.metadata_arch,
        )
        metadata_to_save.update(sai_metadata)
        metadata_to_save = {key: self._metadata_value(value) for key, value in metadata_to_save.items()}

        state_dict = normalize_compiled_state_dict(raw_model.state_dict())
        floating_dtypes = {tensor.dtype for tensor in state_dict.values() if tensor.is_floating_point()}
        if any(dtype != save_dtype for dtype in floating_dtypes):
            raise ValueError(
                f"full-finetune checkpoint contains floating tensors outside the save dtype {save_dtype}: "
                f"{sorted(str(dtype) for dtype in floating_dtypes)}"
            )

        if args.mem_eff_save:
            mem_eff_save_file(state_dict, ckpt_file, metadata_to_save)
        else:
            save_file(state_dict, ckpt_file, metadata=metadata_to_save)

        if args.huggingface_repo_id is not None:
            huggingface_utils.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        self.on_post_save(
            args,
            accelerator,
            None,
            forward_model,
            ckpt_name,
            save_dtype,
            metadata_to_save,
            force_sync_upload,
        )

    def _sample_full_finetune_images(
        self,
        accelerator,
        args,
        epoch,
        steps,
        vae,
        raw_model,
        forward_model,
        sample_parameters,
        trainable_dtype,
    ) -> None:
        if not should_sample_images(args, steps, epoch):
            return

        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = None
        if torch.cuda.is_available():
            try:
                cuda_rng_states = torch.cuda.get_rng_state_all()
            except Exception:
                cuda_rng_states = None
        previous_training = raw_model.training

        try:
            self.on_before_sample_images(
                accelerator,
                args,
                epoch,
                steps,
                vae,
                forward_model,
                None,
                sample_parameters,
                trainable_dtype,
            )
            self.sample_images(
                accelerator,
                args,
                epoch,
                steps,
                vae,
                forward_model,
                sample_parameters,
                trainable_dtype,
            )
        finally:
            try:
                self.on_after_sample_images(
                    accelerator,
                    args,
                    epoch,
                    steps,
                    vae,
                    forward_model,
                    None,
                    sample_parameters,
                    trainable_dtype,
                )
            finally:
                if self.blocks_to_swap > 0:
                    raw_model.switch_block_swap_for_training()
                raw_model.train(previous_training)
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
                clean_memory_on_device(accelerator.device)

    @staticmethod
    def _remove_checkpoint(args, ckpt_name) -> None:
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        if os.path.exists(ckpt_file):
            os.remove(ckpt_file)

    def _remove_checkpoint_all_ranks(self, accelerator, args, ckpt_name) -> None:
        _run_main_process_action_all_ranks(
            accelerator,
            f"checkpoint retention for {ckpt_name}",
            lambda: self._remove_checkpoint(args, ckpt_name),
        )

    def _save_step_state(self, accelerator, args, global_step) -> None:
        state_name = train_utils.STEP_STATE_NAME.format(args.output_name, global_step)
        state_dir = os.path.join(args.output_dir, state_name)
        old_state_dir = None
        last_n_steps = args.save_last_n_steps_state if args.save_last_n_steps_state else args.save_last_n_steps
        if last_n_steps is not None:
            remove_step = global_step - last_n_steps - 1
            remove_step -= remove_step % args.save_every_n_steps
            if remove_step > 0:
                old_name = train_utils.STEP_STATE_NAME.format(args.output_name, remove_step)
                old_state_dir = os.path.join(args.output_dir, old_name)

        def retain_and_upload():
            if args.save_state_to_huggingface:
                huggingface_utils.upload(args, state_dir, "/" + state_name)
            if old_state_dir is not None and os.path.exists(old_state_dir):
                shutil.rmtree(old_state_dir)

        save_state_all_ranks(accelerator, args, state_dir, retain_and_upload)

    def _save_epoch_state(self, accelerator, args, epoch_no) -> None:
        state_name = train_utils.EPOCH_STATE_NAME.format(args.output_name, epoch_no)
        state_dir = os.path.join(args.output_dir, state_name)
        old_state_dir = None
        last_n_epochs = args.save_last_n_epochs_state if args.save_last_n_epochs_state else args.save_last_n_epochs
        if last_n_epochs is not None:
            remove_epoch = epoch_no - args.save_every_n_epochs * last_n_epochs
            old_name = train_utils.EPOCH_STATE_NAME.format(args.output_name, remove_epoch)
            old_state_dir = os.path.join(args.output_dir, old_name)

        def retain_and_upload():
            if args.save_state_to_huggingface:
                huggingface_utils.upload(args, state_dir, "/" + state_name)
            if old_state_dir is not None and os.path.exists(old_state_dir):
                shutil.rmtree(old_state_dir)

        save_state_all_ranks(accelerator, args, state_dir, retain_and_upload)

    def _save_final_state(self, accelerator, args) -> None:
        state_name = train_utils.LAST_STATE_NAME.format(args.output_name)
        state_dir = os.path.join(args.output_dir, state_name)

        def upload():
            if args.save_state_to_huggingface:
                huggingface_utils.upload(args, state_dir, "/" + state_name)

        save_state_all_ranks(accelerator, args, state_dir, upload)

    @staticmethod
    def _patch_fused_backward(args, accelerator, optimizer, named_parameters) -> None:
        if not args.fused_backward_pass:
            return

        import musubi_tuner.modules.adafactor_fused as adafactor_fused

        adafactor_fused.patch_adafactor_fused(optimizer)
        parameter_groups = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
        for _, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            parameter_group = parameter_groups[id(parameter)]

            def grad_hook(tensor, group=parameter_group):
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                optimizer.step_param(tensor, group)
                tensor.grad = None

            parameter.register_post_accumulate_grad_hook(grad_hook)

    def train(self, args):
        _require_resume_training_progress(args)
        if not self._validate_args_and_init(args):
            return
        self.validate_full_finetune_model_args(args)

        session_id, training_started_at = self._init_session(args)

        accelerator = prepare_accelerator(args)
        validate_full_finetune_distributed_type(accelerator.distributed_type)
        if args.mixed_precision is None:
            args.mixed_precision = accelerator.mixed_precision
        validate_full_finetune_args(args, accelerator.num_processes)
        trainable_dtype = resolve_trainable_dtype(args)
        attn_mode = self._attention_mode(args)

        train_dataset_group, collator, current_epoch = self._build_dataset(args)
        vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
        sample_parameters, vae = self._prepare_sampling(args, accelerator, vae_dtype)

        blocks_to_swap = 0 if args.blocks_to_swap is None else args.blocks_to_swap
        has_block_swap = blocks_to_swap > 0
        self.blocks_to_swap = blocks_to_swap
        loading_device = "cpu" if has_block_swap else accelerator.device

        raw_model = self.load_full_finetune_transformer(
            accelerator,
            args,
            args.dit,
            attn_mode,
            args.split_attn,
            loading_device,
            trainable_dtype,
        )
        raw_model.requires_grad_(True)
        self.on_transformer_loaded(args, accelerator, raw_model)
        mismatched_parameters = [
            name
            for name, parameter in raw_model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != trainable_dtype
        ]
        if mismatched_parameters:
            raise ValueError(
                "load_full_finetune_transformer must load parameters directly in "
                f"{trainable_dtype}; mismatched parameters: {', '.join(mismatched_parameters[:5])}"
            )

        if args.gradient_checkpointing:
            raw_model.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
        if has_block_swap:
            swap_config = BlockSwapConfig.from_args(args, accelerator.device, supports_backward=True)
            raw_model.enable_block_swap(blocks_to_swap, swap_config)
            raw_model.move_to_device_except_swap_blocks(accelerator.device)
        if args.compile:
            raw_model = self.compile_transformer(args, raw_model)

        named_parameters = [(name, parameter) for name, parameter in raw_model.named_parameters() if parameter.requires_grad]
        if not named_parameters:
            raise ValueError("full finetuning requires at least one trainable transformer parameter")
        params_to_optimize = [{"params": [parameter for _, parameter in named_parameters], "lr": args.learning_rate}]
        optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn = self.get_optimizer(
            args, params_to_optimize
        )

        workers = min(args.max_data_loader_n_workers, os.cpu_count() or 1)
        sampler_generator = torch.Generator().manual_seed(args.seed)
        dataloader_generator = torch.Generator().manual_seed(args.seed)
        train_sampler = torch.utils.data.RandomSampler(train_dataset_group, generator=sampler_generator)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=workers,
            persistent_workers=bool(args.persistent_data_loader_workers and workers > 0),
            generator=dataloader_generator,
        )
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
        train_dataset_group.set_max_train_steps(args.max_train_steps)
        lr_scheduler = self.get_lr_scheduler(args, optimizer, accelerator.num_processes)

        if has_block_swap:
            forward_model = accelerator.prepare(raw_model, device_placement=[False])
        else:
            forward_model = accelerator.prepare(raw_model)
        raw_model = accelerator.unwrap_model(forward_model, keep_fp32_wrapper=False)
        if has_block_swap:
            raw_model.move_to_device_except_swap_blocks(accelerator.device)
            raw_model.prepare_block_swap_before_forward()

        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
        self.raw_model = raw_model
        self.forward_model = forward_model

        progress = TrainingProgress(sampler_seed=args.seed)
        self.training_progress = progress
        accelerator.register_for_checkpointing(progress)
        self._patch_fused_backward(args, accelerator, optimizer, named_parameters)

        resumed = False
        if args.resume:
            try:
                resumed = self.resume_from_local_or_hf_if_specified(accelerator, args)
            except RuntimeError as error:
                if "custom checkpoint" not in str(error).lower():
                    raise
                raise FullFineTuneResumeProgressError(
                    "--resume state is missing the registered TrainingProgress checkpoint"
                ) from error
            if progress.sampler_seed is None:
                raise FullFineTuneResumeProgressError(
                    "--resume TrainingProgress checkpoint does not contain the effective sampler seed"
                )
            args.seed = progress.sampler_seed

        self.on_train_start(args, accelerator, None, forward_model, optimizer)
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        metadata, minimum_metadata = self._build_full_finetune_metadata(
            args,
            session_id,
            training_started_at,
            train_dataset_group,
            train_dataloader,
            num_train_epochs,
            optimizer_name,
            optimizer_args,
        )
        self.full_metadata = metadata
        self.minimum_metadata = minimum_metadata

        if accelerator.is_main_process:
            init_kwargs = {}
            if args.wandb_run_name:
                init_kwargs["wandb"] = {"name": args.wandb_run_name}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            sanitized_config = train_utils.get_sanitized_config_or_none(args) if hasattr(args, "log_config") else None
            accelerator.init_trackers(
                "full_finetune" if args.log_tracker_name is None else args.log_tracker_name,
                config=sanitized_config,
                init_kwargs=init_kwargs,
            )

        progress_bar = tqdm(
            range(args.max_train_steps),
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc="steps",
        )
        if progress.global_step:
            progress_bar.update(progress.global_step)

        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
        loss_recorder = train_utils.LossRecorder()
        save_dtype = train_utils.resolve_save_dtype(args.save_precision, full_bf16=args.full_bf16)

        raw_model.train()
        optimizer_train_fn()
        if should_sample_images(args, progress.global_step, epoch=0):
            optimizer_eval_fn()
            self._sample_full_finetune_images(
                accelerator,
                args,
                0,
                progress.global_step,
                vae,
                raw_model,
                forward_model,
                sample_parameters,
                trainable_dtype,
            )
            optimizer_train_fn()
        if accelerator.trackers:
            accelerator.log({}, step=progress.global_step)
        clean_memory_on_device(accelerator.device)

        for epoch in range(progress.epoch, num_train_epochs):
            if progress.global_step >= args.max_train_steps and progress.next_batch < len(train_dataloader):
                break

            current_epoch.value = epoch + 1
            epoch_seed = args.seed + epoch
            sampler_generator.manual_seed(epoch_seed)
            dataloader_generator.manual_seed(epoch_seed)

            epoch_dataloader = train_dataloader
            first_batch = 0
            if resumed:
                first_batch = progress.next_batch
                if first_batch >= len(train_dataloader):
                    epoch_dataloader = ()
                else:
                    epoch_dataloader = accelerator.skip_first_batches(train_dataloader, first_batch)
                resumed = False

            progress.epoch = epoch
            progress.next_batch = first_batch

            epoch_exhausted = False
            for step, batch in enumerate(epoch_dataloader, start=first_batch):
                if progress.global_step >= args.max_train_steps:
                    break

                latents = batch["latents"]
                with accelerator.accumulate(forward_model):
                    latents = self.scale_shift_latents(latents)
                    noise = torch.randn_like(latents)
                    loss, loss_metrics = self.process_batch(
                        args,
                        accelerator,
                        forward_model,
                        None,
                        batch,
                        latents,
                        noise,
                        noise_scheduler,
                        trainable_dtype,
                        trainable_dtype,
                        vae,
                        progress.global_step,
                    )
                    accelerator.backward(loss)

                    if not args.fused_backward_pass:
                        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                            accelerator.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
                        if has_block_swap and args.block_swap_optimizer_patch_params:
                            for parameter_group in optimizer.param_groups:
                                for parameter in parameter_group["params"]:
                                    if parameter.grad is not None and parameter.device != parameter.grad.device:
                                        parameter.grad = parameter.grad.to(parameter.device, non_blocking=True)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        lr_scheduler.step()

                    self.on_post_optimizer_step(
                        args,
                        accelerator,
                        None,
                        forward_model,
                        accelerator.sync_gradients,
                        progress.global_step,
                    )

                progress.next_batch = step + 1
                if accelerator.sync_gradients:
                    progress.global_step += 1
                    progress_bar.update(1)

                    should_sample = should_sample_images(args, progress.global_step, epoch=None)
                    should_save = args.save_every_n_steps is not None and progress.global_step % args.save_every_n_steps == 0
                    if should_sample or should_save:
                        optimizer_eval_fn()
                        if should_sample:
                            self._sample_full_finetune_images(
                                accelerator,
                                args,
                                None,
                                progress.global_step,
                                vae,
                                raw_model,
                                forward_model,
                                sample_parameters,
                                trainable_dtype,
                            )
                        if should_save:
                            accelerator.wait_for_everyone()
                            ckpt_name = train_utils.get_step_ckpt_name(args.output_name, progress.global_step)
                            self._save_full_finetune_model(
                                accelerator,
                                args,
                                raw_model,
                                forward_model,
                                ckpt_name,
                                metadata,
                                minimum_metadata,
                                progress.global_step,
                                epoch,
                                save_dtype,
                            )
                            if args.save_state:
                                self._save_step_state(accelerator, args, progress.global_step)
                            remove_step = train_utils.get_remove_step_no(args, progress.global_step)
                            if remove_step is not None:
                                self._remove_checkpoint_all_ranks(
                                    accelerator,
                                    args,
                                    train_utils.get_step_ckpt_name(args.output_name, remove_step),
                                )
                        optimizer_train_fn()

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                average_loss = loss_recorder.moving_average
                progress_bar.set_postfix(avr_loss=average_loss)
                if accelerator.trackers:
                    logs = self.generate_step_logs(
                        args,
                        current_loss,
                        average_loss,
                        lr_scheduler,
                        None,
                        optimizer,
                    )
                    logs.update(loss_metrics)
                    logs.update(self.extra_step_logs(args, logs))
                    accelerator.log(logs, step=progress.global_step)

                if progress.global_step >= args.max_train_steps:
                    epoch_exhausted = progress.next_batch >= len(train_dataloader)
                    break
            else:
                epoch_exhausted = True

            if not epoch_exhausted:
                break

            progress.epoch = epoch + 1
            progress.next_batch = 0
            if accelerator.trackers and loss_recorder.loss_list:
                accelerator.log({"loss/epoch": loss_recorder.moving_average}, step=epoch + 1)
            accelerator.wait_for_everyone()

            optimizer_eval_fn()
            if args.save_every_n_epochs is not None:
                should_save_epoch = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if should_save_epoch:
                    ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, epoch + 1)
                    self._save_full_finetune_model(
                        accelerator,
                        args,
                        raw_model,
                        forward_model,
                        ckpt_name,
                        metadata,
                        minimum_metadata,
                        progress.global_step,
                        epoch + 1,
                        save_dtype,
                    )
                    if args.save_state:
                        self._save_epoch_state(accelerator, args, epoch + 1)
                    remove_epoch = train_utils.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch is not None:
                        self._remove_checkpoint_all_ranks(
                            accelerator,
                            args,
                            train_utils.get_epoch_ckpt_name(args.output_name, remove_epoch),
                        )

            self._sample_full_finetune_images(
                accelerator,
                args,
                epoch + 1,
                progress.global_step,
                vae,
                raw_model,
                forward_model,
                sample_parameters,
                trainable_dtype,
            )
            optimizer_train_fn()

        progress_bar.close()
        optimizer_eval_fn()
        if args.save_state or args.save_state_on_train_end:
            self._save_final_state(accelerator, args)

        accelerator.wait_for_everyone()
        ckpt_name = train_utils.get_last_ckpt_name(args.output_name)
        self._save_full_finetune_model(
            accelerator,
            args,
            raw_model,
            forward_model,
            ckpt_name,
            metadata,
            minimum_metadata,
            progress.global_step,
            num_train_epochs,
            save_dtype,
            force_sync_upload=True,
        )
        accelerator.end_training()
