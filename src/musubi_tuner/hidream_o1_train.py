import argparse
import json
import math
from multiprocessing import Value
import os
import random
import time

import toml
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from musubi_tuner import hidream_o1_train_network
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.hidream_o1_train_network import HiDreamO1NetworkTrainer
from musubi_tuner.hv_train_network import (
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_MINIMUM_KEYS,
    clean_memory_on_device,
    collator_class,
    compute_loss_weighting_for_sd3,
    prepare_accelerator,
    read_config_from_file,
    set_seed,
    setup_parser_common,
    should_sample_images,
)
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.utils import huggingface_utils, model_utils, sai_model_spec, train_utils
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file

import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HiDreamO1Trainer(HiDreamO1NetworkTrainer):
    def __init__(self):
        super().__init__()

    def _validate_full_finetune_args(self, args: argparse.Namespace):
        if args.dataset_config is None:
            raise ValueError("dataset_config is required")
        if args.dit is None:
            raise ValueError("path to DiT model is required")

        if args.fp8_base or args.fp8_scaled:
            raise ValueError("HiDream-O1 full finetuning does not support --fp8_base / --fp8_scaled.")

        if args.sage_attn:
            raise ValueError("SageAttention doesn't support training currently. Use the default attention path or --flash_attn.")
        if args.xformers or args.flash3:
            raise ValueError("HiDream-O1 full finetuning supports the default attention path and --flash_attn only.")

        lora_only_args = {
            "--network_weights": args.network_weights,
            "--network_module": args.network_module,
            "--network_dim": args.network_dim,
            "--network_alpha": None if args.network_alpha == 1 else args.network_alpha,
            "--network_dropout": args.network_dropout,
            "--network_args": args.network_args,
            "--dim_from_weights": args.dim_from_weights,
            "--scale_weight_norms": args.scale_weight_norms,
            "--base_weights": args.base_weights,
            "--base_weights_multiplier": args.base_weights_multiplier,
        }
        specified_lora_args = [name for name, value in lora_only_args.items() if value not in (None, False, [])]
        if specified_lora_args:
            raise ValueError("HiDream-O1 full finetuning does not use LoRA/network arguments: " + ", ".join(specified_lora_args))

    def _set_t2i_visual_dummy_skip(self, transformer, enabled: bool):
        inner_model = getattr(transformer, "model", None)
        if inner_model is not None and hasattr(inner_model, "skip_t2i_visual_dummy"):
            inner_model.skip_t2i_visual_dummy = enabled

    def _freeze_inactive_modules_for_dataset(self, transformer, has_control: bool) -> tuple[int, int]:
        self._set_t2i_visual_dummy_skip(transformer, False)
        if has_control:
            logger.info("HiDream-O1 control/reference data detected; visual encoder remains trainable.")
            return 0, 0

        visual = getattr(transformer, "visual", None)
        if visual is None:
            logger.warning("HiDream-O1 visual encoder was not found; no T2I visual parameters were frozen.")
            return 0, 0

        frozen_params = 0
        frozen_numel = 0
        for parameter in visual.parameters():
            if parameter.requires_grad:
                frozen_params += 1
                frozen_numel += parameter.numel()
                parameter.requires_grad_(False)

        self._set_t2i_visual_dummy_skip(transformer, True)
        logger.info(
            "Pure T2I dataset detected; froze HiDream-O1 visual encoder "
            f"({frozen_params} tensors, {frozen_numel} parameters) and disabled dummy visual training pass."
        )
        return frozen_params, frozen_numel

    def train(self, args):
        if torch.cuda.is_available():
            if args.cuda_allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 on CUDA")
            if args.cuda_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark")

        self._validate_full_finetune_args(args)
        self.handle_model_specific_args(args)

        args.full_fp16 = False
        args.dit_dtype = "bfloat16" if args.full_bf16 else "float32"

        if args.show_timesteps:
            self.show_timesteps(args)
            return

        session_id = random.randint(0, 2**32)
        training_started_at = time.time()

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        if args.num_timestep_buckets is not None:
            logger.info(f"Using timestep bucketing. Number of buckets: {args.num_timestep_buckets}")
        self.num_timestep_buckets = args.num_timestep_buckets

        current_epoch = Value("i", 0)

        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_utils.load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args, architecture=self.architecture)
        train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            blueprint.dataset_group, training=True, num_timestep_buckets=self.num_timestep_buckets, shared_epoch=current_epoch
        )
        # Driven by --task (set on self._i2i_training in handle_model_specific_args). Consistency with the
        # actual dataset is enforced in call_dit, shared with the LoRA trainer.
        has_control = self._i2i_training

        if train_dataset_group.num_train_items == 0:
            raise ValueError(
                "No training items found in the dataset. Please ensure that the pixel/Text Encoder cache has been created beforehand."
            )

        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = collator_class(current_epoch, ds_for_collator)

        logger.info("preparing accelerator")
        accelerator = prepare_accelerator(args)
        if args.mixed_precision is None:
            args.mixed_precision = accelerator.mixed_precision
            logger.info(f"mixed precision set to {args.mixed_precision}")
        is_main_process = accelerator.is_main_process
        if args.full_bf16 and args.mixed_precision != "bf16":
            raise ValueError("full_bf16 requires mixed_precision='bf16'.")

        dit_dtype = model_utils.str_to_dtype(args.dit_dtype)
        logger.info(f"DiT precision: {dit_dtype}")

        sample_parameters = None
        vae = None
        if args.sample_prompts:
            sample_parameters = self.process_sample_prompts(args, accelerator, args.sample_prompts)
            vae = self.load_vae(args, vae_dtype=torch.float32, vae_path=args.vae)
            vae.requires_grad_(False)
            vae.eval()

        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
        self.blocks_to_swap = blocks_to_swap
        loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device

        logger.info(f"Loading DiT model from {args.dit}")
        attn_mode = "flash" if args.flash_attn else "torch"
        dit_weight_dtype = dit_dtype
        transformer = self.load_transformer(
            accelerator, args, args.dit, attn_mode, args.split_attn, loading_device, dit_weight_dtype
        )

        frozen_visual_params, frozen_visual_numel = self._freeze_inactive_modules_for_dataset(transformer, has_control)

        if blocks_to_swap > 0:
            logger.info(
                f"enable swap {blocks_to_swap} blocks to CPU from device: {accelerator.device}, "
                f"use pinned memory: {args.use_pinned_memory_for_block_swap}"
            )
            transformer.enable_block_swap(
                blocks_to_swap, accelerator.device, supports_backward=True, use_pinned_memory=args.use_pinned_memory_for_block_swap
            )
            transformer.move_to_device_except_swap_blocks(accelerator.device)

        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)

        accelerator.print("prepare optimizer, data loader etc.")

        name_and_params = [(name, parameter) for name, parameter in transformer.named_parameters() if parameter.requires_grad]
        if not name_and_params:
            raise ValueError("No trainable HiDream-O1 parameters were found.")
        params_to_optimize = [{"params": [p for _, p in name_and_params], "lr": args.learning_rate}]
        param_names = [[n for n, _ in name_and_params]]

        n_params = 0
        for group in params_to_optimize:
            for p in group["params"]:
                n_params += p.numel()
        accelerator.print(f"number of trainable parameters: {n_params}")

        optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn = self.get_optimizer(
            args, params_to_optimize
        )

        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs is: {args.max_train_steps}")

        train_dataset_group.set_max_train_steps(args.max_train_steps)
        lr_scheduler = self.get_lr_scheduler(args, optimizer, accelerator.num_processes)

        if args.full_bf16:
            accelerator.print("enable full bf16 training.")

        if blocks_to_swap > 0:
            transformer = accelerator.prepare(transformer, device_placement=[False])
            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)
            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        else:
            transformer = accelerator.prepare(transformer)

        if args.compile:
            transformer = self.compile_transformer(args, transformer)
            transformer.__dict__["_orig_mod"] = transformer

        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
        training_model = transformer
        training_model.train()

        self.resume_from_local_or_hf_if_specified(accelerator, args)

        if args.fused_backward_pass:
            import musubi_tuner.modules.adafactor_fused as adafactor_fused

            adafactor_fused.patch_adafactor_fused(optimizer)

            for param_group, param_name_group in zip(optimizer.param_groups, param_names):
                for parameter, param_name in zip(param_group["params"], param_name_group):
                    if parameter.requires_grad:

                        def create_grad_hook(p_name, p_group):
                            def grad_hook(tensor: torch.Tensor):
                                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                    accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                                optimizer.step_param(tensor, p_group)
                                tensor.grad = None

                            return grad_hook

                        parameter.register_post_accumulate_grad_hook(create_grad_hook(param_name, param_group))

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        accelerator.print("running training")
        accelerator.print(f"  num train items: {train_dataset_group.num_train_items}")
        accelerator.print(f"  num batches per epoch: {len(train_dataloader)}")
        accelerator.print(f"  num epochs: {num_train_epochs}")
        accelerator.print(f"  batch size per device: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}")
        accelerator.print(f"  gradient accumulation steps: {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps: {args.max_train_steps}")

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
            SS_METADATA_KEY_BASE_MODEL_VERSION: self.architecture_full_name,
            "ss_mixed_precision": args.mixed_precision,
            "ss_seed": args.seed,
            "ss_training_comment": args.training_comment,
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_full_fp16": False,
            "ss_full_bf16": bool(args.full_bf16),
            "ss_has_control": bool(has_control),
            "ss_frozen_visual_params": frozen_visual_numel,
            "ss_weighting_scheme": args.weighting_scheme,
            "ss_logit_mean": args.logit_mean,
            "ss_logit_std": args.logit_std,
            "ss_mode_scale": args.mode_scale,
            "ss_guidance_scale": args.guidance_scale,
            "ss_timestep_sampling": args.timestep_sampling,
            "ss_sigmoid_scale": args.sigmoid_scale,
            "ss_discrete_flow_shift": args.discrete_flow_shift,
            "ss_model_type": args.model_type,
            "ss_noise_scale_start": args.noise_scale_start,
            "ss_noise_scale_end": args.noise_scale_end,
            "ss_noise_clip_std": args.noise_clip_std,
        }

        if args.dino_loss_weight > 0:
            metadata.update(
                {
                    "ss_dino_loss_weight": args.dino_loss_weight,
                    "ss_dino_loss_backend": args.dino_loss_backend,
                    "ss_dino_loss_model_type": args.dino_loss_model_type,
                    "ss_dino_loss_layer": args.dino_loss_layer,
                    "ss_dino_loss_feature_mode": args.dino_loss_feature_mode,
                    "ss_dino_loss_resize": args.dino_loss_resize,
                    "ss_dino_loss_use_norm": args.dino_loss_use_norm,
                    "ss_dino_loss_use_gram": args.dino_loss_use_gram,
                    "ss_dino_loss_every_n_steps": args.dino_loss_every_n_steps,
                }
            )

        datasets_metadata = []
        for dataset in train_dataset_group.datasets:
            datasets_metadata.append(dataset.get_metadata())
        metadata["ss_datasets"] = json.dumps(datasets_metadata)

        logger.info(f"set DiT model name for metadata: {args.dit}")
        sd_model_name = args.dit
        if os.path.exists(sd_model_name):
            sd_model_name = os.path.basename(sd_model_name)
        metadata["ss_sd_model_name"] = sd_model_name

        metadata = {k: str(v) for k, v in metadata.items()}

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
                "fine-tuning" if args.log_tracker_name is None else args.log_tracker_name,
                config=train_utils.get_sanitized_config_or_none(args),
                init_kwargs=init_kwargs,
            )

        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")

        epoch_to_start = 0
        global_step = 0
        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")

        loss_recorder = train_utils.LossRecorder()
        del train_dataset_group

        def save_model(
            ckpt_name: str, unwrapped_model, steps, epoch_no, force_sync_upload=False, use_memory_efficient_saving=False
        ):
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
                is_lora=False,
                custom_arch=args.metadata_arch,
            )

            metadata_to_save.update(sai_metadata)

            has_self_ref_orig_mod_module = (
                hasattr(unwrapped_model, "_modules")
                and "_orig_mod" in unwrapped_model._modules
                and unwrapped_model._modules["_orig_mod"] is unwrapped_model
            )
            if has_self_ref_orig_mod_module:
                del unwrapped_model._modules["_orig_mod"]

            try:
                state_dict = unwrapped_model.state_dict()
            finally:
                if has_self_ref_orig_mod_module:
                    unwrapped_model._modules["_orig_mod"] = unwrapped_model

            if any("_orig_mod." in key for key in state_dict.keys()):
                logger.info("detected compiled model, getting original model state dict for saving")
                state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}

            if use_memory_efficient_saving:
                mem_eff_save_file(state_dict, ckpt_file, metadata_to_save)
            else:
                save_file(state_dict, ckpt_file, metadata_to_save)

            if args.huggingface_repo_id is not None:
                huggingface_utils.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        if should_sample_images(args, global_step, epoch=0):
            optimizer_eval_fn()
            self.sample_images(accelerator, args, 0, global_step, vae, transformer, sample_parameters, dit_dtype)
            optimizer_train_fn()
        if len(accelerator.trackers) > 0:
            accelerator.log({}, step=0)

        unwrapped_transformer = accelerator.unwrap_model(transformer)
        first_param = next(iter(unwrapped_transformer.parameters()), None)
        logger.info(
            "DiT dtype: %s, device: %s",
            first_param.dtype if first_param is not None else None,
            first_param.device if first_param is not None else accelerator.device,
        )

        clean_memory_on_device(accelerator.device)
        optimizer_train_fn()

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
            current_epoch.value = epoch + 1
            metadata["ss_epoch"] = str(epoch + 1)

            for step, batch in enumerate(train_dataloader):
                latents = batch["latents"]

                with accelerator.accumulate(training_model):
                    latents = self.scale_shift_latents(latents)
                    noise = torch.randn_like(latents)

                    noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
                        args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
                    )

                    weighting = compute_loss_weighting_for_sd3(
                        args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, dit_dtype
                    )

                    dit_output = self.call_dit(
                        args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, dit_dtype
                    )
                    model_pred, target = dit_output.pred, dit_output.target
                    loss = torch.nn.functional.mse_loss(model_pred.to(dit_dtype), target.to(dit_dtype), reduction="none")

                    if weighting is not None:
                        loss = loss * weighting

                    loss = loss.mean()
                    loss, dino_logs = self.apply_dino_loss(
                        args, loss, model_pred, target, dit_output.extra["pixel_grid_hw"], global_step
                    )

                    accelerator.backward(loss)

                    if not args.fused_backward_pass:
                        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                            accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                        if blocks_to_swap > 0 and args.block_swap_optimizer_patch_params:
                            unwrapped_optimizer = accelerator.unwrap_model(optimizer)
                            for group in unwrapped_optimizer.param_groups:
                                for param in group["params"]:
                                    if param.grad is not None and param.device != param.grad.device:
                                        param.grad = param.grad.to(param.device, non_blocking=True)

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        lr_scheduler.step()

                keys_scaled, mean_norm, maximum_norm = None, None, None

                if accelerator.sync_gradients:
                    if global_step == 0:
                        progress_bar.reset()
                    progress_bar.update(1)
                    global_step += 1

                    should_sampling = should_sample_images(args, global_step, epoch=None)
                    should_saving = args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0

                    if should_sampling or should_saving:
                        optimizer_eval_fn()
                        if should_sampling:
                            self.sample_images(accelerator, args, None, global_step, vae, transformer, sample_parameters, dit_dtype)

                        if should_saving:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process:
                                ckpt_name = train_utils.get_step_ckpt_name(args.output_name, global_step)
                                save_model(
                                    ckpt_name,
                                    accelerator.unwrap_model(transformer),
                                    global_step,
                                    epoch,
                                    use_memory_efficient_saving=args.mem_eff_save,
                                )

                                if args.save_state:
                                    train_utils.save_and_remove_state_stepwise(args, accelerator, global_step)

                                remove_step_no = train_utils.get_remove_step_no(args, global_step)
                                if remove_step_no is not None:
                                    remove_ckpt_name = train_utils.get_step_ckpt_name(args.output_name, remove_step_no)
                                    remove_model(remove_ckpt_name)
                        optimizer_train_fn()

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss = loss_recorder.moving_average
                progress_bar.set_postfix(**{"avr_loss": avr_loss})

                if len(accelerator.trackers) > 0:
                    logs = self.generate_step_logs(
                        args, current_loss, avr_loss, lr_scheduler, None, optimizer, keys_scaled, mean_norm, maximum_norm
                    )
                    logs.update(dino_logs)
                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if len(accelerator.trackers) > 0:
                accelerator.log({"loss/epoch": loss_recorder.moving_average}, step=epoch + 1)

            accelerator.wait_for_everyone()

            optimizer_eval_fn()
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, epoch + 1)
                    save_model(
                        ckpt_name,
                        accelerator.unwrap_model(transformer),
                        global_step,
                        epoch + 1,
                        use_memory_efficient_saving=args.mem_eff_save,
                    )

                    remove_epoch_no = train_utils.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, remove_epoch_no)
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        train_utils.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            self.sample_images(accelerator, args, epoch + 1, global_step, vae, transformer, sample_parameters, dit_dtype)
            optimizer_train_fn()

        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            transformer = accelerator.unwrap_model(transformer)

        accelerator.end_training()
        optimizer_eval_fn()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            train_utils.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_utils.get_last_ckpt_name(args.output_name)
            save_model(
                ckpt_name,
                transformer,
                global_step,
                num_train_epochs,
                force_sync_upload=True,
                use_memory_efficient_saving=args.mem_eff_save,
            )
            logger.info("model saved.")


def hidream_o1_finetune_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--full_bf16", action="store_true", help="Enable full bfloat16 training for HiDream-O1")
    parser.add_argument("--fused_backward_pass", action="store_true", help="Use fused backward pass for Adafactor optimizer")
    parser.add_argument(
        "--mem_eff_save",
        action="store_true",
        help="Enable memory efficient model saving. Optimizer state saving still uses the normal accelerator state path.",
    )
    parser.add_argument(
        "--block_swap_optimizer_patch_params",
        action="store_true",
        help=(
            "Patch optimizer parameters for block swap when blocks_to_swap > 0. Only works for some optimizers. "
            "Not needed when using --fused_backward_pass."
        ),
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = hidream_o1_train_network.hidream_o1_setup_parser(parser)
    parser = hidream_o1_finetune_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if args.vae is not None:
        logger.warning("HiDream-O1 is a pixel-space model; --vae is ignored.")
    if args.vae_dtype is not None:
        logger.warning("HiDream-O1 is a pixel-space model; --vae_dtype is ignored.")

    trainer = HiDreamO1Trainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
