import argparse
import json
import math
from multiprocessing import Value
import os
import random
import time
from typing import Optional

import toml
import torch
from tqdm import tqdm
from accelerate import Accelerator
from safetensors.torch import save_file

from musubi_tuner import qwen_image_train_network
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.qwen_image import qwen_image_model, qwen_image_utils
from musubi_tuner.hv_train_network import (
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_MINIMUM_KEYS,
    collator_class,
    compute_loss_weighting_for_sd3,
    clean_memory_on_device,
    prepare_accelerator,
    setup_parser_common,
    read_config_from_file,
    should_sample_images,
    set_seed,
)
import logging

from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer
from musubi_tuner.utils import huggingface_utils, model_utils, sai_model_spec, train_utils
from musubi_tuner.utils.device_utils import synchronize_device
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, load_safetensors, mem_eff_save_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QwenImageTrainer(QwenImageNetworkTrainer):
    def __init__(self):
        super().__init__()

    # region model specific

    def handle_model_specific_args(self, args):
        super().handle_model_specific_args(args)

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
        if "-00001-of-00" in dit_path:
            logger.info("Pruned model detection is disabled because the weights are split into multiple files.")
            model = qwen_image_model.load_qwen_image_model(
                accelerator.device,
                dit_path,
                attn_mode,
                split_attn,
                args.model_version == "edit-2511",
                loading_device,
                dit_weight_dtype,
                args.fp8_scaled,
                disable_numpy_memmap=args.disable_numpy_memmap,
            )
            return model

        # Pruned model with sparse index support
        block_indices = set()
        with MemoryEfficientSafeOpen(dit_path) as f:
            for key in f.keys():
                if key.startswith("transformer_blocks."):
                    block_indices.add(int(key.split(".")[1]))

        total_num_blocks = len(block_indices)
        block_indices = sorted(list(block_indices))
        block_index_map = None
        if total_num_blocks != 60:
            logger.info(f"Detected pruned DiT model with {total_num_blocks} blocks")
            block_index_map = {block_indices[i]: i for i in range(total_num_blocks)}
            logger.info(f"Block index map: {block_index_map}")

        # create model
        model = qwen_image_model.create_model(
            attn_mode, split_attn, args.model_version == "edit-2511", dit_weight_dtype, num_layers=total_num_blocks
        )

        # load weights from disk
        logger.info(f"Loading weights from {dit_path}")
        if block_index_map is None:
            # uses official safetensors loader
            state_dict = load_safetensors(
                dit_path,
                device=loading_device,
                disable_mmap=True,
                dtype=dit_weight_dtype,
                disable_numpy_memmap=args.disable_numpy_memmap,
            )
        else:
            loading_device = torch.device(loading_device) if loading_device is not None else None
            state_dict = {}
            with MemoryEfficientSafeOpen(dit_path, disable_numpy_memmap=args.disable_numpy_memmap) as f:
                for key in f.keys():
                    state_dict_key = key
                    if key.startswith("transformer_blocks."):
                        block_index = int(key.split(".")[1])
                        new_block_index = block_index_map[block_index]
                        state_dict_key = key.replace(f"transformer_blocks.{block_index}.", f"transformer_blocks.{new_block_index}.")

                    state_dict[state_dict_key] = f.get_tensor(key, device=loading_device, dtype=dit_weight_dtype)
            synchronize_device(loading_device)

        # Add after line 121 (after synchronize_device)
        if "__index_timestep_zero__" in state_dict:  # ComfyUI flag for edit-2511
            assert args.model_version == "edit-2511", (
                "Found __index_timestep_zero__ in state_dict, the model must be '2511' variant. Use --model_version edit-2511"
            )
            state_dict.pop("__index_timestep_zero__")

        info = model.load_state_dict(state_dict, strict=True, assign=True)
        logger.info(f"Loaded DiT model from {dit_path}, info={info}")

        return model

    # endregion model specific

    def train(self, args):
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

        # check model specific arguments
        self.handle_model_specific_args(args)

        # QwenImageNetworkTrainer forces dit_dtype to bfloat16, so we set it here
        args.dit_dtype = "bfloat16" if args.full_bf16 else "float32"

        # show timesteps for debugging
        if args.show_timesteps:
            self.show_timesteps(args)
            return

        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        # setup_logging(args, reset=True)

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # Load dataset config
        if args.num_timestep_buckets is not None:
            logger.info(f"Using timestep bucketing. Number of buckets: {args.num_timestep_buckets}")
        self.num_timestep_buckets = args.num_timestep_buckets  # None or int, None makes all the behavior same as before

        current_epoch = Value("i", 0)

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

        # prepare accelerator
        logger.info("preparing accelerator")
        accelerator = prepare_accelerator(args)
        if args.mixed_precision is None:
            args.mixed_precision = accelerator.mixed_precision
            logger.info(f"mixed precision set to {args.mixed_precision} / mixed precisionを{args.mixed_precision}に設定")
        is_main_process = accelerator.is_main_process

        # prepare dtype
        dit_dtype = model_utils.str_to_dtype(args.dit_dtype)  # bfloat16 (mixed precision) or float32
        logger.info(f"DiT precision: {dit_dtype}")

        # get embedding for sampling images
        vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
        sample_parameters = None
        vae = None
        if args.sample_prompts:
            sample_parameters = self.process_sample_prompts(args, accelerator, args.sample_prompts)

            # Load VAE model for sampling images: VAE is loaded to cpu to save gpu memory
            vae = self.load_vae(args, vae_dtype=vae_dtype, vae_path=args.vae)
            vae.requires_grad_(False)
            vae.eval()

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
        transformer = self.load_transformer(accelerator, args, args.dit, attn_mode, args.split_attn, loading_device, dit_dtype)

        if blocks_to_swap > 0:
            logger.info(f"enable swap {blocks_to_swap} blocks to CPU from device: {accelerator.device}")
            transformer.enable_block_swap(
                blocks_to_swap, accelerator.device, supports_backward=True, use_pinned_memory=args.use_pinned_memory_for_block_swap
            )
            transformer.move_to_device_except_swap_blocks(accelerator.device)

        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)

        # prepare optimizer, data loader etc.
        accelerator.print("prepare optimizer, data loader etc.")

        name_and_params = list(transformer.named_parameters())
        # single param group for now
        params_to_optimize = []
        params_to_optimize.append({"params": [p for _, p in name_and_params], "lr": args.learning_rate})
        param_names = [[n for n, _ in name_and_params]]

        # calculate number of trainable parameters
        n_params = 0
        for group in params_to_optimize:
            for p in group["params"]:
                n_params += p.numel()

        accelerator.print(f"number of trainable parameters: {n_params}")

        # trainable_params, lr_descriptions = network.prepare_optimizer_params(unet_lr=args.learning_rate)
        optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn = self.get_optimizer(
            args, params_to_optimize
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

        # prepare training model. accelerator does some magic here

        # experimental feature: train the model with parameters in fp16/bf16
        args.full_fp16 = False
        # if args.full_fp16:
        #     assert (
        #         args.mixed_precision == "fp16"
        #     ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        #     accelerator.print("enable full fp16 training.")
        #     network_dtype = weight_dtype
        #     network.to(network_dtype)
        if args.full_bf16:
            assert args.mixed_precision == "bf16", (
                "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            )
            accelerator.print("enable full bf16 training.")

        if blocks_to_swap > 0:
            transformer = accelerator.prepare(transformer, device_placement=[not blocks_to_swap > 0])
            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        else:
            transformer = accelerator.prepare(transformer)

        if args.compile:
            transformer = self.compile_transformer(args, transformer)
            transformer.__dict__["_orig_mod"] = transformer  # for annoying accelerator checks

        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
        training_model = transformer

        if args.full_fp16:
            # patch accelerator for fp16 training
            # def patch_accelerator_for_fp16_training(accelerator):
            org_unscale_grads = accelerator.scaler._unscale_grads_

            def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
                return org_unscale_grads(optimizer, inv_scale, found_inf, True)

            accelerator.scaler._unscale_grads_ = _unscale_grads_replacer

        # resume from local or huggingface. accelerator.step is set
        self.resume_from_local_or_hf_if_specified(accelerator, args)  # accelerator.load_state(args.resume)

        # patch for fused backward pass, adafactor only
        if args.fused_backward_pass:
            # use fused optimizer for backward pass: other optimizers will be supported in the future
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
            # "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
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

        datasets_metadata = []
        # tag_frequency = {}  # merge tag frequency for metadata editor # TODO support tag frequency
        for dataset in train_dataset_group.datasets:
            dataset_metadata = dataset.get_metadata()
            datasets_metadata.append(dataset_metadata)

        metadata["ss_datasets"] = json.dumps(datasets_metadata)

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
                "fine-tuning" if args.log_tracker_name is None else args.log_tracker_name,
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
        save_dtype = dit_dtype

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

            # temporarily remove self-referencing _orig_mod to avoid infinite recursion in state_dict()
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
                # restore _orig_mod after state_dict() if it was removed
                if has_self_ref_orig_mod_module:
                    unwrapped_model._modules["_orig_mod"] = unwrapped_model

            # if model is compiled, get original model state dict
            if "transformer_blocks.0._orig_mod.attn.add_k_proj.bias" in state_dict:
                logger.info("detected compiled model, getting original model state dict for saving")
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

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

        # For --sample_at_first
        if should_sample_images(args, global_step, epoch=0):
            optimizer_eval_fn()
            self.sample_images(accelerator, args, 0, global_step, vae, transformer, sample_parameters, dit_dtype)
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

        optimizer_train_fn()

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            for step, batch in enumerate(train_dataloader):
                latents = batch["latents"]

                with accelerator.accumulate(training_model):
                    latents = self.scale_shift_latents(latents)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)

                    # calculate model input and timesteps
                    noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
                        args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
                    )

                    weighting = compute_loss_weighting_for_sd3(
                        args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, dit_dtype
                    )

                    model_pred, target = self.call_dit(
                        args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, dit_dtype
                    )
                    loss = torch.nn.functional.mse_loss(model_pred.to(dit_dtype), target, reduction="none")

                    if weighting is not None:
                        loss = loss * weighting

                    loss = loss.mean()  # mean loss over all elements in batch

                    accelerator.backward(loss)

                    if not args.fused_backward_pass:
                        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                            params_to_clip = transformer.parameters()
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        # optimizer.step() and optimizer.zero_grad() are called in the optimizer hook
                        lr_scheduler.step()

                # if args.scale_weight_norms:
                #     keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(transformer).apply_max_norm_regularization(
                #         args.scale_weight_norms, accelerator.device
                #     )
                #     max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                # else:
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
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if len(accelerator.trackers) > 0:
                    logs = self.generate_step_logs(
                        args, current_loss, avr_loss, lr_scheduler, None, optimizer, keys_scaled, mean_norm, maximum_norm
                    )
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

            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
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


def qwen_image_finetune_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Qwen-Image fine-tuning specific parser setup"""
    parser.add_argument("--full_bf16", action="store_true", help="Enable full bfloat16 training for Qwen-Image")
    parser.add_argument("--fused_backward_pass", action="store_true", help="Use fused backward pass for Adafactor optimizer")
    parser.add_argument(
        "--mem_eff_save",
        action="store_true",
        help="Enable memory efficient saving (saving states requires use normal saving, so it takes same amount of memory even with this option enabled)",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = qwen_image_train_network.qwen_image_setup_parser(parser)
    parser = qwen_image_finetune_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"  # make bfloat16 as default for VAE, this should be checked

    if args.fp8_base or args.fp8_scaled:
        logger.warning("FP8 training is not supported for fine-tuning. Set --fp8-base or --fp8-scaled to False.")
        args.fp8_base = False
        args.fp8_scaled = False

    qwen_image_utils.resolve_model_version_args(args)

    trainer = QwenImageTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
