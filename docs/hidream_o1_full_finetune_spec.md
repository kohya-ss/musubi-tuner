# HiDream-O1 Full Finetune Spec

This spec defines the first full-parameter finetuning path for HiDream-O1. It follows the existing Qwen-Image and Z-Image
full finetune scripts, while keeping HiDream-O1-specific pixel-space behavior in the HiDream trainer subclass.

## Goal

Add a dedicated full finetune entry point for HiDream-O1:

- Script: `src/musubi_tuner/hidream_o1_train.py`.
- Base class: `HiDreamO1NetworkTrainer`.
- Input model: the existing single checkpoint passed as `--dit`.
- Output model: a full HiDream-O1 checkpoint, not a LoRA adapter.

The implementation must reuse the existing HiDream-O1 cache, model-loading, sampling, timestep, DINO auxiliary loss, flash
attention, and block-swap behavior already implemented for LoRA training.

## Non-Goals

- Do not add another latent cache path. HiDream-O1 remains a pixel-space model and uses `cache_pixel`.
- Do not add separate DiT/text-encoder path arguments. The single checkpoint plus official repo config/tokenizer remains the
  loading contract.
- Do not implement user-facing partial-module finetuning or train/freeze presets in this pass.
- Do not implement trainable FP8 parameters in this pass. Direct optimizer steps on FP8 parameters are not reliable with the
  current optimizer stack, so trainable FP8 tensors must be promoted back to the training dtype before optimizer creation.
- Do not refactor Qwen-Image/Z-Image full finetune into a shared base in this pass. The first implementation should be a
  conservative sibling script.

## Source Patterns To Follow

Use these existing files as the template:

- `src/musubi_tuner/qwen_image_train.py`
- `src/musubi_tuner/zimage_train.py`

The HiDream-O1 script should copy the same high-level structure:

1. Validate required args.
2. Build dataset group from cached pixel/text data.
3. Prepare accelerator.
4. Load the full transformer.
5. Enable block swap and gradient checkpointing.
6. Build one optimizer parameter group from trainable `transformer.named_parameters()`.
7. Train with the same `call_dit()` and noise/timestep path used by LoRA training.
8. Save the full model state dict.

## CLI Contract

Parser setup:

```python
parser = setup_parser_common()
parser = hidream_o1_train_network.hidream_o1_setup_parser(parser)
parser = hidream_o1_finetune_setup_parser(parser)
```

Add these full-finetune-specific options:

- `--full_bf16`: load and train the model weights in bfloat16.
- `--fused_backward_pass`: use the existing fused Adafactor backward path.
- `--mem_eff_save`: use memory-efficient safetensors saving for model weights.
- `--block_swap_optimizer_patch_params`: move gradients to parameter devices before `optimizer.step()` when using block swap
  without fused backward.

Validation:

- `--dataset_config` is required.
- `--dit` is required.
- Allow `--fp8_base` only as an experimental memory path. Load weights as FP8, then promote all trainable parameters back to
  the training dtype before building the optimizer. For pure T2I datasets, the frozen visual encoder may remain FP8.
- Reject `--fp8_scaled` for full finetuning. The current scaled FP8 monkey patch is an inference/frozen-base path, not a
  trainable full-finetune path.
- Reject SageAttention for training, matching Qwen-Image/Z-Image.
- Reject `--xformers` and `--flash3`; HiDream-O1 full finetuning supports the default attention path and `--flash_attn`.
- `--full_bf16` requires `--mixed_precision bf16`.
- If `--fused_backward_pass` is used, document that gradient accumulation and max grad norm are not reliable; recommend
  `--gradient_accumulation_steps 1 --max_grad_norm 0`.
- LoRA-only arguments such as `--network_module`, `--network_weights`, `--network_dim`, `--network_alpha`, `--base_weights`,
  and `--dim_from_weights` should be rejected when explicitly provided, so a copied LoRA command does not silently train the
  wrong thing.

## Model Loading

Reuse `HiDreamO1NetworkTrainer.load_transformer()`:

- `args.model_type` selects the official config/tokenizer repo.
- `--dit` may point to a Comfy-style single safetensors checkpoint or a local/HF model directory supported by
  `hidream_o1_utils.load_model()`.
- The processor is loaded through `hidream_o1_utils.load_processor(model_type=args.model_type)`.
- No VAE is loaded; `load_vae()` returns `_NoVAE`.

For full finetuning:

- Set `args.dit_dtype = "float32"` by default.
- Set `args.dit_dtype = "bfloat16"` when `--full_bf16` is enabled.
- When `--fp8_base` is enabled, use FP8 as the initial load dtype, then promote every trainable FP8 parameter back to
  `args.dit_dtype` before optimizer creation. This is a quantized initialization path, not numerically identical to loading the
  checkpoint directly in bf16/fp32.
- Do not call `transformer.eval()` or `transformer.requires_grad_(False)`.
- Because the HiDream-O1 loader returns an eval-mode model for inference reuse, explicitly switch the prepared training model
  back to train mode before the training loop.

## Trainable Parameters

The implementation trains every parameter returned by `transformer.named_parameters()` whose `requires_grad` remains true:

- Qwen3VL text decoder / generation backbone.
- Qwen3VL vision encoder for control/reference datasets.
- Token embeddings.
- HiDream-O1 pixel input/output adapters: `x_embedder`, `final_layer2`.
- HiDream-O1 timestep embedder: `t_embedder1`.

`lm_head` is an `Identity` for single-checkpoint HiDream-O1 loading and has no trainable parameters.

Pure T2I datasets freeze the Qwen3VL visual encoder and skip the dummy visual training pass. The visual encoder is inactive
without `pixel_values`, `image_grid_thw`, and `latents_control_*`, so keeping it in the optimizer only wastes memory and can
create misleading zero-gradient behavior. Users who need visual encoder adaptation should use control/reference datasets so the
existing control path is exercised.

## Cache Contract

Use the existing HiDream-O1 cache contract unchanged:

- Pixel cache from `hidream_o1_cache_pixel.py`.
- Text/token cache from `hidream_o1_cache_text_encoder_outputs.py`.
- Target image tensors are pixel patch tokens, not VAE latents.
- Control/reference datasets require both:
  - `latents_control_*` in the pixel cache,
  - `pixel_values` and `image_grid_thw` in the text cache.

The full finetune script should produce the same clear cache-mismatch errors as `hidream_o1_train_network.py`.

## Training Loop

The loop should mirror Qwen-Image/Z-Image, with HiDream-O1 hooks:

1. Read `latents = batch["latents"]`.
2. `latents = self.scale_shift_latents(latents)`.
3. Sample `noise = torch.randn_like(latents)`.
4. Build `noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(...)`.
5. Compute loss weighting with `compute_loss_weighting_for_sd3(...)`.
6. Run `model_pred, target = self.call_dit(...)`.
7. Compute base MSE:
   ```python
   loss = torch.nn.functional.mse_loss(model_pred.to(dit_dtype), target.to(dit_dtype), reduction="none")
   ```
8. Apply weighting when present.
9. Reduce to scalar with `loss.mean()`.
10. Call `self.apply_auxiliary_losses(...)` before backward so HiDream-O1 DINO loss remains available.
11. `accelerator.backward(loss)`.
12. Step optimizer and scheduler using the same fused/non-fused branches as Z-Image.

The `block_swap_optimizer_patch_params` branch should match Z-Image:

- Only run when `blocks_to_swap > 0`.
- Only needed when not using `--fused_backward_pass`.
- Move `param.grad` to `param.device` before `optimizer.step()`.

## Sampling

Reuse `HiDreamO1NetworkTrainer.sample_images()` and `do_inference()`:

- Samples should work with the full model checkpoint in memory.
- Block swap must switch to inference mode before sampling and back to training mode after sampling.
- Existing sample prompt fields remain valid, including `control_image_path`, `editing_scheduler`, and `layout_bboxes`.
- No LoRA merge or LoRA multiplier logic is involved.

## Auxiliary DINO Loss

Keep DINO loss scoped to HiDream-O1:

- The full finetune script should call `self.apply_auxiliary_losses(...)`.
- CLI options are inherited from `hidream_o1_setup_parser()`.
- Dependency remains optional through `.[hidream_o1]`.
- DINO logs should appear through `generate_step_logs()` as `loss/base`, `loss/dino`, and `loss/dino_weighted`.

## Saving

Save full model checkpoints, not adapters:

- Use `safetensors.torch.save_file()` by default.
- Use `mem_eff_save_file()` when `--mem_eff_save` is enabled.
- Strip compile `_orig_mod.` prefixes before saving.
- Temporarily remove self-referencing `_orig_mod` before `state_dict()` if needed, matching Qwen-Image/Z-Image.
- Include SAI metadata with `is_lora=False` and `architecture=self.architecture`.

Metadata should include the normal training fields plus HiDream-O1-specific fields:

- `ss_model_type`
- `ss_noise_scale_start`
- `ss_noise_scale_end`
- `ss_noise_clip_std`
- `ss_full_bf16`
- `ss_fp8_base`
- `ss_fp8_frozen_visual`
- `ss_fp8_promoted_buffers`
- `ss_has_control`
- `ss_frozen_visual_params`
- DINO settings when `--dino_loss_weight > 0`

The saved checkpoint should be loadable by:

```bash
python src/musubi_tuner/hidream_o1_generate_image.py --dit path/to/last.safetensors --model_type full ...
```

## Recommended Initial Command

Full model, lower memory:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hidream_o1_train.py \
    --dit path/to/checkpoints/hidream_o1_image_bf16.safetensors \
    --dataset_config path/to/dataset.toml \
    --model_type full \
    --timestep_sampling uniform --weighting_scheme none \
    --noise_scale_start 8.0 --noise_scale_end 8.0 --noise_clip_std 0.0 \
    --full_bf16 \
    --optimizer_type adafactor --learning_rate 1e-6 --fused_backward_pass \
    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" \
    --max_grad_norm 0 --lr_scheduler constant_with_warmup --lr_warmup_steps 10 \
    --gradient_checkpointing --flash_attn \
    --blocks_to_swap 24 --use_pinned_memory_for_block_swap \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name hidream_o1_full_finetune
```

Dev model uses:

```bash
--model_type dev --noise_scale_start 7.5 --noise_scale_end 7.5 --noise_clip_std 2.5
```

## Linus Review Notes

- Do not make this a mode inside `hidream_o1_train_network.py`. The existing project convention is a dedicated full finetune
  script next to the LoRA trainer.
- Do not silently ignore LoRA args. Silent no-ops are worse than an early error when users copy old LoRA commands.
- The full finetune script must call the HiDream DINO hook. Otherwise `--dino_loss_*` appears in the parser but has no effect.
- `--fp8_base` must never leave trainable parameters in FP8. It is acceptable only when trainable tensors are promoted back to
  the training dtype before optimizer creation; `--fp8_scaled` remains rejected until there is a real trainable scaled-FP8
  strategy.
- Keep single-checkpoint loading. Adding separate text encoder paths would regress the current HiDream-O1 contract.
- Use the Z-Image block-swap optimizer patch, not a new offload workaround.
- Saving must produce a full checkpoint reloadable by HiDream inference without requiring LoRA merge.

## Verification Plan

Static checks:

1. `ruff format --check .`
2. `ruff check src/musubi_tuner/hidream_o1_train.py src/musubi_tuner/hidream_o1_train_network.py`
3. `python -m py_compile src/musubi_tuner/hidream_o1_train.py src/musubi_tuner/hidream_o1_train_network.py`

CPU/light smoke checks:

1. `python src/musubi_tuner/hidream_o1_train.py --help`
2. `python src/musubi_tuner/hidream_o1_train.py --show_timesteps --dataset_config ... --dit ...`
3. Parser rejects `--network_module networks.lora_hidream_o1`.
4. Parser accepts `--fp8_base` and logs trainable FP8 promotion.
5. Parser rejects `--fp8_scaled`.

GPU checks when a checkpoint is available:

1. One tiny T2I overfit run reaches backward and saves a step checkpoint.
2. One control/reference run exercises `pixel_values`, `image_grid_thw`, and `latents_control_*`.
3. `--sample_at_first` and `--sample_every_n_steps` generate images.
4. Saved checkpoint reloads with `hidream_o1_generate_image.py`.
5. `--blocks_to_swap` works with either `--fused_backward_pass` or `--block_swap_optimizer_patch_params`.
6. Optional `--dino_loss_weight` changes logs and still backprops.
