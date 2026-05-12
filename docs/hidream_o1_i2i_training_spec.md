# HiDream-O1 I2I/Control Training Spec

This spec defines the minimal behavior required for HiDream-O1 control/reference training.

## Goal

HiDream-O1 control training must mirror the inference path. A dataset control image is not just a LoRA target selector; it must be used as both:

- Qwen3-VL visual input through `pixel_values` and `image_grid_thw`.
- Reference pixel patch tokens appended after the noisy target patch tokens.

The training loss is computed only on the target image patch segment.

## Cache Contract

Pixel cache:

- Target image: `latents_1x{height_patches}x{width_patches}_{dtype}`.
- Control/reference image: `latents_control_{index}_{height_patches}x{width_patches}_{dtype}`.

Text cache:

- T2I datasets cache prompt `input_ids`.
- Control datasets cache `input_ids` with Qwen image placeholders, plus `pixel_values` and `image_grid_thw` from the official processor.
- Optional cached `input_embeds` are still just initial token embeddings. Image placeholder positions are replaced by the vision encoder during training.

Existing control datasets need both caches rebuilt after this contract changes.

## Training Forward

For each item:

1. Patchify the target image to the normal training target.
2. Add noise only to the target patch tokens.
3. Concatenate control/reference patch tokens after the noisy target tokens for `vinputs`.
4. Build the same RoPE/layout shape as inference:
   - processor image grids for Qwen3-VL condition images,
   - target patch grid,
   - reference patch grids.
5. Mark target generation tokens as type `1`, reference patch tokens as type `2`, and timestep tokens as type `3`.
6. Run the model with `pixel_values` and `image_grid_thw` when control data exists.
7. Slice `x_pred` back to the target patch length before computing loss.

## LoRA Targets

Dataset-driven target selection:

- T2I: Qwen3VL decoder blocks plus HiDream pixel input/output adapters.
- I2I/control: T2I targets plus Qwen3-VL visual encoder layers.

Token embeddings and `lm_head` remain excluded. The Qwen3VL decoder is the shared denoising backbone for text and generated image tokens, so LoRA on decoder attention/MLP projections is still the main training path.
