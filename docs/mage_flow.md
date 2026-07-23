# Mage-Flow

Musubi Tuner has experimental LoRA training and inference support for
[Microsoft Mage-Flow](https://microsoft.github.io/Mage/flow/) and Mage-Flow-Edit.
The model implementation is pinned to upstream Mage commit
`ea7109b3515ddd995c2e1212656dc1bc3a9607b7`.

This integration has passed synthetic contract and tiny-model tests. It has not
yet passed the real released-weight parity checklist in this repository, so it
is not a general-availability compatibility claim.

## Supported Scope

- Mage-Flow T2I and Mage-Flow-Edit with one to three ordered references
- LoRA training only
- BF16, scaled FP8, gradient checkpointing, block swap and `torch.compile`
- PyTorch SDPA by default and optional FlashAttention 2
- fixed released DiT dimensions and Qwen3-VL-4B conditioning
- MageVAE image latents with 128 channels and 16x spatial downsampling
- current bucket training with a 16-pixel image step

Not supported:

- full-model fine-tuning
- public native-resolution packing
- checkpoint directories or sharded component paths
- inferred T2I/Edit or Base/aligned/Turbo identity from filenames
- SageAttention, xFormers, FlashAttention 3/4
- official content screening or Gaussian-Shading watermarking
- unverified Comfy-Org checkpoint layouts

## Component Files

Pass one regular `.safetensors` file for each component:

```text
--dit            Mage-Flow DiT
--vae            combined MageVAE encoder and decoder
--text_encoder   Qwen3-VL-4B backbone
```

Directories and shard lists are rejected. The loaders inspect every key, shape
and dtype before allocating the released model and then load strictly. The Qwen
language-model head is not needed because conditioning uses only the final
backbone hidden state.

The current loader recognizes the pinned Microsoft key layouts documented by
the command errors. It does not guess a layout from a filename or tensor shape.

## Dataset

Use the regular image dataset configuration. Target images and captions are the
same as for other image architectures. For Edit, add `control_directory`; the
target is the desired edited image and controls are the source references.

Multiple controls use the existing indexed naming:

```text
target/image1.png
target/image1.txt
controls/image1_0.png
controls/image1_1.png
controls/image1_2.png
```

Indices are semantic order and must be contiguous from zero. Edit accepts one,
two or three controls. See [Dataset Configuration](./dataset_config.md#image-dataset-with-control-images).

### Bucket Training Versus Native Packing

The first release deliberately keeps Musubi's public bucket pipeline. A batch
therefore contains compatible target sizes; Edit also separates batches by
reference count and ordered reference shapes. This preserves existing collator
behavior and predictable memory use.

Inside the Mage model, every batch is converted to the official-style packed
contract:

```text
image_tokens + image_cu_seqlens
text_tokens  + text_cu_seqlens
image_shapes + target_token_mask
```

This internal ABI already accepts heterogeneous segment lengths and prevents
cross-sample attention. It is not the same as native-resolution packing:
native packing would form a batch directly from unrelated resolutions according
to a token budget, without first grouping samples into equal-resolution
buckets. That future scheduler can use the retained contract, but this release
does not change the public data loader.

## Cache

Use the same mode for both cache passes. Omit `--is_edit` for T2I.

```bash
python mage_flow_cache_latents.py \
  --dataset_config dataset.toml \
  --vae path/to/mage_vae.safetensors \
  --vae_dtype bfloat16 \
  --batch_size 1

python mage_flow_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder path/to/qwen3_vl_4b.safetensors \
  --processor Qwen/Qwen3-VL-4B-Instruct \
  --text_encoder_dtype bfloat16 \
  --batch_size 1
```

For Edit, add `--is_edit` to both commands:

```bash
python mage_flow_cache_latents.py \
  --is_edit \
  --dataset_config edit_dataset.toml \
  --vae path/to/mage_vae.safetensors \
  --vae_dtype bfloat16

python mage_flow_cache_text_encoder_outputs.py \
  --is_edit \
  --dataset_config edit_dataset.toml \
  --text_encoder path/to/qwen3_vl_4b.safetensors
```

MageVAE posterior samples are derived from the architecture, stable item key,
target/control role and `--seed`. Reordering items or changing cache batch size
therefore does not change an item's cached latent.

## Train

The trainer defaults to the fixed Mage LoRA module, BF16, shifted flow matching
with shift 6, and no loss weighting:

```bash
accelerate launch mage_flow_train_network.py \
  --dataset_config dataset.toml \
  --dit path/to/mage_flow_dit.safetensors \
  --network_dim 32 \
  --network_alpha 32 \
  --learning_rate 1e-4 \
  --max_train_steps 1000 \
  --output_dir output \
  --output_name mage_flow_lora \
  --sdpa
```

Add `--is_edit` for Mage-Flow-Edit. The target follows
`x_t = (1-t)z + t*epsilon`, and the velocity target is `epsilon-z`. Edit
references remain clean, share the sample timestep modulation, and are excluded
from loss.

LoRA is limited to attention and image/text feed-forward linear layers inside
the 12 repeated transformer blocks. Modulation, normalization, input, timestep
and output projections are excluded. Include/exclude patterns cannot expand
this supported scope.

Useful memory options:

```text
--gradient_checkpointing
--blocks_to_swap 0..10
--block_swap_h2d_only --gradient_checkpointing
--fp8_base --fp8_scaled
--compile
```

Plain `--fp8_base` is rejected. Scaled FP8 quantizes only supported repeated
block attention/MLP weights; modulation, norms and global projections stay in
the compute dtype.

Use `--flash_attn` instead of `--sdpa` for optional FlashAttention 2. The
package remains an optional user installation.

## Generate

T2I Base-oriented defaults are 30 steps and CFG 5:

```bash
python mage_flow_generate_image.py \
  --dit path/to/mage_flow_dit.safetensors \
  --vae path/to/mage_vae.safetensors \
  --text_encoder path/to/qwen3_vl_4b.safetensors \
  --prompt "a glass greenhouse above a quiet city" \
  --negative_prompt " " \
  --width 1024 \
  --height 1024 \
  --steps 30 \
  --cfg_scale 5 \
  --seed 42 \
  --output output.png
```

For Edit, repeat `--control_image` in semantic order:

```bash
python mage_flow_generate_image.py \
  --is_edit \
  --dit path/to/mage_flow_edit_dit.safetensors \
  --vae path/to/mage_vae.safetensors \
  --text_encoder path/to/qwen3_vl_4b.safetensors \
  --control_image source.png \
  --control_image style_reference.png \
  --prompt "replace the sky with a late afternoon sky" \
  --max_size 1024 \
  --steps 30 \
  --cfg_scale 5 \
  --seed 42 \
  --output edit.png
```

Edit size precedence is explicit width plus height, then `--max_size` applied to
the primary reference aspect ratio, then the primary source size. Dimensions
are rounded down to multiples of 16. Qwen visual conditioning caps each
reference's long edge at 384 pixels; MageVAE receives references resized to the
output size.

Load adapters with `--lora_weight one.safetensors two.safetensors` and optional
matching `--lora_multiplier`. T2I/Edit architecture metadata must match the
selected mode. Cross-mode loading requires
`--allow_mage_architecture_mismatch`.

Recommended explicit schedules:

| Family | Steps | CFG |
|---|---:|---:|
| Base | 30 | 5.0 |
| aligned | 20 (Edit: 30) | 5.0 |
| Turbo | 4 | 1.0 |

## Real-Weight Validation

Before treating a component set as compatible, verify:

1. each of DiT, MageVAE and Qwen is one readable safetensors file;
2. strict header and state-dict loading succeeds without ignored keys;
3. T2I conditioning matches the pinned official output after the 34-token drop;
4. Edit conditioning matches after the 64-token drop with one and three refs;
5. fixed packed DiT inputs match official BF16 outputs;
6. one-step and short Euler outputs match before watermark insertion;
7. one LoRA step, save/reload, T2I sample and one/three-reference Edit sample succeed;
8. SDPA, optional FA2, scaled FP8, checkpointing, block swap and compile are
   checked on the intended GPU.

Until that checklist is run with released files, this support remains
experimental.
