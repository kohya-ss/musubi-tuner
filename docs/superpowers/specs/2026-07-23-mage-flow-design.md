# Mage-Flow and Mage-Flow-Edit Integration Design

- Status: Approved for implementation planning
- Date: 2026-07-23
- Target branch: `codex/mage-flow`, based on `upstream/main` at `8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6`

## 1. Decision Summary

Add experimental LoRA training and inference support for both Microsoft Mage-Flow text-to-image and Mage-Flow-Edit. The first release will:

- consume the DiT, MageVAE, and Qwen3-VL-4B weights as three component-level, single `.safetensors` files;
- use the existing Musubi image bucket pipeline for training;
- preserve Mage-Flow's packed, variable-length model interface so native-resolution packing can be added later without changing model or cache contracts;
- support Mage-Flow-Edit with one to three ordered reference images;
- keep text encoder and VAE ownership outside the DiT, following Musubi's existing cache/train lifecycle;
- train LoRA adapters only;
- avoid global dependency upgrades and port only the required model math to Musubi's current runtime;
- fail explicitly on mode, cache, checkpoint, or tensor-contract mismatches.

The first release does not implement heterogeneous native-resolution image batches. It packs the equal-length samples produced by an existing bucket into the same variable-length interface used by the official model.

## 2. First-Principles Review

### 2.1 Is the problem real?

Yes. Musubi currently has no Mage-Flow model, MageVAE, prompt-conditioning, edit-reference, or LoRA integration. A user cannot train or sample a Mage-Flow adapter by selecting an existing architecture.

### 2.2 Is there a simpler solution?

Importing the official package directly is superficially shorter but is not a compatible solution. The pinned official package asks for Torch 2.13, Diffusers 0.37 or newer, Transformers 5.3 through 5.5, and newer supporting packages. Musubi currently pins Torch extras from 2.5 through 2.12, Diffusers 0.32.1, and Transformers 4.57.6. Replacing the shared runtime to accommodate one model has a much larger blast radius than porting the small model-specific math surface.

Duplicating separate T2I and Edit stacks is also unnecessary. The two released families have the same DiT tensor structure. Their real differences are conditioning, reference-latent layout, loss masking, cache identity, and sampling defaults. Those belong behind an explicit mode, not in copied model code.

### 2.3 What can this break?

The dangerous surfaces are the shared dataset collator, dependency pins, existing architecture cache identities, generic LoRA behavior, and large-checkpoint loading. This design protects them by:

- making shared dataset changes additive registration only;
- reusing the current bucket batching behavior unchanged;
- making all Mage-Flow packing inside Mage-specific code;
- keeping existing dependency pins;
- using a Mage-specific LoRA target policy;
- inspecting safetensors headers and shapes before allocating full models;
- assigning different architecture identities to T2I and Edit caches;
- refusing silent fallbacks or filename-based inference.

## 3. Sources of Truth

Implementation behavior is derived from these sources, in this order:

1. [Microsoft Mage-Flow project page](https://microsoft.github.io/Mage/flow/)
2. [Microsoft Mage repository pinned at `ea7109b3515ddd995c2e1212656dc1bc3a9607b7`](https://github.com/microsoft/Mage/tree/ea7109b3515ddd995c2e1212656dc1bc3a9607b7/mage_flow)
3. [Mage-Flow technical report at the same pinned revision](https://github.com/microsoft/Mage/blob/ea7109b3515ddd995c2e1212656dc1bc3a9607b7/assets/mage_flow_tech_report.pdf)
4. Released component configs in the official Hugging Face repositories:
   - `microsoft/Mage-Flow-Base`
   - `microsoft/Mage-Flow`
   - `microsoft/Mage-Flow-Turbo`
   - `microsoft/Mage-Flow-Edit-Base`
   - `microsoft/Mage-Flow-Edit`
   - `microsoft/Mage-Flow-Edit-Turbo`

The source revision is pinned because the upstream `main` branch can move during implementation. Ported source must retain the Microsoft MIT copyright and license notice where required.

### 3.1 Fixed released architecture

The public component loader supports the released 4B architecture, not arbitrary user-provided configs.

| Parameter | Value |
|---|---:|
| DiT input channels | 128 |
| DiT output channels | 128 |
| Text context dimension | 2560 |
| Hidden size | 3072 |
| Transformer depth | 12 |
| Attention heads | 24 |
| Head dimension | 128 |
| RoPE axes | `[16, 56, 56]` |
| Patch size | 1 |
| Effective text limit | 2048 tokens |
| Static flow shift | 6.0 |
| MageVAE latent channels | 128 |
| MageVAE spatial downsample | 16 |

The low-level model constructor may accept a small injected config for unit tests. The user-facing loader must construct and validate only the fixed released config.

### 3.2 Prompt contracts

T2I and Edit use the exact official prompt templates from the pinned source.

- T2I drops the first 34 system-template tokens after Qwen3-VL encoding.
- Edit drops the first 64 system-template tokens.
- The cached conditioning is the final Qwen3-VL hidden state only, with shape `[L, 2560]` after the template prefix is removed.
- The effective post-prefix length must be between 1 and 2048.
- Edit feeds the ordered reference images through Qwen3-VL's visual path and caps each visual-conditioning image's long edge at 384 pixels, preserving aspect ratio.

The tokenizer and processor are non-weight assets. Following the existing Krea 2 integration, they come from `Qwen/Qwen3-VL-4B-Instruct` or its local Hugging Face cache while `--text_encoder` remains a single local safetensors weight file.

## 4. Module and Process Boundaries

### 4.1 Mage-specific package

Create `src/musubi_tuner/mage_flow/` with these ownership boundaries:

| File | Responsibility |
|---|---|
| `model.py` | Pure NR-MMDiT model, fixed released config, packed forward contract, checkpointing, block swap and repeated-block hooks |
| `layers.py` | Dual-stream blocks, RMSNorm, modulation, timestep embeddings, feed-forward layers and frame-aware multi-axis RoPE |
| `attention.py` | Packed joint attention using PyTorch SDPA or optional FlashAttention 2 |
| `mage_vae.py` | MageVAE encoder/decoder, deterministic posterior sampling hook and component loading |
| `text_encoder.py` | Qwen3-VL-4B config, single-file weight loading, exact T2I/Edit templates and multimodal conditioning |
| `utils.py` | Fixed config, component key normalization, header validation, pack/unpack helpers and sampling utilities |
| `__init__.py` | Package marker and intentionally small public exports |

Do not copy the official monolithic `MageFlowModel` ownership model. Musubi must be able to load and free the VAE, text encoder, and DiT independently.

### 4.2 User entry points

Add package entry points and matching thin root wrappers:

- `src/musubi_tuner/mage_flow_cache_latents.py`
- `src/musubi_tuner/mage_flow_cache_text_encoder_outputs.py`
- `src/musubi_tuner/mage_flow_train_network.py`
- `src/musubi_tuner/mage_flow_generate_image.py`
- `mage_flow_cache_latents.py`
- `mage_flow_cache_text_encoder_outputs.py`
- `mage_flow_train_network.py`
- `mage_flow_generate_image.py`

The root wrappers only import and call the package `main`, matching current repository convention.

Add the Mage-specific LoRA adapter at:

- `src/musubi_tuner/networks/lora_mage_flow.py`

T2I and Edit share all four executable modules. `--is_edit` is required to select Edit behavior. Without it, the mode is T2I. The code must not infer mode from checkpoint names, file paths, reference-image presence, or state-dict shapes.

### 4.3 Required component arguments

The model-facing commands use:

```text
--dit FILE.safetensors
--vae FILE.safetensors
--text_encoder FILE.safetensors
```

Each argument names one regular safetensors file containing one component. Directory loading, Hub model loading, pickle checkpoints, split weight directories, and a combined all-in-one model file are outside this release.

The cache commands only require the component they execute. Training consumes cached text and latents, so it does not keep the VAE or text encoder resident. A text encoder or VAE is loaded during training only when an existing sample-generation workflow actually needs it.

## 5. Checkpoint Contract

### 5.1 Key normalization

All component key conversion lives in one Mage-specific normalizer per component. The normalizer accepts only explicitly documented layouts and returns one canonical internal layout. It must never use fuzzy suffix matching or choose a layout from the filename.

Initial accepted layouts are:

- the pinned official Mage-Flow component key layout exported into one safetensors file;
- the canonical internal layout used by the new Musubi modules;
- a future Comfy-Org layout only after a real released file has been inspected and a fixture has been added.

The absence of a current Comfy-Org mapping is intentional. Guessing future keys creates an untestable compatibility claim.

### 5.2 Header-first validation

Before constructing or allocating the full component, open the safetensors header and validate keys, dtypes, ranks, and required shapes.

DiT validation must prove:

- exactly 12 indexed transformer blocks are present;
- all fixed dimensions in section 3.1 match;
- required input, output, timestep, attention, MLP, modulation and norm tensors exist;
- block indices are contiguous from 0 through 11;
- no unknown layout was selected.

MageVAE validation must prove:

- encoder keys include the canonical `student.dconv_encoder` component;
- decoder keys include the canonical `pipeline` component when decoding is requested;
- the encoder output represents packed mean and log-variance for 128 latent channels;
- only explicitly listed upstream training-only extras may be ignored;
- no latent normalization from another VAE family is applied.

Qwen3-VL validation must prove:

- text hidden size is 2560;
- the language model has 36 layers and the fixed Qwen3-VL-4B dimensions;
- the visual component required by Edit is present;
- tied `lm_head.weight` omission is allowed only when it can be re-tied to the input embedding;
- an 8B or otherwise shape-incompatible Qwen checkpoint is rejected.

On failure, report the component, detected layout, expected shape, actual shape, and the first ten missing or unexpected keys. Do not continue with `strict=False` warnings after a structural mismatch.

## 6. Architecture and Cache Identity

Add two architecture identities:

| Short name | Full cache name | Meaning |
|---|---|---|
| `mf` | `mage_flow` | Text-to-image |
| `mfe` | `mage_flow_edit` | One-to-three-reference editing |

Short names contain no underscore, as required by `dataset/architectures.py`. Separate full names prevent T2I and Edit cache files from being mistaken for one another even when target paths and captions are the same.

MageVAE requires image dimensions divisible by 16. Register a 16-pixel bucket step for both identities. This is an additive registration; the resolution and batching behavior of every existing architecture remains unchanged.

## 7. Cache Data Contract

### 7.1 Latent cache

Each target cache contains:

| Mode | Safetensors key | Tensor shape | Meaning |
|---|---|---|---|
| T2I/Edit | `latents_1x{H}x{W}_bfloat16` | `[128, H, W]` | One sampled target posterior latent |
| Edit | `latents_control_{i}_1x{Hi}x{Wi}_bfloat16` | `[128, Hi, Wi]` | Ordered clean reference latent, `i = 0..N-1` |

Edit requires `1 <= N <= 3`. Reference indices must be contiguous and start at zero. Tensor order is the semantic reference order and must not be sorted by path or shape.

The cache stores one sampled posterior latent, not both mean and log-variance. Storing moments would approximately double latent cache size without helping a trainer that intentionally uses one fixed cache realization.

### 7.2 Deterministic MageVAE posterior sampling

MageVAE caching samples the posterior as the official path does, but makes the cached result stable across worker count, cache batch size, and item order.

For every target or reference tensor:

1. Form a UTF-8 identity from architecture full name, `item_key`, and role (`target` or `control:{i}`).
2. Compute SHA-256 over that identity.
3. Convert the first eight digest bytes to an unsigned integer.
4. Add the cache command's `--seed` modulo the Torch generator's supported 63-bit range.
5. Use a per-item `torch.Generator` to draw epsilon and compute `mean + exp(0.5 * logvar) * epsilon`.

Python's randomized `hash()` and mutation of global RNG state are forbidden for this contract.

### 7.3 Text encoder cache

Each text cache contains exactly one Mage conditioning tensor:

| Mode | Safetensors key | Tensor shape | Meaning |
|---|---|---|---|
| T2I/Edit | `varlen_mage_flow_embed_bfloat16` | `[L, 2560]` | Final Qwen3-VL hidden state after the mode-specific template prefix is removed |

The `varlen_` prefix is deliberate. `BucketBatchManager` already leaves such values as a list instead of padding or stacking them. T2I conditions only on text. Edit conditions Qwen3-VL on the caption/instruction plus the same ordered one-to-three reference images used by the latent cache.

The text encoder is frozen, used under `torch.no_grad()`, and released after caching. Padding is never stored.

### 7.4 Metadata and validation

Latent and text cache metadata must use the exact full architecture name and the repository's current cache `format_version`. Mage-specific preflight validation checks metadata and tensors before the first training step.

Validation rules are:

- T2I rejects datasets or caches containing control images instead of ignoring them.
- Edit rejects a sample with zero or more than three controls instead of falling back to T2I.
- Edit rejects missing, duplicate, or non-contiguous control indices.
- Target and controls must have 128 channels.
- Text conditioning must be rank 2 with final dimension 2560 and length 1 through 2048.
- Cache metadata architecture must match the command mode exactly.
- NaN or infinite conditioning and latent values are errors for Mage-Flow caches; they are not silently converted into a trainable sample.

## 8. Bucket Training and the Future Packing Contract

### 8.1 What official native-resolution packing does

For sample `i`, let its latent token length be `Li = Hi * Wi`. Official native packing concatenates different samples into one sequence:

```text
image_tokens:    [1, sum(Li), 128]
img_cu_seqlens: [0, L0, L0+L1, ...]
```

Text uses the same representation with its own token lengths and cumulative boundaries. Variable-length attention uses those boundaries so sample `i` cannot attend to sample `j`. A scheduler can therefore fill a step to a token budget with different resolutions and aspect ratios without rectangular image padding.

The important effect is resource utilization, not a different loss. A batch can contain, for example, a short wide latent and a large square latent in one transformer call while preserving sample isolation.

### 8.2 How current Musubi buckets differ

Musubi selects a resolution bucket first and forms a batch from samples with a compatible target shape. Edit batches are additionally split by reference count and each reference shape. The current collator therefore stacks regular tensors safely, but all image segments in one batch have the same target length.

Buckets reduce padding waste across the dataset. They do not solve the residual under-fill caused by keeping every sample in a step at one resolution. Official native packing schedules by total tokens and permits heterogeneous `Li` values in the same step.

### 8.3 First-release bridge

The first release keeps `BucketBatchManager` unchanged. A bucketed target batch arrives as `[B, 128, H, W]` and Mage-specific training code converts it to:

```text
image_tokens:    [1, B*H*W, 128]
img_cu_seqlens: [0, H*W, 2*H*W, ..., B*H*W]
```

Text segments are already truly variable length and are concatenated without padding. The model always receives cumulative sequence lengths, including for batch size one.

This is not native-resolution packing because the current image segment lengths are equal. It is nevertheless the same model ABI. A future token-budget sampler and collator can provide heterogeneous segment lengths without changing checkpoint loading, cache tensor formats, model forward, attention, edit masking, or LoRA.

### 8.4 Packed forward contract

In this contract, `Li` means the complete image-token segment owned by sample `i`. For T2I it is the target `H*W`; for Edit it is the target token count plus every reference token count. The Mage-specific packer produces a validated internal object with:

| Field | Shape/type | Invariant |
|---|---|---|
| `image_tokens` | `[1, sum(Li), 128]` | Sample segments are contiguous |
| `image_cu_seqlens` | int32 `[B+1]` | Starts at zero, strictly increasing, final value equals token count |
| `text_tokens` | `[1, sum(Ti), 2560]` | Same sample order as images |
| `text_cu_seqlens` | int32 `[B+1]` | Same boundary invariants |
| `image_shapes` | `list[list[tuple[int, int, int]]]` | Outer length is `B`; each inner list is ordered target then references |
| `timesteps` | `[B]` | One scalar per isolated sample |
| `target_token_mask` | bool `[sum(Li)]` | True only for tokens participating in loss or scheduler updates |

For T2I, each sample contains one frame and every image token is a target. For Edit, sample `i` is ordered as:

```text
[noisy target, clean reference 0, ..., clean reference N-1]
```

Every shape tuple is `(1, H, W)`: the first value is a frame count, not a frame coordinate. The tuple's position inside its sample list supplies the RoPE frame coordinate, which resets for every sample. The target is position 0 and references are positions 1 through N in semantic order. The sum of each tuple's `1*H*W` product must equal that sample's image-segment length. Attention boundaries isolate complete samples; frame coordinates distinguish target and references within a sample. Only target tokens are selected by `target_token_mask` for loss and denoising updates.

The packer must prove all length and shape invariants before calling the first transformer block. Attention code must not reconstruct boundaries from assumed equal resolutions.

## 9. Attention Backends

The supported first-release backends are:

- PyTorch scaled dot-product attention, selected by `--sdpa`, as the dependency-free default;
- FlashAttention 2 variable-length attention, selected by the repository's `--flash_attn` flag when the optional package is installed.

SDPA implements packed isolation by iterating or grouping sample slices internally and calling `torch.nn.functional.scaled_dot_product_attention` on each isolated joint text/image sequence. It must never run one unmasked attention operation over the concatenated samples.

FlashAttention 2 uses cumulative lengths directly. Text and image boundaries are joined per sample before the varlen kernel, then outputs are split back into their original streams.

SageAttention, xFormers, FlashAttention 4, custom fused kernels, split-attention approximations, and unmasked concatenated SDPA are rejected with an actionable error. Backend selection is explicit; unavailable optional dependencies do not silently change numerical paths.

## 10. Training Semantics

### 10.1 Flow-matching target

Use the standard Mage-Flow path where clean latent `z` is at sigma 0 and noise `epsilon` is at sigma 1:

```text
x_t = (1 - t) * z + t * epsilon
target = epsilon - z
```

The released sampler updates with:

```text
x_next = x + (sigma_next - sigma) * model_output
```

Because inference moves from sigma 1 toward sigma 0, the mathematically consistent model output is `epsilon - z`. The technical report text that writes the opposite sign is not used as the executable contract. A deterministic one-step Euler test must lock this direction so a future refactor cannot reverse it silently.

For Edit, noise is added only to target latents. Reference latents remain clean at every timestep, and loss is reduced only over target tokens.

### 10.2 Default training settings

The documented starting point is:

```text
mixed_precision = bf16
timestep_sampling = shift
discrete_flow_shift = 6.0
weighting_scheme = none
```

Reuse Musubi's existing flow timestep and loss plumbing where its math matches this contract. Do not introduce an independent generic trainer.

Training the Base and Edit-Base checkpoints is the supported recommendation. The aligned and Turbo checkpoints may share shapes, but Turbo is a distilled four-step model and LoRA training on it is not presented as an equivalent shortcut.

### 10.3 LoRA scope

Only LoRA training is supported in the first release. The default target set is the attention and feed-forward `nn.Linear` modules inside each of the 12 `MageFlowTransformerBlock` instances:

- image Q/K/V and output projections;
- text Q/K/V and output projections in joint attention;
- image feed-forward projections;
- text feed-forward projections.

The default excludes:

- `img_mod` and `txt_mod` modulation projections;
- normalization parameters;
- global image/text input projections;
- timestep projections;
- final adaptive normalization and output projection;
- VAE and text encoder modules.

The adapter supports existing include and exclude pattern overrides. Documentation recommends rank and alpha 32 as a starting point, but code does not hardcode them.

T2I and Edit DiTs use the same LoRA key layout. Saved metadata records the exact architecture identity. Loading a T2I adapter in Edit mode or the reverse requires an explicit Mage architecture-mismatch override and carries no quality guarantee. Base, aligned, and Turbo variant identity is not guessed from filenames.

## 11. Sampling Semantics

The generation command supports T2I and Edit through the same `--is_edit` mode switch. Defaults are Base-oriented; users select aligned or Turbo behavior by explicitly supplying steps and CFG rather than relying on checkpoint-name inference.

| Family | Recommended steps | Recommended CFG |
|---|---:|---:|
| Mage-Flow-Base | 30 | 5.0 |
| Mage-Flow aligned | 20 | 5.0 |
| Mage-Flow-Turbo | 4 | 1.0 |
| Mage-Flow-Edit-Base | 30 | 5.0 |
| Mage-Flow-Edit aligned | 30 | 5.0 |
| Mage-Flow-Edit-Turbo | 4 | 1.0 |

CFG positive and negative branches use the same packed model ABI. When fused, doubled segments receive correct independent boundaries.

Edit accepts one to three repeated `--control_image` arguments. Output size precedence is:

1. explicit `--width` and `--height` together;
2. `--max_size` applied to the primary reference aspect ratio;
3. the primary reference's source size.

Final output dimensions are aligned to 16. Reference order is preserved. The primary reference is index zero. The Qwen visual path uses the 384-pixel long-edge cap while MageVAE references use the chosen output size or their validated cached shape.

The official content-policy screen and Gaussian-Shading watermark are product-layer behavior, not model math, and are not included in Musubi core. Consequently, matching a command-line seed does not promise pixel-identical output to the official pipeline. Fixed tensors before watermark insertion are the parity boundary.

## 12. Memory and Runtime Features

- Gradient checkpointing applies to all 12 repeated transformer blocks.
- `--blocks_to_swap` accepts 0 through 10, preserving two resident blocks as required by the existing swap design.
- `--fp8_base` is accepted only with `--fp8_scaled`; plain unscaled FP8 is rejected.
- Scaled FP8 conversion applies only to explicitly supported repeated-block linear weights. Norms, modulation arithmetic, LoRA weights, and numerically sensitive projections remain in their supported higher precision.
- LoRA modules are attached to the BF16 base model before eligible base linears are converted to scaled FP8.
- Repeated transformer blocks expose the repository's compile hook.
- Block swap, compile, checkpointing and FP8 combinations must either pass their gated composition tests or fail at argument validation. They must not fail after allocating the 4B model.
- The text encoder is used for cache generation and then freed.
- MageVAE remains on CPU outside latent caching or sample encode/decode windows when an existing training sampling flow needs it.

No dependency pin in `pyproject.toml` is raised for this integration. The port must run on the current Musubi dependency surface. Optional FlashAttention remains an optional user installation.

## 13. Error Handling

All errors below occur during argument or cache/checkpoint preflight where possible:

- `--is_edit` with zero or more than three controls;
- T2I mode with any control data;
- one of `--width` or `--height` without the other;
- image dimensions not divisible by 16 after the documented sizing step;
- wrong cache architecture metadata;
- wrong latent channel count, text width, rank, length or non-finite values;
- non-contiguous reference keys;
- component path that is not one readable safetensors file;
- unknown key layout, missing required key or fixed-config shape mismatch;
- unsupported attention backend;
- FlashAttention selected but unavailable;
- plain FP8, out-of-range block swap or unsupported memory-feature composition;
- LoRA architecture mismatch without an explicit override.

Error messages name the bad sample, cache path, component or flag. Assertions are reserved for internal invariants; user data and files raise descriptive exceptions.

## 14. Test Strategy

### 14.1 CPU tests with tiny injected models

Add deterministic tests for:

- image and text pack/unpack round trips;
- cumulative-length validation;
- frame-aware RoPE for T2I and ordered Edit references;
- equal-size packed forward versus separate per-sample forwards;
- packed sample isolation by perturbing one sample and proving another output is unchanged;
- SDPA isolation with different text lengths;
- T2I all-target masking;
- Edit target-only loss masking for one, two and three references;
- the `epsilon - z` target and one-step Euler sign;
- deterministic VAE posterior sampling across item order and cache batch size;
- invalid mode, control count, shape, dtype and metadata errors.

### 14.2 Cache tests

Use temporary safetensors files to test:

- T2I target latent and varlen text round trip;
- Edit target, one/two/three ordered controls and multimodal text round trip;
- no text padding stored;
- architecture identity mismatch rejection;
- non-contiguous control rejection;
- existing `BucketBatchManager` behavior for varlen text and control-shape grouping;
- equal-size bucket conversion into the future-compatible packed ABI.

These tests must not require changing the common collator.

### 14.3 Loader tests

Synthetic header/key fixtures verify:

- canonical official component layouts are accepted;
- prefix normalization is deterministic;
- missing, unexpected and wrong-shape keys are rejected before allocation;
- a Qwen3-VL-8B-shaped fixture is rejected;
- allowed tied or training-only exceptions are narrow and documented;
- one file is required per component.

Do not add a fabricated Comfy-Org fixture. Add that compatibility only when the real release exists.

### 14.4 LoRA tests

Verify:

- the default target enumeration contains attention and both stream FFNs in all 12 blocks;
- modulation, norms and global projections are excluded;
- include/exclude overrides remain functional;
- save/load round trips preserve keys and architecture metadata;
- applying a nonzero adapter changes a tiny model output;
- T2I/Edit metadata mismatch is rejected without the explicit override.

### 14.5 Entrypoint and CUDA-gated tests

Extend the top-level entrypoint test so all four root wrappers import cleanly and respond to `--help` without loading weights.

CUDA-gated tests cover:

- SDPA versus FlashAttention 2 within an appropriate BF16 tolerance;
- scaled FP8 conversion and LoRA dtype separation;
- block swap limits and execution;
- repeated-block compile;
- supported feature compositions.

### 14.6 Real-weight manual parity tests

Large weights remain external to CI. A documented opt-in test or script takes component paths from environment variables and covers:

- MageVAE encode/decode;
- T2I Qwen embedding parity after the 34-token drop;
- Edit multimodal embedding parity after the 64-token drop for one and three references;
- fixed-input DiT forward parity against pinned upstream `ea7109b`;
- fixed-noise one-step and short multi-step sampler parity before official watermarking;
- T2I smoke generation at 512 and 1024;
- Edit smoke generation with one and three references;
- LoRA save, unload, reload and output reproduction.

CI proves contracts and math. It does not claim final visual quality without real weights.

## 15. Documentation

Add `docs/mage_flow.md` following the existing bilingual English/Japanese documentation style. It must cover:

- experimental status and attribution;
- the three single-file component arguments;
- how tokenizer/processor assets are resolved;
- separate T2I and Edit caching commands;
- Edit dataset examples with one to three ordered references;
- bucket training and the fact that heterogeneous native packing is not yet implemented;
- LoRA defaults and exclusions;
- Base, aligned and Turbo inference settings;
- SDPA, optional FlashAttention 2, FP8 and block swap restrictions;
- absence of the official content screen and watermark;
- the real-weight smoke-test procedure.

Link it from both `README.md` and `README.ja.md`. Examples must use explicit `--is_edit`, component files, steps and CFG where variant behavior matters.

## 16. Explicit Non-Goals

The first release does not include:

- heterogeneous native-resolution image packing or a token-budget sampler;
- changes to the public dataset collator or existing architecture batching semantics;
- Diffusers directory or Hugging Face Hub model loading;
- an all-in-one combined model checkpoint;
- split component directories or pickle checkpoints;
- guessed compatibility with an unreleased Comfy-Org key layout;
- SageAttention, xFormers, FlashAttention 4 or a custom CUDA kernel;
- official content screening or Gaussian-Shading watermarking;
- a GUI;
- full DiT fine-tuning;
- VAE or text encoder training;
- RL alignment or Turbo distillation;
- a promise of pixel-identical seeded output from the official product pipeline.

## 17. Expected Repository Touch Points

Implementation is expected to be additive and localized to:

- the new `mage_flow` package and four command modules;
- four root wrappers;
- `networks/lora_mage_flow.py`;
- architecture constant and 16-pixel bucket-step registration;
- Mage-specific cache serialization helpers;
- README links and `docs/mage_flow.md`;
- focused Mage-Flow tests plus the existing top-level entrypoint test.

No generic abstraction is required unless implementation proves unavoidable duplication with an existing, stable helper. In particular, the shared bucket collator is not redesigned for a feature intentionally deferred from this release.

## 18. Acceptance Criteria

The implementation is acceptable when all of the following are true:

1. All four entry points provide `--help` without importing or allocating model weights.
2. T2I caches, runs a tiny one-step LoRA training test, saves the adapter, reloads it, and performs fixed-noise sampling through the packed ABI.
3. Edit does the same for one and three ordered reference images, with loss and scheduler updates restricted to target tokens.
4. The first-release trainer uses existing same-resolution buckets and makes no semantic change to the common collator.
5. The model, cache and attention interfaces accept heterogeneous segment lengths in unit tests even though the production data loader does not yet construct such image batches.
6. Checkpoint and cache structural mismatches fail before a large allocation or first training step.
7. SDPA is correct without an optional attention package, and FlashAttention 2 passes its gated parity test when installed.
8. LoRA targets and exclusions match section 10.3 and round-trip through safetensors metadata.
9. The existing test suite remains green.
10. `pyproject.toml` dependency pins are unchanged.
11. Real-weight manual tests document results against pinned official revision `ea7109b` before the integration is declared ready for general use.
12. No claim of Comfy-Org compatibility is made until an actual released file passes a loader fixture and smoke test.
