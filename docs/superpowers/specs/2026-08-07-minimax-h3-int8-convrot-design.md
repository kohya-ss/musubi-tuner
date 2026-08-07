# MiniMax-H3 INT8 ConvRot Support Design

Date: 2026-08-07

Status: Approved after artifact scope, automatic detection, architecture, validation, and test review

Branch: `codex/minimax-h3-int8-convrot`

Base: `kohya-ss/musubi-tuner@8918fc29d0e2c087042db6e2ee1edda6541ab243`

## 1. Summary

Add native loading and execution for the Comfy-Org MiniMax-H3 INT8 ConvRot artifacts. The supported matrix is:

- Full FL2VA and Ref2VA INT8 ConvRot transformers.
- Pruned-AdaLN FL2VA and Ref2VA INT8 ConvRot transformers.
- The truncated Qwen3-VL-32B MiniMax-H3 INT8 ConvRot text encoder.
- MiniMax-H3 LoRA training, text caching, scheduled training samples, and standalone generation.

Prequantized artifacts are detected from their tensor protocol. Users do not pass `--convrot_int8`. The loader reads each layer's `.comfy_quant` payload, validates its companion tensors, and applies the existing ConvRot runtime from PR #1008 with the group size declared by that layer.

The implementation keeps Krea2's working on-load quantization contract intact. Shared code is added only for Comfy-style prequantized artifact inspection and per-layer runtime patching. External Comfy `weight_scale` keys normalize to the existing internal `scale_weight` name, so the runtime keeps one scale-buffer contract.

Pruned transformers need a model change independent of quantization. They replace the standard time embedder and full-width AdaLN inputs with a shared `[1025, 8]` curve table and 8-wide AdaLN projections. This representation is implemented for the requested pruned INT8 artifacts; accepting pruned BF16 files is outside this change's acceptance matrix.

## 2. Source Anchors

Repository baseline and prior MiniMax-H3 design:

- <https://github.com/kohya-ss/musubi-tuner/tree/8918fc29d0e2c087042db6e2ee1edda6541ab243>
- `docs/superpowers/specs/2026-08-03-minimax-h3-support-design.md`

ConvRot implementation merged before this work:

- PR: <https://github.com/kohya-ss/musubi-tuner/pull/1008>
- Merge commit: `7d832bffa64f4b5e624ac719c8fd3122c843d3cc`
- Reviewed PR head: `5b31bd756322de4a11c530306a963d3f703571ac`

Released artifact repository:

- <https://huggingface.co/Comfy-Org/MiniMax-H3>
- <https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models>
- <https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders>

ComfyUI is an artifact and numerical reference only:

- Reviewed commit: `0ab8332bfa41c695b1c104a6535ff1fde81c7939`
- MiniMax-H3 model: <https://github.com/Comfy-Org/ComfyUI/blob/0ab8332bfa41c695b1c104a6535ff1fde81c7939/comfy/ldm/minimax/model.py>
- Quantization protocol: <https://github.com/Comfy-Org/ComfyUI/blob/0ab8332bfa41c695b1c104a6535ff1fde81c7939/comfy/quant_ops.py>
- Qwen3-VL adapter: <https://github.com/Comfy-Org/ComfyUI/blob/0ab8332bfa41c695b1c104a6535ff1fde81c7939/comfy/text_encoders/minimax.py>

Existing Apache-2.0 ConvRot kernels remain the implementation source. GPL-licensed ComfyUI code is not copied.

## 3. Goals

- Detect supported prequantized files from tensor content rather than filenames or CLI flags.
- Stream INT8 weights without materializing a BF16 copy.
- Preserve the mixed-precision islands in the released artifacts.
- Use the group size declared by every quantized layer.
- Keep quantized base layers as `nn.Linear` instances so existing LoRA targeting, block swap, and compile exclusion continue to work.
- Support the standard and pruned AdaLN transformer layouts.
- Load and execute the INT8 Qwen3-VL text encoder for text cache and direct prompt encoding.
- Keep BF16 MiniMax-H3 behavior unchanged.
- Fail before model execution when a quantized artifact is malformed or unsupported.

## 4. Non-Goals

- Dynamically quantizing a MiniMax-H3 BF16 checkpoint at startup.
- NVFP4/AWQ text encoder loading.
- FP8 transformer loading.
- INT4 or mixed INT4/INT8 ConvRot.
- Quantizing either VAE.
- Training the text encoder or either VAE.
- Destructively merging LoRA deltas into INT8 base tensors.
- Accepting unknown `comfy_quant` formats through a dequantized fallback.
- Downloading the 20-51 GB artifacts as part of CI.
- Replacing the PR #1008 Triton kernels with `comfy-kitchen` or adding it as a dependency.

`--base_weights` is a destructive base merge and is rejected when the transformer is prequantized. `--network_weights` remains supported because it loads the trainable additive LoRA branch. Standalone generation also uses attached LoRA branches for a prequantized transformer instead of `_merge_lora_weights`.

## 5. Released Artifact Contract

### 5.1 Layer protocol

Every quantized Linear is represented by three sibling keys:

```text
<module>.weight        I8  [out_features, in_features]
<module>.weight_scale  F32 [out_features, 1]
<module>.comfy_quant   U8  [payload_bytes]
```

The UTF-8 payload currently has this schema:

```json
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

The loader validates the values, not the serialized byte count or JSON whitespace. It accepts only:

- `format == "int8_tensorwise"`
- `convrot is True`
- `convrot_groupsize >= 4` and an exact power of four
- `in_features % convrot_groupsize == 0`

The adapter performs the power-of-four check before the kernel can run and reports the checkpoint path and normalized layer name on failure. Floating-point layers without `.comfy_quant` are valid mixed-precision islands. An INT8 weight without a complete and valid sibling triple is invalid.

`weight_scale` is the external Comfy key. Inspection normalizes it to the repository's existing `scale_weight` name before model matching and loading. The scale remains FP32 from the source file through the registered buffer and fused epilogue; it is never included in a blanket model-dtype cast.

### 5.2 Full transformers

The full FL2VA and Ref2VA artifacts contain 250 quantized layers:

- Four attention/MLP Linears in each of 50 main blocks use group size 256.
- The AdaLN projection in each main block uses group size 64.

The token refiner, patch projections, output heads, norms, and time embedder remain floating point. Full files carry the released transformer config metadata and use the standard 2688-wide time embedding.

### 5.3 Pruned transformers

The pruned artifacts contain 200 quantized layers: the four attention/MLP Linears in each main block. AdaLN is not quantized. Its representation is:

```text
adaln_t_table                         F32 [1025, 8]
blocks.N.adaln_proj.linear.weight     F16 [96768, 8]
blocks.N.adaln_proj.linear.bias       F16 [96768]
final_layer.adaln_proj.linear.weight  F16 [10752, 8]
final_layer.adaln_proj.linear.bias    F16 [10752]
```

These files omit the standard `time_embedder.*` tensors and may omit config metadata. The loader therefore derives pruned mode from `adaln_t_table`, then validates all other released dimensions from tensor shapes. It does not infer arbitrary H3 variants.

### 5.4 Qwen3-VL text encoder

The text encoder contains 350 quantized language-model Linears across the retained 50 layers:

- `self_attn.q_proj`, `k_proj`, `v_proj`, and `o_proj`
- `mlp.gate_proj`, `up_proj`, and `down_proj`

The current artifact declares group size 256 for these layers. Embeddings, norms, rotary tensors, and visual components remain floating point. The existing `model.*` to `language_model.*` key transform applies to weight, scale, and control keys before module matching.

## 6. Shared ConvRot Artifact Adapter

`src/musubi_tuner/modules/convrot_int8_utils.py` gains a small artifact model:

```python
@dataclass(frozen=True)
class ConvRotInt8LayerSpec:
    module_path: str
    weight_key: str
    scale_key: str
    groupsize: int


@dataclass(frozen=True)
class ConvRotInt8Artifact:
    layers: Mapping[str, ConvRotInt8LayerSpec]
    control_keys: frozenset[str]
```

Inspection happens before the meta model is populated. It uses the existing `MemoryEfficientSafeOpen.header` entries for dtype and shape and only materializes the small U8 control tensors. It does not add another safetensors header reader. Key prefixes and architecture key transforms are applied consistently to all three sibling keys, including the canonical `weight_scale` to `scale_weight` transform.

The adapter prepares a model by locating every declared module, requiring an exact `nn.Linear`, replacing its meta weight contract with an INT8 `nn.Parameter(..., requires_grad=False)`, registering an FP32 `scale_weight` buffer, storing the per-module group size, and binding the shared ConvRot forward. Setting `requires_grad=False` during meta construction is mandatory because `load_state_dict(assign=True)` preserves the destination parameter's gradient flag and PyTorch cannot create a gradient-bearing INT8 Parameter. The existing Krea2 helper remains callable with its state-dict-driven API and creates equivalent layer specs internally.

Both Krea2 and prequantized artifacts use the single internal name `scale_weight`, so `convrot_int8_linear_forward_patch` needs no scale-name indirection. `ConvRotInt8LinearFn` already accepts group size as an argument, so the numerical kernel does not need an architecture branch.

The prepared model exposes `is_convrot_int8 = True` and the number of patched layers for integration checks and logging.

## 7. Strict Streaming Loader

`src/musubi_tuner/minimax_h3/checkpoint.py` accepts an optional inspected artifact adapter.

Loading follows this order:

1. Resolve the safetensors file or indexed shards.
2. Normalize raw keys, including `weight_scale` to `scale_weight`, and inspect the ConvRot sibling triples.
3. Inspect transformer structure and select full or pruned configuration.
4. Create the module under `accelerate.init_empty_weights()`.
5. Prepare all declared INT8 Linear modules before computing the expected state dict.
6. Stream weight and scale tensors shard by shard with `assign=True`; scales and control tensors are excluded from the general floating-point cast policy.
7. Mark validated `.comfy_quant` keys as consumed control data without registering them as model buffers.
8. Apply the existing missing, unexpected, shape, and dtype checks to every model tensor.
9. Move the populated model to its requested loading device and set evaluation mode.

The loader keeps three distinct accounting sets:

- `seen_normalized_keys` includes model and control keys and detects duplicates across shards.
- `loaded_model_keys` drives missing-model-key checks.
- `consumed_control_keys` records only control keys declared by the inspected artifact.

This prevents a control key from becoming an unexpected model key without losing duplicate detection. Strict dtype checking understands the inspected INT8 layers and checks source dtype for fixed-precision islands. `load_h3_text_encoder` enables strict dtype checking. Its FP32 scales must match an FP32 destination buffer and remain bit-for-bit unchanged; `dtype=torch.bfloat16` applies only to eligible floating model tensors, not `scale_weight`.

## 8. Pruned AdaLN Runtime

`MiniMaxH3Config` gains an optional `adaln_curve_grid`. Standard models keep `time_embed_dim=2688` and construct `TimeEmbedder`. Pruned models use `time_embed_dim=8`, register `adaln_t_table`, omit `TimeEmbedder`, and construct 8-input AdaLN projections.

For each unique timestep `t` in `[0, 1]`, the pruned path computes:

```python
t_fp32 = t.to(device=table.device, dtype=torch.float32)
table_fp32 = table.to(dtype=torch.float32)
position = t_fp32.clamp(0.0, 1.0) * (table_fp32.shape[0] - 1)
lower = position.floor().long().clamp(max=table_fp32.shape[0] - 2)
fraction = position - lower.to(position.dtype)
timestep_embedding = torch.lerp(
    table_fp32[lower],
    table_fp32[lower + 1],
    fraction.unsqueeze(1),
)
```

The whole coordinate calculation and interpolation stay FP32 even when the caller supplies BF16 timesteps. Computing `t * 1024` in BF16 rounds many interior positions to integers and silently degrades interpolation to table lookup. Standard AdaLN continues to apply SiLU inside `AdalnProj`. Pruned AdaLN consumes the interpolated curve coordinate directly and skips that SiLU. Both endpoints and non-grid interior positions are part of the numerical contract.

## 9. Runtime Integration

### 9.1 Transformer training

`load_h3_transformer` automatically detects and prepares ConvRot files. The trainer no longer rejects a non-BF16 source artifact when it has a valid inspected ConvRot contract. Floating-point activations and the LoRA branch remain BF16.

Add `--convrot_int8_bwd {bf16,int8}` with default `bf16`:

- `bf16` transiently dequantizes the rotated base weight for `grad_x`.
- `int8` uses the fused INT8 backward path and requires Triton.

The option does not enable quantization. Supplying `int8` for a non-ConvRot transformer is rejected. Existing `--base_weights` merges are rejected for an INT8 base; normal LoRA creation, `--network_weights`, gradient checkpointing, and block swap remain supported.

### 9.2 Compilation and block swap

Quantized Linears are excluded from `torch.compile`, matching Krea2. The existing offloader streams only Linear `.weight` tensors. During block-swap preparation it first moves each whole block to the execution device, which places norms, biases, and `scale_weight` buffers there, then returns only swapped weights to CPU. Scale buffers therefore remain resident on the execution device while INT8 weights swap.

The released full transformer keeps about 30.1 MiB of FP32 block scales resident; the pruned transformer keeps about 11.6 MiB. This is an explicit fixed cost, not swapped storage. Tests verify device consistency before and after swap rather than claiming that scale buffers move with each weight transfer. Per-layer group sizes are immutable attributes and require no device transfer.

### 9.3 Standalone and scheduled generation

Generation uses the same automatic transformer detection. BF16 transformers intentionally retain the existing destructive LoRA merge because it pays the merge cost once and avoids LoRA matrix multiplications on every denoising step. Changing that established performance behavior is outside this feature. INT8 tensors cannot receive a floating LoRA delta without dequantization and requantization, which would alter the released base artifact, so INT8 transformers create inference LoRA modules, apply them as additive branches, load their state dicts, and keep the modules alive for the sampling lifetime. Multiple requested LoRAs remain independent additive branches with their configured multipliers.

Training-time scheduled samples already execute through the attached training network and need no separate merge path.

### 9.4 Text encoder

`load_h3_text_encoder` applies the artifact adapter after the existing Qwen config truncation and layer-50 final-norm removal. INT8 is accepted only for valid ConvRot triples, `strict_dtype=True` is required, and every `scale_weight` remains FP32 even though ordinary eligible model tensors load at BF16. The encoder remains frozen and evaluation-only, so no text-encoder backward mode is exposed.

Text cache and direct prompt encoding share this loader. No cache format changes are needed because cached hidden states remain BF16.

### 9.5 Existing R1 gates

Implementation removes or replaces all code gates that currently defer this feature:

- `checkpoint.py` no longer rejects every `.weight_scale` or `.comfy_quant` suffix; the inspected adapter validates and consumes them.
- `model.py` removes the metadata-wide substring search for `convrot`, `int8`, `fp8`, or `quantized`. Tensor protocol and exact config fields determine support, so unrelated metadata comments cannot reject a BF16 file.
- `model.py` replaces the `adaln_curve_grid` rejection with the strict full/pruned structural contract.
- `load_h3_transformer` continues to require BF16 floating activations, but its error text no longer claims that all checkpoint weights must be BF16.

## 10. Error Contract

Raise `ValueError` before execution for:

- Malformed or non-object `.comfy_quant` JSON.
- Unknown quantization format or `convrot != true`.
- Missing weight, scale, or control siblings.
- INT8 weights outside a validated sibling triple.
- Weight and scale dtype or shape mismatches.
- A group size that is not a power of four, is unsupported by the shared kernel, or does not divide the input width.
- A declared module that does not exist or is not exactly `nn.Linear`.
- Duplicate raw or normalized keys.
- Mixed full and pruned AdaLN structures.
- A pruned table other than `[1025, 8]` or inconsistent AdaLN input widths.
- `--convrot_int8_bwd int8` without an INT8 transformer or without Triton.
- Destructive base-weight merges requested for an INT8 transformer.

Messages include the checkpoint path and normalized layer name. Unsupported quantized formats do not silently dequantize.

## 11. Test Strategy

### 11.1 Shared adapter tests

Extend `tests/test_krea2_convrot_int8.py` or add a focused shared test module to cover:

- Comfy sibling discovery with `weight_scale`.
- Canonical `weight_scale` to `scale_weight` normalization without changing the shared forward.
- Exact payload parsing independent of JSON whitespace.
- Different group sizes in adjacent layers.
- Context-rich rejection of group sizes such as 128 and 512 before kernel entry.
- An INT8 meta Parameter with `requires_grad=False` surviving `assign=True`.
- FP32 scales with values not exactly representable in BF16 remaining bit-for-bit unchanged.
- Duplicate control keys across shards being rejected without entering model missing/unexpected accounting.
- Krea2 `scale_weight` compatibility through the same internal name.
- INT8 forward and BF16 backward parity against dequantized references.
- CPU fallback and Triton availability gates.
- Every error listed in the artifact error contract.

### 11.2 MiniMax-H3 transformer tests

Use tiny model configs and synthetic safetensors files to cover:

- Automatic full INT8 detection without a CLI format flag.
- Automatic pruned INT8 detection without config metadata.
- Standard and curve AdaLN construction.
- Curve interpolation at both endpoints and non-grid interior positions when input timesteps are FP32 and BF16; both must match the FP32 reference and differ from nearest-neighbor lookup.
- Per-layer group size propagation, including a group-64 AdaLN layer.
- Forward and LoRA backward through a patched main block.
- Scale buffers remaining on the execution device while block weights swap.
- Compile exclusion when the loaded model is ConvRot.
- Rejection of destructive base merges.
- Attached LoRA inference over an INT8 base.

### 11.3 Text encoder tests

Patch the Transformers factory with a tiny Qwen-shaped module and verify:

- `model.*` keys normalize to `language_model.*` for all sibling tensors.
- Mixed quantized and floating layers load strictly.
- FP32 scale values survive `dtype=torch.bfloat16` loading unchanged.
- The patched Linears execute during layer-50 extraction.
- Text cache and direct encoding entry points accept the artifact automatically.

### 11.4 Regression and artifact checks

Run at minimum:

```text
pytest tests/test_krea2_convrot_int8.py
pytest tests/test_minimax_h3_model.py tests/test_minimax_h3_training.py tests/test_minimax_h3_text_encoder.py
pytest tests/test_minimax_h3_sampling.py tests/test_minimax_h3_cache_contract.py
pytest
```

Before completion, fetch only the official safetensors headers and control payload ranges to compare key counts, dtypes, shapes, group sizes, and full/pruned classification against this contract. A full numerical smoke test is reported only if the actual multi-gigabyte artifacts are present locally.

## 12. Documentation

Update `docs/minimax_h3.md` and the README update entry with:

- The accepted full, pruned, and text-encoder filenames.
- Automatic format detection.
- Triton requirements and CPU fallback behavior.
- `--convrot_int8_bwd` semantics.
- Compatibility with LoRA training, attached-LoRA generation, block swap, and gradient checkpointing.
- The intentional BF16 merged-LoRA versus INT8 attached-LoRA generation tradeoff.
- Rejection of dynamic BF16 quantization, NVFP4/AWQ, and destructive base merges.

Remove the R1 statements that all quantized and pruned artifacts are deferred.

## 13. Acceptance Criteria

- All five released INT8 ConvRot files in scope classify from content without filename checks: full and pruned FL2VA/Ref2VA transformers plus the Qwen3-VL text encoder.
- Full transformers patch 250 declared Linear layers with their per-layer group sizes.
- Pruned transformers patch 200 declared Linear layers and use `[1025, 8]` curve interpolation.
- The text encoder patches 350 declared language-model Linears.
- BF16 transformer and text-encoder tests remain unchanged in behavior.
- LoRA gradients reach trainable adapter parameters over an INT8 base.
- Standalone LoRA generation does not mutate INT8 base tensors.
- Under block swap, INT8 weights stream while every FP32 scale buffer stays on the execution device.
- Invalid or unsupported quantization fails before forward execution.
- Focused tests, the full test suite, formatter checks, and `git diff --check` pass, or any unrelated baseline failure is reported with evidence.
