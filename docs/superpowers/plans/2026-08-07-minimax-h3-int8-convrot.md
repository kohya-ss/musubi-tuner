# MiniMax-H3 INT8 ConvRot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load and run the released full and pruned MiniMax-H3 INT8 ConvRot transformers and Qwen3-VL text encoder automatically in training, caching, and generation paths.

**Architecture:** A shared ConvRot artifact adapter inspects Comfy tensor triples from safetensors headers, canonicalizes `weight_scale` to `scale_weight`, and prepares exact `nn.Linear` modules before streamed assignment. MiniMax-H3's loader selects either its existing BF16 dtype contract or an artifact-aware conversion contract; the model adds a pruned AdaLN curve path while all entry points reuse automatic artifact detection.

**Tech Stack:** Python 3.10+, PyTorch, Accelerate meta initialization, safetensors, Triton-backed ConvRot kernels with eager CPU fallback, pytest, Ruff.

## Global Constraints

- Branch from `kohya-ss/musubi-tuner` dev commit `8918fc29d0e2c087042db6e2ee1edda6541ab243` and preserve unrelated worktree changes.
- Detect prequantized artifacts only from `.comfy_quant` tensor content; do not add a format-enabling CLI flag.
- Accept only `format="int8_tensorwise"`, `convrot=true`, and group sizes that are exact powers of four, at least 4, and divide the Linear input width.
- Keep one internal scale name, `scale_weight`, and preserve every ConvRot scale as FP32 without a blanket dtype cast.
- Keep patched modules as exact `nn.Linear` instances so LoRA targeting, block swap, and compile exclusion retain their current contracts.
- Support released full FL2VA/Ref2VA, pruned FL2VA/Ref2VA, and Qwen3-VL-32B MiniMax-H3 INT8 ConvRot files; do not add FP8, NVFP4/AWQ, INT4, or dynamic BF16 quantization.
- Keep the existing Krea2 on-load ConvRot quantization behavior and tests intact.
- Use test-first red-green cycles for every production behavior.

## File Map

- `src/musubi_tuner/modules/convrot_int8_utils.py`: shared artifact dataclasses, inspection, validation, and module preparation; retains Krea2 quantization and runtime forward.
- `src/musubi_tuner/minimax_h3/checkpoint.py`: MiniMax-H3 key normalization, streamed assignment, key accounting, and path-specific dtype policies.
- `src/musubi_tuner/minimax_h3/model.py`: full/pruned structure classification, pruned AdaLN model construction, FP32 curve interpolation, and automatic transformer loading.
- `src/musubi_tuner/minimax_h3/text_encoder.py`: automatic Qwen3-VL ConvRot inspection and loading.
- `src/musubi_tuner/minimax_h3_train_network.py`: backward-mode option, destructive-merge rejection, and compile exclusion.
- `src/musubi_tuner/minimax_h3_generate_video.py`: BF16 merge versus INT8 attached-LoRA generation.
- `tests/test_convrot_int8_artifact.py`: shared artifact protocol and loader-policy tests.
- `tests/test_minimax_h3_model.py`: pruned model, transformer loading, block-swap, and numerical tests.
- `tests/test_minimax_h3_text_encoder.py`: synthetic 50-layer Qwen-shaped INT8 loading tests.
- `tests/test_minimax_h3_training.py`: parser, validation, compile, and LoRA-gradient tests.
- `tests/test_minimax_h3_sampling.py`: attached-LoRA generation and non-mutating INT8 base tests.
- `docs/minimax_h3.md`, `README.md`: supported files, automatic detection, runtime tradeoffs, and limitations.

---

### Task 1: Shared ConvRot Artifact Adapter

**Files:**
- Modify: `src/musubi_tuner/modules/convrot_int8_utils.py`
- Create: `tests/test_convrot_int8_artifact.py`
- Regression: `tests/test_krea2_convrot_int8.py`

**Interfaces:**
- Produces: `ConvRotInt8LayerSpec`, `ConvRotInt8Artifact`, `canonicalize_convrot_int8_key`, `inspect_convrot_int8_artifact`, and `prepare_convrot_int8_model`.
- Preserves: `ConvRotInt8Quantizer`, `ConvRotInt8LinearFn`, `convrot_int8_linear_forward_patch`, and `apply_convrot_int8_monkey_patch`.

- [ ] **Step 1: Write failing protocol-inspection tests**

Create synthetic safetensors with two modules and exact payload bytes:

```python
def _payload(groupsize: int, *, whitespace: bool = False) -> torch.Tensor:
    separators = (", ", ": ") if whitespace else (",", ":")
    raw = json.dumps(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize},
        separators=separators,
    ).encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.uint8)


def test_inspector_canonicalizes_comfy_scales_and_keeps_per_layer_groups(tmp_path):
    path = tmp_path / "artifact.safetensors"
    save_file(
        {
            "root.a.weight": torch.zeros(8, 16, dtype=torch.int8),
            "root.a.weight_scale": torch.ones(8, 1, dtype=torch.float32),
            "root.a.comfy_quant": _payload(4, whitespace=True),
            "root.b.weight": torch.zeros(12, 64, dtype=torch.int8),
            "root.b.weight_scale": torch.ones(12, 1, dtype=torch.float32),
            "root.b.comfy_quant": _payload(16),
        },
        path,
    )

    artifact = inspect_convrot_int8_artifact([path], key_normalizer=lambda key: key.removeprefix("root."))

    assert artifact.layers["a"].scale_key == "a.scale_weight"
    assert artifact.layers["a"].groupsize == 4
    assert artifact.layers["b"].groupsize == 16
    assert artifact.control_keys == frozenset({"a.comfy_quant", "b.comfy_quant"})
```

Add parameterized failures for malformed JSON, non-object JSON, wrong format, `convrot=false`, groups 128 and 512, missing siblings, I8 weights outside a triple, U8/F32/I8 dtype mismatches, scale shape mismatches, and indivisible input widths. Assert that errors contain the checkpoint path and normalized module name.

- [ ] **Step 2: Run the shared tests and verify the expected import failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_convrot_int8_artifact.py -q`

Expected: collection fails because the new artifact symbols do not exist.

- [ ] **Step 3: Implement immutable artifact records and header inspection**

Add these public records and signatures:

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

    @property
    def weight_keys(self) -> frozenset[str]:
        return frozenset(layer.weight_key for layer in self.layers.values())

    @property
    def scale_keys(self) -> frozenset[str]:
        return frozenset(layer.scale_key for layer in self.layers.values())


def canonicalize_convrot_int8_key(key: str) -> str:
    if key.endswith(".weight_scale"):
        return key.removesuffix(".weight_scale") + ".scale_weight"
    return key
```

Implement `inspect_convrot_int8_artifact(files: Iterable[str | Path], *, key_normalizer: Callable[[str], str] | None = None, disable_numpy_memmap: bool = False) -> ConvRotInt8Artifact | None`. Use `MemoryEfficientSafeOpen.header` for dtype and shape. Materialize only `.comfy_quant` tensors, decode them as UTF-8 JSON, validate exact protocol values, and aggregate siblings across shards. Canonicalize external `.weight_scale` before duplicate detection. Return `None` only when there are no controls, no external scales, and no INT8 weights.

- [ ] **Step 4: Write failing module-preparation tests**

Test a meta model containing two exact `nn.Linear` modules. Assert that preparation creates INT8 Parameters with `requires_grad=False`, FP32 `scale_weight` buffers, per-layer group sizes, the shared forward, and model-level `is_convrot_int8`/`convrot_int8_layer_count` attributes. Add rejection tests for a missing module and an `nn.Linear` subclass.

- [ ] **Step 5: Implement model preparation and refactor Krea2 patching through it**

Add `prepare_convrot_int8_model(model: nn.Module, artifact: ConvRotInt8Artifact, *, bwd_mode: str = "bf16") -> nn.Module`. For each layer, require `type(module) is nn.Linear`, replace the existing meta weight with `nn.Parameter(torch.empty(module.weight.shape, device=module.weight.device, dtype=torch.int8), requires_grad=False)`, register an FP32 `[out_features, 1]` `scale_weight`, bind `convrot_int8_linear_forward_patch`, and store `_convrot_groupsize` and `_convrot_bwd_mode`. Build equivalent layer specs from Krea2's canonical `.scale_weight` state dict inside `apply_convrot_int8_monkey_patch` so one preparation path owns the runtime contract.

- [ ] **Step 6: Run shared and Krea2 tests**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_convrot_int8_artifact.py tests/test_krea2_convrot_int8.py -q`

Expected: CPU tests pass; CUDA/Triton tests skip when unavailable.

- [ ] **Step 7: Commit the shared adapter**

```powershell
git add src/musubi_tuner/modules/convrot_int8_utils.py tests/test_convrot_int8_artifact.py tests/test_krea2_convrot_int8.py
git commit -m "feat: inspect prequantized ConvRot artifacts"
```

### Task 2: Artifact-Aware Streaming Loader

**Files:**
- Modify: `src/musubi_tuner/minimax_h3/checkpoint.py`
- Modify: `tests/test_convrot_int8_artifact.py`
- Regression: `tests/test_minimax_h3_model.py`

**Interfaces:**
- Consumes: `ConvRotInt8Artifact`, `canonicalize_convrot_int8_key`, `inspect_convrot_int8_artifact`, `prepare_convrot_int8_model`.
- Produces: `inspect_safetensors_convrot_int8` and new `convrot_artifact: ConvRotInt8Artifact | None` / `convrot_bwd_mode: str` keyword parameters on `load_safetensors_module`.

- [ ] **Step 1: Write failing streaming-loader tests**

Cover these behaviors with tiny models and one- or two-shard safetensors:

```python
def test_artifact_loader_preserves_fp32_scale_and_converts_compute_islands(tmp_path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
            self.compute = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
            self.register_buffer("fixed", torch.empty(3, dtype=torch.float32))

    scale = torch.tensor([[1.000123], [0.999321]], dtype=torch.float32)
    payload = json.dumps(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}
    ).encode("utf-8")
    path = tmp_path / "selective.safetensors"
    save_file(
        {
            "quant.weight": torch.zeros(2, 4, dtype=torch.int8),
            "quant.weight_scale": scale,
            "quant.comfy_quant": torch.tensor(list(payload), dtype=torch.uint8),
            "compute.weight": torch.ones(4, 4, dtype=torch.float16),
            "fixed": torch.arange(3, dtype=torch.float32),
        },
        path,
    )
    artifact = inspect_safetensors_convrot_int8([path])
    loaded = load_safetensors_module(
        Tiny,
        [path],
        device="cpu",
        dtype=None,
        strict_dtype=True,
        convrot_artifact=artifact,
    )
    assert loaded.quant.weight.dtype is torch.int8
    assert loaded.compute.weight.dtype is torch.bfloat16
    assert loaded.fixed.dtype is torch.float32
    assert torch.equal(loaded.quant.scale_weight, scale)
```

Add tests proving that control keys do not become unexpected model keys, duplicate normalized control keys across shards fail, `loaded_model_keys` drives missing checks, an F16 fixed-FP32 island fails, and ordinary non-artifact `strict_dtype=True` retains exact comparison.

- [ ] **Step 2: Run the focused loader tests and verify they fail on the R1 gate**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_convrot_int8_artifact.py -q`

Expected: artifact loading fails at the current `.weight_scale`/`.comfy_quant` rejection or the new keyword is unknown.

- [ ] **Step 3: Add one shared key-normalization path and the inspection wrapper**

Implement:

```python
def _normalize_loaded_key(key, key_prefixes, key_transform):
    key = _normalize_checkpoint_key(key, key_prefixes)
    if key_transform is not None:
        key = key_transform(key)
    return canonicalize_convrot_int8_key(key)
```

Implement `inspect_safetensors_convrot_int8(files, *, key_prefixes=(), key_transform=None, disable_mmap=False) -> ConvRotInt8Artifact | None` and pass the same normalization callable to inspection and streaming so weight, scale, and control keys cannot drift.

- [ ] **Step 4: Replace the single `seen` set and implement artifact dtype policy**

Use `seen_normalized_keys`, `loaded_model_keys`, and `consumed_control_keys`. Prepare the meta model before computing `expected_state`. For an artifact:

```python
if key in artifact.weight_keys:
    if tensor.dtype is not torch.int8:
        dtype_mismatches.append(f"{key}: expected torch.int8, got {tensor.dtype}")
elif key in artifact.scale_keys:
    if tensor.dtype is not torch.float32:
        dtype_mismatches.append(f"{key}: expected torch.float32, got {tensor.dtype}")
    target_dtype = None
elif tensor.is_floating_point():
    if tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        dtype_mismatches.append(f"{key}: unsupported source dtype {tensor.dtype}")
    target_dtype = dtype if dtype is not None else expected_state[key].dtype
    if target_dtype is torch.float32 and tensor.dtype is not torch.float32:
        dtype_mismatches.append(f"{key}: expected fixed torch.float32, got {tensor.dtype}")
```

For BF16 destination tensors, accept only F16/BF16 source storage and cast once. For artifact controls, validate that the canonical key belongs to `control_keys`, mark it consumed, and do not assign it. Keep the old exact strict comparison only when `convrot_artifact is None`.

- [ ] **Step 5: Run loader and existing MiniMax model tests**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_convrot_int8_artifact.py tests/test_minimax_h3_model.py -q`

Expected: all tests pass, including the existing exact-dtype and missing-RoPE regressions.

- [ ] **Step 6: Commit the streaming loader**

```powershell
git add src/musubi_tuner/minimax_h3/checkpoint.py tests/test_convrot_int8_artifact.py tests/test_minimax_h3_model.py
git commit -m "feat: stream MiniMax-H3 ConvRot tensors"
```

### Task 3: Pruned AdaLN Model Runtime

**Files:**
- Modify: `src/musubi_tuner/minimax_h3/model.py`
- Modify: `tests/test_minimax_h3_model.py`
- Modify: `tests/test_minimax_h3_packing.py`

**Interfaces:**
- Produces: `MiniMaxH3Config.adaln_curve_grid`, `MiniMaxH3Config.is_pruned`, the `apply_silu` keyword on `AdalnProj`, and `MiniMaxH3Model._timestep_embeddings`.

- [ ] **Step 1: Write failing pruned-construction and interpolation tests**

Extend the tiny config helper to create `time_embed_dim=8, adaln_curve_grid=1025`. Assert the pruned state dict contains `adaln_t_table` with shape `[1025, 8]`, omits every `time_embedder.*` key, and keeps 8-input block/final AdaLN weights.

Add a deterministic table test:

```python
table = torch.arange(1025, dtype=torch.float32)[:, None].repeat(1, 8)
model.adaln_t_table.copy_(table)
actual = model._timestep_embeddings(torch.tensor([0.0, 0.30017, 1.0]), torch.device("cpu"))
position = torch.tensor([0.0, 0.30017, 1.0]) * 1024
expected = position[:, None].repeat(1, 8).to(model.dtype)
torch.testing.assert_close(actual, expected)
assert not torch.equal(actual[1].float(), table[position[1].round().long()])
```

Assert rank-zero timesteps fail with a shape-specific message, packed production timesteps are one-dimensional FP32, and BF16 input is compared against a reference built from `t_bf16.float()`.

- [ ] **Step 2: Run the pruned tests and verify missing-config failures**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_minimax_h3_model.py tests/test_minimax_h3_packing.py -q`

Expected: failures show that `adaln_curve_grid`, `adaln_t_table`, and `_timestep_embeddings` do not exist.

- [ ] **Step 3: Add the strict pruned configuration and model structure**

Add `adaln_curve_grid: int | None = None`. Require exactly 1025 rows when set and require `time_embed_dim == 8`. Standard mode constructs `TimeEmbedder`; pruned mode registers an FP32 `adaln_t_table` and sets `time_embedder = None` so no standard keys enter the state dict.

Add `apply_silu: bool = True` to `AdalnProj`. Standard blocks pass `True`; pruned block/final projections pass `False` and consume the curve coordinate directly.

- [ ] **Step 4: Implement FP32 interpolation followed by explicit compute cast**

Add:

```python
def _timestep_embeddings(self, unique_timesteps: torch.Tensor, execution_device: torch.device) -> torch.Tensor:
    if unique_timesteps.ndim != 1:
        raise ValueError("MiniMax-H3 unique timesteps must be a one-dimensional tensor")
    if not self.config.is_pruned:
        return self.time_embedder(unique_timesteps.to(execution_device)).to(self.dtype)
    table = self.adaln_t_table
    t_fp32 = unique_timesteps.to(device=table.device, dtype=torch.float32)
    position = t_fp32.clamp(0.0, 1.0) * (table.shape[0] - 1)
    lower = position.floor().long().clamp(max=table.shape[0] - 2)
    fraction = position - lower.to(position.dtype)
    embedding_fp32 = torch.lerp(table.float()[lower], table.float()[lower + 1], fraction[:, None])
    return embedding_fp32.to(self.dtype)
```

Replace the current forward call at `model.py:791` with this method. Add a BF16 pruned forward test outside autocast that records the AdaLN Linear input dtype and requires BF16.

- [ ] **Step 5: Run model and packing tests**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_minimax_h3_model.py tests/test_minimax_h3_packing.py -q`

Expected: full and pruned tests pass.

- [ ] **Step 6: Commit pruned AdaLN runtime**

```powershell
git add src/musubi_tuner/minimax_h3/model.py tests/test_minimax_h3_model.py tests/test_minimax_h3_packing.py
git commit -m "feat: add MiniMax-H3 pruned AdaLN runtime"
```

### Task 4: Automatic Transformer Classification and Loading

**Files:**
- Modify: `src/musubi_tuner/minimax_h3/model.py`
- Modify: `tests/test_minimax_h3_model.py`
- Modify: `tests/test_convrot_int8_artifact.py`

**Interfaces:**
- Consumes: `inspect_safetensors_convrot_int8` and artifact-aware `load_safetensors_module`.
- Produces: the `convrot_int8_bwd: str = "bf16"` keyword on automatic `load_h3_transformer` and full/pruned header classification.

- [ ] **Step 1: Write failing full/pruned automatic-loading tests**

Build tiny full and pruned state dicts, quantize selected exact Linears with `quantize_int8_convrot_weight`, rename canonical scales to external `weight_scale`, add payload tensors, and save synthetic files. Monkeypatch the published-config parser to return tiny released-shaped substitutes only where the test is about loading rather than published dimensions.

Assert:

- Full files classify from valid controls while unrelated metadata text containing `int8` no longer rejects them.
- Pruned files classify from `adaln_t_table` without config metadata.
- A file containing both `adaln_t_table` and `time_embedder.*` fails.
- A table with a non-F32 dtype or shape other than `[1025, 8]` fails before model construction.
- Pruned F16 AdaLN source tensors become BF16; table, patch/output projection, RoPE, and scales stay FP32.
- Patched layer counts and per-layer group sizes equal the synthetic artifact declaration.

- [ ] **Step 2: Run transformer-loading tests and verify the current R1 rejections**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_minimax_h3_model.py tests/test_convrot_int8_artifact.py -q`

Expected: failures come from metadata marker, pruned deferral, or quantized checkpoint rejection.

- [ ] **Step 3: Replace metadata substring rejection with structural classification**

Read normalized tensor header entries through `MemoryEfficientSafeOpen.header`. Keep the current exact config validation for a full transformer. When `adaln_t_table` is present, require a detected ConvRot artifact, exact FP32 `[1025, 8]`, no `time_embedder.*` keys, 8-wide block/final AdaLN inputs, and construct `MiniMaxH3Config(time_embed_dim=8, adaln_curve_grid=1025)`. Reject pruned non-ConvRot files as outside this feature.

- [ ] **Step 4: Integrate inspection and artifact-aware loading**

Extend the signature:

```python
def load_h3_transformer(
    checkpoint_path,
    *,
    device,
    dtype=torch.bfloat16,
    attn_mode="torch",
    split_attn=False,
    convrot_int8_bwd="bf16",
    disable_mmap=False,
) -> MiniMaxH3Model:
    files = resolve_safetensors_files(checkpoint_path)
    artifact = inspect_safetensors_convrot_int8(files, disable_mmap=disable_mmap)
    config = _classify_h3_transformer(files, artifact)
    return load_safetensors_module(
        lambda: MiniMaxH3Model(config, attn_mode=attn_mode, split_attn=split_attn, dtype=dtype),
        files,
        device=device,
        dtype=None,
        strict_dtype=artifact is None,
        convrot_artifact=artifact,
        convrot_bwd_mode=convrot_int8_bwd,
        disable_mmap=disable_mmap,
    )
```

Resolve files once, inspect controls once, classify full/pruned from metadata and headers, then pass the artifact and backward mode into `load_safetensors_module`. Keep BF16 as the activation/compute dtype requirement but change the error text so it does not claim every source weight must be BF16.

- [ ] **Step 5: Add block-swap and compile-visible state assertions**

Assert quantized block weights remain INT8, `scale_weight` is a registered FP32 buffer, `_assert_block_device` includes it, and model attributes expose automatic ConvRot state. Keep group size as a plain immutable integer attribute.

- [ ] **Step 6: Run transformer regression tests**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_minimax_h3_model.py tests/test_minimax_h3_packing.py tests/test_convrot_int8_artifact.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit transformer integration**

```powershell
git add src/musubi_tuner/minimax_h3/model.py tests/test_minimax_h3_model.py tests/test_convrot_int8_artifact.py
git commit -m "feat: load MiniMax-H3 ConvRot transformers"
```

### Task 5: Qwen3-VL INT8 ConvRot Text Encoder

**Files:**
- Modify: `src/musubi_tuner/minimax_h3/text_encoder.py`
- Modify: `tests/test_minimax_h3_text_encoder.py`
- Modify: `tests/test_convrot_int8_artifact.py`

**Interfaces:**
- Consumes: automatic artifact inspection and artifact-aware streamed loading.
- Produces: unchanged `load_h3_text_encoder` API with automatic BF16/INT8 behavior.

- [ ] **Step 1: Write a failing synthetic Qwen-shaped loading test**

Patch the imported `transformers` module with a tiny config/model pair that honors the requested 50 retained layers. Each language layer has `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`; add a small embedding and visual floating island. Save external `model.layers.*` triples for all 350 Linears, with non-BF16-representable FP32 scales.

Assert:

```python
loaded = load_h3_text_encoder(path, processor_path="fake", device="cpu", dtype=torch.bfloat16)
assert loaded.is_convrot_int8
assert loaded.convrot_int8_layer_count == 350
assert loaded.language_model.layers[0].self_attn.q_proj.weight.dtype is torch.int8
assert loaded.language_model.layers[0].self_attn.q_proj.scale_weight.dtype is torch.float32
assert torch.equal(loaded.language_model.layers[0].self_attn.q_proj.scale_weight, original_scale)
assert loaded.visual.weight.dtype is torch.bfloat16
```

Add a BF16 nonquantized regression through the same FP32-meta factory and a failure where one external scale is F16.

- [ ] **Step 2: Run the text encoder tests and verify the quantized deferral**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_minimax_h3_text_encoder.py tests/test_convrot_int8_artifact.py -q`

Expected: the synthetic INT8 artifact fails at the current loader rejection.

- [ ] **Step 3: Inspect with the existing text key transform and stream selectively**

In `load_h3_text_encoder`, resolve files once, call `inspect_safetensors_convrot_int8` with `normalize_h3_text_encoder_key`, and pass the result to `load_safetensors_module`. Keep `strict_dtype=False`; ordinary floating tensors still convert to requested BF16, while the artifact policy validates every I8/F32/U8 triple and exempts canonical `scale_weight` from casting.

- [ ] **Step 4: Run text encoding and cache-entry regressions**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_minimax_h3_text_encoder.py tests/test_minimax_h3_cache_contract.py -q`

Expected: tests pass in a dependency-correct environment; if local optional packages prevent collection, record the pre-existing import failure and run the isolated fake-Transformers tests directly.

- [ ] **Step 5: Commit text encoder support**

```powershell
git add src/musubi_tuner/minimax_h3/text_encoder.py tests/test_minimax_h3_text_encoder.py tests/test_convrot_int8_artifact.py
git commit -m "feat: load MiniMax-H3 ConvRot text encoder"
```

### Task 6: Training, Compile, and LoRA Generation Integration

**Files:**
- Modify: `src/musubi_tuner/minimax_h3_train_network.py`
- Modify: `src/musubi_tuner/minimax_h3_generate_video.py`
- Modify: `tests/test_minimax_h3_training.py`
- Modify: `tests/test_minimax_h3_sampling.py`

**Interfaces:**
- Consumes: model-level `is_convrot_int8`, automatic transformer loading, and existing `lora_minimax_h3` factory.
- Produces: training `--convrot_int8_bwd {bf16,int8}` and `_apply_lora_weights` for non-destructive INT8 generation.

- [ ] **Step 1: Write failing trainer argument and compile tests**

Assert the parser defaults `convrot_int8_bwd` to `bf16`, accepts `int8`, and does not add `--convrot_int8`. Test `on_transformer_loaded` rejects `convrot_int8_bwd=int8` for a BF16 transformer and rejects `base_weights` for an INT8 transformer. Record `model_utils.compile_transformer` arguments and require `disable_linear=True` whenever `transformer.is_convrot_int8`, independent of block swap.

- [ ] **Step 2: Run the trainer tests and verify missing option/hook failures**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_minimax_h3_training.py -q`

Expected: parser and integration assertions fail before implementation.

- [ ] **Step 3: Implement training integration**

Add the backward choice to `minimax_h3_setup_parser`, pass it to `load_h3_transformer`, and add `on_transformer_loaded`. Keep existing FP8 flags rejected, but remove wording that all quantized bases are deferred. Reject destructive `base_weights` only after automatic detection. Update compile exclusion to:

```python
disable_linear = bool(self.blocks_to_swap) or bool(getattr(transformer, "is_convrot_int8", False))
```

- [ ] **Step 4: Write and pass an INT8-base LoRA gradient test**

Prepare a tiny model's four target block Linears with the shared artifact adapter, assign quantized weights/scales, freeze the base, apply `lora_minimax_h3.create_arch_network`, and run forward/backward with gradient checkpointing. Require a nonzero LoRA gradient and no base-weight gradient.

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_minimax_h3_training.py tests/test_krea2_convrot_int8.py -q`

- [ ] **Step 5: Write failing generation branch tests**

Test that BF16 transformers still call `_merge_lora_weights`, while INT8 transformers call a new attached path. With a tiny real LoRA state, snapshot every INT8 base tensor, attach one or more networks, run a forward, and assert base tensors remain byte-identical.

- [ ] **Step 6: Implement attached LoRA inference for INT8 bases**

Add:

```python
def _apply_lora_weights(transformer, args, device) -> list[nn.Module]:
    networks = []
    for index, path in enumerate(args.lora_weight or []):
        multipliers = args.lora_multiplier or []
        includes = args.include_patterns or []
        excludes = args.exclude_patterns or []
        multiplier = multipliers[index] if index < len(multipliers) else 1.0
        include = includes[index] if index < len(includes) else None
        exclude = excludes[index] if index < len(excludes) else None
        state = filter_lora_state_dict(load_file(path), include, exclude)
        network = lora_minimax_h3.create_arch_network_from_weights(
            multiplier, state, unet=transformer, for_inference=True
        )
        if not network.unet_loras:
            raise ValueError(f"MiniMax-H3 LoRA {path} contains no compatible target modules")
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        network.load_state_dict(state, strict=True)
        network.eval().requires_grad_(False).to(device)
        networks.append(network)
    return networks
```

In `run_generation`, load first, branch on `is_convrot_int8`, retain the returned networks for the sampling lifetime, and leave the BF16 destructive merge unchanged. Update CLI help so `--dit` and `--text_encoder` do not claim BF16-only files.

- [ ] **Step 7: Run training and sampling tests**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_minimax_h3_training.py tests/test_minimax_h3_sampling.py -q`

Expected: tests pass in the project dependency environment.

- [ ] **Step 8: Commit runtime integration**

```powershell
git add src/musubi_tuner/minimax_h3_train_network.py src/musubi_tuner/minimax_h3_generate_video.py tests/test_minimax_h3_training.py tests/test_minimax_h3_sampling.py
git commit -m "feat: integrate MiniMax-H3 ConvRot runtime"
```

### Task 7: Documentation, Artifact Audit, and Final Verification

**Files:**
- Modify: `docs/minimax_h3.md`
- Modify: `README.md`
- Verify: all changed source and test files

**Interfaces:**
- Consumes: completed automatic loading and runtime behavior.
- Produces: user-facing support matrix and reproducible verification evidence.

- [ ] **Step 1: Update the supported artifact table and usage text**

List these exact released files:

```text
minimax_h3_fl2va_int8_convrot.safetensors
minimax_h3_ref2va_int8_convrot.safetensors
minimax_h3_fl2va_pruned_int8_convrot.safetensors
minimax_h3_ref2va_pruned_int8_convrot.safetensors
qwen3vl_32b_minimax_h3_int8_convrot.safetensors
```

Document automatic detection, Triton/eager fallback, `--convrot_int8_bwd`, full/pruned behavior, FP32 resident scales under block swap, attached LoRA generation for INT8, retained merge for BF16, and rejection of FP8/NVFP4/AWQ/destructive base merges. Remove statements that ConvRot and pruned AdaLN are deferred.

- [ ] **Step 2: Run Ruff and whitespace checks**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check src/musubi_tuner/modules/convrot_int8_utils.py src/musubi_tuner/minimax_h3/checkpoint.py src/musubi_tuner/minimax_h3/model.py src/musubi_tuner/minimax_h3/text_encoder.py src/musubi_tuner/minimax_h3_train_network.py src/musubi_tuner/minimax_h3_generate_video.py tests/test_convrot_int8_artifact.py tests/test_minimax_h3_model.py tests/test_minimax_h3_text_encoder.py tests/test_minimax_h3_training.py tests/test_minimax_h3_sampling.py
python -m ruff format --check src/musubi_tuner/modules/convrot_int8_utils.py src/musubi_tuner/minimax_h3/checkpoint.py src/musubi_tuner/minimax_h3/model.py src/musubi_tuner/minimax_h3/text_encoder.py src/musubi_tuner/minimax_h3_train_network.py src/musubi_tuner/minimax_h3_generate_video.py tests/test_convrot_int8_artifact.py tests/test_minimax_h3_model.py tests/test_minimax_h3_text_encoder.py tests/test_minimax_h3_training.py tests/test_minimax_h3_sampling.py
git diff --check
```

Expected: exit 0.

- [ ] **Step 3: Run focused and full tests**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_convrot_int8_artifact.py tests/test_krea2_convrot_int8.py -q
python -m pytest tests/test_minimax_h3_model.py tests/test_minimax_h3_packing.py tests/test_minimax_h3_text_encoder.py -q
python -m pytest tests/test_minimax_h3_training.py tests/test_minimax_h3_sampling.py tests/test_minimax_h3_cache_contract.py -q
python -m pytest -q
```

Record exact pass/skip/failure counts. Distinguish implementation failures from the already observed local dependency issues: broken `flash_attn_2_cuda` loading and incompatible installed `huggingface-hub==1.24.0` versus Transformers' `<1.0` requirement.

- [ ] **Step 4: Audit official headers without downloading tensor payloads**

Use HTTP range requests against `Comfy-Org/MiniMax-H3` to read the first eight bytes, safetensors JSON header, and each small `.comfy_quant` payload range. Assert full transformer counts are 250, pruned counts are 200, text counts are 350, all scales are F32, all controls are U8, table shape is `[1025, 8]`, and declared group sizes satisfy the runtime validator.

- [ ] **Step 5: Review the implementation against every acceptance criterion**

Read `docs/superpowers/specs/2026-08-07-minimax-h3-int8-convrot-design.md` line by line. Map each criterion to a passing test, artifact-header result, or explicitly reported environment blocker. Inspect `git diff upstream/dev...HEAD` for unrelated changes and accidental metadata churn.

- [ ] **Step 6: Commit documentation and any final verified corrections**

```powershell
git add docs/minimax_h3.md README.md
git commit -m "docs: document MiniMax-H3 ConvRot support"
```
