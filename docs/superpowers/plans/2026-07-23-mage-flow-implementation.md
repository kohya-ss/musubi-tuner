# Mage-Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add experimental, LoRA-only Mage-Flow and Mage-Flow-Edit training, caching, and sampling with a packed model ABI while keeping Musubi's public bucket data pipeline unchanged.

**Architecture:** The integration owns all packing, model math, component validation, and mode-specific behavior under `musubi_tuner.mage_flow`. The public data loader continues to emit same-resolution bucket batches; a Mage-specific packer converts those tensors and variable-length text lists into the same cumulative-length ABI used by the official model. T2I and Edit share one transformer and four commands, with `--is_edit` selecting distinct cache identities, reference packing, loss masking, and prompt conditioning.

**Tech Stack:** Python 3.10+, PyTorch, Diffusers 0.32.1, Transformers 4.57.6, Safetensors 0.4.5, Accelerate, pytest.

## Global Constraints

- Preserve the Microsoft MIT notice on code ported from Mage commit `ea7109b3515ddd995c2e1212656dc1bc3a9607b7`.
- Do not change `pyproject.toml` dependency pins.
- Accept only one regular `.safetensors` file for each of `--dit`, `--vae`, and `--text_encoder`.
- The released loader is fixed to DiT `128/128/2560/3072/12/24`, RoPE axes `[16, 56, 56]`, patch size `1`, text limit `2048`, and MageVAE latent channels/downsample `128/16`.
- Use architecture identities `mf/mage_flow` and `mfe/mage_flow_edit`; register a per-architecture bucket step of `16` without changing existing entries.
- Keep `BucketBatchManager` and the public collator behavior unchanged.
- Edit accepts exactly one through three ordered references.
- Reference latents stay clean and are excluded from loss and scheduler updates, but share the sample timestep modulation with the target.
- Train LoRA adapters only; do not add full-model fine-tuning.
- Do not claim Comfy-Org compatibility until a released file has a loader fixture.
- Implement SDPA as the required backend and FlashAttention 2 as optional; do not add Sage, xFormers, FlashAttention 4, or dependency requirements.

---

### Task 1: Architecture Identity and Packed Data Contract

**Files:**
- Create: `src/musubi_tuner/mage_flow/__init__.py`
- Create: `src/musubi_tuner/mage_flow/utils.py`
- Modify: `src/musubi_tuner/dataset/architectures.py`
- Modify: `src/musubi_tuner/dataset/bucket.py`
- Test: `tests/test_mage_flow_contracts.py`

**Interfaces:**
- Produces: `MageFlowConfig`, `PackedMageFlowInputs`, `pack_training_batch(...)`, `validate_packed_inputs(...)`, `architecture_for_mode(is_edit: bool)`.
- `pack_training_batch(targets, text, timesteps, controls=None)` accepts target tensors `[B,128,H,W]`, a list of `[Ti,2560]`, `[B]` timesteps, and ordered control batches `[B,128,Hi,Wi]`.

- [ ] **Step 1: Write failing identity and packing tests**

```python
def test_edit_pack_keeps_reference_order_and_one_sample_timestep():
    targets = torch.zeros(2, 4, 2, 3)
    refs = [torch.ones(2, 4, 1, 2), torch.full((2, 4, 1, 1), 2.0)]
    packed = pack_training_batch(targets, [torch.zeros(2, 7), torch.zeros(3, 7)],
                                 torch.tensor([0.2, 0.8]), controls=refs)
    assert packed.image_cu_seqlens.tolist() == [0, 9, 18]
    assert packed.image_shapes == [[(1, 2, 3), (1, 1, 2), (1, 1, 1)]] * 2
    assert packed.target_token_mask.tolist() == [True] * 6 + [False] * 3 + [True] * 6 + [False] * 3
    assert packed.timesteps.tolist() == pytest.approx([0.2, 0.8])
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_contracts.py -q`

Expected: collection fails because `musubi_tuner.mage_flow.utils` does not exist.

- [ ] **Step 3: Add the immutable packed object and validation**

```python
@dataclass(frozen=True)
class PackedMageFlowInputs:
    image_tokens: torch.Tensor
    image_cu_seqlens: torch.Tensor
    text_tokens: torch.Tensor
    text_cu_seqlens: torch.Tensor
    image_shapes: list[list[tuple[int, int, int]]]
    timesteps: torch.Tensor
    target_token_mask: torch.Tensor

    def validate(self, image_dim: int, text_dim: int) -> None:
        validate_packed_inputs(self, image_dim=image_dim, text_dim=text_dim)
```

The packer flattens each sample as target then controls, concatenates samples on dimension 1, creates CPU-independent int32 cumulative lengths on the tensor device, and validates finite values, dimensions, lengths, and strictly increasing boundaries.

- [ ] **Step 4: Register only the two new identities**

```python
ARCHITECTURE_MAGE_FLOW = "mf"
ARCHITECTURE_MAGE_FLOW_FULL = "mage_flow"
ARCHITECTURE_MAGE_FLOW_EDIT = "mfe"
ARCHITECTURE_MAGE_FLOW_EDIT_FULL = "mage_flow_edit"
```

Add both short names with value `16` to `BucketSelector.ARCHITECTURE_STEPS_MAP`; the regression test snapshots every pre-existing key/value.

- [ ] **Step 5: Run focused and dataset regression tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_contracts.py tests/test_dataset.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/musubi_tuner/mage_flow src/musubi_tuner/dataset/architectures.py src/musubi_tuner/dataset/bucket.py tests/test_mage_flow_contracts.py
git commit -m "feat: add Mage-Flow packed data contract"
```

### Task 2: Packed SDPA and NR-MMDiT Core

**Files:**
- Create: `src/musubi_tuner/mage_flow/attention.py`
- Create: `src/musubi_tuner/mage_flow/layers.py`
- Create: `src/musubi_tuner/mage_flow/model.py`
- Test: `tests/test_mage_flow_attention.py`
- Test: `tests/test_mage_flow_model.py`

**Interfaces:**
- Consumes: `PackedMageFlowInputs`.
- Produces: `packed_attention(q, k, v, cu_seqlens, backend="sdpa")` and `MageFlow(MageFlowConfig).forward(packed) -> [1,sum(Li),C]`.

- [ ] **Step 1: Write failing SDPA isolation and dispatch tests**

```python
def test_equal_lengths_use_one_batched_sdpa(monkeypatch):
    calls = []
    monkeypatch.setattr(F, "scaled_dot_product_attention",
                        lambda q, k, v, **kw: calls.append(q.shape) or v)
    out = packed_attention(q, k, v, torch.tensor([0, 3, 6], dtype=torch.int32))
    assert calls == [(2, 2, 3, 4)]
    assert out.shape == q.shape

def test_heterogeneous_segments_cannot_cross_attend():
    out = packed_attention(q, k, v, torch.tensor([0, 1, 4], dtype=torch.int32))
    changed = packed_attention(q, k, v.index_fill(0, torch.tensor([1, 2, 3]), 999),
                               torch.tensor([0, 1, 4], dtype=torch.int32))
    torch.testing.assert_close(out[0], changed[0])
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_attention.py -q`

Expected: import failure for `mage_flow.attention`.

- [ ] **Step 3: Implement the backend**

```python
lengths = cu_seqlens[1:] - cu_seqlens[:-1]
if torch.all(lengths == lengths[0]):
    batched = q.reshape(batch, length, heads, dim).permute(0, 2, 1, 3)
    return F.scaled_dot_product_attention(batched, ...).permute(0, 2, 1, 3).reshape_as(q)
```

Heterogeneous lengths are grouped by identical query/key length when possible and otherwise dispatched per isolated segment. The optional FA2 path imports `flash_attn_varlen_func` lazily and reports an actionable error when unavailable.

- [ ] **Step 4: Write tiny-model parity and timestep tests**

```python
def test_tiny_model_batch_matches_individual_samples():
    model = MageFlow(MageFlowConfig.tiny(image_dim=4, text_dim=7, hidden_size=16,
                                         depth=2, num_heads=2, axes_dim=(4, 2, 2)))
    batched = model(pack)
    singles = torch.cat([model(slice_packed_sample(pack, i)) for i in range(2)], dim=1)
    torch.testing.assert_close(batched, singles, rtol=1e-5, atol=1e-5)
```

The Edit test changes the sample timestep and proves both target and reference outputs change, while the packer's reference values remain unchanged.

- [ ] **Step 5: Port the pinned model math and pass tests**

Port the timestep embedding, RMSNorm, frame-aware multi-axis RoPE, dual-stream attention, modulation, image/text MLPs, output AdaLN, and checkpointed block loop from the pinned Microsoft source. Replace its global backend shim with `packed_attention`; keep canonical module names including `transformer_blocks`, `attn`, `img_mlp`, `txt_mlp`, `img_mod`, and `txt_mod`.

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_attention.py tests/test_mage_flow_model.py -q`

Expected: PASS on CPU in float32.

- [ ] **Step 6: Commit**

```bash
git add src/musubi_tuner/mage_flow/attention.py src/musubi_tuner/mage_flow/layers.py src/musubi_tuner/mage_flow/model.py tests/test_mage_flow_attention.py tests/test_mage_flow_model.py
git commit -m "feat: add packed Mage-Flow transformer"
```

### Task 3: Component Header Validation and DiT Loading

**Files:**
- Modify: `src/musubi_tuner/mage_flow/utils.py`
- Modify: `src/musubi_tuner/mage_flow/model.py`
- Test: `tests/test_mage_flow_checkpoints.py`

**Interfaces:**
- Produces: `inspect_component(path, component, require_decoder=False)`, `normalize_dit_state_dict(sd)`, and `load_mage_flow_transformer(path, device, dtype, attn_mode, fp8_scaled)`.

- [ ] **Step 1: Write synthetic safetensors header tests**

```python
def test_dit_header_rejects_non_contiguous_blocks_before_construction(tmp_path, monkeypatch):
    path = write_tiny_header(tmp_path, block_indices=(0, 2))
    monkeypatch.setattr(model, "MageFlow", lambda *_: pytest.fail("allocated model"))
    with pytest.raises(ComponentValidationError, match=r"blocks.*0.*2"):
        load_mage_flow_transformer(path, device="cpu")
```

Cover non-safetensors files, directories, unknown prefixes, wrong fixed dimensions, 11/13/non-contiguous blocks, and a valid canonical tiny fixture injected through a private test config.

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_checkpoints.py -q`

Expected: imports fail for the loader symbols.

- [ ] **Step 3: Implement strict header-first validation**

Use `safetensors.safe_open(path, framework="pt", device="cpu")` to inspect keys, dtypes, and shapes before constructing a module. Normalize only exact `transformer.` and canonical unprefixed layouts. Error messages include component, layout, expected and actual shapes, and at most ten missing/unexpected keys. Load with `strict=True`; recognized upstream extras are removed explicitly.

- [ ] **Step 4: Add scaled-FP8 conversion**

Reuse `musubi_tuner.utils.model_utils` scaled-FP8 helpers and reject `--fp8_base` without `--fp8_scaled`. Only eligible block linear weights are quantized; norms and modulation remain bf16.

- [ ] **Step 5: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_checkpoints.py tests/test_mage_flow_model.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/musubi_tuner/mage_flow/model.py src/musubi_tuner/mage_flow/utils.py tests/test_mage_flow_checkpoints.py
git commit -m "feat: validate and load Mage-Flow checkpoints"
```

### Task 4: MageVAE and Deterministic Latent Cache

**Files:**
- Create: `src/musubi_tuner/mage_flow/mage_vae.py`
- Create: `src/musubi_tuner/mage_flow_cache_latents.py`
- Create: `mage_flow_cache_latents.py`
- Modify: `src/musubi_tuner/dataset/cache_io.py`
- Test: `tests/test_mage_flow_cache.py`

**Interfaces:**
- Produces: `posterior_seed(architecture, item_key, role, seed)`, `sample_posterior(mean, logvar, generator)`, `MageVAE.encode_moments`, `MageVAE.decode`, and `save_latent_cache_mage_flow(...)`.

- [ ] **Step 1: Write failing deterministic sampling and cache tests**

```python
def test_posterior_is_stable_across_order_and_batching():
    first = encode_fake(items, order=(0, 1), seed=42)
    second = encode_fake(items, order=(1, 0), seed=42)
    torch.testing.assert_close(first["a"], second["a"])
    assert not torch.equal(first["a"], encode_fake(items, order=(0, 1), seed=43)["a"])
```

Also test target/control role separation, keys, metadata identity, 128 channels, control count 1-3, contiguous indices, non-finite rejection, and exact ordered values.

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_cache.py -q`

Expected: import failure for MageVAE/cache helpers.

- [ ] **Step 3: Port MageVAE and deterministic posterior sampling**

Port the inference subset of `mage_vae.py` from the pinned source with the Microsoft MIT notice. Adapt `_load_state_dict` to one safetensors file and expose moments separately:

```python
def encode(self, x, generators=None):
    mean, logvar = self.encode_moments(x).chunk(2, dim=1)
    if generators is None:
        return mean
    return torch.cat([sample_posterior(mean[i:i+1], logvar[i:i+1], generators[i])
                      for i in range(len(generators))])
```

- [ ] **Step 4: Implement the cache command**

Subclass the existing cache-latents flow, require image dimensions divisible by 16, encode targets and controls in semantic order, and save `[128,H,W]` tensors using architecture `mage_flow` or `mage_flow_edit`.

- [ ] **Step 5: Run tests and command help**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_cache.py -q`

Run: `.venv\Scripts\python.exe mage_flow_cache_latents.py --help`

Expected: both PASS/exit 0 without model allocation.

- [ ] **Step 6: Commit**

```bash
git add src/musubi_tuner/mage_flow/mage_vae.py src/musubi_tuner/mage_flow_cache_latents.py mage_flow_cache_latents.py src/musubi_tuner/dataset/cache_io.py tests/test_mage_flow_cache.py
git commit -m "feat: add MageVAE latent caching"
```

### Task 5: Qwen3-VL Conditioning and Text Cache

**Files:**
- Create: `src/musubi_tuner/mage_flow/text_encoder.py`
- Create: `src/musubi_tuner/mage_flow_cache_text_encoder_outputs.py`
- Create: `mage_flow_cache_text_encoder_outputs.py`
- Modify: `src/musubi_tuner/dataset/cache_io.py`
- Test: `tests/test_mage_flow_text_encoder.py`

**Interfaces:**
- Produces: `load_mage_flow_text_encoder(path, device, dtype, is_edit)`, `encode_prompts(...) -> list[Tensor[L,2560]]`, and `save_text_encoder_output_cache_mage_flow(...)`.

- [ ] **Step 1: Write failing template, prefix, and loader tests**

```python
@pytest.mark.parametrize(("is_edit", "drop"), [(False, 34), (True, 64)])
def test_conditioning_drops_exact_prefix(fake_backbone, is_edit, drop):
    out = encode_prompts(fake_backbone, ["instruction"], refs=None if not is_edit else [[image]], is_edit=is_edit)
    torch.testing.assert_close(out[0], fake_backbone.hidden[0, drop:drop + out[0].shape[0]])
```

Test final hidden state only, length 1-2048, Edit one-to-three refs with long edge at most 384, T2I control rejection, visual component validation, pinned 4B dimensional signature, and optional `lm_head.weight` ignored without allocation or re-tie.

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_text_encoder.py -q`

Expected: import failure for `mage_flow.text_encoder`.

- [ ] **Step 3: Implement the current-Transformers adapter**

Construct `Qwen3VLModel` from the pinned 4B config, load the canonical backbone keys strictly from the single safetensors file, and use `AutoProcessor` only for non-weight assets. Build the exact official T2I/Edit chat templates, call the backbone under `torch.no_grad()`, gather valid final hidden states, and remove 34 or 64 prefix tokens.

- [ ] **Step 4: Implement text caching**

Save each result as `varlen_mage_flow_embed_bfloat16`; provide the same ordered controls to Qwen visual conditioning in Edit mode. Do not pad stored tensors.

- [ ] **Step 5: Run tests and help**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_text_encoder.py tests/test_mage_flow_cache.py -q`

Run: `.venv\Scripts\python.exe mage_flow_cache_text_encoder_outputs.py --help`

Expected: PASS/exit 0.

- [ ] **Step 6: Commit**

```bash
git add src/musubi_tuner/mage_flow/text_encoder.py src/musubi_tuner/mage_flow_cache_text_encoder_outputs.py mage_flow_cache_text_encoder_outputs.py src/musubi_tuner/dataset/cache_io.py tests/test_mage_flow_text_encoder.py tests/test_mage_flow_cache.py
git commit -m "feat: cache Mage-Flow Qwen conditioning"
```

### Task 6: T2I LoRA Training and Sampling

**Files:**
- Create: `src/musubi_tuner/networks/lora_mage_flow.py`
- Create: `src/musubi_tuner/mage_flow_train_network.py`
- Create: `mage_flow_train_network.py`
- Create: `src/musubi_tuner/mage_flow_generate_image.py`
- Create: `mage_flow_generate_image.py`
- Test: `tests/test_mage_flow_training.py`
- Test: `tests/test_mage_flow_lora.py`

**Interfaces:**
- Produces: `MageFlowNetworkTrainer`, standard Musubi LoRA `create_network*` functions, `build_scheduler`, and Euler sampling through the packed ABI.

- [ ] **Step 1: Write failing flow sign and LoRA scope tests**

```python
def test_one_euler_step_uses_epsilon_minus_z():
    z = torch.tensor([2.0])
    eps = torch.tensor([5.0])
    x = eps.clone()
    assert euler_step(x, eps - z, sigma=1.0, sigma_next=0.0).item() == pytest.approx(2.0)

def test_lora_targets_only_block_attention_and_ffn(tiny_model):
    names = mage_flow_lora_target_names(tiny_model)
    assert any(".attn.to_q" in name for name in names)
    assert any(".img_mlp" in name for name in names)
    assert not any("img_mod" in name or "txt_mod" in name or "norm" in name for name in names)
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_training.py tests/test_mage_flow_lora.py -q`

Expected: import failures for trainer and LoRA module.

- [ ] **Step 3: Implement T2I trainer hooks**

`call_dit` converts `[B,128,1,H,W]` into a packed object, feeds `timesteps / 1000`, unpacks only target tokens, and returns `DiTOutput(pred, target=noise-latents)`. Preflight checks mode-specific metadata, finite values, conditioning shape, and absence of controls before the first step.

- [ ] **Step 4: Implement exact LoRA selection and round trip**

Subclass the existing LoRA implementation but enumerate only `nn.Linear` descendants of each `MageFlowTransformerBlock.attn`, `.img_mlp`, and `.txt_mlp`. Save `ss_architecture=mage_flow` or `mage_flow_edit` and verify reload produces identical fixed-input output.

- [ ] **Step 5: Implement packed Euler sampling**

Use `build_scheduler(steps, shift=6.0)` and `x_next = x + (sigma_next - sigma) * velocity`. Decode only target latent shapes. The standalone command loads text encoder, DiT, and VAE sequentially and supports fixed seeds.

- [ ] **Step 6: Run tiny end-to-end tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_training.py tests/test_mage_flow_lora.py -q`

Expected: one tiny optimizer step lowers/changes trainable LoRA weights, safetensors round trip succeeds, and fixed-noise sample is deterministic.

- [ ] **Step 7: Commit**

```bash
git add src/musubi_tuner/networks/lora_mage_flow.py src/musubi_tuner/mage_flow_train_network.py mage_flow_train_network.py src/musubi_tuner/mage_flow_generate_image.py mage_flow_generate_image.py tests/test_mage_flow_training.py tests/test_mage_flow_lora.py
git commit -m "feat: train and sample Mage-Flow LoRA"
```

### Task 7: Edit Training and Sampling

**Files:**
- Modify: `src/musubi_tuner/mage_flow_train_network.py`
- Modify: `src/musubi_tuner/mage_flow_generate_image.py`
- Modify: `tests/test_mage_flow_training.py`
- Test: `tests/test_mage_flow_edit.py`

**Interfaces:**
- Consumes: ordered `latents_control_0..N`, `PackedMageFlowInputs.target_token_mask`.
- Produces: shared `--is_edit` behavior for one-to-three references.

- [ ] **Step 1: Write failing one-ref, three-ref, mask, and timestep tests**

```python
@pytest.mark.parametrize("count", [1, 3])
def test_edit_updates_only_target(count, tiny_edit_batch):
    packed, original_refs = make_edit_inputs(tiny_edit_batch(count))
    next_tokens = scheduler_step_targets_only(packed.image_tokens, velocity, packed.target_token_mask, 1.0, 0.0)
    torch.testing.assert_close(next_tokens[~packed.target_token_mask], original_refs)
```

Also assert zero/four refs and missing/duplicate/non-contiguous keys raise, reference order survives packing, and one sample timestep is repeated by modulation across the complete target-plus-reference segment.

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_edit.py -q`

Expected: Edit path rejects or mishandles controls.

- [ ] **Step 3: Implement Edit trainer and sampler**

Collect control cache keys by exact numeric index, pack target then references, keep references clean while noising the target, compute MSE only over target outputs, and update only targets in Euler sampling. Require explicit `--is_edit`; never infer mode.

- [ ] **Step 4: Run Edit and T2I regressions**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_edit.py tests/test_mage_flow_training.py tests/test_mage_flow_model.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/musubi_tuner/mage_flow_train_network.py src/musubi_tuner/mage_flow_generate_image.py tests/test_mage_flow_edit.py tests/test_mage_flow_training.py
git commit -m "feat: add Mage-Flow-Edit training and sampling"
```

### Task 8: Block Swap, Compile, and Runtime Matrix

**Files:**
- Modify: `src/musubi_tuner/mage_flow/model.py`
- Modify: `src/musubi_tuner/mage_flow_train_network.py`
- Modify: `tests/test_mage_flow_model.py`
- Test: `tests/test_mage_flow_runtime.py`

**Interfaces:**
- Produces: `enable_block_swap(num_blocks, device, supports_backward)`, `prepare_block_swap_before_forward()`, and compile of `transformer_blocks`.

- [ ] **Step 1: Write failing runtime feature tests**

```python
def test_block_swap_bounds_and_compile_scope(tiny_model, monkeypatch):
    with pytest.raises(ValueError, match="0 through 10"):
        tiny_model.enable_block_swap(11, "cpu")
    compiled = []
    monkeypatch.setattr(torch, "compile", lambda m, **kw: compiled.append(m) or m)
    compile_repeated_blocks(tiny_model, dynamic=True)
    assert compiled == list(tiny_model.transformer_blocks)
```

Add CUDA-gated SDPA/FA2 parity, bf16 checkpointed backward, block-swap backward, scaled-FP8 finite-output, and compile smoke tests.

- [ ] **Step 2: Run CPU tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_runtime.py -q`

Expected: missing runtime methods.

- [ ] **Step 3: Implement runtime hooks**

Reuse the repository's block offloader and `model_utils.compile_transformer`; allow `0..10` swapped blocks for the fixed 12-block model. Prepare swaps before each forward, preserve checkpointing, and ensure compiled names normalize `._orig_mod.` during state handling.

- [ ] **Step 4: Run CPU and available CUDA tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_runtime.py tests/test_mage_flow_attention.py tests/test_mage_flow_model.py -q`

Expected: CPU tests pass; environment-gated CUDA tests either pass or report SKIPPED with their dependency/device reason.

- [ ] **Step 5: Commit**

```bash
git add src/musubi_tuner/mage_flow/model.py src/musubi_tuner/mage_flow_train_network.py tests/test_mage_flow_model.py tests/test_mage_flow_runtime.py
git commit -m "feat: add Mage-Flow runtime memory features"
```

### Task 9: Entrypoints, Documentation, and Full Regression

**Files:**
- Modify: `tests/test_top_level_entrypoints.py`
- Create: `docs/mage_flow.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-23-mage-flow-design.md`
- Test: `tests/test_mage_flow_entrypoints.py`

**Interfaces:**
- Produces: four discoverable root commands and user documentation with explicit experimental/manual-parity status.

- [ ] **Step 1: Write failing lazy-import help tests**

```python
@pytest.mark.parametrize("script", [
    "mage_flow_cache_latents.py",
    "mage_flow_cache_text_encoder_outputs.py",
    "mage_flow_train_network.py",
    "mage_flow_generate_image.py",
])
def test_help_does_not_allocate_models(script):
    result = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow_entrypoints.py tests/test_top_level_entrypoints.py -q`

Expected: newly registered Mage commands are absent or fail.

- [ ] **Step 3: Finish command parsers and docs**

Document component conversion, cache order, exact T2I/Edit invocations, LoRA-only status, 16-pixel buckets, packed-vs-native behavior, SDPA/optional FA2, FP8 constraints, block swap, compile, known exclusions, pinned upstream revision, and a real-weight manual test checklist. Mark general availability blocked until real component files pass that checklist.

- [ ] **Step 4: Run focused suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mage_flow*.py tests/test_top_level_entrypoints.py -q`

Expected: PASS with only documented environment skips.

- [ ] **Step 5: Prove shared contracts and dependencies are unchanged**

Run: `git diff 8934cfb -- pyproject.toml src/musubi_tuner/dataset/image_video_dataset.py`

Expected: no output.

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: PASS with no new failures.

- [ ] **Step 6: Run static checks and inspect the final diff**

Run: `.venv\Scripts\python.exe -m compileall -q src/musubi_tuner/mage_flow src/musubi_tuner/mage_flow_*.py`

Expected: exit 0.

Run: `git diff --check`

Expected: exit 0 with no output.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/mage_flow.md docs/superpowers/specs/2026-07-23-mage-flow-design.md tests/test_mage_flow_entrypoints.py tests/test_top_level_entrypoints.py
git commit -m "docs: document Mage-Flow integration"
```

## Final Verification Record

Record these exact facts in the final report:

- Focused Mage test count, pass count, and gated skip reasons.
- Full-suite result.
- `git diff --check` result.
- Confirmation that `pyproject.toml` and `dataset/image_video_dataset.py` have no diff from `8934cfb`.
- Manual real-weight parity remains explicitly unverified unless the user supplies the three released component safetensors; do not convert that absence into a compatibility claim.
