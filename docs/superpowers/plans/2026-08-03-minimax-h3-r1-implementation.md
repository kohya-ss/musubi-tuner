# MiniMax-H3 R1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native BF16 MiniMax-H3 T2VA, FL2VA, and JSONL Ref2VA caching, LoRA training, block swap, and joint video/audio sampling.

**Architecture:** Keep shared dataset and trainer changes limited to architecture registration and frame-count normalization. Put all H3 media contracts, model code, packing, checkpoint loading, and sampling in `musubi_tuner.minimax_h3`; expose four architecture-specific commands that reuse the repository's common cache and trainer lifecycle.

**Tech Stack:** Python 3.10-3.12, PyTorch, Accelerate, Transformers 4.57.6, PyAV, safetensors, pytest.

## Global Constraints

- R1 accepts BF16 transformer weights only; ConvRot, prequantized INT8, and pruned AdaLN remain deferred to R2.
- The released model configuration is fixed to `in_channels=24`, `audio_in_channels=32`, `hidden_size=5376`, `num_layers=50`, `num_attention_heads=56`, `attention_head_dim=128`, and `text_dim=5120`; patching `[1,2,2]` makes the video projection input width 96.
- Supported tasks are `t2va`, `fl2va`, and JSONL-only `ref2va`.
- Target video frames use `F = 17 * n + 5`; target audio latent length uses exact integer arithmetic `A = (10 * F + 3) // 6`.
- Cache keys must follow the repository's `latents_` and `varlen_` collation contracts exactly.
- Every H3 dataset uses `batch_size = 1`; gradient accumulation is the R1 effective-batch mechanism.
- Training uses independent video/audio shifts, one scalar noise amount for the single item, raw dataward velocity targets, video mean MSE for every sample, and optional audio mean MSE selected by the cache policy scalar.
- Ported model code is derived from Apache-2.0 Diffusers PR #14355 at pinned commit `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`; ComfyUI is validation-only.
- BF16 block swap is required for both training and inference.
- Every production behavior is introduced through a red-green-refactor test cycle.

---

## File Map

- `src/musubi_tuner/dataset/architectures.py`: H3 short/full names and architecture-aware frame rounding.
- `src/musubi_tuner/dataset/bucket.py`: H3 32-pixel resolution steps.
- `src/musubi_tuner/dataset/image_video_dataset.py`: H3 24 fps construction path and shared frame helper calls.
- `src/musubi_tuner/dataset/cache_io.py`: H3 latent and text cache writers.
- `src/musubi_tuner/minimax_h3/media.py`: JSONL records, canonical references, audio selection, and exact target geometry.
- `src/musubi_tuner/minimax_h3/video_vae.py`: released video VAE and normalization boundary.
- `src/musubi_tuner/minimax_h3/audio_vae.py`: released stereo audio VAE and posterior mode boundary.
- `src/musubi_tuner/minimax_h3/text_encoder.py`: Qwen3-VL presentation, pre-norm layer-50 extraction, tags, and token bound.
- `src/musubi_tuner/minimax_h3/packing.py`: modality rows, tags, timesteps, FP64 position grid, and unpacking.
- `src/musubi_tuner/minimax_h3/model.py`: BF16 H3 transformer and block-swap lifecycle.
- `src/musubi_tuner/minimax_h3/checkpoint.py`: artifact discovery and strict sharded loading.
- `src/musubi_tuner/minimax_h3/sampling.py`: paired schedules, joint denoising, decode, and mux helpers.
- `src/musubi_tuner/minimax_h3_cache_latents.py`: dual-VAE cache command.
- `src/musubi_tuner/minimax_h3_cache_text_encoder_outputs.py`: multimodal Qwen cache command.
- `src/musubi_tuner/minimax_h3_train_network.py`: dataset batch-size gate, trainer hooks, CLI, and LoRA target contract.
- `src/musubi_tuner/minimax_h3_generate_video.py`: standalone T2VA/FL2VA/Ref2VA generation.
- Root `minimax_h3_*.py` files: thin module entrypoint wrappers.
- `src/musubi_tuner/networks/lora_minimax_h3.py`: H3 LoRA include/exclude policy.
- `tests/test_minimax_h3_*.py`: synthetic contract, packing, model, training, and sampling coverage.

### Task 1: Architecture And Frame-Grid Wiring

**Files:**
- Modify: `src/musubi_tuner/dataset/architectures.py`
- Modify: `src/musubi_tuner/dataset/bucket.py`
- Modify: `src/musubi_tuner/dataset/image_video_dataset.py`
- Modify: `src/musubi_tuner/training/trainer_base.py`
- Create: `tests/test_minimax_h3_dataset.py`

**Interfaces:**
- Produces: `ARCHITECTURE_MINIMAX_H3`, `ARCHITECTURE_MINIMAX_H3_FULL`, and `round_down_frame_count(frame_count: int, architecture: str, vae_frame_stride: int) -> int`.
- Produces: `VideoDataset.TARGET_FPS_MINIMAX_H3 == 24.0` and `BucketSelector.RESOLUTION_STEPS_MINIMAX_H3 == 32`.

- [x] **Step 1: Write failing registration and helper tests**

```python
def test_h3_architecture_and_bucket_step():
    assert ARCHITECTURE_MINIMAX_H3 == "mmh3"
    assert ARCHITECTURE_MINIMAX_H3_FULL == "minimax_h3"
    assert BucketSelector.ARCHITECTURE_STEPS_MAP[ARCHITECTURE_MINIMAX_H3] == 32


@pytest.mark.parametrize((frames, expected), [(5, 5), (21, 5), (22, 22), (38, 22), (39, 39), (56, 56)])
def test_h3_frame_count_rounds_to_17n_plus_5(frames, expected):
    assert round_down_frame_count(frames, ARCHITECTURE_MINIMAX_H3, 4) == expected


def test_frame_helper_has_no_default_stride_and_preserves_stride_one():
    with pytest.raises(TypeError):
        round_down_frame_count(8, ARCHITECTURE_KREA2)
    assert round_down_frame_count(8, ARCHITECTURE_KREA2, 1) == 8
    assert round_down_frame_count(8, ARCHITECTURE_QWEN_IMAGE, 1) == 8
```

- [x] **Step 2: Run the tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_dataset.py -v`

Expected: collection or assertion failure because H3 constants and `round_down_frame_count` do not exist.

- [x] **Step 3: Implement the shared registration and required-argument helper**

```python
ARCHITECTURE_MINIMAX_H3 = "mmh3"
ARCHITECTURE_MINIMAX_H3_FULL = "minimax_h3"


def round_down_frame_count(frame_count: int, architecture: str, vae_frame_stride: int) -> int:
    if architecture == ARCHITECTURE_MINIMAX_H3:
        if frame_count < 5:
            raise ValueError("MiniMax-H3 requires at least 5 frames")
        return 5 + ((frame_count - 5) // 17) * 17
    return 1 + ((frame_count - 1) // vae_frame_stride) * vae_frame_stride
```

Import the H3 constant in `bucket.py`, register step 32, add `TARGET_FPS_MINIMAX_H3 = 24.0`, and replace all three inline frame expressions with explicit calls that pass `self.vae_frame_stride`.

- [x] **Step 4: Add construction and call-site regression tests**

Patch only external media probing in the test, construct an H3 `VideoDataset`, and assert target FPS 24 plus target frame preservation. Exercise the `frame_extraction="full"` normalization and `NetworkTrainer.sample_image_inference` normalization through small test subclasses so the three call sites cannot drift.

- [x] **Step 5: Run focused and existing dataset tests**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_dataset.py tests/test_ideogram4_synthetic.py -v`

Expected: PASS, including stride-one regressions.

- [x] **Step 6: Commit**

```powershell
git add src/musubi_tuner/dataset/architectures.py src/musubi_tuner/dataset/bucket.py src/musubi_tuner/dataset/image_video_dataset.py src/musubi_tuner/training/trainer_base.py tests/test_minimax_h3_dataset.py
git commit -m "feat: register MiniMax-H3 dataset geometry"
```

### Task 2: H3 Cache Schema And Media Contract

**Files:**
- Modify: `src/musubi_tuner/dataset/cache_io.py`
- Create: `src/musubi_tuner/minimax_h3/__init__.py`
- Create: `src/musubi_tuner/minimax_h3/media.py`
- Create: `tests/test_minimax_h3_cache_contract.py`

**Interfaces:**
- Consumes: architecture constants and frame helper from Task 1.
- Produces: `audio_latent_frames(frame_count: int) -> int`, `waveform_samples(audio_frames: int) -> int`, `H3Reference`, `H3Record`, `load_h3_jsonl_records(path)`, and architecture-specific cache writers.

- [x] **Step 1: Write failing exact-geometry tests**

```python
@pytest.mark.parametrize((frames, audio_frames), [(5, 8), (22, 37), (39, 65), (56, 93)])
def test_audio_grid_uses_integer_identity(frames, audio_frames):
    assert audio_latent_frames(frames) == audio_frames
    assert waveform_samples(audio_frames) == audio_frames * 800
```

Add tests that reject malformed JSONL, ambiguous same-stem sidecars, explicit undecodable audio without fallback, reference counts above 9 images/3 videos/3 audio-bearing/12 total, Ref2VA without a visual reference, and reference video duration outside 2-15 seconds. Missing target audio is accepted as an unsupervised record under the final Task 13 contract.

- [x] **Step 2: Run cache-contract tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_cache_contract.py -v`

Expected: import failure because `musubi_tuner.minimax_h3.media` does not exist.

- [x] **Step 3: Implement canonical media records and exact geometry**

```python
def audio_latent_frames(frame_count: int) -> int:
    if frame_count < 5 or (frame_count - 5) % 17:
        raise ValueError(f"Invalid MiniMax-H3 frame count: {frame_count}; expected 17*n+5")
    return (10 * frame_count + 3) // 6


def waveform_samples(audio_frames: int) -> int:
    return audio_frames * 800
```

Use dataclasses with canonical absolute paths and ordered references. Resolve target audio as explicit `audio_path`, then one exact same-stem sidecar, then embedded stream, then `None`; preserve failures for explicitly selected sources. Keep the shared `VideoJsonlDatasource` tuple contract unchanged.

- [x] **Step 4: Write failing cache-key round-trip tests**

Use temporary safetensors with target video `[24,Fv,Hv,Wv]`, target audio `[32,2,A]`, ordered condition keys, `varlen_mmh3_hidden_states`, and `varlen_mmh3_token_tags`. Assert `BucketBatchManager` produces `batch["latents"]`, stacked `batch["latents_audio"]`, and list-valued text entries.

- [x] **Step 5: Add H3 cache writers**

```python
def save_latent_cache_minimax_h3(
    item_info: ItemInfo,
    tensors: dict[str, torch.Tensor],
    metadata: Optional[dict[str, str]] = None,
):
    save_latent_cache_common(item_info, tensors, ARCHITECTURE_MINIMAX_H3_FULL, metadata)


def save_text_encoder_output_cache_minimax_h3(
    item_info: ItemInfo,
    tensors: dict[str, torch.Tensor],
    metadata: Optional[dict[str, str]] = None,
):
    save_text_encoder_output_cache_common(item_info, tensors, ARCHITECTURE_MINIMAX_H3_FULL, False, metadata)
```

Extend both common writers with an optional final metadata mapping, merge it before the required architecture/item fields, and leave all existing positional calls and shared key parsing unchanged.

- [x] **Step 6: Run focused tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_cache_contract.py tests/test_minimax_h3_dataset.py -v`

```powershell
git add src/musubi_tuner/dataset/cache_io.py src/musubi_tuner/minimax_h3 tests/test_minimax_h3_cache_contract.py
git commit -m "feat: define MiniMax-H3 cache contracts"
```

### Task 3: Released Video And Audio VAEs

**Files:**
- Create: `src/musubi_tuner/minimax_h3/video_vae.py`
- Create: `src/musubi_tuner/minimax_h3/audio_vae.py`
- Create: `src/musubi_tuner/minimax_h3/checkpoint.py`
- Create: `tests/test_minimax_h3_vae.py`

**Interfaces:**
- Produces: `load_video_vae(path, device, dtype)`, `load_audio_vae(path, device, dtype)`, `encode_video(...)`, `decode_video(...)`, `encode_audio_mode(...)`, and `decode_audio(...)`.
- Produces: strict checkpoint helpers that accept a safetensors file or HF snapshot directory and load on CPU before transfer.

- [x] **Step 1: Write failing posterior-boundary tests**

Create tiny encoder doubles returning moments. Assert target video sampling is reproducible from `(cache_seed, canonical_item_key)`, visual conditions use seed 42 and FP16 round-trip before normalization, and audio stores posterior mode `[B,32,2,A]` without evaluating or sampling `logs_proj`.

- [x] **Step 2: Run VAE tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_vae.py -v`

Expected: import failure because the H3 VAE modules do not exist.

- [x] **Step 3: Adapt the released modules from Apache-compatible sources**

Adapt the video and audio autoencoders from Diffusers PR #14355 at commit `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` and the audio module's documented DAC/BigVGAN/alias-free sources. Preserve parameter names needed by the published checkpoints and keep normalization constants at the encode/decode boundary. ComfyUI may be used only for numerical comparison, not as implementation source.

```python
@torch.no_grad()
def encode_audio_mode(vae, waveform: torch.Tensor) -> torch.Tensor:
    encoded = vae.encode(waveform)
    mode = encoded.mode if hasattr(encoded, "mode") else encoded[0]
    if mode.ndim != 4 or mode.shape[1:3] != (32, 2):
        raise ValueError(f"Expected audio latent [B,32,2,A], got {tuple(mode.shape)}")
    return mode
```

- [x] **Step 4: Add strict state-dict and geometry tests**

Assert missing keys, unexpected structural keys, wrong channel widths, unsupported dtype, and non-H3 config fields fail before forward. Use a tiny saved shard index to verify CPU streaming and deterministic key normalization.

- [x] **Step 5: Run VAE and checkpoint tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_vae.py -v`

```powershell
git add src/musubi_tuner/minimax_h3/video_vae.py src/musubi_tuner/minimax_h3/audio_vae.py src/musubi_tuner/minimax_h3/checkpoint.py tests/test_minimax_h3_vae.py
git commit -m "feat: add MiniMax-H3 autoencoders"
```

### Task 4: Dual-VAE Latent Cache Command

**Files:**
- Create: `src/musubi_tuner/minimax_h3_cache_latents.py`
- Create: `minimax_h3_cache_latents.py`
- Modify: `tests/test_minimax_h3_cache_contract.py`
- Modify: `tests/test_top_level_entrypoints.py`

**Interfaces:**
- Consumes: Task 2 media records/cache writers and Task 3 VAE boundaries.
- Produces: cache tensors and metadata for T2VA, FL2VA, and Ref2VA.

- [x] **Step 1: Write failing synthetic command tests**

Use tiny fake VAEs and decoded media to assert exact target crop timestamps, `A * 800` stereo samples, target keys, FL first/last keys, ordered numbered reference keys, and metadata fingerprints. Assert Ref2VA fails before either VAE call when limits are invalid.

- [x] **Step 2: Run command tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_cache_contract.py tests/test_top_level_entrypoints.py -v`

Expected: missing H3 entrypoint and command module.

- [x] **Step 3: Implement cache batching and CLI**

Reuse `BlueprintGenerator`, bucket crop/resize, PyAV decode, and the common cache writer. Add required `--video_vae`, `--audio_vae`, and `--task {t2va,fl2va,ref2va}` arguments plus `--allow_experimental_duration`. Later Task 13 supersedes the original mandatory-target-audio behavior while keeping the Audio VAE mandatory.

```python
def build_latent_tensors(record, task, video_vae, audio_vae, cache_seed):
    tensors = encode_target_pair(record, video_vae, audio_vae, cache_seed)
    if task == "fl2va":
        tensors.update(encode_first_last_conditions(record, video_vae))
    elif task == "ref2va":
        tensors.update(encode_ordered_references(record, video_vae, audio_vae))
    return tensors
```

- [x] **Step 4: Add the root wrapper and run tests**

The root wrapper imports `main` from `musubi_tuner.minimax_h3_cache_latents` and executes it under `if __name__ == "__main__"`.

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_cache_contract.py tests/test_top_level_entrypoints.py -v`

- [x] **Step 5: Commit**

```powershell
git add src/musubi_tuner/minimax_h3_cache_latents.py minimax_h3_cache_latents.py tests/test_minimax_h3_cache_contract.py tests/test_top_level_entrypoints.py
git commit -m "feat: cache MiniMax-H3 audio video latents"
```

### Task 5: Qwen3-VL Layer-50 Text Cache

**Files:**
- Create: `src/musubi_tuner/minimax_h3/text_encoder.py`
- Create: `src/musubi_tuner/minimax_h3_cache_text_encoder_outputs.py`
- Create: `minimax_h3_cache_text_encoder_outputs.py`
- Create: `tests/test_minimax_h3_text_encoder.py`
- Modify: `src/musubi_tuner/dataset/image_video_dataset.py`
- Modify: `tests/test_minimax_h3_dataset.py`
- Modify: `tests/test_top_level_entrypoints.py`

**Interfaces:**
- Produces: `build_presentation(record, task)`, `extract_layer_50_pre_norm(output, model)`, `build_token_tags(processed)`, and `encode_h3_presentation(...) -> tuple[Tensor[L,5120], Tensor[L]]`.

- [x] **Step 1: Write failing presentation and layer-index tests**

Lock non-chat T2VA/FL2VA/Ref2VA strings and timestamp formatting in golden fixtures. Assert `hidden_states[0]` is treated as embeddings and index 50 as the state after exactly 50 layers. For a 50-layer truncated model, assert the last decoder state is captured before final norm.

Also lock crop-specific H3 text-cache paths (`{item_key}_{frame_pos}-{frame_count}_mmh3_te.safetensors`) so FL2VA crops cannot overwrite one another.

- [x] **Step 2: Write failing tag and size-bound tests**

Build a synthetic expanded multimodal token sequence. Assert ordinary tokens/labels are tag 1, every vision span and both flanking vision tokens are tag 0, no tag 2 is emitted, and `L=32769` raises with modality counts and payload estimate.

- [x] **Step 3: Run text tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_text_encoder.py -v`

Expected: missing text encoder module.

- [x] **Step 4: Implement processor presentation and extraction**

Load Qwen3-VL BF16 through Transformers, pass `output_hidden_states=True`, and preserve the pre-final-norm convention. Build tags from processor expansion boundaries rather than token text guesses.

```python
MAX_TEXT_ROWS = 32768
TEXT_WIDTH = 5120


def validate_text_rows(hidden_states: torch.Tensor, token_tags: torch.Tensor) -> None:
    if hidden_states.ndim != 2 or hidden_states.shape[1] != TEXT_WIDTH:
        raise ValueError(f"Expected hidden states [L,{TEXT_WIDTH}], got {tuple(hidden_states.shape)}")
    if token_tags.shape != (hidden_states.shape[0],) or token_tags.dtype != torch.int64:
        raise ValueError("MiniMax-H3 token tags must be int64 [L]")
    if hidden_states.shape[0] > MAX_TEXT_ROWS:
        raise ValueError("MiniMax-H3 text presentation exceeds 32768 rows")
```

- [x] **Step 5: Implement cache CLI and root wrapper**

Save `varlen_mmh3_hidden_states_{dtype}` and `varlen_mmh3_token_tags_int64` with tokenizer, processor, pre-norm layer index, presentation, and tag-algorithm fingerprints.

- [x] **Step 6: Run tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_text_encoder.py tests/test_top_level_entrypoints.py -v`

```powershell
git add src/musubi_tuner/minimax_h3/text_encoder.py src/musubi_tuner/minimax_h3_cache_text_encoder_outputs.py minimax_h3_cache_text_encoder_outputs.py tests/test_minimax_h3_text_encoder.py tests/test_top_level_entrypoints.py
git commit -m "feat: cache MiniMax-H3 multimodal text states"
```

### Task 6: Packed Joint Sequence And FP64 Rotary Clock

**Files:**
- Create: `src/musubi_tuner/minimax_h3/packing.py`
- Create: `tests/test_minimax_h3_packing.py`

**Interfaces:**
- Produces: `pack_video_rows`, `pack_audio_rows`, `build_h3_layout`, `build_position_grid`, `build_timestep_rows`, and `unpack_targets`.
- Consumes: cached latent/audio/text tensor contracts from Tasks 2-5.

- [x] **Step 1: Write failing row-order and count tests**

```python
def test_audio_rows_are_channel_major():
    x = torch.arange(32 * 2 * 3).reshape(1, 32, 2, 3)
    rows = pack_audio_rows(x)
    expected = x.permute(0, 2, 3, 1).reshape(1, 6, 32)
    torch.testing.assert_close(rows, expected)
```

Assert video patch width 96, target video row count `Fv*(Hv//2)*(Wv//2)`, condition order, reference cursor order, target slices, and exact packed-row totals for all tasks.

- [x] **Step 2: Write failing tag/timestep/AdaLN tests**

Assert text tags are preserved, latent rows use video tag 0/audio tag 2, target and condition timesteps follow the clean-coefficient rules, block AdaLN indexes use `3*timestep_index+tag`, and FinalLayer indexes directly select video/audio timestep indexes without tag offsets.

- [x] **Step 3: Write failing FP64 position tests**

Construct small T2VA, FL2VA, and Ref2VA fixtures and compare the full position tensor against explicit double-precision formulas, including `(5/3)*(1,4,4,4,4)` temporal cycles, normalized spatial axes, FL anchors, and monotonically advanced reference cursors.

- [x] **Step 4: Run packing tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_packing.py -v`

Expected: missing packing module.

- [x] **Step 5: Implement immutable layout descriptors and packing**

Use dataclasses containing named slices and per-role geometry. Perform all clock calculations in `torch.float64`, then pass FP64 coordinates to rotary frequency construction; do not create a layout signature or padding path.

- [x] **Step 6: Run tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_packing.py -v`

```powershell
git add src/musubi_tuner/minimax_h3/packing.py tests/test_minimax_h3_packing.py
git commit -m "feat: pack MiniMax-H3 joint modality rows"
```

### Task 7: BF16 Transformer, Checkpoint Loading, And Block Swap

**Files:**
- Create: `src/musubi_tuner/minimax_h3/model.py`
- Modify: `src/musubi_tuner/minimax_h3/checkpoint.py`
- Create: `tests/test_minimax_h3_model.py`

**Interfaces:**
- Consumes: `H3PackedLayout` and row/timestep/position builders from Task 6.
- Produces: `MiniMaxH3Config`, `MiniMaxH3Model.forward(...)`, gradient checkpointing methods, and the standard block-swap lifecycle used by `NetworkTrainer`.

- [x] **Step 1: Write failing tiny-model forward tests**

Instantiate a reduced config with widths divisible by its head count while keeping separate text/video/audio projections. Assert output target slices reconstruct video and audio shapes, block AdaLN respects three modalities, FinalLayer respects one modality, structural tensors retain their batch axes, and `B != 1` is rejected.

- [x] **Step 2: Run model tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_model.py -v`

Expected: missing model module.

- [x] **Step 3: Adapt the BF16 transformer from Diffusers**

Adapt `transformer_minimax_h3.py` from Diffusers PR #14355 at commit `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`, preserving published checkpoint parameter names and the exact refiner, attention, MLP, AdaLN, rotary, and output-head math. Use repository attention helpers for `torch`, `flash`, `flash3`, `sageattn`, and `xformers` only where their semantics match. ComfyUI remains an independent numerical reference.

```python
@dataclass(frozen=True)
class MiniMaxH3Config:
    in_channels: int = 24
    audio_in_channels: int = 32
    hidden_size: int = 5376
    num_layers: int = 50
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    text_dim: int = 5120
```

- [x] **Step 4: Write failing block-swap lifecycle tests**

Patch `create_offloader` with the repository test double and assert `enable_block_swap`, `move_to_device_except_swap_blocks`, inference/training switches, `prepare_block_swap_before_forward`, per-block wait, and submit order. After every wait, assert active block parameters and required buffers are on the execution device.

- [x] **Step 5: Implement block swap using the existing offloader**

Follow the established Wan/Z-Image lifecycle. Do not add an H3 offloader adapter. Keep non-swapped modules and required buffers on the accelerator through `move_to_device_except_swap_blocks`; validate devices immediately after wait.

- [x] **Step 6: Add strict released-config and state-dict tests**

Assert the published config fields match the R1 contract, transformer input projection width is 96, audio projection width is 32, and structural missing/unexpected keys fail. Reject FP8/INT8/ConvRot metadata in R1 with a message pointing to deferred R2.

- [x] **Step 7: Run tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_model.py tests/test_minimax_h3_packing.py -v`

```powershell
git add src/musubi_tuner/minimax_h3/model.py src/musubi_tuner/minimax_h3/checkpoint.py tests/test_minimax_h3_model.py
git commit -m "feat: add MiniMax-H3 BF16 transformer"
```

### Task 8: LoRA Policy And Dual-Modality Trainer

**Files:**
- Create: `src/musubi_tuner/networks/lora_minimax_h3.py`
- Create: `src/musubi_tuner/minimax_h3_train_network.py`
- Create: `minimax_h3_train_network.py`
- Create: `tests/test_minimax_h3_training.py`
- Modify: `tests/test_top_level_entrypoints.py`

**Interfaces:**
- Consumes: model, cache, and packing APIs from Tasks 2-7.
- Produces: `MiniMaxH3NetworkTrainer`, `validate_h3_dataset_batch_size`, H3 CLI arguments, and LoRA module creation compatible with the common network loader.

- [x] **Step 1: Write failing batch-size gate tests**

Build synthetic dataset managers without cache files. Accept only `batch_size = 1`; reject any other configured value before model loading with the dataset index and a gradient-accumulation recommendation. Assert the gate performs no cache reads.

- [x] **Step 2: Write failing process/loss tests**

Use a tiny fake transformer and real tensors at `B=1`. Assert video/audio noises are independently sampled, model times use shifted `1-sigma`, targets are `latents-noise`, condition augmentation uses fresh per-step noise with deterministic condition resets, and loss is `mean(video_mse)` plus optional `mean(audio_mse)` selected through `DiTOutput.extra`. Assert runtime calls reject `B != 1`.

- [x] **Step 3: Run trainer tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_training.py -v`

Expected: missing H3 trainer module.

- [x] **Step 4: Implement CLI validation and the dataset batch-size gate**

Accept only uniform timestep sampling, `weighting_scheme=none`, and discrete flow shift 1 in the generic scheduler. Add `--h3_shift_video 12.0`, `--h3_shift_audio 3.0`, `--h3_visual_cond_clean 0.999`, and `--h3_audio_cond_clean 1.0`. Override `_build_dataset`, call `super()`, then reject any H3 dataset whose configured batch size is not one without opening safetensors.

- [x] **Step 5: Implement dual-modality trainer hooks**

```python
def compute_loss(self, args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step):
    video_loss = F.mse_loss(output.pred.to(network_dtype), output.target.to(network_dtype))
    if output.extra["audio_loss_weight"].item() == 0.0:
        return video_loss, {"loss/video": video_loss.detach(), "loss/audio": video_loss.detach().new_zeros(())}
    audio_loss = F.mse_loss(
        output.extra["audio_pred"].to(network_dtype),
        output.extra["audio_target"].to(network_dtype),
    )
    return video_loss + audio_loss, {"loss/video": video_loss.detach(), "loss/audio": audio_loss.detach()}
```

Return the video pair through standard `DiTOutput.pred/target` and the audio pair through `extra`. Override `process_batch` so audio noise is never inferred from the video-shaped base noise.

- [x] **Step 6: Write and implement LoRA target tests**

Assert `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, and `mlp.fc2` in each of the 50 main transformer blocks are discoverable with stable names. Exclude AdaLN, time conditioning, input/output projections, refiner blocks, VAEs, and text modules. Verify forward/backward gradients reach LoRA parameters under block swap.

- [x] **Step 7: Run focused tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_training.py tests/test_minimax_h3_model.py tests/test_top_level_entrypoints.py -v`

```powershell
git add src/musubi_tuner/networks/lora_minimax_h3.py src/musubi_tuner/minimax_h3_train_network.py minimax_h3_train_network.py tests/test_minimax_h3_training.py tests/test_top_level_entrypoints.py
git commit -m "feat: train MiniMax-H3 LoRA with joint AV loss"
```

### Task 9: Joint AV Sampling And Generation Command

**Files:**
- Create: `src/musubi_tuner/minimax_h3/sampling.py`
- Create: `src/musubi_tuner/minimax_h3_generate_video.py`
- Create: `minimax_h3_generate_video.py`
- Create: `tests/test_minimax_h3_sampling.py`
- Modify: `tests/test_top_level_entrypoints.py`

**Interfaces:**
- Consumes: all model, text, VAE, media, and packing APIs.
- Produces: `build_shifted_schedule`, `sample_joint_av`, and a standalone generation CLI.

- [x] **Step 1: Write failing scheduler and Euler tests**

Assert video shift 12 and audio shift 3 create distinct paired schedules from the same base discretization. For a fixed prediction, assert each update is exactly `x_next = x + (sigma_i - sigma_next) * pred`; reject negative-prediction and audio-slope transforms.

- [x] **Step 2: Write failing deterministic condition-noise tests**

Assert request seed drives target video/audio initialization, every visual condition receives the same reset noise stream, audio conditions use seed plus one, and changing the seed changes all stochastic inputs.

- [x] **Step 3: Run sampling tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_sampling.py -v`

Expected: missing sampling module.

- [x] **Step 4: Implement joint denoising and decode**

Build text and condition rows once, update target video/audio latents with their own sigma deltas at every paired step, preserve the model's raw dataward velocity, decode both VAEs, synchronize duration, and mux audio with video. Keep PyAV muxing behind a replaceable callable boundary tested with argument assertions.

- [x] **Step 5: Implement the generation CLI and root wrapper**

Support T2VA, external first/last images for FL2VA, and JSONL ordered references for Ref2VA. Wire LoRA loading, attention choice, block swap, BF16 checkpoint validation, seed, dimensions, frame count, steps, and output path.

- [x] **Step 6: Run focused tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_minimax_h3_sampling.py tests/test_top_level_entrypoints.py -v`

```powershell
git add src/musubi_tuner/minimax_h3/sampling.py src/musubi_tuner/minimax_h3_generate_video.py minimax_h3_generate_video.py tests/test_minimax_h3_sampling.py tests/test_top_level_entrypoints.py
git commit -m "feat: generate MiniMax-H3 video with audio"
```

### Task 10: Full Verification, Documentation, And PR Update

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-03-minimax-h3-support-design.md` only if implementation exposes a reviewed discrepancy.
- Modify: H3 source/tests only for defects reproduced by verification.

**Interfaces:**
- Consumes: the complete R1 implementation.
- Produces: user-facing command examples and recorded automated/manual acceptance evidence.

- [x] **Step 1: Run formatting and lint checks**

Run: `.venv\Scripts\python -m ruff format --check src/musubi_tuner/minimax_h3 src/musubi_tuner/minimax_h3_*.py src/musubi_tuner/networks/lora_minimax_h3.py tests/test_minimax_h3_*.py`

Run: `.venv\Scripts\python -m ruff check src/musubi_tuner/minimax_h3 src/musubi_tuner/minimax_h3_*.py src/musubi_tuner/networks/lora_minimax_h3.py tests/test_minimax_h3_*.py`

- [x] **Step 2: Run the complete automated suite**

Run: `.venv\Scripts\python -m pytest -v`

Expected: all existing and H3 tests pass without new warnings.

- [ ] **Step 3: Run published-artifact smoke validation**

Use `Comfy-Org/MiniMax-H3` BF16 transformer, video VAE, audio VAE, and matching Qwen3-VL artifacts. Validate strict load, one tiny T2VA forward at batch 1, one LoRA forward/backward with block swap, and one short joint AV sample. The acceptance run does not add a batch-size matrix or repeat all three tasks at multiple batch sizes.

Published-header validation on 2026-08-03 found zero missing, unexpected, shape-mismatched, or dtype-mismatched tensors in both BF16 transformers, and zero key/shape mismatches in the video VAE, audio VAE, and BF16 Qwen3-VL-32B text encoder. Full tensor-body forward validation remains open because the approximately 120 GB artifact set is not present locally.

- [x] **Step 4: Document exact commands and limitations**

Add cache, text-cache, train, and generate examples to `README.md`. State JSONL-only Ref2VA, target-audio fallback policy, 24 fps and `17*n+5`, 32-pixel geometry, 32768 text-row limit, BF16-only R1, and deferred ConvRot R2.

- [x] **Step 5: Inspect the final diff and commit**

Run: `git diff --check`

Run: `git status --short`

```powershell
git add README.md docs/superpowers src tests minimax_h3_*.py
git commit -m "docs: document MiniMax-H3 R1 workflows"
```

- [x] **Step 6: Push and update the Draft PR**

Run: `git push origin codex/minimax-h3-support`

Update PR #1018 with the implemented scope, test commands/results, published-artifact validation, and the explicit R2 deferrals.

### Task 11: Final Review Hardening

**Files:**
- Modify: `src/musubi_tuner/minimax_h3_train_network.py`
- Modify: `src/musubi_tuner/minimax_h3/packing.py`
- Modify: `src/musubi_tuner/minimax_h3/model.py`
- Modify: `tests/test_minimax_h3_training.py`
- Modify: `tests/test_minimax_h3_model.py`
- Modify: `tests/test_minimax_h3_sampling.py`
- Modify: `docs/minimax_h3.md`
- Modify: `docs/superpowers/specs/2026-08-03-minimax-h3-support-design.md`

- [x] **Step 1: Replace structural preflight with the strict R1 gate**

Delete structural fingerprint construction and all multi-item compatibility machinery. Require configured dataset `batch_size = 1` without opening cache files; keep runtime/model defense-in-depth checks and explicit structural batch axes.

- [x] **Step 2: Remove repeated layout work without assuming one global layout**

Vectorize AdaLN run discovery. Keep the R1 modulation run plan flat for its single supported packed sequence while retaining batch axes on tensors. Cache completed rotation tables in a bounded layout/device/dtype LRU and clear it across module moves or checkpoint loads. Do not cache timestep values that change every step.

- [x] **Step 3: Harden checkpoint and modulation semantics**

Register `rope.inv_freq` as an empty checkpoint-owned buffer. Replace gate concatenation with one preallocated residual clone plus slice updates. Verify FP64 in-place scale/shift/gate outputs and gradients, including the RMSNorm weight, against an out-of-place segmented reference; the reported in-place backward failure is not reproducible.

- [x] **Step 4: Make tests independently collectible and metadata explicit**

Give training and sampling tests their own `src` path setup. Record the equal-modality loss policy, retain the intentional T2VA-to-FL2VA base-family mapping, and document the two sources of non-bitwise ComfyUI parity.

- [x] **Step 5: Audit and record Apache-compatible provenance**

Pin Diffusers PR #14355 at `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`, map each ported module to its source file, and add retained Apache-2.0 headers. Rewrite the video VAE from the Diffusers implementation while preserving published checkpoint names. Treat ComfyUI only as an independent numerical reference. Remove operation-factory aliases and unrelated framework comments.

- [x] **Step 6: Run complete verification, commit, push, and refresh PR #1018**

### Task 12: Restore Training-Time Joint AV Sampling

**Files:**
- Add: `src/musubi_tuner/minimax_h3/generation_inputs.py`
- Modify: `src/musubi_tuner/minimax_h3_generate_video.py`
- Modify: `src/musubi_tuner/minimax_h3_train_network.py`
- Modify: `tests/test_minimax_h3_training.py`
- Modify: `docs/minimax_h3.md`
- Modify: `docs/superpowers/specs/2026-08-03-minimax-h3-support-design.md`

- [x] **Step 1: Reproduce the omitted interface**

Replace the regression that expected `--sample_prompts` to fail with tests requiring H3 sampling assets, a live-transformer joint denoise, sequential video/audio decode, and muxed output.

- [x] **Step 2: Share generation input preparation**

Extract record loading, visual decoding, and visual/audio condition encoding from the standalone CLI into an H3 package module so standalone and training-time generation keep one task/layout contract. Keep record parsing separate from pixel decoding and make audio encoding depend only on stored reference-video frame counts.

- [x] **Step 3: Implement H3-specific trainer hooks**

Override sampling preparation instead of changing the shared single-VAE contract. Load each canonical record once. Encode all text presentations before the transformer is allocated, then deliberately re-decode visual pixels for Video-VAE encoding instead of retaining hundreds of MB per prompt across model teardown. Prepare task-specific condition latents, retain both VAEs on CPU, and return `None` for the shared VAE slot. Override per-prompt inference to use the live transformer/LoRA, produce joint target latents, decode each modality with its own VAE in sequence, restore model placement/mode, and mux the result.

- [x] **Step 4: Document training sample inputs and lifecycle**

Document T2VA, FL2VA, and Ref2VA prompt fields, required sampling artifacts, geometry/duration validation, block-swap reuse, and the absence of CFG.

- [x] **Step 5: Run full verification, commit, push, and refresh PR #1018**

### Task 13: Add Per-Sample Video-Only Audio Supervision

**Files:**
- Modify: `src/musubi_tuner/minimax_h3/media.py`
- Modify: `src/musubi_tuner/minimax_h3_cache_latents.py`
- Modify: `src/musubi_tuner/dataset/cache_io.py`
- Modify: `src/musubi_tuner/minimax_h3_train_network.py`
- Modify: `tests/test_minimax_h3_cache_contract.py`
- Modify: `tests/test_minimax_h3_training.py`
- Modify: `docs/minimax_h3.md`
- Modify: `docs/superpowers/specs/2026-08-03-minimax-h3-support-design.md`

- [x] **Step 1: Make target audio optional without weakening explicit-source errors**

Accept a missing target audio stream as unknown. In cache-only `--h3_video_only` mode, bypass target-audio fields, sidecars, probing, decoding, and fingerprinting while leaving Ref2VA reference audio unchanged.

- [x] **Step 2: Cache Audio-VAE silence with an unsupervised scalar**

Keep target-audio rows structurally present. Encode duration-matched FP32 stereo silence through the real Audio VAE and write scalar `mmh3_audio_loss_weight_float32`, with `1.0` only for real target audio and `0.0` for missing or intentionally ignored audio. Require consistent three-valued provenance metadata.

- [x] **Step 3: Make equivalent unsupervised caches reusable**

Retain exact provenance for diagnostics, but compare only the supervision class for `--skip_existing`. Treat complete legacy policy/scalar absence as supervised real audio and reject partial or contradictory new states.

- [x] **Step 4: Select the loss branch per sample**

Validate the collated scalar before transformer execution. Continue joint audio/video noising and forward topology for every sample, but do not evaluate audio MSE at all when the scalar is zero. Record the optional-loss policy and supervised fraction in artifact metadata.

- [x] **Step 5: Scan supervision once before model allocation**

Read paths from constructed batch managers, count repeats with one `Counter`, open each unique cache once, and read only metadata plus the four-byte scalar. Log `supervised_audio_fraction` without reciprocal renormalization.

- [x] **Step 6: Bound cache warnings and document shared-weight risk**

Warn for at most the first ten missing-audio records, then report cache-item policy totals. Document that video-only LoRA updates shared single-stream attention/MLP weights and therefore do not preserve base-model audio behavior.

- [x] **Step 7: Run full verification, commit, and push without posting a PR reply**
