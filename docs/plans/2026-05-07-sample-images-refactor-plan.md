# Sample Images Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace 8–9 loose positional args in `sample_images`/`sample_image_inference` with typed `SamplingContext` + `SamplePrompt` dataclasses, and replace the lambda around-hook with simple before/after lifecycle hooks.

**Architecture:** New `sampling.py` introduces both dataclasses. `trainer_base.py` adopts them and swaps the around-hook for two simple hooks. Architecture-specific `process_sample_prompts` methods return `list[SamplePrompt]` instead of `list[dict]`. `do_inference` is untouched throughout.

**Tech Stack:** Python 3.10+, dataclasses, pytest (`uv run --extra cu130 pytest`)

**Design doc:** `docs/plans/2026-05-07-sample-images-refactor-design.md`

---

## Task 1: Add `sampling.py` with `SamplePrompt` and `SamplingContext`

**Files:**
- Create: `src/musubi_tuner/training/sampling.py`
- Create: `tests/training/test_sampling.py`

**Step 1: Write the failing test**

```python
# tests/training/test_sampling.py
import torch
import pytest
from unittest.mock import MagicMock


def test_sample_prompt_defaults():
    from musubi_tuner.training.sampling import SamplePrompt
    p = SamplePrompt(prompt="test", width=512, height=512)
    assert p.prompt == "test"
    assert p.width == 512
    assert p.height == 512
    assert p.sample_steps == 20
    assert p.seed is None
    assert p.ctx_vec is None
    assert p.image_path is None
    assert p.control_video_path is None
    assert p.control_image_path is None


def test_sample_prompt_all_fields():
    from musubi_tuner.training.sampling import SamplePrompt
    tensor = torch.zeros(1, 77, 768)
    p = SamplePrompt(
        prompt="cat",
        width=256,
        height=256,
        sample_steps=10,
        seed=42,
        guidance_scale=3.5,
        discrete_flow_shift=1.0,
        cfg_scale=7.0,
        negative_prompt="ugly",
        enum=1,
        ctx_vec=tensor,
        negative_ctx_vec=tensor,
        image_path="/tmp/img.png",
        control_video_path="/tmp/vid.mp4",
        control_image_path=["/tmp/a.png"],
    )
    assert p.seed == 42
    assert p.ctx_vec is tensor
    assert p.image_path == "/tmp/img.png"
    assert p.control_image_path == ["/tmp/a.png"]


def test_sampling_context_fields():
    from musubi_tuner.training.sampling import SamplingContext, SamplePrompt
    mock_accel = MagicMock()
    mock_args = MagicMock()
    mock_vae = MagicMock()
    mock_transformer = MagicMock()
    mock_network = MagicMock()
    prompts = [SamplePrompt(prompt="x", width=64, height=64)]

    ctx = SamplingContext(
        accelerator=mock_accel,
        args=mock_args,
        epoch=1,
        steps=100,
        vae=mock_vae,
        transformer=mock_transformer,
        network=mock_network,
        sample_prompts=prompts,
        dit_dtype=torch.bfloat16,
    )
    assert ctx.steps == 100
    assert ctx.epoch == 1
    assert ctx.dit_dtype == torch.bfloat16
    assert len(ctx.sample_prompts) == 1


def test_sampling_context_epoch_none():
    from musubi_tuner.training.sampling import SamplingContext, SamplePrompt
    ctx = SamplingContext(
        accelerator=MagicMock(),
        args=MagicMock(),
        epoch=None,
        steps=0,
        vae=MagicMock(),
        transformer=MagicMock(),
        network=MagicMock(),
        sample_prompts=[],
        dit_dtype=torch.float16,
    )
    assert ctx.epoch is None
```

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/sample-images-refactor
uv run --extra cu130 pytest tests/training/test_sampling.py -v
```
Expected: `ModuleNotFoundError` — `sampling` doesn't exist yet.

**Step 3: Write `sampling.py`**

```python
# src/musubi_tuner/training/sampling.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from accelerate import Accelerator

if TYPE_CHECKING:
    import argparse


@dataclass
class SamplePrompt:
    # required
    prompt: str
    width: int
    height: int
    # common optional
    sample_steps: int = 20
    seed: int | None = None
    guidance_scale: float = 5.0
    discrete_flow_shift: float = 1.0
    cfg_scale: float | None = None
    negative_prompt: str | None = None
    enum: int = 0
    # precomputed text embeddings (populated by process_sample_prompts)
    ctx_vec: torch.Tensor | None = None
    negative_ctx_vec: torch.Tensor | None = None
    # architecture-specific (flat, all optional)
    image_path: str | None = None
    control_video_path: str | None = None
    control_image_path: list[str] | None = None


@dataclass
class SamplingContext:
    accelerator: Accelerator
    args: argparse.Namespace
    epoch: int | None
    steps: int
    vae: torch.nn.Module
    transformer: torch.nn.Module
    network: torch.nn.Module
    sample_prompts: list[SamplePrompt]
    dit_dtype: torch.dtype
```

Also create `tests/__init__.py` and `tests/training/__init__.py` if they don't exist:

```bash
mkdir -p tests/training
touch tests/__init__.py tests/training/__init__.py
```

**Step 4: Run test to verify it passes**

```bash
uv run --extra cu130 pytest tests/training/test_sampling.py -v
```
Expected: 4 tests PASS.

**Step 5: Commit**

```bash
git add src/musubi_tuner/training/sampling.py tests/ 
git commit -m "feat: add SamplePrompt and SamplingContext dataclasses"
```

---

## Task 2: Replace `on_sample_images` with before/after hooks in `trainer_base.py`

**Files:**
- Modify: `src/musubi_tuner/training/trainer_base.py:1178-1191`
- Modify: `tests/training/test_sampling.py` (add hook tests)

**Step 1: Write the failing tests**

Add to `tests/training/test_sampling.py`:

```python
def _make_trainer():
    """Import and instantiate a minimal concrete subclass for hook testing."""
    from musubi_tuner.training.trainer_base import NetworkTrainer
    from musubi_tuner.training.sampling import SamplingContext, SamplePrompt

    class ConcreteTrainer(NetworkTrainer):
        def process_sample_prompts(self, args, accelerator, sample_prompts):
            return []
        def do_inference(self, *a, **kw):
            raise NotImplementedError

    return ConcreteTrainer(), SamplingContext(
        accelerator=MagicMock(),
        args=MagicMock(),
        epoch=None,
        steps=0,
        vae=MagicMock(),
        transformer=MagicMock(),
        network=MagicMock(),
        sample_prompts=[],
        dit_dtype=torch.bfloat16,
    )


def test_on_before_sample_images_default_returns_ctx():
    trainer, ctx = _make_trainer()
    returned = trainer.on_before_sample_images(ctx)
    assert returned is ctx


def test_on_after_sample_images_default_is_noop():
    trainer, ctx = _make_trainer()
    trainer.on_after_sample_images(ctx)  # should not raise


def test_on_before_sample_images_can_be_overridden():
    from musubi_tuner.training.trainer_base import NetworkTrainer
    from musubi_tuner.training.sampling import SamplingContext, SamplePrompt

    class PatchingTrainer(NetworkTrainer):
        def process_sample_prompts(self, args, accelerator, sample_prompts):
            return []
        def do_inference(self, *a, **kw):
            raise NotImplementedError
        def on_before_sample_images(self, ctx):
            ctx.steps = 999
            return ctx

    trainer = PatchingTrainer()
    ctx = SamplingContext(
        accelerator=MagicMock(), args=MagicMock(), epoch=None, steps=0,
        vae=MagicMock(), transformer=MagicMock(), network=MagicMock(),
        sample_prompts=[], dit_dtype=torch.bfloat16,
    )
    result = trainer.on_before_sample_images(ctx)
    assert result.steps == 999
```

**Step 2: Run test to verify it fails**

```bash
uv run --extra cu130 pytest tests/training/test_sampling.py::test_on_before_sample_images_default_returns_ctx -v
```
Expected: `AttributeError: 'ConcreteTrainer' object has no attribute 'on_before_sample_images'`

**Step 3: Replace the hook in `trainer_base.py`**

Find `on_sample_images` at line 1178. Replace the entire method with two new hooks:

```python
def on_before_sample_images(self, ctx: "SamplingContext") -> "SamplingContext":
    """Called before the sampling loop. Return ctx, optionally modified.

    Override to e.g. swap student weights for EMA (teacher) weights before
    sampling begins.
    """
    return ctx

def on_after_sample_images(self, ctx: "SamplingContext") -> None:
    """Called after the sampling loop. Restore any state changed in on_before.

    Override to restore student weights after EMA sampling, etc.
    """
    pass
```

Add to imports at top of `trainer_base.py` (near other TYPE_CHECKING imports or direct imports):
```python
from musubi_tuner.training.sampling import SamplePrompt, SamplingContext
```

**Step 4: Run test to verify it passes**

```bash
uv run --extra cu130 pytest tests/training/test_sampling.py -v
```
Expected: all tests PASS.

**Step 5: Commit**

```bash
git add src/musubi_tuner/training/trainer_base.py tests/training/test_sampling.py
git commit -m "feat: replace on_sample_images around-hook with on_before/after_sample_images"
```

---

## Task 3: Update `sample_images` to accept `SamplingContext`

**Files:**
- Modify: `src/musubi_tuner/training/trainer_base.py:803-860`

The internal behavior is unchanged — only the signature and how variables are sourced change.

**Step 1: No new tests needed** — behavior is identical, covered by existing hook tests. The signature change is validated by the call site update in the same task.

**Step 2: Update `sample_images` signature and body**

Replace lines 803–860 with:

```python
def sample_images(self, ctx: SamplingContext) -> None:
    """Architecture-independent sample images."""
    if not should_sample_images(ctx.args, ctx.steps, ctx.epoch):
        return

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {ctx.steps}")
    if not ctx.sample_prompts:
        logger.error(f"No prompt file / プロンプトファイルがありません: {ctx.args.sample_prompts}")
        return

    distributed_state = PartialState()

    # Use the unwrapped model
    transformer = ctx.accelerator.unwrap_model(ctx.transformer)
    transformer.switch_block_swap_for_inference()

    save_dir = os.path.join(ctx.args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    ctx = self.on_before_sample_images(ctx)
    try:
        if distributed_state.num_processes <= 1:
            with torch.no_grad(), ctx.accelerator.autocast():
                for prompt in ctx.sample_prompts:
                    self.sample_image_inference(ctx, prompt, save_dir)
                    clean_memory_on_device(ctx.accelerator.device)
        else:
            per_process_params = [
                ctx.sample_prompts[i :: distributed_state.num_processes]
                for i in range(distributed_state.num_processes)
            ]
            with torch.no_grad():
                with distributed_state.split_between_processes(per_process_params) as slices:
                    for prompt in slices[0]:
                        self.sample_image_inference(ctx, prompt, save_dir)
                        clean_memory_on_device(ctx.accelerator.device)
    finally:
        self.on_after_sample_images(ctx)

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    transformer.switch_block_swap_for_training()
    clean_memory_on_device(ctx.accelerator.device)
```

**Step 3: Update `_do_sample` at line 1909**

Replace the existing `_do_sample` function:

```python
def _do_sample(epoch_arg, steps_arg):
    ctx = SamplingContext(
        accelerator=accelerator,
        args=args,
        epoch=epoch_arg,
        steps=steps_arg,
        vae=vae,
        transformer=transformer,
        network=network,
        sample_prompts=sample_parameters or [],
        dit_dtype=dit_dtype,
    )
    self.sample_images(ctx)
```

**Step 4: Run tests**

```bash
uv run --extra cu130 pytest tests/training/test_sampling.py -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add src/musubi_tuner/training/trainer_base.py
git commit -m "feat: update sample_images to accept SamplingContext"
```

---

## Task 4: Update `sample_image_inference` to accept `SamplingContext` + `SamplePrompt`

**Files:**
- Modify: `src/musubi_tuner/training/trainer_base.py:862-1001`

`do_inference` is **untouched** — still called with unpacked positional args.

**Step 1: Replace `sample_image_inference` signature and body**

Replace the method at line 862. Source all values from `ctx` and `prompt` instead of individual args. The call to `do_inference` remains identical:

```python
def sample_image_inference(
    self,
    ctx: SamplingContext,
    prompt: SamplePrompt,
    save_dir: str,
) -> None:
    """Architecture-independent single-prompt inference."""
    assert isinstance(prompt, SamplePrompt), f"expected SamplePrompt, got {type(prompt)}"

    width = (prompt.width // 8) * 8
    height = (prompt.height // 8) * 8
    # 1, 5, 9, 13, ... For HunyuanVideo and Wan2.1
    frame_count = (prompt.frame_count - 1) // self.vae_frame_stride * self.vae_frame_stride + 1

    if self.i2v_training:
        if prompt.image_path is None:
            logger.error("No image_path for i2v model / i2vモデルのサンプル画像生成にはimage_pathが必要です")
            return
    if self.control_training:
        if prompt.control_video_path is None:
            logger.error(
                "No control_video_path for control model / controlモデルのサンプル画像生成にはcontrol_video_pathが必要です"
            )
            return

    device = ctx.accelerator.device
    if prompt.seed is not None:
        torch.manual_seed(prompt.seed)
        torch.cuda.manual_seed(prompt.seed)
        generator = torch.Generator(device=device).manual_seed(prompt.seed)
    else:
        torch.seed()
        torch.cuda.seed()
        generator = torch.Generator(device=device).manual_seed(torch.initial_seed())

    guidance_scale = prompt.guidance_scale if prompt.guidance_scale is not None else self.default_guidance_scale
    discrete_flow_shift = prompt.discrete_flow_shift if prompt.discrete_flow_shift is not None else self.default_discrete_flow_shift

    logger.info(f"prompt: {prompt.prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"frame count: {frame_count}")
    logger.info(f"sample steps: {prompt.sample_steps}")
    logger.info(f"guidance scale: {guidance_scale}")
    logger.info(f"discrete flow shift: {discrete_flow_shift}")
    if prompt.seed is not None:
        logger.info(f"seed: {prompt.seed}")

    do_classifier_free_guidance = prompt.negative_prompt is not None
    if do_classifier_free_guidance:
        logger.info(f"negative prompt: {prompt.negative_prompt}")
        logger.info(f"cfg scale: {prompt.cfg_scale}")

    if self.i2v_training:
        logger.info(f"image path: {prompt.image_path}")
    if self.control_training:
        logger.info(f"control video path: {prompt.control_video_path}")

    transformer = ctx.accelerator.unwrap_model(ctx.transformer)
    has_self_ref_orig_mod = getattr(transformer, "_orig_mod", None) is transformer
    was_train = transformer.training if not has_self_ref_orig_mod else True
    if not has_self_ref_orig_mod:
        transformer.eval()

    video = self.do_inference(
        ctx.accelerator,
        ctx.args,
        prompt,
        ctx.vae,
        ctx.dit_dtype,
        transformer,
        discrete_flow_shift,
        prompt.sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        prompt.cfg_scale,
        image_path=prompt.image_path,
        control_video_path=prompt.control_video_path,
    )

    if not has_self_ref_orig_mod:
        transformer.train(was_train)

    if video is None:
        logger.error("No video generated / 生成された動画がありません")
        return

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{ctx.epoch:06d}" if ctx.epoch is not None else f"{ctx.steps:06d}"
    seed_suffix = "" if prompt.seed is None else f"_{prompt.seed}"
    save_path = (
        f"{'' if ctx.args.output_name is None else ctx.args.output_name + '_'}"
        f"{num_suffix}_{prompt.enum:02d}_{ts_str}{seed_suffix}"
    )

    wandb_tracker = None
    try:
        wandb_tracker = ctx.accelerator.get_tracker("wandb")
        try:
            import wandb
        except ImportError:
            raise ImportError("No wandb / wandb がインストールされていないようです")
    except Exception:
        wandb = None

    if video.shape[2] == 1:
        image_paths = save_images_grid(video, save_dir, save_path, n_rows=video.shape[0], create_subdir=False)
        if wandb_tracker is not None and wandb is not None:
            for ip in image_paths:
                wandb_tracker.log({f"sample_{prompt.enum}": wandb.Image(ip)}, step=ctx.steps)
    else:
        video_path = os.path.join(save_dir, save_path) + ".mp4"
        save_videos_grid(video, video_path)
        if wandb_tracker is not None and wandb is not None:
            wandb_tracker.log({f"sample_{prompt.enum}": wandb.Video(video_path)}, step=ctx.steps)

    ctx.vae.to("cpu")
    clean_memory_on_device(device)
```

Note: `prompt.frame_count` — add `frame_count: int = 1` to `SamplePrompt` in `sampling.py` if not already present.

**Step 2: Add `frame_count` to `SamplePrompt`**

In `src/musubi_tuner/training/sampling.py`, add after `height`:
```python
frame_count: int = 1
```

**Step 3: Run tests**

```bash
uv run --extra cu130 pytest tests/training/test_sampling.py -v
```
Expected: all PASS.

**Step 4: Commit**

```bash
git add src/musubi_tuner/training/trainer_base.py src/musubi_tuner/training/sampling.py
git commit -m "feat: update sample_image_inference to accept SamplingContext + SamplePrompt"
```

---

## Task 5: Migrate `process_sample_prompts` — `flux_2_train_network.py`

**Files:**
- Read: `src/musubi_tuner/flux_2_train_network.py` (find `process_sample_prompts`)
- Modify: same file

**Step 1: Read the current implementation**

```bash
grep -n "def process_sample_prompts\|SamplePrompt\|return " src/musubi_tuner/flux_2_train_network.py | head -30
```

**Step 2: Update the return type**

The method currently builds and returns `list[dict]`. Change it to build and return `list[SamplePrompt]`.

Pattern — instead of:
```python
sample_dict = {"prompt": prompt_text, "width": width, "ctx_vec": ctx_vec, ...}
samples.append(sample_dict)
```

Use:
```python
from musubi_tuner.training.sampling import SamplePrompt
sample = SamplePrompt(prompt=prompt_text, width=width, ctx_vec=ctx_vec, ...)
samples.append(sample)
```

Update the return type annotation to `list[SamplePrompt]`.

**Step 3: Run tests**

```bash
uv run --extra cu130 pytest tests/training/test_sampling.py -v
```
Expected: all PASS.

**Step 4: Commit**

```bash
git add src/musubi_tuner/flux_2_train_network.py
git commit -m "feat: flux_2 process_sample_prompts returns list[SamplePrompt]"
```

---

## Task 6: Migrate remaining architectures' `process_sample_prompts`

Repeat the Task 5 pattern for each file. Same approach: replace dict construction with `SamplePrompt(...)`. Commit per file.

Files to migrate (in order — simpler ones first):
1. `src/musubi_tuner/hv_train_network.py`
2. `src/musubi_tuner/wan_train_network.py`
3. `src/musubi_tuner/zimage_train_network.py`
4. `src/musubi_tuner/qwen_image_train_network.py`
5. `src/musubi_tuner/flux_kontext_train_network.py`
6. `src/musubi_tuner/fpack_train_network.py`
7. `src/musubi_tuner/hv_1_5_train_network.py`
8. `src/musubi_tuner/kandinsky5_train_network.py`
9. `src/musubi_tuner/hv_train.py`

For each:
```bash
# read the file first
grep -n "process_sample_prompts\|return\b" src/musubi_tuner/<file>.py | head -40
# after editing:
uv run --extra cu130 pytest tests/training/test_sampling.py -v
git add src/musubi_tuner/<file>.py
git commit -m "feat: <arch> process_sample_prompts returns list[SamplePrompt]"
```

---

## Task 7: Update self-flow skeleton to use new hooks

**Files:**
- Modify: `src/musubi_tuner/flux_2_train_network_self_flow.py:228-246`

**Step 1: Replace `on_sample_images` with before/after hooks**

Remove lines 228–246 (`on_sample_images`). Add:

```python
def on_before_sample_images(self, ctx: "SamplingContext") -> "SamplingContext":
    """Swap to EMA (teacher) weights before sampling when Self-Flow is active."""
    if not ctx.args.self_flow or self.ema_lora_state is None:
        return ctx
    network = ctx.accelerator.unwrap_model(ctx.network)
    self._saved_student_state = {k: v.clone() for k, v in network.state_dict().items()}
    network.load_state_dict(self.ema_lora_state)
    return ctx

def on_after_sample_images(self, ctx: "SamplingContext") -> None:
    """Restore student weights after EMA sampling."""
    if self._saved_student_state is None:
        return
    network = ctx.accelerator.unwrap_model(ctx.network)
    network.load_state_dict(self._saved_student_state)
    self._saved_student_state = None
```

Add `_saved_student_state: dict | None = None` to `__init__` alongside the other self-flow state fields.

**Step 2: Run tests**

```bash
uv run --extra cu130 pytest tests/training/test_sampling.py -v
```
Expected: all PASS.

**Step 3: Commit**

```bash
git add src/musubi_tuner/flux_2_train_network_self_flow.py
git commit -m "feat: update self-flow skeleton to use on_before/after_sample_images"
```

---

## Verification

After all tasks complete:

```bash
uv run --extra cu130 pytest tests/ -v
# Verify no stray references to old on_sample_images hook
grep -rn "on_sample_images" src/musubi_tuner/
# Should return nothing
```
