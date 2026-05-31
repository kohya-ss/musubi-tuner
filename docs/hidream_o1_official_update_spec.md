# HiDream-O1 Official Update Sync Spec

Upstream source:

- Repository: https://github.com/HiDream-ai/HiDream-O1-Image
- Checked branch: `main`
- Checked commit: `1dbe80c8b3ec93322ccd01bc9fa3280803b9d2dd`
- Upstream update note: May 13, 2026 inference/pipeline update for faster IP inference, layout/skeleton conditioning, and Dev editing scheduler behavior.

## Goal

Sync the upstream HiDream-O1 inference behavior that matters for Musubi Tuner without overwriting Musubi-specific training and checkpoint support.

The update must preserve:

- single-checkpoint loading through `--dit`,
- LoRA merge/training support,
- `blocks_to_swap`,
- optional cached `input_embeds`,
- `--flash_attn` as an explicit Musubi option,
- existing pixel/control cache contracts.

## Upstream Behaviors To Sync

### Dev Editing Scheduler

Official behavior:

- `model_type=full`: 50 steps, guidance scale `5.0`, shift `3.0`, default FlowUniPC scheduler.
- `model_type=dev`, no editing or multi-reference IP: 28 steps, guidance scale `0.0`, shift `1.0`, flash scheduler with `DEFAULT_TIMESTEPS`.
- `model_type=dev`, exactly one reference image: default to `flow_match` scheduler with `DEFAULT_TIMESTEPS`, guidance scale `0.0`, shift `1.0`.
- `--editing_scheduler flash` can force the old flash scheduler for Dev editing.
- Official CLI only forwards `noise_scale_start`, `noise_scale_end`, and `noise_clip_std` to the Dev `flash` branch. The Dev
  `flow_match` editing branch uses `generate_image()` defaults unless those values are passed by a direct API caller.

Musubi changes required:

- Wire `flow_match` into `hidream_o1.pipeline.build_scheduler()` using the existing project dependency
  `diffusers.FlowMatchEulerDiscreteScheduler`; do not add another scheduler implementation.
- Add `--editing_scheduler {flow_match,flash}` to `hidream_o1_generate_image.py`.
- In training sample inference, allow per-sample `editing_scheduler` and default Dev one-reference sampling to `flow_match`.
- Pass noise-scale kwargs only for the `flash` scheduler path when mirroring official CLI behavior.
- Keep full model behavior unchanged.

Existing scheduler inventory:

- `musubi_tuner.modules.scheduling_flow_match_discrete.FlowMatchDiscreteScheduler` already exists for other training paths.
- `musubi_tuner.qwen_image.qwen_image_utils.FlowMatchEulerDiscreteScheduler` already contains a local Qwen-Image copy.
- HiDream should not duplicate either class. The minimal change is an import/routing update in HiDream's `build_scheduler()`.

### Layout Conditioning

Official behavior:

- `layout_bboxes` can be a JSON string or a JSON file path.
- Input format is xxyy relative coordinates: `[x1, x2, y1, y2]`.
- Supported wrappers include `{"layout_bboxes": ...}`, `{"bboxes": ...}`, `{"boxes": ...}`, and `{"bbox_list": ...}`.
- Layout conditioning creates colored bordered reference images plus one black layout image with colored boxes, then runs the normal multi-reference path.

Musubi changes required:

- Port the upstream layout helper functions into `hidream_o1.utils`.
- Add `layout_bboxes` argument to `generate_image()`.
- Add `--layout_bboxes` to standalone inference.
- Allow `layout_bboxes` in training sample prompt configs.
- For cached control training, layout synthesis is an inference/sampling feature only unless the dataset explicitly materializes those layout images as control files.

### IP Inference Acceleration

Official behavior:

- For reference-image generation, Qwen3-VL image embeddings and DeepStack image embeddings are computed once and reused across denoising steps.
- `generate_image()` stores `cond_image_embeds` and `cond_deepstack_image_embeds` from the first forward pass and passes them back as `precomputed_*` inputs.

Musubi changes required:

- Extend generation forward to accept `precomputed_image_embeds` and `precomputed_deepstack_image_embeds`.
- Return `cond_image_embeds` and `cond_deepstack_image_embeds` in model outputs.
- Reuse precomputed embeddings only in inference/no-grad sampling. Training must still recompute visual embeddings so visual LoRA gets gradients.

### DeepStack Visual Injection In Generation

Official behavior:

- Generation forward now mirrors normal Qwen3-VL forward:
  - build `visual_pos_masks`,
  - carry `deepstack_visual_embeds`,
  - inject DeepStack visual features into decoder layers,
  - pass those tensors through both standard attention and flash-attention paths.
- The T2I dummy vision path touches DeepStack outputs too, keeping visual/deepstack params on a live zero-gradient path.

Musubi changes required:

- Patch `Qwen3VLModel._forward_generation()` instead of only `pipeline.py`.
- Preserve Musubi's optional `inputs_embeds` argument while adding the upstream DeepStack logic.
- Extend `_run_decoder_flash()` call to receive `visual_pos_masks` and `deepstack_visual_embeds`.
- Pass `visual_pos_masks` and `deepstack_visual_embeds` into `language_model()` on the non-flash path.
- Include DeepStack outputs in the T2I dummy-gradient scalar.

This is not just a speed patch: it affects correctness for I2I/control training because control visual features must reach the same decoder injection points as official inference.

## Non-Goals

- Do not change the training timestep sampling policy in this sync.
- Do not change resolution snapping in this sync.
- Do not require a full HuggingFace model directory; `--dit` single-checkpoint support remains required.
- Do not force flash attention on. Official inference hardcodes flash in one path, but Musubi should keep the existing `--flash_attn` switch for compatibility.
- Do not implement synthetic layout generation inside dataset cache unless explicitly requested later.

## Linus Review Notes

- Highest-risk bug if skipped: syncing only `pipeline.py` would speed up nothing important for training and would still leave generation forward missing DeepStack visual injection.
- Highest-risk regression if copied blindly: official `_forward_generation()` no longer accepts Musubi's cached `inputs_embeds`; losing that would break the HiDream text embedding cache path.
- `precomputed_*` embeddings must be inference-only. Reusing detached embeddings during training would silently kill visual LoRA gradients.
- Layout support is user-facing inference plumbing. It should not mutate dataset cache semantics.
- Scheduler selection should be centralized so standalone inference and training sample inference do not drift.
- Do not copy a new FlowMatch scheduler into HiDream. The repository already has FlowMatch scheduler code and a pinned
  `diffusers` dependency; HiDream only needs to route `scheduler_name="flow_match"` to an existing implementation.
- Treat scheduler choice, `DEFAULT_TIMESTEPS`, and noise kwargs as one recipe. Passing Dev flash noise kwargs into the
  Dev `flow_match` editing branch would silently diverge from official code.

## Verification Plan

1. `ruff check` on modified HiDream files.
2. `python -m compileall` on modified HiDream files.
3. Unit-level smoke checks:
   - `build_scheduler(..., scheduler_name="flow_match")` constructs without error when diffusers is installed.
   - layout JSON string and JSON file parsing both produce expected PIL reference image counts.
   - Dev scheduler selection returns `flow_match` for exactly one ref image and `flash` for zero or multiple refs.
   - `_forward_generation()` still accepts `inputs_embeds`.
4. Integration checks where GPU/model is available:
   - T2I sample still works.
   - Dev edit with one ref uses `flow_match`.
   - multi-ref IP sampling reuses visual embeddings after the first step.
   - control training forward still recomputes visual features and produces gradients for visual LoRA targets.
