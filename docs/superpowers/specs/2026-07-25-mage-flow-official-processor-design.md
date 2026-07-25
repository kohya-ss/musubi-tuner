# Mage-Flow Official Processor Loading Design

- Status: Approved for implementation
- Date: 2026-07-25
- Target: `codex/mage-flow`, then merge into `qinglong`

## Problem

Comfy-Org now publishes the three single-file components used by this
integration:

- `diffusion_models/mage_flow*_bf16.safetensors`
- `text_encoders/qwen3vl_4b_bf16.safetensors`
- `vae/mage_flow_vae_bf16.safetensors`

The current Mage-Flow commands still expose `--processor`, even though users
should not select tokenizer or image-preprocessor assets independently from
the released model family. Mage-Flow has no `--tokenizer` argument today, and
one must not be added.

## Decision

Remove `--processor` from every Mage-Flow command:

- text-encoder cache
- image generation
- training-time sample generation

Keep `--text_encoder` because it identifies the actual Comfy-Org model weight
file. `--dit` and `--vae` also remain unchanged.

`load_mage_flow_text_encoder()` will no longer accept `processor_source`.
Instead, it will always construct `AutoProcessor` from:

- repository: `microsoft/Mage-Flow`
- revision: `faca09c18c1c19458e7fbc3f7bce6f7a7d4d01a9`
- subfolder: `text_encoder`

This official directory contains the tokenizer, chat template, image
preprocessor, and video preprocessor configuration used by the released
Mage-Flow pipeline. Hugging Face caching continues to provide offline reuse
after the first successful download.

## Alternatives Rejected

### Keep `--processor` with a better default

This preserves an unsupported configuration surface. A user could pair the
Comfy-Org weights with processor assets from an incompatible Qwen revision and
silently change tokenization or image conditioning.

### Infer processor assets beside the Comfy-Org weight

The Comfy-Org repository intentionally contains only repackaged weight files;
it does not ship tokenizer or processor configuration. Local-path inference
would fail for the documented layout.

### Follow `microsoft/Mage-Flow` main without a revision

This removes manual input but makes cached conditioning change when upstream
updates files. Training caches must be reproducible, so the release commit is
pinned.

## Data Flow

1. The user passes the Comfy-Org single-file text encoder through
   `--text_encoder`.
2. Header validation and strict backbone loading remain unchanged.
3. The loader obtains official processor assets from the pinned
   `microsoft/Mage-Flow/text_encoder` directory.
4. Cache and generation paths reuse the same `MageFlowTextEncoder` object and
   therefore the same tokenizer and image preprocessor contract.

## Compatibility And Errors

- Existing commands that omit `--processor` continue to behave automatically.
- Commands that still pass `--processor` fail argument parsing instead of
  pretending the override was honored.
- No generic parser, public dataset pipeline, dependency pin, or other model
  architecture changes.
- Network and cache failures from `AutoProcessor.from_pretrained()` remain
  explicit; there is no silent fallback to another Qwen repository.

## Tests

- Parser tests prove that cache, generation, and training parsers expose
  neither `--processor` nor `--tokenizer`.
- A loader test proves the exact repository, revision, and subfolder passed to
  `AutoProcessor.from_pretrained()`.
- Existing strict checkpoint, conditioning, cache, and generation tests remain
  green.
- English and Japanese documentation examples no longer mention
  `--processor` and document the automatic pinned source.

## Non-Goals

- Automatically downloading the three large model weight files.
- Replacing `--dit`, `--vae`, or `--text_encoder` with a model-root argument.
- Adding support for sharded text-encoder weights.
- Adding Turbo or INT8 ConvRot runtime support.
