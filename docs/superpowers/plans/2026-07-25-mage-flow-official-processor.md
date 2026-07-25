# Mage-Flow Official Processor Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove user-selectable Mage-Flow processor/tokenizer configuration and always load the released processor assets from a pinned official Mage-Flow repository.

**Architecture:** Keep the Comfy-Org text encoder as an explicitly selected single safetensors weight file. Move processor ownership into `musubi_tuner.mage_flow.text_encoder`, where one helper loads the official `microsoft/Mage-Flow/text_encoder` assets at a fixed revision; all cache, generation, and training-sample entrypoints call the text encoder loader without processor overrides.

**Tech Stack:** Python 3.10+, argparse, Transformers `AutoProcessor`, Hugging Face Hub caching, pytest, Ruff.

## Global Constraints

- Keep `--dit`, `--vae`, and `--text_encoder`.
- Remove `--processor` from all Mage-Flow parsers.
- Do not add `--tokenizer`.
- Use repository `microsoft/Mage-Flow`.
- Use revision `faca09c18c1c19458e7fbc3f7bce6f7a7d4d01a9`.
- Use subfolder `text_encoder`.
- Do not change dependency pins, generic parsers, public dataset collation, or other architectures.
- Keep the existing single-safetensors strict weight loader.

---

### Task 1: Pin Official Processor Ownership In The Text Encoder

**Files:**
- Modify: `src/musubi_tuner/mage_flow/text_encoder.py`
- Test: `tests/test_mage_flow_text_encoder.py`

**Interfaces:**
- Consumes: `AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)`.
- Produces: `load_mage_flow_processor()` and a `load_mage_flow_text_encoder(path, *, device, dtype)` signature without `processor_source`.

- [ ] **Step 1: Write the failing processor contract test**

Add module and loader imports:

```python
import inspect

import musubi_tuner.mage_flow.text_encoder as text_encoder_module
from musubi_tuner.mage_flow.text_encoder import (
    load_mage_flow_processor,
    load_mage_flow_text_encoder,
)
```

Add:

```python
def test_processor_loader_uses_pinned_official_mage_assets(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_from_pretrained(repo_id, **kwargs):
        captured.update(repo_id=repo_id, **kwargs)
        return sentinel

    monkeypatch.setattr(text_encoder_module.AutoProcessor, "from_pretrained", fake_from_pretrained)

    assert load_mage_flow_processor() is sentinel
    assert captured == {
        "repo_id": "microsoft/Mage-Flow",
        "revision": "faca09c18c1c19458e7fbc3f7bce6f7a7d4d01a9",
        "subfolder": "text_encoder",
    }
    assert "processor_source" not in inspect.signature(load_mage_flow_text_encoder).parameters
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -q tests/test_mage_flow_text_encoder.py::test_processor_loader_uses_pinned_official_mage_assets
```

Expected: collection fails because `load_mage_flow_processor` does not exist, proving the fixed-source contract is absent.

- [ ] **Step 3: Implement the fixed processor loader**

Replace the Qwen repository constants with:

```python
MAGE_FLOW_REPO_ID = "microsoft/Mage-Flow"
MAGE_FLOW_REPO_REVISION = "faca09c18c1c19458e7fbc3f7bce6f7a7d4d01a9"
MAGE_FLOW_TEXT_ENCODER_SUBFOLDER = "text_encoder"
```

Add:

```python
def load_mage_flow_processor():
    return AutoProcessor.from_pretrained(
        MAGE_FLOW_REPO_ID,
        revision=MAGE_FLOW_REPO_REVISION,
        subfolder=MAGE_FLOW_TEXT_ENCODER_SUBFOLDER,
    )
```

Remove `processor_source` from `load_mage_flow_text_encoder()` and replace its conditional source logic with:

```python
processor = load_mage_flow_processor()
```

- [ ] **Step 4: Run the text encoder tests and verify GREEN**

Run:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -q tests/test_mage_flow_text_encoder.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/musubi_tuner/mage_flow/text_encoder.py tests/test_mage_flow_text_encoder.py
git commit -m "fix: pin Mage-Flow processor assets"
```

---

### Task 2: Remove Processor CLI And Call-Site Plumbing

**Files:**
- Modify: `src/musubi_tuner/mage_flow_cache_text_encoder_outputs.py`
- Modify: `src/musubi_tuner/mage_flow_generate_image.py`
- Modify: `src/musubi_tuner/mage_flow_train_network.py`
- Modify: `tests/test_mage_flow_entrypoints.py`
- Modify: `tests/test_mage_flow_training.py`

**Interfaces:**
- Consumes: `load_mage_flow_text_encoder(path, *, device, dtype)` from Task 1.
- Produces: cache, generation, and training parsers with no `--processor` or `--tokenizer` actions.

- [ ] **Step 1: Write failing parser and call-site tests**

In `tests/test_mage_flow_entrypoints.py`, import the cache and training modules:

```python
import musubi_tuner.mage_flow_cache_text_encoder_outputs as cache_text_module
import musubi_tuner.mage_flow_train_network as train_module
```

Add:

```python
@pytest.mark.parametrize(
    "parser",
    [
        cache_text_module.setup_parser(),
        generate_module.setup_parser(),
        train_module.mage_flow_setup_parser(train_module.setup_parser_common()),
    ],
)
def test_mage_parsers_do_not_expose_processor_or_tokenizer(parser):
    option_strings = {option for action in parser._actions for option in action.option_strings}

    assert "--processor" not in option_strings
    assert "--tokenizer" not in option_strings
```

Add a generation call test that constructs valid arguments without a processor
attribute and stops at the text encoder boundary:

```python
def test_generation_loads_text_encoder_without_processor_override(monkeypatch):
    class StopAfterTextEncoder(Exception):
        pass

    captured = {}
    parser = generate_module.setup_parser()
    args = parser.parse_args(
        [
            "--dit",
            "dit.safetensors",
            "--vae",
            "vae.safetensors",
            "--text_encoder",
            "text.safetensors",
            "--prompt",
            "test",
            "--device",
            "cpu",
        ]
    )

    def stop_loader(path, **kwargs):
        captured.update(path=path, **kwargs)
        raise StopAfterTextEncoder

    monkeypatch.setattr(generate_module, "load_mage_flow_text_encoder", stop_loader)

    with pytest.raises(StopAfterTextEncoder):
        generate_module.generate(args, parser)

    assert captured == {
        "path": "text.safetensors",
        "device": torch.device("cpu"),
        "dtype": torch.bfloat16,
    }
```

In `tests/test_mage_flow_training.py`, add:

```python
def test_training_sample_prompts_loads_fixed_processor_contract(monkeypatch):
    captured = {}
    trainer = MageFlowNetworkTrainer()
    trainer.is_edit = False
    encoder = torch.nn.Module()

    monkeypatch.setattr(train_module, "load_prompts", lambda _path: [{"prompt": "test"}])
    monkeypatch.setattr(
        train_module,
        "load_mage_flow_text_encoder",
        lambda path, **kwargs: captured.update(path=path, **kwargs) or encoder,
    )
    monkeypatch.setattr(
        train_module,
        "encode_conditioning",
        lambda *_args, **_kwargs: [torch.zeros(1, 2560), torch.zeros(1, 2560)],
    )
    monkeypatch.setattr(train_module, "clean_memory_on_device", lambda _device: None)

    trainer.process_sample_prompts(
        SimpleNamespace(text_encoder="text.safetensors"),
        SimpleNamespace(device=torch.device("cpu")),
        "prompts.toml",
    )

    assert captured == {
        "path": "text.safetensors",
        "device": torch.device("cpu"),
        "dtype": torch.bfloat16,
    }
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_mage_flow_entrypoints.py::test_mage_parsers_do_not_expose_processor_or_tokenizer `
  tests/test_mage_flow_entrypoints.py::test_generation_loads_text_encoder_without_processor_override `
  tests/test_mage_flow_training.py::test_training_sample_prompts_loads_fixed_processor_contract
```

Expected: parser test reports `--processor`; generation and training paths fail
because they still read `args.processor`.

- [ ] **Step 3: Remove the CLI actions and overrides**

In the cache entrypoint:

- remove the Qwen repository constant import;
- remove `parser.add_argument("--processor", ...)`;
- call `load_mage_flow_text_encoder(args.text_encoder, device=device, dtype=dtype)`.

In generation:

- remove the Qwen repository constant import;
- remove the `--processor` parser action;
- remove `processor_source=args.processor` from the loader call.

In training:

- remove the `--processor` parser action;
- delete `processor_source` and conditional `loader_kwargs` assembly;
- call:

```python
encoder = load_mage_flow_text_encoder(
    args.text_encoder,
    device=accelerator.device,
    dtype=torch.bfloat16,
)
```

- [ ] **Step 4: Run entrypoint and training tests and verify GREEN**

Run:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -q tests/test_mage_flow_entrypoints.py tests/test_mage_flow_training.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add `
  src/musubi_tuner/mage_flow_cache_text_encoder_outputs.py `
  src/musubi_tuner/mage_flow_generate_image.py `
  src/musubi_tuner/mage_flow_train_network.py `
  tests/test_mage_flow_entrypoints.py `
  tests/test_mage_flow_training.py
git commit -m "fix: remove Mage-Flow processor CLI"
```

---

### Task 3: Update Documentation And Verify The Branch

**Files:**
- Modify: `docs/mage_flow.md`

**Interfaces:**
- Consumes: the fixed official processor behavior from Tasks 1 and 2.
- Produces: English and Japanese instructions matching the actual CLI.

- [ ] **Step 1: Update user documentation**

Remove `--processor Qwen/Qwen3-VL-4B-Instruct` from both language examples.
Document that tokenizer, chat-template, and image-processor assets are loaded
automatically from pinned `microsoft/Mage-Flow/text_encoder`, while
`--text_encoder` continues to select the Comfy-Org safetensors weight.

- [ ] **Step 2: Prove the obsolete surface is gone**

Run:

```powershell
rg -n -- '--processor|--tokenizer|processor_source|args\\.processor|QWEN3_VL_4B_INSTRUCT_REPO_ID' `
  src/musubi_tuner/mage_flow `
  src/musubi_tuner/mage_flow_cache_text_encoder_outputs.py `
  src/musubi_tuner/mage_flow_generate_image.py `
  src/musubi_tuner/mage_flow_train_network.py `
  docs/mage_flow.md
```

Expected: no matches.

- [ ] **Step 3: Run Mage-Flow tests**

Run:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -q tests/test_mage_flow_*.py
```

Expected: all required tests pass; the optional FlashAttention 2 test may skip
when its package is absent.

- [ ] **Step 4: Run static and repository verification**

Run:

```powershell
$files = @(git diff --name-only upstream/main -- '*.py')
.\.venv\Scripts\python.exe -m ruff check @files
.\.venv\Scripts\python.exe -m ruff format --check @files
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src/musubi_tuner/mage_flow
git diff --check
git diff upstream/main -- pyproject.toml
```

Expected:

- Ruff passes;
- all repository tests pass except the documented optional dependency skip;
- compileall and diff-check pass;
- `pyproject.toml` has no diff from `upstream/main`.

- [ ] **Step 5: Commit**

```powershell
git rm `
  docs/superpowers/specs/2026-07-25-mage-flow-official-processor-design.md `
  docs/superpowers/plans/2026-07-25-mage-flow-official-processor.md
git add docs/mage_flow.md
git commit -m "docs: document automatic Mage-Flow processor loading"
```

The internal design and execution plan remain in branch history but are not
part of the upstream PR's final tree.

- [ ] **Step 6: Update delivery branches**

Push `codex/mage-flow` to update upstream PR #1010. After its CI succeeds,
merge `codex/mage-flow` into `qinglong`, run the integration test set, and push
`qinglong` without force.
