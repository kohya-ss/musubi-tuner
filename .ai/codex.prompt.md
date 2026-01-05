## About This File

This file provides guidance to Codex CLI when working with code in this repository.

## Guidelines

### Coding Style & Naming Conventions
- Style: PEP 8, 4‑space indentation, limit lines to ~120 chars.
- Naming: snake_case for files/functions (`*_train_network.py`, `*_generate_*`), PascalCase for classes.
- Types/Docs: Prefer type hints for public APIs and short docstrings describing args/returns.
- Formatting: No formatter configured; keep diffs small and consistent with surrounding code.

### Testing Guidelines
- Current state: No formal test suite.
- If adding tests, use `pytest`, place under `tests/` mirroring `src/musubi_tuner/` and name files `test_*.py`.
- Run (uv): `uv run pytest -q`. Run (pip): `pytest -q`.
- Prefer small, deterministic unit tests around data utilities and argument parsing.

### Commit & Pull Request Guidelines
- Commits: Use Conventional Commit style seen in history (`feat:`, `fix:`, `doc:`). Write clear, scoped messages.
- PRs: Include a summary, rationale, linked issue(s), and reproduction commands (e.g., the exact `python ... --args`). Add screenshots/log snippets when relevant.
- Docs: Update related files in `docs/` when changing behavior or flags.

### Security & Configuration Tips
- Large files: Do not commit datasets, model weights, or logs (`logs/` is ignored). Use external storage.
- Credentials: Keep any tokens/keys out of the repo and environment‑specific.
- CUDA: Choose the matching extra (`cu124`, `cu128` or `cu130`) for your driver; verify with `torch.cuda.is_available()`.
