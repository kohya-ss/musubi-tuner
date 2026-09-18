"""Import-direction guard for MiniMax-H3: ``musubi_tuner.minimax_h3`` is the importable
package and the top-level ``minimax_h3_*.py`` files are entry points, so the package, the
trainer and the generation script must never import from a cache script (the shared media
helpers live in ``minimax_h3/media.py`` and ``minimax_h3/checkpoint.py``)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from musubi_tuner.minimax_h3.media import TARGET_FPS, TEXT_VISUAL_FPS, TEXT_VISUAL_FRAME_STRIDE

SRC = Path(__file__).resolve().parents[1] / "src" / "musubi_tuner"
GUARDED = sorted((SRC / "minimax_h3").glob("*.py")) + [SRC / "minimax_h3_train_network.py", SRC / "minimax_h3_generate_video.py"]


def _imported_modules(path: Path) -> list[str]:
    modules = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


@pytest.mark.parametrize("path", GUARDED, ids=lambda path: path.name)
def test_package_and_runtime_scripts_do_not_import_the_cache_scripts(path):
    offenders = [module for module in _imported_modules(path) if module.startswith("musubi_tuner.minimax_h3_cache_")]
    assert offenders == [], f"{path.name} imports from a cache entry point: {offenders}"


def test_text_visual_clock_is_derived_from_the_native_frame_rate():
    # reference videos enter the Qwen3-VL presentation at 2 fps, sampled from the 24 fps decode
    assert TEXT_VISUAL_FPS == 2
    assert TEXT_VISUAL_FRAME_STRIDE == TARGET_FPS // TEXT_VISUAL_FPS == 12
