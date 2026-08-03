from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PINNED_DIFFUSERS_COMMIT = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"


@pytest.mark.parametrize(
    "relative_path",
    (
        "src/musubi_tuner/minimax_h3/model.py",
        "src/musubi_tuner/minimax_h3/video_vae.py",
        "src/musubi_tuner/minimax_h3/packing.py",
        "src/musubi_tuner/minimax_h3/sampling.py",
        "src/musubi_tuner/minimax_h3/text_encoder.py",
    ),
)
def test_ported_module_records_apache_diffusers_provenance(relative_path):
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    assert "Licensed under the Apache License, Version 2.0" in source
    assert "Hugging Face Diffusers PR #14355" in source
    assert PINNED_DIFFUSERS_COMMIT in source
    assert "ComfyUI is used only as an independent numerical reference" in source


def test_minimax_modules_do_not_keep_foreign_operation_factory_or_decoder_names():
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src/musubi_tuner/minimax_h3").glob("*.py"))

    assert "ops = nn" not in source
    assert "ops." not in source
    assert "LTXVAudioVAEDecode" not in source
