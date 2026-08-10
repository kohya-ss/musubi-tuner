import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.minimax_h3.media import H3AudioSource, H3Record, H3Reference
from musubi_tuner.minimax_h3.text_encoder import (
    IMAGE_PLACEHOLDER,
    VIDEO_PLACEHOLDER,
    H3TextVisual,
    build_presentation,
    build_token_tags,
    load_h3_text_encoder,
    normalize_h3_text_encoder_key,
    validate_text_rows,
)

try:
    from musubi_tuner.minimax_h3_cache_text_encoder_outputs import _text_cache_metadata
except ImportError as error:
    _text_cache_metadata = None
    _text_cache_import_error = str(error)
else:
    _text_cache_import_error = None


def _record(tmp_path: Path, references=()) -> H3Record:
    return H3Record(
        video_path=tmp_path / "target.mp4",
        caption="A bright scene with clear sound.",
        references=tuple(references),
        jsonl_line=1,
    )


def _visual(frame_count: int, timestamps=None) -> H3TextVisual:
    return H3TextVisual(
        frames=torch.zeros(frame_count, 32, 32, 3),
        timestamps=None if timestamps is None else tuple(timestamps),
    )


def test_t2va_and_fl2va_presentations_are_non_chat_golden_strings(tmp_path: Path):
    record = _record(tmp_path)

    t2va = build_presentation(record, "t2va")
    fl2va = build_presentation(record, "fl2va", {"first": _visual(1), "last": _visual(1)})

    assert t2va.text == record.caption
    assert t2va.images == ()
    assert t2va.videos == ()
    assert fl2va.text == f"<Picture 1>: {IMAGE_PLACEHOLDER}<Picture 2>: {IMAGE_PLACEHOLDER}{record.caption}"
    assert len(fl2va.images) == 2
    assert fl2va.videos == ()


def test_ref2va_presentation_preserves_jsonl_order_and_timestamp_format(tmp_path: Path):
    image = tmp_path / "face.png"
    video = tmp_path / "motion.mp4"
    soundtrack = tmp_path / "motion.wav"
    voice = tmp_path / "voice.wav"
    references = (
        H3Reference(type="image", path=image),
        H3Reference(
            type="video",
            path=video,
            audio=H3AudioSource(soundtrack, embedded=False),
            duration_seconds=2.0,
        ),
        H3Reference(
            type="audio",
            path=voice,
            audio=H3AudioSource(voice, embedded=False),
            duration_seconds=1.0,
        ),
    )
    record = _record(tmp_path, references)

    presentation = build_presentation(
        record,
        "ref2va",
        {
            image: _visual(1),
            video: _visual(3, [0.0, 0.5, 1.0]),
        },
    )

    assert presentation.text == (
        f"<Picture 1>: {IMAGE_PLACEHOLDER}"
        f"<Audio 1>: <Video 1>: <0.2 seconds>{VIDEO_PLACEHOLDER}"
        f"<1.0 seconds>{VIDEO_PLACEHOLDER}"
        f"<Audio 2>: {record.caption}"
    )
    assert presentation.processor_text == (
        f"<Picture 1>: {IMAGE_PLACEHOLDER}<Audio 1>: <Video 1>: {VIDEO_PLACEHOLDER}<Audio 2>: {record.caption}"
    )
    assert len(presentation.images) == 1
    assert [tuple(video_block.shape) for video_block in presentation.videos] == [
        (3, 32, 32, 3),
    ]


def test_token_tags_cover_expanded_vision_rows_and_both_flanking_tokens():
    processed = {
        "input_ids": torch.tensor(
            [[10, 151652, 151655, 151655, 151653, 20, 151652, 151656, 151653, 30]],
            dtype=torch.int64,
        )
    }

    tags = build_token_tags(processed)

    torch.testing.assert_close(tags, torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.int64))
    assert 2 not in tags


def test_text_row_limit_reports_modality_counts_and_bf16_payload():
    hidden_states = torch.empty(32769, 5120, dtype=torch.bfloat16, device="meta")
    token_tags = torch.ones(32769, dtype=torch.int64)
    token_tags[:3] = 0

    with pytest.raises(
        ValueError,
        match=r"32769.*vision_rows=3.*text_rows=32766.*320\.0 MiB",
    ):
        validate_text_rows(hidden_states, token_tags)


def test_text_row_validation_rejects_audio_tags():
    hidden_states = torch.zeros(3, 5120, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_text_rows(hidden_states, torch.tensor([1, 2, 0], dtype=torch.int64))


def test_released_text_checkpoint_key_mapping_keeps_visual_tower_separate():
    assert normalize_h3_text_encoder_key("model.layers.49.mlp.down_proj.weight") == (
        "language_model.layers.49.mlp.down_proj.weight"
    )
    assert normalize_h3_text_encoder_key("model.embed_tokens.weight") == "language_model.embed_tokens.weight"
    assert normalize_h3_text_encoder_key("visual.patch_embed.proj.weight") == "visual.patch_embed.proj.weight"


def test_text_cache_metadata_distinguishes_requested_storage_dtype():
    if _text_cache_metadata is None:
        pytest.skip(f"local optional dependency mismatch: {_text_cache_import_error}")
    common = {
        "task": "t2va",
        "crop_start": 0,
        "processor_identity": "processor",
        "text_encoder_identity": "encoder",
        "presentation_identity": "presentation",
    }

    bf16 = _text_cache_metadata(**common, cache_dtype="bf16")
    float32 = _text_cache_metadata(**common, cache_dtype="float32")

    assert bf16["cache_dtype"] == "bf16"
    assert float32["cache_dtype"] == "float32"
    assert bf16["cache_format"] == "minimax-h3-text-v2"
    assert bf16 != float32
    assert set(bf16) == {
        "task",
        "crop_start_frame",
        "cache_format",
        "text_encoder_fingerprint",
        "processor_fingerprint",
        "presentation_fingerprint",
        "cache_dtype",
    }


class _FakeQwen3VLConfig:
    @classmethod
    def from_pretrained(cls, _path, subfolder=None):
        del subfolder
        return SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=5120,
                num_hidden_layers=64,
                use_cache=True,
            )
        )


class _FakeSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(256, 2, bias=False))


class _FakeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(256, 2, bias=False))


class _FakeLanguageLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeSelfAttention()
        self.mlp = _FakeMlp()
        self.input_layernorm = nn.LayerNorm(256)
        self.post_attention_layernorm = nn.LayerNorm(256)


class _FakeQwen3VLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.embed_tokens = nn.Embedding(8, 256)
        self.language_model.layers = nn.ModuleList([_FakeLanguageLayer() for _ in range(config.text_config.num_hidden_layers)])
        self.language_model.norm = nn.LayerNorm(256)
        self.visual = nn.Linear(256, 4, bias=False)


def _install_fake_transformers(monkeypatch):
    fake = types.ModuleType("transformers")
    fake.Qwen3VLConfig = _FakeQwen3VLConfig
    fake.Qwen3VLModel = _FakeQwen3VLModel
    monkeypatch.setitem(sys.modules, "transformers", fake)


def _text_convrot_payload() -> torch.Tensor:
    raw = json.dumps(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
        separators=(",", ":"),
    ).encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.uint8)


def _fake_text_state(*, quantized: bool):
    config = _FakeQwen3VLConfig.from_pretrained("fake")
    config.text_config.num_hidden_layers = 50
    model = _FakeQwen3VLModel(config)
    del model.language_model.norm
    state = {key: tensor.detach().to(torch.bfloat16) for key, tensor in model.state_dict().items()}
    original_scale = torch.tensor([[1.000123], [0.999321]], dtype=torch.float32)

    if quantized:
        for index in range(50):
            modules = (
                *(f"language_model.layers.{index}.self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj")),
                *(f"language_model.layers.{index}.mlp.{name}" for name in ("gate_proj", "up_proj", "down_proj")),
            )
            for module_path in modules:
                del state[f"{module_path}.weight"]
                state[f"{module_path}.weight"] = torch.zeros(2, 256, dtype=torch.int8)
                state[f"{module_path}.weight_scale"] = original_scale.clone()
                state[f"{module_path}.comfy_quant"] = _text_convrot_payload()

    external = {}
    for key, tensor in state.items():
        external_key = "model." + key.removeprefix("language_model.") if key.startswith("language_model.") else key
        external[external_key] = tensor
    return external, original_scale


def test_load_h3_text_encoder_auto_detects_all_350_convrot_linears(tmp_path, monkeypatch):
    _install_fake_transformers(monkeypatch)
    state, original_scale = _fake_text_state(quantized=True)
    checkpoint = tmp_path / "qwen3vl-int8-convrot.safetensors"
    save_file(state, checkpoint)

    loaded = load_h3_text_encoder(
        checkpoint,
        device="cpu",
        dtype=torch.bfloat16,
    )

    q_proj = loaded.language_model.layers[0].self_attn.q_proj
    assert loaded.is_convrot_int8
    assert loaded.convrot_int8_layer_count == 350
    assert q_proj.weight.dtype is torch.int8
    assert q_proj.scale_weight.dtype is torch.float32
    assert torch.equal(q_proj.scale_weight, original_scale)
    assert loaded.language_model.embed_tokens.weight.dtype is torch.bfloat16
    assert loaded.visual.weight.dtype is torch.bfloat16


def test_load_h3_text_encoder_keeps_existing_bf16_conversion_for_ordinary_files(tmp_path, monkeypatch):
    _install_fake_transformers(monkeypatch)
    state, _scale = _fake_text_state(quantized=False)
    checkpoint = tmp_path / "qwen3vl-bf16.safetensors"
    save_file(state, checkpoint)

    loaded = load_h3_text_encoder(
        checkpoint,
        device="cpu",
        dtype=torch.bfloat16,
    )

    assert not getattr(loaded, "is_convrot_int8", False)
    assert loaded.language_model.layers[0].self_attn.q_proj.weight.dtype is torch.bfloat16
    assert loaded.visual.weight.dtype is torch.bfloat16


def test_load_h3_text_encoder_rejects_non_fp32_convrot_scale(tmp_path, monkeypatch):
    _install_fake_transformers(monkeypatch)
    state, _scale = _fake_text_state(quantized=True)
    state["model.layers.0.self_attn.q_proj.weight_scale"] = torch.ones(2, 1, dtype=torch.float16)
    checkpoint = tmp_path / "qwen3vl-bad-scale.safetensors"
    save_file(state, checkpoint)

    with pytest.raises(ValueError, match=r"scale.*FP32|scale.*F32"):
        load_h3_text_encoder(
            checkpoint,
            device="cpu",
            dtype=torch.bfloat16,
        )


def test_load_h3_text_encoder_accepts_nonpublished_convrot_layers_permissively(tmp_path, monkeypatch):
    # the pre-quantized file dictates which layers are INT8; a quantized layer outside the
    # published 350-layer scope loads and patches as long as the module exists
    _install_fake_transformers(monkeypatch)
    state, _scale = _fake_text_state(quantized=True)
    del state["visual.weight"]
    state["visual.weight"] = torch.zeros(4, 256, dtype=torch.int8)
    state["visual.weight_scale"] = torch.ones(4, 1, dtype=torch.float32)
    state["visual.comfy_quant"] = _text_convrot_payload()
    checkpoint = tmp_path / "qwen3vl-extra-convrot.safetensors"
    save_file(state, checkpoint)

    loaded = load_h3_text_encoder(
        checkpoint,
        device="cpu",
        dtype=torch.bfloat16,
    )

    assert loaded.visual.weight.dtype is torch.int8
    assert loaded.convrot_int8_layer_count == 351


def test_load_h3_text_encoder_installs_identity_final_norm_for_layer_50_convention(tmp_path, monkeypatch):
    # the hidden-state convention: 50 decoder layers + Identity final norm, so the model's
    # last_hidden_state is exactly the layer-50 pre-norm state (no capture hook involved)
    _install_fake_transformers(monkeypatch)
    state, _scale = _fake_text_state(quantized=False)
    checkpoint = tmp_path / "qwen3vl-bf16.safetensors"
    save_file(state, checkpoint)

    loaded = load_h3_text_encoder(
        checkpoint,
        device="cpu",
        dtype=torch.bfloat16,
    )

    assert isinstance(loaded.language_model.norm, nn.Identity)
    assert len(loaded.language_model.layers) == 50


def test_load_h3_text_encoder_rejects_streaming_without_cuda(tmp_path, monkeypatch):
    _install_fake_transformers(monkeypatch)
    state, _scale = _fake_text_state(quantized=False)
    checkpoint = tmp_path / "qwen3vl-bf16.safetensors"
    save_file(state, checkpoint)

    with pytest.raises(ValueError, match="CUDA"):
        load_h3_text_encoder(
            checkpoint,
            device="cpu",
            dtype=torch.bfloat16,
            blocks_to_swap=50,
        )


def test_load_h3_text_encoder_rejects_convrot_layer_missing_from_the_model(tmp_path, monkeypatch):
    _install_fake_transformers(monkeypatch)
    state, _scale = _fake_text_state(quantized=True)
    state["missing_tower.weight"] = torch.zeros(4, 256, dtype=torch.int8)
    state["missing_tower.weight_scale"] = torch.ones(4, 1, dtype=torch.float32)
    state["missing_tower.comfy_quant"] = _text_convrot_payload()
    checkpoint = tmp_path / "qwen3vl-missing-convrot.safetensors"
    save_file(state, checkpoint)

    with pytest.raises(ValueError, match=r"missing module missing_tower"):
        load_h3_text_encoder(
            checkpoint,
            device="cpu",
            dtype=torch.bfloat16,
        )
