import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_mage_flow
from musubi_tuner.dataset.image_video_dataset import ItemInfo
import musubi_tuner.mage_flow.text_encoder as text_encoder_module
from musubi_tuner.mage_flow.text_encoder import (
    EDIT_IMAGE_PLACEHOLDER,
    MAGE_FLOW_EDIT_PROMPT_TEMPLATE,
    MAGE_FLOW_PROMPT_TEMPLATE,
    MageFlowTextEncoder,
    encode_conditioning,
    normalize_qwen_state_dict,
)


def test_processor_loader_uses_pinned_official_mage_assets(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_from_pretrained(repo_id, **kwargs):
        captured.update(repo_id=repo_id, **kwargs)
        return sentinel

    monkeypatch.setattr(text_encoder_module.AutoProcessor, "from_pretrained", fake_from_pretrained)
    loader = getattr(text_encoder_module, "load_mage_flow_processor", None)

    assert loader is not None
    assert loader() is sentinel
    assert captured == {
        "repo_id": "microsoft/Mage-Flow",
        "revision": "faca09c18c1c19458e7fbc3f7bce6f7a7d4d01a9",
        "subfolder": "text_encoder",
    }
    assert "processor_source" not in inspect.signature(text_encoder_module.load_mage_flow_text_encoder).parameters


class FakeProcessor:
    def __init__(self, total_length):
        self.total_length = total_length
        self.calls = []
        self.tokenizer = SimpleNamespace(padding_side="left")

    def __call__(self, *, text, images=None, **kwargs):
        self.calls.append({"text": text, "images": images, **kwargs})
        return {
            "input_ids": torch.arange(self.total_length).unsqueeze(0),
            "attention_mask": torch.ones(1, self.total_length, dtype=torch.long),
        }


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, input_ids, **kwargs):
        self.calls.append({"input_ids": input_ids, **kwargs})
        length = input_ids.shape[1]
        hidden = torch.zeros(1, length, 2560)
        hidden[0, :, 0] = torch.arange(length)
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.mark.parametrize(
    ("is_edit", "drop", "template"),
    [
        (False, 34, MAGE_FLOW_PROMPT_TEMPLATE),
        (True, 64, MAGE_FLOW_EDIT_PROMPT_TEMPLATE),
    ],
)
def test_conditioning_uses_final_hidden_state_and_drops_exact_prefix(is_edit, drop, template):
    processor = FakeProcessor(drop + 3)
    encoder = MageFlowTextEncoder(FakeBackbone(), processor)
    references = [[np.zeros((32, 32, 3), dtype=np.uint8)]] if is_edit else None

    result = encode_conditioning(encoder, ["make it blue"], references=references, is_edit=is_edit)

    assert result[0].shape == (3, 2560)
    assert result[0][:, 0].tolist() == [drop, drop + 1, drop + 2]
    assert processor.calls[0]["text"][0].startswith(template.split("{}")[0])
    assert encoder.backbone.calls[0]["output_hidden_states"] is False
    assert "lm_head" not in dict(encoder.named_modules())


def test_edit_prompt_preserves_reference_order_and_caps_long_edge_at_384():
    processor = FakeProcessor(67)
    encoder = MageFlowTextEncoder(FakeBackbone(), processor)
    references = [
        [
            np.full((800, 400, 3), 1, dtype=np.uint8),
            np.full((200, 900, 3), 2, dtype=np.uint8),
            np.full((100, 100, 3), 3, dtype=np.uint8),
        ]
    ]

    encode_conditioning(encoder, ["combine them"], references=references, is_edit=True)

    call = processor.calls[0]
    assert call["text"][0].count(EDIT_IMAGE_PLACEHOLDER) == 3
    assert [image.getpixel((0, 0))[0] for image in call["images"]] == [1, 2, 3]
    assert [max(image.size) for image in call["images"]] == [384, 384, 100]


@pytest.mark.parametrize(
    ("is_edit", "references", "match"),
    [
        (False, [[np.zeros((8, 8, 3), dtype=np.uint8)]], "T2I"),
        (True, [[]], "between 1 and 3"),
        (True, [[np.zeros((8, 8, 3), dtype=np.uint8)] * 4], "between 1 and 3"),
    ],
)
def test_conditioning_rejects_mode_and_reference_contract_mismatches(is_edit, references, match):
    encoder = MageFlowTextEncoder(FakeBackbone(), FakeProcessor(70))
    with pytest.raises(ValueError, match=match):
        encode_conditioning(encoder, ["prompt"], references=references, is_edit=is_edit)


def test_conditioning_rejects_empty_or_overlong_effective_sequence():
    with pytest.raises(ValueError, match="between 1 and 2048"):
        encode_conditioning(
            MageFlowTextEncoder(FakeBackbone(), FakeProcessor(34)),
            ["empty"],
            is_edit=False,
        )
    with pytest.raises(ValueError, match="between 1 and 2048"):
        encode_conditioning(
            MageFlowTextEncoder(FakeBackbone(), FakeProcessor(34 + 2049)),
            ["long"],
            is_edit=False,
        )


def test_qwen_normalizer_ignores_lm_head_without_comfy_key_guessing():
    state = {
        "model.language_model.embed_tokens.weight": torch.zeros(2, 3),
        "model.visual.patch_embed.proj.weight": torch.zeros(1),
        "lm_head.weight": torch.ones(2, 3),
    }

    normalized = normalize_qwen_state_dict(state)

    assert set(normalized) == {"language_model.embed_tokens.weight", "visual.patch_embed.proj.weight"}
    with pytest.raises(ValueError, match="unknown"):
        normalize_qwen_state_dict({"model.embed_tokens.weight": torch.zeros(2, 3)})


def test_text_cache_serializer_writes_only_finite_bfloat16_final_hidden_state(tmp_path):
    item = ItemInfo(
        item_key="item",
        caption="caption",
        original_size=(32, 32),
        latent_cache_path=str(tmp_path / "latent.safetensors"),
    )
    item.text_encoder_output_cache_path = str(tmp_path / "text.safetensors")
    embedding = torch.zeros(3, 2560, dtype=torch.bfloat16)

    save_text_encoder_output_cache_mage_flow(item, embedding, is_edit=True)

    tensors = load_file(item.text_encoder_output_cache_path)
    assert list(tensors) == ["varlen_mage_flow_embed_bfloat16"]
    with safe_open(item.text_encoder_output_cache_path, framework="pt", device="cpu") as handle:
        assert handle.metadata()["architecture"] == "mage_flow_edit"
    embedding[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        save_text_encoder_output_cache_mage_flow(item, embedding, is_edit=True)
