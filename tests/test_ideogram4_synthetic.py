import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.dataset.architectures import ARCHITECTURE_IDEOGRAM4
from musubi_tuner.dataset.bucket import BucketSelector
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_ideogram4
from musubi_tuner.ideogram4 import ideogram4_utils
from musubi_tuner.ideogram4.ideogram4_quantized_loading import Fp8Linear, load_fp8_state_dict, swap_linears_to_fp8
from musubi_tuner.networks import lora_ideogram4


def _has_fp8():
    return hasattr(torch, "float8_e4m3fn")


@unittest.skipUnless(_has_fp8(), "torch.float8_e4m3fn is not available")
class Ideogram4Fp8Tests(unittest.TestCase):
    def test_weight_scale_swaps_and_loads_fp8_linear(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2, bias=False)

        model = Tiny()
        state = {
            "linear.weight": torch.ones(2, 3, dtype=torch.float8_e4m3fn),
            "linear.weight_scale": torch.full((2,), 0.5, dtype=torch.float32),
        }
        swap_linears_to_fp8(model, state, compute_dtype=torch.float32)
        self.assertIsInstance(model.linear, Fp8Linear)
        load_fp8_state_dict(model, state, device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(len(list(model.linear.parameters())), 0)
        out = model.linear(torch.ones(1, 3))
        self.assertEqual(tuple(out.shape), (1, 2))

    def test_scale_weight_does_not_trigger_official_fp8_swap(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2, bias=False)

        model = Tiny()
        state = {
            "linear.weight": torch.ones(2, 3, dtype=torch.float8_e4m3fn),
            "linear.scale_weight": torch.full((2,), 0.5, dtype=torch.float32),
        }
        swap_linears_to_fp8(model, state, compute_dtype=torch.float32)
        self.assertIsInstance(model.linear, nn.Linear)

    def test_lora_discovers_fp8_linear_and_changes_output(self):
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = Fp8Linear(4, 4, bias=False, compute_dtype=torch.float32)
                self.o = Fp8Linear(4, 4, bias=False, compute_dtype=torch.float32)
                self.qkv.weight.fill_(1.0)
                self.o.weight.fill_(1.0)
                self.qkv.weight_scale.fill_(0.01)
                self.o.weight_scale.fill_(0.01)

        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = Fp8Linear(4, 4, bias=False, compute_dtype=torch.float32)
                self.w2 = Fp8Linear(4, 4, bias=False, compute_dtype=torch.float32)
                self.w3 = Fp8Linear(4, 4, bias=False, compute_dtype=torch.float32)
                for module in (self.w1, self.w2, self.w3):
                    module.weight.fill_(1.0)
                    module.weight_scale.fill_(0.01)

        class Ideogram4TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = Attention()
                self.feed_forward = FeedForward()

        class TinyRoot(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Ideogram4TransformerBlock()])

        root = TinyRoot()
        network = lora_ideogram4.create_arch_network(1.0, 2, 1.0, None, [], root)
        self.assertGreater(len(network.unet_loras), 0)
        network.apply_to(None, root, apply_text_encoder=False, apply_unet=True)

        first_lora = network.unet_loras[0]
        first_lora.lora_down.weight.data.fill_(1.0)
        first_lora.lora_up.weight.data.fill_(1.0)

        x = torch.ones(1, 4)
        network.set_multiplier(0.0)
        y0 = root.layers[0].attention.qkv(x)
        network.set_multiplier(1.0)
        y1 = root.layers[0].attention.qkv(x)
        self.assertFalse(torch.allclose(y0, y1))


class Ideogram4InputAndCacheTests(unittest.TestCase):
    def test_bucket_selector_accepts_ideogram4_architecture(self):
        selector = BucketSelector((1024, 1024), enable_bucket=True, architecture=ARCHITECTURE_IDEOGRAM4)

        self.assertEqual(selector.reso_steps, 16)
        self.assertEqual(selector.get_bucket_resolution((1024, 1024)), (1024, 1024))

    def test_build_inputs_and_patchify_roundtrip(self):
        features = [torch.ones(3, 8), torch.ones(5, 8)]
        inputs = ideogram4_utils.build_sequence_inputs_from_features(features, 512, 512, device=torch.device("cpu"))
        self.assertEqual(inputs["num_image_tokens"], 32 * 32)
        self.assertEqual(inputs["max_text_tokens"], 5)
        self.assertEqual(tuple(inputs["position_ids"].shape), (2, 5 + 32 * 32, 3))
        self.assertEqual(int((inputs["indicator"] == 3).sum().item()), 8)
        self.assertEqual(int((inputs["indicator"] == 2).sum().item()), 2 * 32 * 32)

        latents = torch.arange(2 * 32 * 4 * 6, dtype=torch.float32).reshape(2, 32, 4, 6)
        token_grid = ideogram4_utils.patchify_vae_latents(latents)
        self.assertEqual(tuple(token_grid.shape), (2, 128, 2, 3))
        restored = ideogram4_utils.unpatchify_vae_latents(token_grid, 2, 3)
        self.assertTrue(torch.equal(latents, restored))

    def test_text_cache_metadata_and_flow_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            class DummyItem:
                item_key = "sample"
                caption = "caption"

            item = DummyItem()
            item.text_encoder_output_cache_path = os.path.join(tmp, "sample_i4_te.safetensors")
            features = torch.ones(4, 53248, dtype=torch.bfloat16)
            save_text_encoder_output_cache_ideogram4(item, features, "bf16")
            with safe_open(item.text_encoder_output_cache_path, framework="pt") as f:
                metadata = f.metadata()
                self.assertEqual(metadata["architecture"], "ideogram4")
                self.assertEqual(metadata["text_cache_dtype"], "bf16")
                self.assertEqual(metadata["num_text_tokens"], "4")
                self.assertIn("varlen_i4_llm_features_bfloat16", f.keys())

            features_fp32 = torch.ones(2, 53248, dtype=torch.float32)
            save_text_encoder_output_cache_ideogram4(item, features_fp32, "float32")
            with safe_open(item.text_encoder_output_cache_path, framework="pt") as f:
                metadata = f.metadata()
                keys = set(f.keys())
                self.assertEqual(metadata["text_cache_dtype"], "float32")
                self.assertEqual(metadata["num_text_tokens"], "2")
                self.assertIn("varlen_i4_llm_features_float32", keys)
                self.assertNotIn("varlen_i4_llm_features_bfloat16", keys)

        latents = torch.zeros(1, 2, 2, 2)
        noise = torch.ones_like(latents)
        target = noise - latents
        t = torch.full((1, 1, 1, 1), 0.25)
        noisy = (1 - t) * latents + t * noise
        moved = noisy + target * (0.0 - 0.25)
        self.assertTrue(torch.allclose(moved, latents))

    def test_denoising_steps_move_from_noise_to_data(self):
        params = ideogram4_utils.PRESETS["V4_DEFAULT_20"]
        schedule = ideogram4_utils.get_schedule_for_resolution((512, 512), known_mean=params.mu, std=params.std)
        intervals = ideogram4_utils.make_step_intervals(params.num_steps)
        steps = list(ideogram4_utils._iter_denoising_steps(params, schedule, intervals))

        self.assertEqual(len(steps), params.num_steps)
        self.assertGreater(steps[0][0], steps[0][1])
        self.assertAlmostEqual(steps[0][2], 7.0)
        self.assertGreater(steps[-1][0], steps[-1][1])
        self.assertAlmostEqual(steps[-1][2], 3.0)


class Ideogram4QwenMaskTests(unittest.TestCase):
    def test_qwen_causal_mask_helper_supports_current_signature(self):
        original = ideogram4_utils.create_causal_mask
        calls = {}

        def fake_create_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids=None):
            calls["config"] = config
            calls["input_embeds"] = input_embeds
            calls["attention_mask"] = attention_mask
            calls["cache_position"] = cache_position
            calls["past_key_values"] = past_key_values
            calls["position_ids"] = position_ids
            return torch.ones(1, 1, 2, 2)

        try:
            ideogram4_utils.create_causal_mask = fake_create_causal_mask
            language_model = SimpleNamespace(config=object())
            inputs_embeds = torch.randn(1, 2, 4)
            attention_mask = torch.ones(1, 2, dtype=torch.long)
            position_ids = torch.arange(2).unsqueeze(0)
            cache_position = torch.arange(2)
            mask = ideogram4_utils._create_qwen_causal_mask(
                language_model, inputs_embeds, attention_mask, position_ids, cache_position
            )
        finally:
            ideogram4_utils.create_causal_mask = original

        self.assertEqual(tuple(mask.shape), (1, 1, 2, 2))
        self.assertIs(calls["config"], language_model.config)
        self.assertIs(calls["input_embeds"], inputs_embeds)
        self.assertIs(calls["attention_mask"], attention_mask)
        self.assertIs(calls["cache_position"], cache_position)
        self.assertIsNone(calls["past_key_values"])
        self.assertIs(calls["position_ids"], position_ids)


if __name__ == "__main__":
    unittest.main()
