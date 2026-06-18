import argparse
import importlib
import os
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
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
from musubi_tuner import ideogram4_cache_text_encoder_outputs
from musubi_tuner.ideogram4 import ideogram4_utils
from musubi_tuner.ideogram4.ideogram4_scheduler import SamplerParameters
from musubi_tuner.networks import lora_ideogram4


class Ideogram4LoraDiscoveryTests(unittest.TestCase):
    def test_lora_discovers_block_linears_and_changes_output(self):
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = nn.Linear(4, 4, bias=False)
                self.o = nn.Linear(4, 4, bias=False)

        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Linear(4, 4, bias=False)
                self.w2 = nn.Linear(4, 4, bias=False)
                self.w3 = nn.Linear(4, 4, bias=False)

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

    def test_apply_lora_weights_changes_conditional_output(self):
        import copy

        from safetensors.torch import save_file

        from musubi_tuner import ideogram4_generate_image

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = nn.Linear(4, 4, bias=False)
                self.o = nn.Linear(4, 4, bias=False)

        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Linear(4, 4, bias=False)
                self.w2 = nn.Linear(4, 4, bias=False)
                self.w3 = nn.Linear(4, 4, bias=False)

        class Ideogram4TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = Attention()
                self.feed_forward = FeedForward()

        class TinyRoot(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Ideogram4TransformerBlock()])

        torch.manual_seed(0)
        root_ref = TinyRoot()
        root_test = copy.deepcopy(root_ref)  # identical frozen base, no hooks

        # Build a "trained" LoRA on the reference root and give it a nonzero effect.
        network = lora_ideogram4.create_arch_network(1.0, 2, 1.0, None, [], root_ref)
        network.apply_to(None, root_ref, apply_text_encoder=False, apply_unet=True)
        for lora in network.unet_loras:
            nn.init.normal_(lora.lora_up.weight, std=0.5)
        weights_sd = {k: v.contiguous() for k, v in network.state_dict().items()}

        x = torch.ones(1, 4)
        y_base = root_test.layers[0].attention.qkv(x).clone()

        with tempfile.TemporaryDirectory() as tmp:
            lora_path = os.path.join(tmp, "lora.safetensors")
            save_file(weights_sd, lora_path)
            args = SimpleNamespace(
                lora_weight=[lora_path],
                lora_multiplier=[1.0],
                include_patterns=None,
                exclude_patterns=None,
            )
            ideogram4_generate_image._apply_lora_weights(root_test, args, torch.device("cpu"))

        y_lora = root_test.layers[0].attention.qkv(x)
        self.assertFalse(torch.allclose(y_base, y_lora))


class Ideogram4InputAndCacheTests(unittest.TestCase):
    def test_bucket_selector_accepts_ideogram4_architecture(self):
        selector = BucketSelector((1024, 1024), enable_bucket=True, architecture=ARCHITECTURE_IDEOGRAM4)

        self.assertEqual(selector.reso_steps, 16)
        self.assertEqual(selector.get_bucket_resolution((1024, 1024)), (1024, 1024))

    def test_qwen3_vl_metadata_loads_from_public_qwen_repo(self):
        calls = {}
        original_tokenizer = ideogram4_utils.AutoTokenizer
        original_config = ideogram4_utils.AutoConfig
        original_model = ideogram4_utils.AutoModel
        original_validate = ideogram4_utils.validate_local_safetensors
        original_load_state_dict = ideogram4_utils._load_state_dict
        original_init_empty_weights = ideogram4_utils.init_empty_weights

        class FakeTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls["tokenizer"] = (args, kwargs)
                return object()

        class FakeConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls["config"] = (args, kwargs)
                return object()

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()

            def load_state_dict(self, state_dict, strict=False, assign=True):
                calls["load_state_dict"] = (state_dict, strict, assign)
                return [], []

            def to(self, *, device, dtype):
                calls["to"] = (device, dtype)

            def eval(self):
                calls["eval"] = True

        class FakeAutoModel:
            @staticmethod
            def from_config(config, trust_remote_code=True):
                calls["model"] = (config, trust_remote_code)
                return FakeModel()

        class NoOpContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        try:
            ideogram4_utils.AutoTokenizer = FakeTokenizer
            ideogram4_utils.AutoConfig = FakeConfig
            ideogram4_utils.AutoModel = FakeAutoModel
            ideogram4_utils.validate_local_safetensors = lambda path: {}
            ideogram4_utils._load_state_dict = lambda path, device="cpu", disable_mmap=False: {}
            ideogram4_utils.init_empty_weights = lambda: NoOpContext()

            ideogram4_utils.load_ideogram4_tokenizer()
            ideogram4_utils.load_ideogram4_text_encoder(
                "qwen3vl_8b_bf16.safetensors",
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        finally:
            ideogram4_utils.AutoTokenizer = original_tokenizer
            ideogram4_utils.AutoConfig = original_config
            ideogram4_utils.AutoModel = original_model
            ideogram4_utils.validate_local_safetensors = original_validate
            ideogram4_utils._load_state_dict = original_load_state_dict
            ideogram4_utils.init_empty_weights = original_init_empty_weights

        self.assertEqual(calls["tokenizer"][0], ("Qwen/Qwen3-VL-8B-Instruct",))
        self.assertNotIn("subfolder", calls["tokenizer"][1])
        self.assertTrue(calls["tokenizer"][1]["trust_remote_code"])
        self.assertEqual(calls["config"][0], ("Qwen/Qwen3-VL-8B-Instruct",))
        self.assertNotIn("subfolder", calls["config"][1])
        self.assertTrue(calls["config"][1]["trust_remote_code"])

    def test_text_encoder_state_dict_remaps_qwen3_vl_model_prefixes(self):
        state = {
            "lm_head.weight": torch.full((1,), -1.0),
            "model.lm_head.weight": torch.full((1,), -1.0),
            "model.embed_tokens.weight": torch.ones(2, 2),
            "model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 2.0),
            "model.language_model.norm.bias": torch.full((2,), 5.0),
            "model.visual.patch_embed.proj.weight": torch.full((2, 2), 3.0),
            "language_model.norm.weight": torch.full((2,), 4.0),
        }

        normalized = ideogram4_utils._normalize_qwen3_vl_state_dict_for_automodel(state)

        self.assertNotIn("lm_head.weight", normalized)
        self.assertNotIn("model.lm_head.weight", normalized)
        self.assertTrue(torch.equal(normalized["language_model.embed_tokens.weight"], state["model.embed_tokens.weight"]))
        self.assertTrue(
            torch.equal(
                normalized["language_model.layers.0.self_attn.q_proj.weight"],
                state["model.layers.0.self_attn.q_proj.weight"],
            )
        )
        self.assertTrue(torch.equal(normalized["language_model.norm.bias"], state["model.language_model.norm.bias"]))
        self.assertTrue(torch.equal(normalized["visual.patch_embed.proj.weight"], state["model.visual.patch_embed.proj.weight"]))
        self.assertTrue(torch.equal(normalized["language_model.norm.weight"], state["language_model.norm.weight"]))

    def test_text_encoder_loader_materializes_missing_meta_tensors(self):
        original_config = ideogram4_utils.AutoConfig
        original_model = ideogram4_utils.AutoModel
        original_validate = ideogram4_utils.validate_local_safetensors
        original_load_state_dict = ideogram4_utils._load_state_dict
        original_init_empty_weights = ideogram4_utils.init_empty_weights

        class FakeConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return object()

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = nn.Module()
                self.language_model.loaded = nn.Linear(2, 2, device="meta")

            def eval(self):
                self.eval_called = True
                return super().eval()

        class FakeAutoModel:
            @staticmethod
            def from_config(config, trust_remote_code=True):
                return FakeModel()

        class NoOpContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        try:
            ideogram4_utils.AutoConfig = FakeConfig
            ideogram4_utils.AutoModel = FakeAutoModel
            ideogram4_utils.validate_local_safetensors = lambda path: {}
            ideogram4_utils._load_state_dict = lambda path, device="cpu", disable_mmap=False: {
                "model.loaded.weight": torch.ones(2, 2),
                "model.loaded.bias": torch.ones(2),
            }
            ideogram4_utils.init_empty_weights = lambda: NoOpContext()

            model = ideogram4_utils.load_ideogram4_text_encoder(
                "qwen3vl_8b_partial.safetensors",
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        finally:
            ideogram4_utils.AutoConfig = original_config
            ideogram4_utils.AutoModel = original_model
            ideogram4_utils.validate_local_safetensors = original_validate
            ideogram4_utils._load_state_dict = original_load_state_dict
            ideogram4_utils.init_empty_weights = original_init_empty_weights

        self.assertFalse(any(param.is_meta for param in model.parameters()))
        self.assertTrue(model.eval_called)
        self.assertTrue(torch.equal(model.language_model.loaded.weight, torch.ones(2, 2)))

    def test_build_inputs_and_patchify_roundtrip(self):
        features = [torch.ones(3, 8), torch.ones(5, 8)]
        inputs = ideogram4_utils.build_sequence_inputs_from_features(features, 512, 512, device=torch.device("cpu"))
        num_image = 32 * 32
        self.assertEqual(inputs["num_image_tokens"], num_image)
        self.assertEqual(inputs["max_text_tokens"], 5)
        self.assertEqual(tuple(inputs["position_ids"].shape), (2, 5 + num_image, 3))
        self.assertEqual(int((inputs["indicator"] == 3).sum().item()), 8)
        self.assertEqual(int((inputs["indicator"] == 2).sum().item()), 2 * num_image)

        # Layout is [image][text][right-padding]: image tokens lead, the text mask covers only the text region.
        self.assertEqual(tuple(inputs["attention_mask"].shape), (2, 5))
        self.assertTrue(torch.all(inputs["indicator"][:, :num_image] == 2))
        self.assertTrue(torch.equal(inputs["attention_mask"].sum(dim=1), torch.tensor([3, 5])))
        # The first valid text token sits right after the image block (no left padding).
        self.assertEqual(int(inputs["indicator"][0, num_image].item()), 3)

        latents = torch.arange(2 * 32 * 4 * 6, dtype=torch.float32).reshape(2, 32, 4, 6)
        token_grid = ideogram4_utils.patchify_vae_latents(latents)
        self.assertEqual(tuple(token_grid.shape), (2, 128, 2, 3))
        restored = ideogram4_utils.unpatchify_vae_latents(token_grid, 2, 3)
        self.assertTrue(torch.equal(latents, restored))

    def test_decode_tokens_keeps_patched_vae_channels(self):
        class FakeDecoder:
            def __init__(self):
                self.seen_shape = None

            def __call__(self, z):
                self.seen_shape = tuple(z.shape)
                return torch.zeros(z.shape[0], 3, z.shape[2] * 8, z.shape[3] * 8)

        class FakeAutoEncoder:
            dtype = torch.float32

            def __init__(self):
                self.decoder = FakeDecoder()

            def decode(self, z):
                raise AssertionError("official pipeline calls autoencoder.decoder directly")

        autoencoder = FakeAutoEncoder()
        tokens = torch.zeros(1, 4, 128)
        images = ideogram4_utils.decode_tokens_to_images(autoencoder, tokens, grid_h=2, grid_w=2)

        self.assertEqual(autoencoder.decoder.seen_shape, (1, 32, 4, 4))
        self.assertEqual(len(images), 1)

    def test_generate_images_initial_sigma_overrides_first_timestep(self):
        class FakeTransformer:
            def __init__(self):
                self.config = SimpleNamespace(in_channels=2)
                self.seen_t = []

            def __call__(self, *, llm_features, x, t, position_ids, attention_mask, indicator):
                del llm_features, position_ids, attention_mask, indicator
                self.seen_t.append(float(t[0].item()))
                return torch.zeros_like(x)

        original_decode = ideogram4_utils.decode_tokens_to_images
        original_preset = ideogram4_utils.PRESETS.get("TEST_INITIAL_SIGMA")
        ideogram4_utils.PRESETS["TEST_INITIAL_SIGMA"] = SamplerParameters(
            num_steps=1,
            guidance_schedule=(1.0,),
            mu=0.0,
            std=1.0,
        )
        ideogram4_utils.decode_tokens_to_images = lambda *args, **kwargs: ["image"]

        try:
            conditional = FakeTransformer()
            unconditional = FakeTransformer()
            images = ideogram4_utils.generate_images(
                conditional_transformer=conditional,
                unconditional_transformer=unconditional,
                autoencoder=object(),
                text_features=[torch.ones(1, 4)],
                height=16,
                width=16,
                sampler_preset="TEST_INITIAL_SIGMA",
                device=torch.device("cpu"),
                seed=1,
                show_progress=False,
                initial_sigma=1.004,
            )
        finally:
            ideogram4_utils.decode_tokens_to_images = original_decode
            if original_preset is None:
                ideogram4_utils.PRESETS.pop("TEST_INITIAL_SIGMA", None)
            else:
                ideogram4_utils.PRESETS["TEST_INITIAL_SIGMA"] = original_preset

        self.assertEqual(images, ["image"])
        self.assertAlmostEqual(conditional.seen_t[0], -0.004, places=6)
        self.assertAlmostEqual(unconditional.seen_t[0], -0.004, places=6)

    def test_generate_images_reuses_conditional_transformer_for_unconditional_embeds_cfg(self):
        class FakeTransformer:
            def __init__(self):
                self.config = SimpleNamespace(in_channels=2)
                self.calls = []

            def __call__(self, *, llm_features, x, t, position_ids, attention_mask, indicator):
                self.calls.append(
                    {
                        "llm_shape": tuple(llm_features.shape),
                        "x_shape": tuple(x.shape),
                        "t": float(t[0].item()),
                        "position_shape": tuple(position_ids.shape),
                        "attention_shape": tuple(attention_mask.shape),
                        "indicator_shape": tuple(indicator.shape),
                    }
                )
                return torch.full_like(x, float(len(self.calls)))

        original_decode = ideogram4_utils.decode_tokens_to_images
        original_preset = ideogram4_utils.PRESETS.get("TEST_SINGLE_DIT_CFG")
        ideogram4_utils.PRESETS["TEST_SINGLE_DIT_CFG"] = SamplerParameters(
            num_steps=1,
            guidance_schedule=(2.0,),
            mu=0.0,
            std=1.0,
        )
        ideogram4_utils.decode_tokens_to_images = lambda *args, **kwargs: ["image"]

        try:
            transformer = FakeTransformer()
            images = ideogram4_utils.generate_images(
                conditional_transformer=transformer,
                autoencoder=object(),
                text_features=[torch.ones(2, 4)],
                unconditional_text_features=[torch.zeros(1, 4)],
                height=16,
                width=16,
                sampler_preset="TEST_SINGLE_DIT_CFG",
                device=torch.device("cpu"),
                seed=1,
                show_progress=False,
            )
        finally:
            ideogram4_utils.decode_tokens_to_images = original_decode
            if original_preset is None:
                ideogram4_utils.PRESETS.pop("TEST_SINGLE_DIT_CFG", None)
            else:
                ideogram4_utils.PRESETS["TEST_SINGLE_DIT_CFG"] = original_preset

        self.assertEqual(images, ["image"])
        self.assertEqual(len(transformer.calls), 2)
        self.assertEqual(transformer.calls[0]["x_shape"], (1, 3, 2))
        self.assertEqual(transformer.calls[0]["llm_shape"], (1, 3, 4))
        self.assertEqual(transformer.calls[1]["x_shape"], (1, 2, 2))
        self.assertEqual(transformer.calls[1]["llm_shape"], (1, 2, 4))

    def test_text_cache_metadata_and_flow_target(self):
        with tempfile.TemporaryDirectory() as tmp:

            class DummyItem:
                item_key = "sample"
                caption = "caption"

            item = DummyItem()
            item.text_encoder_output_cache_path = os.path.join(tmp, "sample_i4_te.safetensors")
            features = torch.ones(4, 53248, dtype=torch.bfloat16)
            save_text_encoder_output_cache_ideogram4(item, features)
            with safe_open(item.text_encoder_output_cache_path, framework="pt") as f:
                metadata = f.metadata()
                self.assertEqual(metadata["architecture"], "ideogram4")
                self.assertIn("varlen_i4_llm_features_bfloat16", f.keys())

            features_fp32 = torch.ones(2, 53248, dtype=torch.float32)
            save_text_encoder_output_cache_ideogram4(item, features_fp32)
            with safe_open(item.text_encoder_output_cache_path, framework="pt") as f:
                keys = set(f.keys())
                self.assertIn("varlen_i4_llm_features_float32", keys)
                self.assertNotIn("varlen_i4_llm_features_bfloat16", keys)

        latents = torch.zeros(1, 2, 2, 2)
        noise = torch.ones_like(latents)
        target = latents - noise
        t = torch.full((1, 1, 1, 1), 0.25)
        noisy = (1 - t) * latents + t * noise
        model_t = 1.0 - t
        moved = noisy + target * (1.0 - model_t)
        self.assertTrue(torch.allclose(moved, latents))

    def test_text_cache_accepts_plain_caption_by_default(self):
        calls = {"validate": 0, "save": 0}
        original_validate = ideogram4_cache_text_encoder_outputs.ideogram4_utils.validate_prompt
        original_encode = ideogram4_cache_text_encoder_outputs.ideogram4_utils.encode_prompt_to_features
        original_save = ideogram4_cache_text_encoder_outputs.save_text_encoder_output_cache_ideogram4

        def fake_validate(prompt, *, warn_only):
            calls["validate"] += 1
            raise ValueError("plain captions are not structured JSON")

        try:
            ideogram4_cache_text_encoder_outputs.ideogram4_utils.validate_prompt = fake_validate
            ideogram4_cache_text_encoder_outputs.ideogram4_utils.encode_prompt_to_features = (
                lambda tokenizer, text_encoder, prompt, device: torch.ones(1, 4)
            )
            ideogram4_cache_text_encoder_outputs.save_text_encoder_output_cache_ideogram4 = (
                lambda item, features: calls.__setitem__("save", calls["save"] + 1)
            )

            item = SimpleNamespace(item_key="sample", caption="ordinary plain prompt")
            ideogram4_cache_text_encoder_outputs.encode_and_save_batch(
                object(),
                object(),
                [item],
                torch.device("cpu"),
                "float32",
                validate_caption_structure=False,
                warn_only=False,
            )
            self.assertEqual(calls["validate"], 0)
            self.assertEqual(calls["save"], 1)

            with self.assertRaisesRegex(ValueError, "plain captions"):
                ideogram4_cache_text_encoder_outputs.encode_and_save_batch(
                    object(),
                    object(),
                    [item],
                    torch.device("cpu"),
                    "float32",
                    validate_caption_structure=True,
                    warn_only=False,
                )
            self.assertEqual(calls["validate"], 1)
        finally:
            ideogram4_cache_text_encoder_outputs.ideogram4_utils.validate_prompt = original_validate
            ideogram4_cache_text_encoder_outputs.ideogram4_utils.encode_prompt_to_features = original_encode
            ideogram4_cache_text_encoder_outputs.save_text_encoder_output_cache_ideogram4 = original_save

    def test_train_parser_registers_dit_dtype(self):
        original_image_video = sys.modules.get("musubi_tuner.dataset.image_video_dataset")
        original_sampling = sys.modules.get("musubi_tuner.training.sampling_prompts")
        original_trainer = sys.modules.get("musubi_tuner.training.trainer_base")
        original_module = sys.modules.pop("musubi_tuner.ideogram4_train_network", None)

        fake_image_video = types.ModuleType("musubi_tuner.dataset.image_video_dataset")
        fake_image_video.ARCHITECTURE_IDEOGRAM4 = "i4"
        fake_image_video.ARCHITECTURE_IDEOGRAM4_FULL = "ideogram4"

        fake_sampling = types.ModuleType("musubi_tuner.training.sampling_prompts")
        fake_sampling.load_prompts = lambda path: []

        fake_trainer = types.ModuleType("musubi_tuner.training.trainer_base")

        class DiTOutput:
            def __init__(self, pred=None, target=None):
                self.pred = pred
                self.target = target

        class NetworkTrainer:
            def __init__(self):
                self.blocks_to_swap = 0

            def train(self, args):
                pass

        fake_trainer.DiTOutput = DiTOutput
        fake_trainer.NetworkTrainer = NetworkTrainer

        try:
            sys.modules["musubi_tuner.dataset.image_video_dataset"] = fake_image_video
            sys.modules["musubi_tuner.training.sampling_prompts"] = fake_sampling
            sys.modules["musubi_tuner.training.trainer_base"] = fake_trainer
            ideogram4_train_network = importlib.import_module("musubi_tuner.ideogram4_train_network")
            parser = ideogram4_train_network.ideogram4_setup_parser(argparse.ArgumentParser())
            args = parser.parse_args([])
        finally:
            if original_module is not None:
                sys.modules["musubi_tuner.ideogram4_train_network"] = original_module
            else:
                sys.modules.pop("musubi_tuner.ideogram4_train_network", None)
            if original_image_video is not None:
                sys.modules["musubi_tuner.dataset.image_video_dataset"] = original_image_video
            else:
                sys.modules.pop("musubi_tuner.dataset.image_video_dataset", None)
            if original_sampling is not None:
                sys.modules["musubi_tuner.training.sampling_prompts"] = original_sampling
            else:
                sys.modules.pop("musubi_tuner.training.sampling_prompts", None)
            if original_trainer is not None:
                sys.modules["musubi_tuner.training.trainer_base"] = original_trainer
            else:
                sys.modules.pop("musubi_tuner.training.trainer_base", None)

        self.assertTrue(hasattr(args, "dit_dtype"))
        self.assertIsNone(args.dit_dtype)
        self.assertAlmostEqual(args.initial_sigma, 1.004)
        self.assertFalse(args.validate_caption_structure)

    def test_generate_parser_allows_missing_unconditional_dit(self):
        from musubi_tuner import ideogram4_generate_image

        parser = ideogram4_generate_image.setup_parser()
        args = parser.parse_args(
            [
                "--dit",
                "conditional.safetensors",
                "--text_encoder",
                "qwen3vl.safetensors",
                "--vae",
                "vae.safetensors",
                "--prompt",
                "caption",
                "--save_path",
                "out.png",
            ]
        )

        self.assertIsNone(args.unconditional_dit)
        self.assertIsNone(args.lora_weight)
        self.assertEqual(args.attn_mode, "torch")
        self.assertFalse(args.split_attn)

    def test_generate_parser_accepts_attn_mode_and_split_attn(self):
        from musubi_tuner import ideogram4_generate_image

        parser = ideogram4_generate_image.setup_parser()
        base = [
            "--dit",
            "conditional.safetensors",
            "--text_encoder",
            "qwen3vl.safetensors",
            "--vae",
            "vae.safetensors",
            "--prompt",
            "caption",
            "--save_path",
            "out.png",
        ]
        args = parser.parse_args(base + ["--attn_mode", "flash", "--split_attn"])
        self.assertEqual(args.attn_mode, "flash")
        self.assertTrue(args.split_attn)
        # "sdpa" is accepted as a choice (normalized to "torch" inside main()).
        args_sdpa = parser.parse_args(base + ["--attn_mode", "sdpa"])
        self.assertEqual(args_sdpa.attn_mode, "sdpa")

    def test_generate_parser_accepts_lora_args(self):
        from musubi_tuner import ideogram4_generate_image

        parser = ideogram4_generate_image.setup_parser()
        args = parser.parse_args(
            [
                "--dit",
                "conditional.safetensors",
                "--text_encoder",
                "qwen3vl.safetensors",
                "--vae",
                "vae.safetensors",
                "--prompt",
                "caption",
                "--save_path",
                "out.png",
                "--lora_weight",
                "a.safetensors",
                "b.safetensors",
                "--lora_multiplier",
                "0.8",
                "1.0",
            ]
        )

        self.assertEqual(args.lora_weight, ["a.safetensors", "b.safetensors"])
        self.assertEqual(args.lora_multiplier, [0.8, 1.0])

    def test_process_batch_uses_common_timestep_sampler_with_normalized_model_latents(self):
        original_image_video = sys.modules.get("musubi_tuner.dataset.image_video_dataset")
        original_sampling = sys.modules.get("musubi_tuner.training.sampling_prompts")
        original_trainer = sys.modules.get("musubi_tuner.training.trainer_base")
        original_module = sys.modules.pop("musubi_tuner.ideogram4_train_network", None)

        fake_image_video = types.ModuleType("musubi_tuner.dataset.image_video_dataset")
        fake_image_video.ARCHITECTURE_IDEOGRAM4 = "i4"
        fake_image_video.ARCHITECTURE_IDEOGRAM4_FULL = "ideogram4"

        fake_sampling = types.ModuleType("musubi_tuner.training.sampling_prompts")
        fake_sampling.load_prompts = lambda path: []

        fake_trainer = types.ModuleType("musubi_tuner.training.trainer_base")

        class DiTOutput:
            def __init__(self, pred=None, target=None):
                self.pred = pred
                self.target = target

        class NetworkTrainer:
            def __init__(self):
                self.blocks_to_swap = 0

            def train(self, args):
                pass

        fake_trainer.DiTOutput = DiTOutput
        fake_trainer.NetworkTrainer = NetworkTrainer

        try:
            sys.modules["musubi_tuner.dataset.image_video_dataset"] = fake_image_video
            sys.modules["musubi_tuner.training.sampling_prompts"] = fake_sampling
            sys.modules["musubi_tuner.training.trainer_base"] = fake_trainer
            ideogram4_train_network = importlib.import_module("musubi_tuner.ideogram4_train_network")

            trainer = ideogram4_train_network.Ideogram4NetworkTrainer()
            captured = {}

            def fake_get_timesteps(args, noise, latents, timesteps, noise_scheduler, device, dtype):
                captured["sampler_args"] = args
                captured["sampler_timesteps"] = timesteps
                captured["sampler_noise_scheduler"] = noise_scheduler
                captured["sampler_device"] = device
                captured["sampler_dtype"] = dtype
                t = torch.full((latents.shape[0],), 0.25, device=device, dtype=dtype)
                noisy_model_input = (1.0 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise
                return noisy_model_input, t * 1000.0 + 1.0

            def fake_call(args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype):
                captured["latents"] = latents.detach().clone()
                captured["noise"] = noise.detach().clone()
                captured["noisy_model_input"] = noisy_model_input.detach().clone()
                captured["timesteps"] = timesteps.detach().clone()
                target = latents - noise
                return DiTOutput(pred=target, target=target)

            trainer.get_noisy_model_input_and_timesteps = fake_get_timesteps
            trainer.call_dit = fake_call

            latents = torch.linspace(-1.0, 1.0, 128 * 32 * 40, dtype=torch.float32).reshape(1, 128, 32, 40)
            args = SimpleNamespace(timestep_sampling="uniform", log_loss_stats=True)
            noise_scheduler = object()
            batch_timesteps = [0.25]
            loss, metrics = trainer.process_batch(
                args,
                SimpleNamespace(device=torch.device("cpu")),
                transformer=None,
                network=None,
                batch={"i4_llm_features": [torch.zeros(1, 8)], "timesteps": batch_timesteps},
                latents=latents,
                noise=torch.empty_like(latents),
                noise_scheduler=noise_scheduler,
                dit_dtype=torch.float32,
                network_dtype=torch.float32,
                vae=None,
                global_step=0,
            )
        finally:
            if original_module is not None:
                sys.modules["musubi_tuner.ideogram4_train_network"] = original_module
            else:
                sys.modules.pop("musubi_tuner.ideogram4_train_network", None)
            if original_image_video is not None:
                sys.modules["musubi_tuner.dataset.image_video_dataset"] = original_image_video
            else:
                sys.modules.pop("musubi_tuner.dataset.image_video_dataset", None)
            if original_sampling is not None:
                sys.modules["musubi_tuner.training.sampling_prompts"] = original_sampling
            else:
                sys.modules.pop("musubi_tuner.training.sampling_prompts", None)
            if original_trainer is not None:
                sys.modules["musubi_tuner.training.trainer_base"] = original_trainer
            else:
                sys.modules.pop("musubi_tuner.training.trainer_base", None)

        self.assertIs(captured["sampler_args"], args)
        self.assertIs(captured["sampler_timesteps"], batch_timesteps)
        self.assertIs(captured["sampler_noise_scheduler"], noise_scheduler)
        self.assertEqual(captured["sampler_device"], torch.device("cpu"))
        self.assertEqual(captured["sampler_dtype"], torch.float32)
        expected_latents = ideogram4_utils.normalize_token_grid(latents)
        self.assertTrue(torch.allclose(captured["latents"], expected_latents))
        expected_noisy = 0.75 * expected_latents + 0.25 * captured["noise"]
        self.assertTrue(torch.allclose(captured["noisy_model_input"], expected_noisy))
        self.assertTrue(torch.equal(captured["timesteps"], torch.full((1,), 251.0)))
        self.assertEqual(float(loss.item()), 0.0)
        self.assertGreater(metrics["loss/zero_pred"], 0.0)
        self.assertGreater(metrics["loss/flipped_pred"], 0.0)
        self.assertAlmostEqual(metrics["loss/pred_target_cosine"], 1.0, places=6)
        self.assertAlmostEqual(metrics["timestep/mean"], 251.0, places=6)

    def test_call_dit_uses_model_time_and_clean_minus_noise_target(self):
        original_image_video = sys.modules.get("musubi_tuner.dataset.image_video_dataset")
        original_sampling = sys.modules.get("musubi_tuner.training.sampling_prompts")
        original_trainer = sys.modules.get("musubi_tuner.training.trainer_base")
        original_module = sys.modules.pop("musubi_tuner.ideogram4_train_network", None)
        imported_module = None
        original_build_inputs = None

        fake_image_video = types.ModuleType("musubi_tuner.dataset.image_video_dataset")
        fake_image_video.ARCHITECTURE_IDEOGRAM4 = "i4"
        fake_image_video.ARCHITECTURE_IDEOGRAM4_FULL = "ideogram4"

        fake_sampling = types.ModuleType("musubi_tuner.training.sampling_prompts")
        fake_sampling.load_prompts = lambda path: []

        fake_trainer = types.ModuleType("musubi_tuner.training.trainer_base")

        class DiTOutput:
            def __init__(self, pred=None, target=None):
                self.pred = pred
                self.target = target

        class NetworkTrainer:
            def __init__(self):
                self.blocks_to_swap = 0

            def train(self, args):
                pass

        fake_trainer.DiTOutput = DiTOutput
        fake_trainer.NetworkTrainer = NetworkTrainer

        try:
            sys.modules["musubi_tuner.dataset.image_video_dataset"] = fake_image_video
            sys.modules["musubi_tuner.training.sampling_prompts"] = fake_sampling
            sys.modules["musubi_tuner.training.trainer_base"] = fake_trainer
            imported_module = importlib.import_module("musubi_tuner.ideogram4_train_network")
            original_build_inputs = imported_module.ideogram4_utils.build_sequence_inputs_from_features

            def fake_build_inputs(text_features, image_height, image_width, device):
                self.assertEqual(len(text_features), 1)
                self.assertEqual((image_height, image_width), (32, 32))
                return {
                    "max_text_tokens": 1,
                    "num_image_tokens": 4,
                    "llm_features": torch.zeros(1, 5, 8, device=device),
                    "position_ids": torch.zeros(1, 5, 3, dtype=torch.long, device=device),
                    "attention_mask": torch.zeros(1, 1, dtype=torch.long, device=device),
                    "indicator": torch.zeros(1, 5, dtype=torch.long, device=device),
                }

            imported_module.ideogram4_utils.build_sequence_inputs_from_features = fake_build_inputs

            class FakeModel:
                def __init__(self):
                    self.config = SimpleNamespace(in_channels=4)
                    self.seen_t = None
                    self.seen_x_shape = None

                def __call__(self, *, llm_features, x, t, position_ids, attention_mask, indicator):
                    del llm_features, position_ids, attention_mask, indicator
                    self.seen_t = t.detach().clone()
                    self.seen_x_shape = tuple(x.shape)
                    return torch.zeros_like(x)

            trainer = imported_module.Ideogram4NetworkTrainer()
            transformer = FakeModel()
            latents = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
            noise = torch.full_like(latents, 3.0)
            noisy_model_input = 0.75 * latents + 0.25 * noise

            output = trainer.call_dit(
                SimpleNamespace(gradient_checkpointing=False, timestep_sampling="uniform"),
                SimpleNamespace(device=torch.device("cpu"), autocast=lambda *a, **k: nullcontext()),
                transformer,
                latents,
                {"i4_llm_features": [torch.zeros(1, 8)]},
                noise,
                noisy_model_input,
                torch.tensor([251.0]),
                torch.float32,
            )
        finally:
            if imported_module is not None and original_build_inputs is not None:
                imported_module.ideogram4_utils.build_sequence_inputs_from_features = original_build_inputs
            if original_module is not None:
                sys.modules["musubi_tuner.ideogram4_train_network"] = original_module
            else:
                sys.modules.pop("musubi_tuner.ideogram4_train_network", None)
            if original_image_video is not None:
                sys.modules["musubi_tuner.dataset.image_video_dataset"] = original_image_video
            else:
                sys.modules.pop("musubi_tuner.dataset.image_video_dataset", None)
            if original_sampling is not None:
                sys.modules["musubi_tuner.training.sampling_prompts"] = original_sampling
            else:
                sys.modules.pop("musubi_tuner.training.sampling_prompts", None)
            if original_trainer is not None:
                sys.modules["musubi_tuner.training.trainer_base"] = original_trainer
            else:
                sys.modules.pop("musubi_tuner.training.trainer_base", None)

        self.assertTrue(torch.allclose(transformer.seen_t, torch.tensor([0.75])))
        self.assertEqual(transformer.seen_x_shape, (1, 5, 4))
        self.assertTrue(torch.equal(output.target, latents - noise))
        self.assertEqual(tuple(output.pred.shape), (1, 4, 2, 2))

    def test_denoising_steps_move_from_noise_to_data(self):
        params = ideogram4_utils.PRESETS["V4_DEFAULT_20"]
        schedule = ideogram4_utils.get_schedule_for_resolution((512, 512), known_mean=params.mu, std=params.std)
        intervals = ideogram4_utils.make_step_intervals(params.num_steps)
        steps = list(ideogram4_utils._iter_denoising_steps(params, schedule, intervals))

        self.assertEqual(len(steps), params.num_steps)
        official_first_index = params.num_steps - 1
        expected_t0 = float(schedule(intervals[official_first_index + 1].unsqueeze(0)).item())
        expected_s0 = float(schedule(intervals[official_first_index].unsqueeze(0)).item())
        expected_t_last = float(schedule(intervals[1].unsqueeze(0)).item())
        expected_s_last = float(schedule(intervals[0].unsqueeze(0)).item())
        self.assertAlmostEqual(steps[0][0], expected_t0)
        self.assertAlmostEqual(steps[0][1], expected_s0)
        self.assertLess(steps[0][0], steps[0][1])
        self.assertAlmostEqual(steps[0][2], 7.0)
        self.assertAlmostEqual(steps[-1][0], expected_t_last)
        self.assertAlmostEqual(steps[-1][1], expected_s_last)
        self.assertLess(steps[-1][0], steps[-1][1])
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
