"""Qwen-Image 2.1 model, conditioning, and training integration tests."""

import copy
import json
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from musubi_tuner.networks import lora_qwen_image_21
from musubi_tuner.qwen_image_21.qwen_image_21_model import (
    QwenImage21Transformer2DModel,
    canonical_weight_hook,
    load_model,
    segmented_attention,
)
from musubi_tuner.qwen_image_21.qwen_image_21_utils import store_linears_in_fp8
from musubi_tuner.utils.safetensors_utils import TensorWeightAdapter, WeightTransformHooks

TINY_CONFIG = dict(
    in_channels=4,
    out_channels=4,
    num_layers=2,
    attention_head_dim=8,
    num_attention_heads=2,
    context_in_dim=12,
    axes_dims_rope=[2, 2, 4],
)


def tiny():
    return QwenImage21Transformer2DModel(**TINY_CONFIG)


def inputs():
    return dict(
        hidden_states=torch.randn(2, 4, 4),
        encoder_hidden_states=torch.randn(2, 5, 12),
        timestep=torch.tensor([0.2, 0.7]),
        img_shapes=[[(1, 2, 2)]],
        txt_seq_lens=[3, 5],
    )


def setUpModule():
    original_threads = torch.get_num_threads()
    unittest.addModuleCleanup(torch.set_num_threads, original_threads)
    torch.set_num_threads(2)


class QwenImage21ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_segmented_attention_matches_dense_outputs_and_gradients(self):
        runs = [[(0, 2, True), (2, 6, False), (6, 8, False), (8, 10, True), (10, 14, False)]]
        tensors = [torch.randn(1, 14, 2, 8, requires_grad=True) for _ in range(3)]
        q, k, v = tensors
        result = segmented_attention(q, k, v, runs)
        mask = torch.ones(14, 14, dtype=torch.bool).tril()
        for start, end, text in runs[0]:
            if not text:
                mask[start:end, start:end] = True
        expected = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask)
        expected = expected.transpose(1, 2).flatten(2)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
        grads = torch.autograd.grad(result.square().sum(), tensors, retain_graph=True)
        ref_grads = torch.autograd.grad(expected.square().sum(), tensors)
        for a, b in zip(grads, ref_grads):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)

    def test_batch_padding_matches_individual_samples(self):
        model, data = tiny(), inputs()
        out = model(**data)
        for b, length in enumerate(data["txt_seq_lens"]):
            row = {
                **data,
                "hidden_states": data["hidden_states"][b : b + 1],
                "encoder_hidden_states": data["encoder_hidden_states"][b : b + 1, :length],
                "timestep": data["timestep"][b : b + 1],
                "txt_seq_lens": [length],
            }
            torch.testing.assert_close(out[b : b + 1], model(**row), atol=1e-6, rtol=1e-5)
        data["encoder_hidden_states"][0, 3:] = 1000
        torch.testing.assert_close(out, model(**data))

    def test_reference_prefix_is_independent_of_target_and_timestep(self):
        model, data = tiny(), inputs()
        data.update(
            reference_latents=[torch.randn(2, 4, 4), torch.randn(2, 4, 4)],
            image_slots=[[1, 1], [2, 2]],
            img_shapes=[[(1, 2, 2)] * 3],
        )
        captured = []
        handle = model.transformer_blocks[-1].register_forward_hook(lambda m, a, o: captured.append(o.detach().clone()))
        out = model(**data)
        data["hidden_states"] = torch.randn_like(data["hidden_states"]) * 10
        data["timestep"] = torch.tensor([0.9, 0.1])
        model(**data)
        handle.remove()
        for b, length in enumerate([11, 13]):
            torch.testing.assert_close(captured[0][b, :length], captured[1][b, :length])
        self.assertTrue(torch.isfinite(out).all())

    def test_checkpoint_lora_gradients_and_reload(self):
        base = tiny().requires_grad_(False)
        initial = copy.deepcopy(base.state_dict())
        model = copy.deepcopy(base)
        network = lora_qwen_image_21.create_arch_network(1, 2, 2, None, [], model)
        self.assertEqual(len(network.unet_loras), 14)  # 7 projections per block
        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        network.requires_grad_(True)
        model.enable_gradient_checkpointing()
        data = inputs()
        before = model(**data).detach()
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.02)
        target = torch.randn_like(before)
        for _ in range(3):
            optimizer.zero_grad()
            F.mse_loss(model(**data), target).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters()))
            optimizer.step()
        self.assertFalse(torch.allclose(before, model(**data)))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, initial[key])
        with tempfile.TemporaryDirectory() as temp:
            filename = str(Path(temp) / "lora.safetensors")
            network.save_weights(filename, torch.float32, {})
            sd = load_file(filename)
            restored = lora_qwen_image_21.create_arch_network_from_weights(1, sd, unet=base)
            restored.apply_to(None, base, apply_text_encoder=False, apply_unet=True)
            restored.load_state_dict(sd, strict=True)
            torch.testing.assert_close(base(**data), model(**data))
        model.disable_gradient_checkpointing()
        torch.testing.assert_close(base(**data), model(**data))

    def test_loader_diffusers_shards_and_comfy_fused(self):
        model = tiny()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "config.json").write_text(json.dumps(TINY_CONFIG))
            sd = model.state_dict()
            keys = list(sd)
            save_file({k: sd[k] for k in keys[::2]}, str(path / "diffusion_pytorch_model-00001-of-00002.safetensors"))
            save_file({k: sd[k] for k in keys[1::2]}, str(path / "diffusion_pytorch_model-00002-of-00002.safetensors"))
            loaded = load_model(temp, dtype=torch.float32)
            data = inputs()
            torch.testing.assert_close(model(**data), loaded(**data))
            for i in range(2):
                prefix = f"transformer_blocks.{i}.img_mlp."
                sd[prefix + "gate_up.weight"] = torch.cat([sd.pop(prefix + "gate_layer.weight"), sd.pop(prefix + "proj.weight")])
            file = path / "comfy.safetensors"
            save_file({"model.diffusion_model." + k: v for k, v in sd.items()}, str(file))
            loaded = load_model(str(file), dtype=torch.float32)
            torch.testing.assert_close(model(**data), loaded(**data))

    def test_invalid_layout_rejected(self):
        model, data = tiny(), inputs()
        data.update(reference_latents=[torch.randn(2, 4, 4)], image_slots=[[1], [1]])
        with self.assertRaisesRegex(ValueError, "same count"):
            model(**data)


class QwenImage21IntegrationTests(unittest.TestCase):
    def test_single_file_qwen3vl_text_encoder_loading(self):
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        from musubi_tuner.qwen_image_21 import qwen_image_21_utils as utils

        config = Qwen3VLConfig(
            text_config=dict(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
            ),
            vision_config=dict(
                depth=1,
                hidden_size=16,
                intermediate_size=32,
                out_hidden_size=16,
                num_heads=2,
                patch_size=16,
                deepstack_visual_indexes=[],
                num_position_embeddings=16,
            ),
            image_token_id=3,
            video_token_id=4,
            vision_start_token_id=5,
            vision_end_token_id=6,
        )
        expected = Qwen3VLForConditionalGeneration(config).model.eval()
        comfy_state = {}
        for key, value in expected.state_dict().items():
            if key.startswith("visual."):
                key = "model." + key
            elif key.startswith("language_model."):
                key = "model." + key[len("language_model.") :]
            comfy_state[key] = value.detach().clone().contiguous()
        comfy_state["lm_head.weight"] = torch.zeros(1)

        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "qwen3vl_8b_bf16.safetensors"
            save_file(comfy_state, str(checkpoint))
            processor = object()
            with (
                patch.object(utils, "QWEN3_VL_8B_INSTRUCT_CONFIG", config.to_dict()),
                patch("transformers.Qwen3VLProcessor.from_pretrained", return_value=processor) as load_processor,
            ):
                actual_processor, actual = utils.load_text_encoder(str(checkpoint), dtype=torch.float32)

        self.assertIs(actual_processor, processor)
        load_processor.assert_called_once_with("Qwen/Qwen-Image-2.1", subfolder="processor")
        self.assertFalse(hasattr(actual, "model"))
        self.assertFalse(actual.training)
        self.assertFalse(any(parameter.requires_grad for parameter in actual.parameters()))
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], value)

    def test_vae_rgba_encode_decode_and_normalization(self):
        import numpy as np

        from musubi_tuner.qwen_image_21 import qwen_image_21_utils as utils
        from musubi_tuner.qwen_image_21.qwen_image_21_autoencoder_kl import AutoencoderKLQwenImage21

        vae = AutoencoderKLQwenImage21(base_dim=4, decoder_base_dim=4, num_res_blocks=1).eval()
        with torch.no_grad():
            image = np.full((32, 32, 3), 128, dtype=np.uint8)
            pixels = utils.image_tensor(image)
            self.assertTrue((pixels[:, 3] == 1).all())
            latent = utils.encode_image(vae, image)
            self.assertEqual(tuple(latent.shape), (1, 64, 1, 2, 2))
            mean, std = utils.latent_stats(vae, latent)
            raw = vae.encode(pixels).latent_dist.mode()
            torch.testing.assert_close(latent * std + mean, raw, atol=1e-6, rtol=1e-5)
            decoded = utils.decode_latents(vae, latent)
            self.assertEqual(tuple(decoded.shape), (1, 4, 32, 32))
            self.assertTrue(torch.isfinite(decoded).all())
            torch.testing.assert_close(utils.unpack_latents(utils.pack_latents(latent), 2, 2), latent)

            images = [image, np.full((64, 32, 4), 64, dtype=np.uint8), np.full_like(image, 192)]
            expected = [utils.encode_image(vae, value)[0] for value in images]
            with patch.object(vae, "encode", wraps=vae.encode) as encode:
                actual = utils.encode_images(vae, images)
            self.assertEqual([call.args[0].shape[0] for call in encode.call_args_list], [2, 1])
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)

            from musubi_tuner.dataset.image_video_dataset import ItemInfo
            from musubi_tuner.qwen_image_21_cache_latents import encode_and_save_batch

            with tempfile.TemporaryDirectory() as temp:
                items = []
                for index, target_index in enumerate([0, 2]):
                    item = ItemInfo(str(index), "edit", (32, 32), (32, 32))
                    item.content = images[target_index]
                    item.control_content = [images[target_index], images[1]]
                    item.latent_cache_path = str(Path(temp) / f"{index}.safetensors")
                    items.append(item)
                encode_and_save_batch(vae, items)
                for item, target_index in zip(items, [0, 2]):
                    cached = load_file(item.latent_cache_path)
                    torch.testing.assert_close(cached["latents_1x2x2_float32"], expected[target_index], atol=1e-5, rtol=1e-4)
                    torch.testing.assert_close(
                        cached["latents_control_0_1x2x2_float32"], expected[target_index], atol=1e-5, rtol=1e-4
                    )
                    torch.testing.assert_close(cached["latents_control_1_1x4x2_float32"], expected[1], atol=1e-5, rtol=1e-4)

    def test_cache_to_trainer_lora_step(self):
        from contextlib import nullcontext

        from musubi_tuner.qwen_image_21_train_network import QwenImage21NetworkTrainer, prepare_conditioning

        config = {**TINY_CONFIG, "in_channels": 64, "out_channels": 64, "context_in_dim": 4096}
        model = QwenImage21Transformer2DModel(**config).requires_grad_(False)
        net = lora_qwen_image_21.create_arch_network(1, 2, 2, None, [], model)
        net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        latents = torch.randn(2, 64, 1, 2, 2)
        batch = dict(
            vl_embed=[torch.randn(3, 4096), torch.randn(5, 4096)],
            latents_control_0=torch.randn_like(latents),
            image_slots=[torch.tensor([1]), torch.tensor([2])],
            reference_grids=[torch.tensor([[2, 2]])] * 2,
            reference_hashes=[torch.zeros(1, 32, dtype=torch.uint8)] * 2,
            vlm_reference_hashes=[torch.zeros(1, 32, dtype=torch.uint8)] * 2,
        )
        noise = torch.randn_like(latents)
        accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
        result = QwenImage21NetworkTrainer().call_dit(
            SimpleNamespace(gradient_checkpointing=True),
            accelerator,
            model,
            latents,
            batch,
            noise,
            0.5 * (latents + noise),
            torch.tensor([500.0, 500.0]),
            torch.float32,
        )
        self.assertEqual(result.pred.shape, latents.shape)
        torch.testing.assert_close(result.target, noise - latents)
        F.mse_loss(result.pred, result.target).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters()))
        batch["reference_grids"][0] = torch.tensor([[2, 4]])
        with self.assertRaisesRegex(ValueError, "disagree"):
            prepare_conditioning(batch, "cpu", torch.float32)

    def test_sampling_with_reference_and_cfg(self):
        from contextlib import nullcontext

        import numpy as np

        from musubi_tuner.qwen_image_21.qwen_image_21_autoencoder_kl import AutoencoderKLQwenImage21
        from musubi_tuner.qwen_image_21_train_network import QwenImage21NetworkTrainer

        vae = AutoencoderKLQwenImage21(base_dim=4, decoder_base_dim=4, num_res_blocks=1).eval()
        model = QwenImage21Transformer2DModel(**{**TINY_CONFIG, "in_channels": 64, "out_channels": 64, "context_in_dim": 4096})
        prompt = {
            "reference_images": [np.full((32, 32, 4), 128, dtype=np.uint8)],
            "positive": (torch.randn(3, 4096), torch.tensor([1]), torch.tensor([[2, 2]])),
            "negative": (torch.randn(4, 4096), torch.tensor([1]), torch.tensor([[2, 2]])),
        }
        accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
        pixels = QwenImage21NetworkTrainer().do_inference(
            accelerator,
            None,
            prompt,
            vae,
            torch.float32,
            model,
            3.0,
            2,
            32,
            32,
            1,
            torch.Generator().manual_seed(42),
            True,
            1.0,
            4.0,
            control_video_path=None,
        )
        self.assertEqual(pixels.shape, (1, 4, 1, 32, 32))
        self.assertTrue(torch.isfinite(pixels).all())
        self.assertTrue(((pixels >= 0) & (pixels <= 1)).all())
        from musubi_tuner.hv_generate_video import save_images_grid

        with tempfile.TemporaryDirectory() as temp:
            path = save_images_grid(pixels, temp, "sample", create_subdir=False)[0]
            with Image.open(path) as image:
                self.assertEqual(image.mode, "RGBA")
                np.testing.assert_array_equal(np.asarray(image.getchannel("A")), (pixels[0, 3, 0].numpy() * 255).astype(np.uint8))

    def test_real_qwen3vl_text_and_reference_encoding(self):
        from PIL import Image
        from tokenizers.pre_tokenizers import ByteLevel
        from transformers import (
            Qwen2TokenizerFast,
            Qwen2VLImageProcessor,
            Qwen3VLConfig,
            Qwen3VLForConditionalGeneration,
            Qwen3VLProcessor,
        )
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

        from musubi_tuner.qwen_image_21.qwen_image_21_utils import encode_prompt, encode_prompts

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            special = [
                "<|endoftext|>",
                "<|im_start|>",
                "<|im_end|>",
                "<|image_pad|>",
                "<|vision_start|>",
                "<|vision_end|>",
                "<|video_pad|>",
            ]
            vocab = {token: i for i, token in enumerate(sorted(ByteLevel.alphabet()) + special)}
            tokenizer = Qwen2TokenizerFast(vocab=vocab, merges=[], additional_special_tokens=special)
            processor = Qwen3VLProcessor(
                tokenizer=tokenizer,
                video_processor=Qwen3VLVideoProcessor(),
                image_processor=Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
            )
            config = Qwen3VLConfig(
                text_config=dict(
                    vocab_size=len(tokenizer),
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
                ),
                vision_config=dict(
                    depth=2,
                    hidden_size=32,
                    intermediate_size=64,
                    out_hidden_size=32,
                    num_heads=4,
                    patch_size=16,
                    deepstack_visual_indexes=[0],
                    num_position_embeddings=16,
                ),
                image_token_id=vocab["<|image_pad|>"],
                video_token_id=vocab["<|video_pad|>"],
                vision_start_token_id=vocab["<|vision_start|>"],
                vision_end_token_id=vocab["<|vision_end|>"],
            )
            encoder = Qwen3VLForConditionalGeneration(config).eval()
            image = Image.new("RGBA", (32, 64), (1, 2, 3, 128))
            for images, expected in [([], []), ([image], [[4, 2]]), ([image, Image.new("RGB", (64, 32))], [[4, 2], [2, 4]])]:
                captured = []
                handle = encoder.model.language_model.norm.register_forward_pre_hook(
                    lambda m, args: captured.append(args[0].detach().clone())
                )
                features, slots, grids = encode_prompt(processor, encoder, "hello", images)
                handle.remove()
                self.assertEqual(grids.tolist(), expected)
                self.assertEqual(len(slots), len(images))
                self.assertEqual(features.shape[-1], 32)
                self.assertTrue(torch.isfinite(features).all())
                # The final assistant suffix is retained and comes from before RMSNorm.
                torch.testing.assert_close(features[-5:], captured[0][0, -5:])
                self.assertEqual(len(encoder.model.language_model.norm._forward_hooks), 0)

            prompts = ["hello", "a longer caption", "edit"]
            references = [[], [image], [image, Image.new("RGB", (64, 32))]]
            expected = [encode_prompt(processor, encoder, prompt, refs) for prompt, refs in zip(prompts, references)]
            with patch.object(encoder.model, "forward", wraps=encoder.model.forward) as forward:
                actual = encode_prompts(processor, encoder, prompts, references)
            self.assertEqual(forward.call_count, 1)
            for left, right in zip(actual, expected):
                for a, b in zip(left, right):
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)

            from musubi_tuner.dataset.image_video_dataset import ItemInfo
            from musubi_tuner.qwen_image_21_cache_text_encoder_outputs import encode_and_save_batch

            items = []
            for index, (prompt, refs) in enumerate(zip(prompts, references)):
                item = ItemInfo(str(index), prompt, (32, 32), (32, 32))
                item.control_content = refs
                item.text_encoder_output_cache_path = str(path / f"{index}_te.safetensors")
                items.append(item)
            with patch.object(encoder.model, "forward", wraps=encoder.model.forward) as forward:
                encode_and_save_batch(processor, encoder, items)
            self.assertEqual(forward.call_count, 1)
            for item, (features, slots, grids) in zip(items, expected):
                cached = load_file(item.text_encoder_output_cache_path)
                torch.testing.assert_close(cached["varlen_vl_embed_float32"], features, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(cached["varlen_image_slots_int64"], slots)
                torch.testing.assert_close(cached["varlen_reference_grids_int64"], grids)

    def test_cache_cli_help_has_no_argument_conflicts(self):
        import io
        from contextlib import redirect_stdout

        from musubi_tuner import qwen_image_21_cache_latents, qwen_image_21_cache_text_encoder_outputs

        for module in (qwen_image_21_cache_latents, qwen_image_21_cache_text_encoder_outputs):
            with self.subTest(module=module.__name__), patch.object(sys, "argv", [module.__name__, "--help"]):
                with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                    module.main()
                self.assertEqual(result.exception.code, 0)

    def test_buckets_and_parser(self):
        from musubi_tuner.dataset.architectures import ARCHITECTURE_QWEN_IMAGE_21
        from musubi_tuner.dataset.bucket import BucketSelector
        from musubi_tuner.qwen_image_21_cache_text_encoder_outputs import setup_parser as setup_cache_parser
        from musubi_tuner.qwen_image_21_train_network import QwenImage21NetworkTrainer, setup_parser

        bucket = BucketSelector((1024, 1024), architecture=ARCHITECTURE_QWEN_IMAGE_21)
        self.assertTrue(all(w % 32 == h % 32 == 0 for w, h in bucket.bucket_resolutions))
        args = setup_parser().parse_args([])
        self.assertEqual(args.network_module, "networks.lora_qwen_image_21")
        args = setup_parser().parse_args(
            ["--sample_prompts", "prompts.txt", "--text_encoder", "encoder.safetensors", "--vae", "vae.safetensors"]
        )
        QwenImage21NetworkTrainer().handle_model_specific_args(args)
        cache_args = setup_cache_parser().parse_args(["--dataset_config", "dataset.toml", "--text_encoder", "encoder.safetensors"])
        self.assertEqual(cache_args.text_encoder, "encoder.safetensors")


class QwenImage21LoadingAndTrainingTests(unittest.TestCase):
    def test_standalone_generation_with_lora(self):
        import numpy as np

        from musubi_tuner import qwen_image_21_generate_image as generate
        from musubi_tuner.qwen_image_21.qwen_image_21_autoencoder_kl import AutoencoderKLQwenImage21

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            config = {**TINY_CONFIG, "in_channels": 64, "out_channels": 64, "context_in_dim": 4096}
            model = QwenImage21Transformer2DModel(**config)
            save_file(model.state_dict(), str(path / "dit.safetensors"))
            (path / "config.json").write_text(json.dumps(config))
            vae = AutoencoderKLQwenImage21(base_dim=4, decoder_base_dim=4, num_res_blocks=1).eval()
            vae.save_pretrained(path / "vae")
            network = lora_qwen_image_21.create_arch_network(1.0, 2, 2, None, None, model)
            network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
            with torch.no_grad():
                for module in network.unet_loras:
                    module.lora_up.weight.fill_(0.1)
            network.save_weights(str(path / "lora.safetensors"), torch.float32, {})
            control = path / "control.png"
            Image.new("RGBA", (32, 32), (255, 0, 0, 128)).save(control)
            args = generate.setup_parser().parse_args(
                [
                    "--dit",
                    str(path / "dit.safetensors"),
                    "--vae",
                    str(path / "vae"),
                    "--text_encoder",
                    "unused",
                    "--prompt",
                    "edit",
                    "--control_image_path",
                    str(control),
                    "--lora_weight",
                    str(path / "lora.safetensors"),
                    "--lora_multiplier",
                    "0.5",
                    "--save_path",
                    str(path / "output"),
                    "--image_size",
                    "32",
                    "32",
                    "--infer_steps",
                    "2",
                    "--seed",
                    "42",
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                ]
            )
            encoded = (torch.randn(3, 4096), torch.tensor([1]), torch.tensor([[2, 2]]))
            with (
                patch.object(generate.qwen_image_21_utils, "load_text_encoder", return_value=(Mock(), Mock())) as load_encoder,
                patch.object(generate.qwen_image_21_utils, "encode_prompt", return_value=encoded),
                patch.object(generate.qwen_image_21_sampling.BucketSelector, "calculate_bucket_resolution", return_value=(32, 32)),
                patch.object(generate, "load_dit_model", wraps=generate.load_dit_model) as load_dit,
            ):
                shared_models = {}
                output = generate.generate(args, shared_models)[0]
                with Image.open(output) as image:
                    self.assertEqual(image.mode, "RGBA")
                    self.assertEqual(image.size, (32, 32))
                    pixels = np.array(image)
                self.assertNotEqual(output, generate.generate(args, shared_models)[0])
                self.assertEqual(load_encoder.call_count, 1)
                self.assertEqual(load_dit.call_count, 1)
            self.assertTrue(((pixels[..., 3] > 0) & (pixels[..., 3] < 255)).any())

    def check_filtered_adapter_merge(self, lycoris):
        from musubi_tuner import qwen_image_21_generate_image as generate

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            model = tiny()
            original = {key: value.clone() for key, value in model.state_dict().items()}
            save_file(original, str(path / "dit.safetensors"))
            (path / "config.json").write_text(json.dumps(TINY_CONFIG))
            weights = {}
            for index in range(2):
                key = f"lora_unet_transformer_blocks_{index}_attn_to_q"
                weights[f"{key}.lora_down.weight"] = torch.ones(2, 16)
                weights[f"{key}.lora_up.weight"] = torch.ones(16, 2)
                weights[f"{key}.alpha"] = torch.tensor(2.0)
            save_file(weights, str(path / "lora.safetensors"))
            args = generate.setup_parser().parse_args(
                [
                    "--dit",
                    str(path / "dit.safetensors"),
                    "--vae",
                    "unused",
                    "--text_encoder",
                    "unused",
                    "--prompt",
                    "test",
                    "--save_path",
                    temp,
                    "--lora_weight",
                    str(path / "lora.safetensors"),
                    "--include_patterns",
                    "transformer_blocks",
                    "--exclude_patterns",
                    "transformer_blocks_1",
                ]
            )
            args.lycoris = lycoris
            merged = generate.load_dit_model(args, torch.device("cpu"), torch.float32, [0.5])
            for key, value in merged.state_dict().items():
                expected = original[key] + 1.0 if key == "transformer_blocks.0.attn.to_q.weight" else original[key]
                torch.testing.assert_close(value, expected)

    def test_inference_lora_filtering(self):
        self.check_filtered_adapter_merge(False)

    def test_file_and_interactive_prompts(self):
        from musubi_tuner import qwen_image_21_generate_image as generate

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prompts.txt"
            path.write_text("# comment\n\nedit --w 64 --ci first.png --ci second.png --l 2 --n blur\ntext --d 7\n")
            argv = [
                "generate",
                "--dit",
                "unused",
                "--vae",
                "unused",
                "--text_encoder",
                "unused",
                "--save_path",
                temp,
                "--image_size",
                "32",
                "32",
            ]
            with patch.object(sys, "argv", argv + ["--from_file", str(path)]), patch.object(generate, "generate") as run:
                generate.main()
            first, second = [call.args[0] for call in run.call_args_list]
            self.assertEqual(first.image_size, [32, 64])
            self.assertEqual(first.control_image_path, ["first.png", "second.png"])
            self.assertEqual((first.guidance_scale, first.negative_prompt), (2, "blur"))
            self.assertEqual(second.image_size, [32, 32])
            self.assertIsNone(second.control_image_path)
            self.assertEqual(second.seed, 7)
            with (
                patch.object(sys, "argv", argv + ["--interactive"]),
                patch("builtins.input", side_effect=["text --s 3 --fs 2", "q"]),
                patch.object(generate, "generate") as run,
            ):
                generate.main()
            self.assertEqual(run.call_count, 1)
            self.assertEqual((run.call_args.args[0].infer_steps, run.call_args.args[0].flow_shift), (3, 2))

    def test_inference_compiled_model(self):
        from musubi_tuner import qwen_image_21_generate_image as generate

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            model = tiny().eval()
            save_file(model.state_dict(), str(path / "dit.safetensors"))
            (path / "config.json").write_text(json.dumps(TINY_CONFIG))
            args = generate.setup_parser().parse_args(
                [
                    "--dit",
                    str(path / "dit.safetensors"),
                    "--vae",
                    "unused",
                    "--text_encoder",
                    "unused",
                    "--prompt",
                    "test",
                    "--save_path",
                    temp,
                    "--compile",
                    "--compile_backend",
                    "eager",
                ]
            )
            compiled = generate.load_dit_model(args, torch.device("cpu"), torch.float32, None)
            batch = inputs()
            with torch.no_grad():
                torch.testing.assert_close(compiled(**batch), model(**batch))

    @unittest.skipUnless(find_spec("lycoris"), "LyCORIS is not installed")
    def test_inference_lycoris_filtering(self):
        self.check_filtered_adapter_merge(True)

    def test_sampling_guidance_and_scheduler(self):
        from contextlib import nullcontext

        import numpy as np
        from diffusers import FlowMatchEulerDiscreteScheduler

        from musubi_tuner import qwen_image_21_train_network as train

        trainer = train.QwenImage21NetworkTrainer()
        trainer.handle_model_specific_args(SimpleNamespace(sample_prompts=None))
        self.assertIsNone(trainer.default_discrete_flow_shift)
        self.assertTrue(train.setup_parser().parse_args([]).split_attn)
        prompt = {
            "reference_images": [],
            "positive": (torch.full((1, 4096), 2.0), torch.empty(0), torch.empty(0)),
            "negative": (torch.ones(1, 4096), torch.empty(0), torch.empty(0)),
        }
        accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
        for shift, cfg_scale, has_negative, expected, calls in [
            (None, None, True, 2.0, 4),
            (None, 2.0, False, 2.0, 4),
            (None, 1.0, True, 2.0, 4),
            (3.0, 2.0, True, 3.0, 8),
        ]:
            with self.subTest(shift=shift, cfg_scale=cfg_scale, has_negative=has_negative):
                scheduler = train.qwen_image_21_sampling.qwen_image_utils.get_scheduler(shift)
                model = Mock(
                    side_effect=lambda **kwargs: torch.ones_like(kwargs["hidden_states"]) * kwargs["encoder_hidden_states"].mean()
                )
                with (
                    patch.object(train.qwen_image_21_sampling.qwen_image_utils, "get_scheduler", return_value=scheduler),
                    patch.object(scheduler, "step", wraps=scheduler.step) as step,
                    patch.object(train.qwen_image_21_utils, "decode_latents", return_value=torch.zeros(1, 4, 32, 32)),
                ):
                    trainer.do_inference(
                        accelerator,
                        None,
                        prompt,
                        Mock(),
                        torch.float32,
                        model,
                        shift,
                        4,
                        32,
                        32,
                        1,
                        torch.Generator().manual_seed(42),
                        has_negative,
                        1.0,
                        cfg_scale,
                    )
                self.assertEqual(model.call_count, calls)
                for call in step.call_args_list:
                    torch.testing.assert_close(call.args[0], torch.full_like(call.args[0], expected))
                reference = FlowMatchEulerDiscreteScheduler(
                    shift=1.0 if shift is None else shift,
                    use_dynamic_shifting=shift is None,
                    base_image_seq_len=256,
                    max_image_seq_len=8192,
                    base_shift=0.5,
                    max_shift=0.9,
                    shift_terminal=0.02,
                )
                mu = 0.5 + (4 - 256) * (0.9 - 0.5) / (8192 - 256)
                reference.set_timesteps(4, device="cpu", sigmas=np.linspace(1.0, 0.25, 4), mu=mu)
                torch.testing.assert_close(scheduler.timesteps, reference.timesteps)
                torch.testing.assert_close(scheduler.sigmas, reference.sigmas)

    def test_lora_diffusers_conversion(self):
        from musubi_tuner.convert_lora import convert_from_diffusers, convert_to_diffusers

        model = tiny()
        modules = [
            name
            for name, module in model.named_modules()
            if name.startswith("transformer_blocks.") and isinstance(module, torch.nn.Linear)
        ]
        modules.extend(["transformer_blocks.0.img_mlp.net.0.proj", "transformer_blocks.0.txt_mlp.net.2"])
        for module_name in modules:
            with self.subTest(module=module_name):
                name = "lora_unet_" + module_name.replace(".", "_")
                down, up = torch.randn(2, 4), torch.randn(4, 2)
                weights = {name + ".lora_down.weight": down, name + ".lora_up.weight": up, name + ".alpha": torch.tensor(1.0)}
                converted = convert_to_diffusers("lora_unet_", "transformer", weights)
                prefix = "transformer." + module_name
                self.assertEqual(set(converted), {prefix + ".lora_A.weight", prefix + ".lora_B.weight"})
                torch.testing.assert_close(
                    converted[prefix + ".lora_B.weight"] @ converted[prefix + ".lora_A.weight"], (up @ down) / 2
                )
                restored = convert_from_diffusers("lora_unet_", converted)
                self.assertEqual(set(restored), set(weights))
                torch.testing.assert_close(
                    restored[name + ".lora_up.weight"] @ restored[name + ".lora_down.weight"], (up @ down) / 2
                )

    def test_fused_weight_split_reads_once(self):
        source = Mock()
        source.keys.return_value = ["model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight"]
        value = torch.randn(12, 4)
        source.get_tensor.return_value = value
        adapter = TensorWeightAdapter(WeightTransformHooks(split_hook=canonical_weight_hook), source)
        gate = adapter.get_tensor("transformer_blocks.0.img_mlp.gate_layer.weight")
        up = adapter.get_tensor("transformer_blocks.0.img_mlp.proj.weight")
        torch.testing.assert_close(torch.cat([gate, up]), value)
        self.assertEqual(source.get_tensor.call_count, 1)
        self.assertEqual(adapter.tensor_cache, {})

    def test_fused_base_lora_merge_precedes_quantization(self):
        model = tiny()
        state = model.state_dict()
        key = "transformer_blocks.0.img_mlp.gate_layer.weight"
        prefix = "lora_unet_transformer_blocks_0_img_mlp_gate_layer"
        down, up = torch.randn(2, state[key].shape[1]), torch.randn(state[key].shape[0], 2)
        adapter = {prefix + ".lora_down.weight": down, prefix + ".lora_up.weight": up, prefix + ".alpha": torch.tensor(2.0)}
        merged = {k: v.clone() for k, v in state.items()}
        merged[key] += 0.3 * (up @ down)
        for i in range(2):
            part = f"transformer_blocks.{i}.img_mlp."
            state[part + "gate_up.weight"] = torch.cat([state.pop(part + "gate_layer.weight"), state.pop(part + "proj.weight")])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "config.json").write_text(json.dumps(TINY_CONFIG))
            save_file({"model.diffusion_model." + k: v for k, v in state.items()}, str(path / "fused.safetensors"))
            save_file(merged, str(path / "merged.safetensors"))
            cases = [(torch.float32, False), (torch.float8_e4m3fn, False)]
            if torch.cuda.is_available():
                cases.append((torch.bfloat16, True))
            for dtype, scaled in cases:
                with self.subTest(dtype=dtype, scaled=scaled):
                    common = dict(dtype=dtype, fp8_scaled=scaled, device="cuda" if scaled else "cpu")
                    actual = load_model(
                        str(path / "fused.safetensors"), lora_weights_list=[adapter], lora_multipliers=[0.3], **common
                    )
                    expected = load_model(str(path / "merged.safetensors"), **common)
                    for name, value in actual.state_dict().items():
                        torch.testing.assert_close(value.float(), expected.state_dict()[name].float())

    def test_fp8_storage_keeps_input_backward(self):
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4))
        store_linears_in_fp8(model)
        self.assertEqual(model[0].weight.dtype, torch.float8_e4m3fn)
        x = torch.randn(2, 8, requires_grad=True)
        model(x).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_qwen_shift_uses_unpacked_latent_resolution(self):
        from musubi_tuner.qwen_image_21_train_network import QwenImage21NetworkTrainer
        from musubi_tuner.training.trainer_base import NetworkTrainer

        args = SimpleNamespace(
            timestep_sampling="qwen_shift",
            min_timestep=None,
            max_timestep=None,
            preserve_distribution_shape=False,
            sigmoid_scale=1.0,
        )
        for height, width, mu in [(16, 16, 0.5), (64, 128, 0.9)]:
            with self.subTest(height=height, width=width):
                latents = torch.empty(1, 64, 1, height, width)
                actual = QwenImage21NetworkTrainer().sample_timesteps(args, 1, [0.5], latents, torch.device("cpu"))
                # A uniform draw of 0.5 maps to zero before the resolution-dependent shift.
                torch.testing.assert_close(actual, torch.tensor([mu]).sigmoid())

                # The shared trainer retains the original 2x2 latent packing.
                legacy_mu = 0.5 + ((height // 2) * (width // 2) - 256) * (0.9 - 0.5) / (8192 - 256)
                legacy = NetworkTrainer().sample_timesteps(args, 1, [0.5], latents, torch.device("cpu"))
                torch.testing.assert_close(legacy, torch.tensor([legacy_mu]).sigmoid())

    def test_sample_prompt_resize_uses_architecture_keyword(self):
        from musubi_tuner import qwen_image_21_train_network as train

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ref.png"
            Image.new("RGBA", (32, 64), "red").save(path)
            args = SimpleNamespace(text_encoder="unused", fp8_vl=False)
            prompts = [{"prompt": "edit", "control_image_path": [str(path)]} for _ in range(2)]
            encoded = (torch.randn(3, 4096), torch.tensor([1]), torch.tensor([[90, 44]]))
            with (
                patch.object(train, "load_prompts", return_value=prompts),
                patch.object(train.qwen_image_21_utils, "load_text_encoder", return_value=(Mock(), Mock())),
                patch.object(train.qwen_image_21_utils, "encode_prompt", return_value=encoded) as encode,
            ):
                result = train.QwenImage21NetworkTrainer().process_sample_prompts(
                    args, SimpleNamespace(device=torch.device("cpu")), "unused"
                )
            height, width = result[0]["reference_images"][0].shape[:2]
            self.assertEqual(width % 32, 0)
            self.assertEqual(height % 32, 0)
            self.assertGreater(height, width)
            self.assertEqual(encode.call_count, 1)

    def test_sample_prompt_cache_preserves_reference_order(self):
        from musubi_tuner.qwen_image_21 import qwen_image_21_sampling as sampling

        with tempfile.TemporaryDirectory() as temp:
            paths = [str(Path(temp) / name) for name in ["red.png", "blue.png"]]
            for path, color in zip(paths, ["red", "blue"]):
                Image.new("RGBA", (32, 32), color).save(path)
            prompts = [{"prompt": "edit", "control_image_path": order} for order in [paths, paths, paths[::-1]]]
            encoded = (torch.randn(3, 4096), torch.tensor([1, 2]), torch.tensor([[64, 64], [64, 64]]))
            with patch.object(sampling.qwen_image_21_utils, "encode_prompt", return_value=encoded) as encode:
                result = sampling.encode_sample_prompts(Mock(), Mock(), prompts)
            self.assertEqual(encode.call_count, 2)
            for call, expected_colors in zip(encode.call_args_list, [[(255, 0, 0), (0, 0, 255)], [(0, 0, 255), (255, 0, 0)]]):
                self.assertEqual([tuple(image[0, 0, :3]) for image in call.args[3]], expected_colors)
            for prompt in result:
                torch.testing.assert_close(prompt["positive"][0], encoded[0])

    def test_sage_grad_enabled_uses_differentiable_sdpa(self):
        from musubi_tuner.qwen_image_21 import qwen_image_21_model as model

        tensors = [torch.randn(1, 6, 2, 8, requires_grad=True) for _ in range(3)]
        segments = [[(0, 2, True), (2, 6, False)]]
        with patch.object(model.attention_backends, "sageattn", side_effect=AssertionError("inference kernel")):
            actual = segmented_attention(*tensors, segments, "sageattn")
            expected = segmented_attention(*tensors, segments)
            torch.testing.assert_close(actual, expected)
            actual.sum().backward()
        self.assertTrue(all(t.grad is not None for t in tensors))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for CPU offload regression")
    def test_checkpoint_and_input_offload_backward(self):
        torch.manual_seed(42)
        baseline = tiny().cuda()
        offloaded = copy.deepcopy(baseline)
        offloaded.enable_gradient_checkpointing(activation_cpu_offloading=True)
        offloaded.img_in_txt_in_offloading = True
        data = inputs()
        data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        expected = baseline(**data)
        actual = offloaded(**data)
        self.assertEqual(actual.device.type, "cuda")
        torch.testing.assert_close(actual, expected)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for (name, left), (_, right) in zip(baseline.named_parameters(), offloaded.named_parameters()):
            self.assertIsNotNone(right.grad, name)
            torch.testing.assert_close(left.grad.cpu(), right.grad.cpu(), atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
