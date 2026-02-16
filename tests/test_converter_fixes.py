"""Tests for convert_lora.py fixes: full-matrix LoKr alpha, hash recompute, and lokr_rank validation."""

import os
import tempfile
import types
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner.convert_lora import convert, convert_from_diffusers


class TestFullMatrixLoKrAlphaSynthesis(unittest.TestCase):
    """convert_from_diffusers must synthesize .alpha for full-matrix LoKr inputs."""

    PREFIX = "lora_unet_"

    def _make_diffusers_lokr_fullmatrix(self, module_name, w1_shape=(2, 4), w2_shape=(4, 4)):
        """Build a diffusers-format full-matrix LoKr state dict (lokr_w2, no lokr_w2_a/b)."""
        return {
            f"diffusion_model.{module_name}.lokr_w1": torch.randn(*w1_shape),
            f"diffusion_model.{module_name}.lokr_w2": torch.randn(*w2_shape),
        }

    def _make_diffusers_lokr_lowrank(self, module_name, w1_shape=(2, 4), w2_out=4, w2_in=4, rank=2):
        """Build a diffusers-format low-rank LoKr state dict (lokr_w2_a + lokr_w2_b)."""
        return {
            f"diffusion_model.{module_name}.lokr_w1": torch.randn(*w1_shape),
            f"diffusion_model.{module_name}.lokr_w2_a": torch.randn(w2_out, rank),
            f"diffusion_model.{module_name}.lokr_w2_b": torch.randn(rank, w2_in),
        }

    def test_fullmatrix_lokr_gets_alpha(self):
        """Full-matrix LoKr (lokr_w2) should get .alpha synthesized."""
        sd = self._make_diffusers_lokr_fullmatrix("blocks.0.cross_attn.k_img", w2_shape=(4, 4))
        result = convert_from_diffusers(self.PREFIX, sd)

        lora_name = "lora_unet_blocks_0_cross_attn_k_img"
        alpha_key = f"{lora_name}.alpha"
        self.assertIn(alpha_key, result, "alpha should be synthesized for full-matrix LoKr")
        self.assertEqual(result[alpha_key].item(), 4.0, "alpha should equal max(w2.shape)")

    def test_fullmatrix_lokr_alpha_equals_max_w2_shape(self):
        """Alpha should be max(w2.shape) for non-square full-matrix w2."""
        sd = self._make_diffusers_lokr_fullmatrix("blocks.0.cross_attn.k_img", w2_shape=(8, 4))
        result = convert_from_diffusers(self.PREFIX, sd)

        lora_name = "lora_unet_blocks_0_cross_attn_k_img"
        self.assertEqual(result[f"{lora_name}.alpha"].item(), 8.0)

    def test_lowrank_lokr_still_gets_alpha(self):
        """Low-rank LoKr (lokr_w2_a) should still get .alpha synthesized (regression check)."""
        sd = self._make_diffusers_lokr_lowrank("blocks.0.cross_attn.k_img", rank=2)
        result = convert_from_diffusers(self.PREFIX, sd)

        lora_name = "lora_unet_blocks_0_cross_attn_k_img"
        alpha_key = f"{lora_name}.alpha"
        self.assertIn(alpha_key, result)
        self.assertEqual(result[alpha_key].item(), 2.0, "alpha should equal rank (w2_a.shape[1])")

    def test_existing_alpha_not_overwritten(self):
        """If diffusers input already has .alpha, it should NOT be overwritten."""
        sd = self._make_diffusers_lokr_fullmatrix("blocks.0.cross_attn.k_img", w2_shape=(4, 4))
        sd["diffusion_model.blocks.0.cross_attn.k_img.alpha"] = torch.tensor(99.0)
        result = convert_from_diffusers(self.PREFIX, sd)

        lora_name = "lora_unet_blocks_0_cross_attn_k_img"
        self.assertEqual(result[f"{lora_name}.alpha"].item(), 99.0, "existing alpha should be preserved")


class TestConverterHashRecompute(unittest.TestCase):
    """convert() must recompute sshs_* hashes for the converted weights."""

    def _make_test_safetensors(self, path, sd=None, metadata=None):
        """Save a minimal safetensors file with optional metadata."""
        if sd is None:
            sd = {"lora_unet_test.lora_down.weight": torch.randn(4, 8), "lora_unet_test.lora_up.weight": torch.randn(8, 4)}
        if metadata is None:
            metadata = {}
        save_file(sd, path, metadata=metadata)

    def test_default_conversion_writes_fresh_hashes(self):
        """target=default should produce fresh sshs_* hashes in output metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            # Create input in diffusers format with stale hashes
            sd = {
                "diffusion_model.blocks.0.cross_attn.k_img.lora_A.weight": torch.randn(4, 8),
                "diffusion_model.blocks.0.cross_attn.k_img.lora_B.weight": torch.randn(8, 4),
            }
            stale_metadata = {"sshs_model_hash": "STALE_HASH", "sshs_legacy_hash": "STALE_LEG"}
            save_file(sd, input_path, metadata=stale_metadata)

            convert(input_path, output_path, "default", None)

            with safe_open(output_path, framework="pt") as f:
                out_metadata = dict(f.metadata() or {})

            self.assertIn("sshs_model_hash", out_metadata)
            self.assertIn("sshs_legacy_hash", out_metadata)
            self.assertNotEqual(out_metadata["sshs_model_hash"], "STALE_HASH", "hash should be recomputed")
            self.assertNotEqual(out_metadata["sshs_legacy_hash"], "STALE_LEG", "legacy hash should be recomputed")

    def test_other_conversion_writes_fresh_hashes(self):
        """target=other should also produce fresh sshs_* hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            # Create input in default format with stale hashes
            sd = {
                "lora_unet_blocks_0_cross_attn_k_img.lora_down.weight": torch.randn(4, 8),
                "lora_unet_blocks_0_cross_attn_k_img.lora_up.weight": torch.randn(8, 4),
                "lora_unet_blocks_0_cross_attn_k_img.alpha": torch.tensor(4.0),
            }
            stale_metadata = {"sshs_model_hash": "OLD_HASH", "sshs_legacy_hash": "OLD_LEG"}
            save_file(sd, input_path, metadata=stale_metadata)

            convert(input_path, output_path, "other", "diffusion_model")

            with safe_open(output_path, framework="pt") as f:
                out_metadata = dict(f.metadata() or {})

            self.assertIn("sshs_model_hash", out_metadata)
            self.assertIn("sshs_legacy_hash", out_metadata)
            self.assertNotEqual(out_metadata["sshs_model_hash"], "OLD_HASH")
            self.assertNotEqual(out_metadata["sshs_legacy_hash"], "OLD_LEG")

    def test_ss_lokr_factor_metadata_preserved(self):
        """ss_lokr_factor in safetensors metadata should survive conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            sd = {
                "diffusion_model.blocks.0.cross_attn.k_img.lokr_w1": torch.randn(2, 4),
                "diffusion_model.blocks.0.cross_attn.k_img.lokr_w2": torch.randn(4, 4),
            }
            metadata = {"ss_lokr_factor": "8", "ss_network_module": "networks.lokr"}
            save_file(sd, input_path, metadata=metadata)

            convert(input_path, output_path, "default", None)

            with safe_open(output_path, framework="pt") as f:
                out_metadata = dict(f.metadata() or {})

            self.assertEqual(out_metadata.get("ss_lokr_factor"), "8", "ss_lokr_factor should survive conversion")
            self.assertEqual(out_metadata.get("ss_network_module"), "networks.lokr")


class TestLoKrRankValidation(unittest.TestCase):
    """convert_z_image_lora_to_comfy should reject non-positive lokr_rank values."""

    def _make_args(self, lokr_rank, src_path, dst_path=None):
        """Build a minimal args namespace matching convert_z_image_lora_to_comfy.main()."""
        return types.SimpleNamespace(
            lokr_rank=lokr_rank,
            reverse=False,
            src_path=src_path,
            dst_path=dst_path or os.path.join(os.path.dirname(src_path), "out.safetensors"),
        )

    def _make_minimal_safetensors(self, path):
        """Create a minimal safetensors file that main() can load."""
        save_file({"dummy.lora_down.weight": torch.randn(4, 8)}, path)

    def test_zero_rank_raises(self):
        from musubi_tuner.networks.convert_z_image_lora_to_comfy import main

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "input.safetensors")
            self._make_minimal_safetensors(src)
            args = self._make_args(0, src)
            with self.assertRaises(ValueError, msg="lokr_rank=0 should raise ValueError"):
                main(args)

    def test_negative_rank_raises(self):
        from musubi_tuner.networks.convert_z_image_lora_to_comfy import main

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "input.safetensors")
            self._make_minimal_safetensors(src)
            args = self._make_args(-1, src)
            with self.assertRaises(ValueError, msg="lokr_rank=-1 should raise ValueError"):
                main(args)

    def test_positive_rank_does_not_raise(self):
        from musubi_tuner.networks.convert_z_image_lora_to_comfy import main

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "input.safetensors")
            dst = os.path.join(tmpdir, "output.safetensors")
            self._make_minimal_safetensors(src)
            args = self._make_args(8, src, dst)
            # Should not raise — no LoKr keys in input means no work, but validation passes
            main(args)


class TestMixedKeyConversion(unittest.TestCase):
    """convert_diffusers_if_needed must preserve non-Diffusers keys in mixed-key dicts."""

    def test_mixed_keys_preserves_lora_unet(self):
        """A dict with both diffusion_model.* and lora_unet_* keys should keep both after conversion."""
        from musubi_tuner.utils.lora_utils import convert_diffusers_if_needed

        sd = {
            # Diffusers-format key
            "diffusion_model.blocks.0.attn.q.lora_A.weight": torch.randn(4, 8),
            "diffusion_model.blocks.0.attn.q.lora_B.weight": torch.randn(8, 4),
            # Already-normalized key
            "lora_unet_blocks_1_attn_k.lora_down.weight": torch.randn(4, 8),
            "lora_unet_blocks_1_attn_k.lora_up.weight": torch.randn(8, 4),
            "lora_unet_blocks_1_attn_k.alpha": torch.tensor(4.0),
        }
        result = convert_diffusers_if_needed(sd)

        # The Diffusers keys should be converted
        self.assertIn("lora_unet_blocks_0_attn_q.lora_down.weight", result)
        self.assertIn("lora_unet_blocks_0_attn_q.lora_up.weight", result)
        # The already-normalized keys must survive
        self.assertIn("lora_unet_blocks_1_attn_k.lora_down.weight", result)
        self.assertIn("lora_unet_blocks_1_attn_k.lora_up.weight", result)
        self.assertIn("lora_unet_blocks_1_attn_k.alpha", result)
        # Original Diffusers keys should NOT be in the result
        self.assertNotIn("diffusion_model.blocks.0.attn.q.lora_A.weight", result)

    def test_pure_default_format_is_noop(self):
        """A dict with only lora_unet_* keys should pass through unchanged."""
        from musubi_tuner.utils.lora_utils import convert_diffusers_if_needed

        sd = {
            "lora_unet_blocks_0_attn_q.lora_down.weight": torch.randn(4, 8),
            "lora_unet_blocks_0_attn_q.lora_up.weight": torch.randn(8, 4),
            "lora_unet_blocks_0_attn_q.alpha": torch.tensor(4.0),
        }
        result = convert_diffusers_if_needed(sd)
        self.assertIs(result, sd, "no-op should return the same dict object")

    def test_metadata_keys_preserved(self):
        """Non-dotted metadata keys should survive mixed-key conversion."""
        from musubi_tuner.utils.lora_utils import convert_diffusers_if_needed

        sd = {
            "lokr_factor": torch.tensor(8),
            "diffusion_model.blocks.0.attn.q.lora_A.weight": torch.randn(4, 8),
            "diffusion_model.blocks.0.attn.q.lora_B.weight": torch.randn(8, 4),
        }
        result = convert_diffusers_if_needed(sd)
        self.assertIn("lokr_factor", result)
        self.assertEqual(result["lokr_factor"].item(), 8)

    def test_first_key_lora_unet_does_not_skip_diffusers(self):
        """Even if lora_unet_* keys appear first, Diffusers keys must still be converted."""
        from musubi_tuner.utils.lora_utils import convert_diffusers_if_needed

        sd = {
            # lora_unet_* key sorts first alphabetically
            "lora_unet_aaa_block.lora_down.weight": torch.randn(4, 8),
            "lora_unet_aaa_block.lora_up.weight": torch.randn(8, 4),
            # Diffusers key sorts second
            "transformer.blocks.0.attn.q.lora_A.weight": torch.randn(4, 8),
            "transformer.blocks.0.attn.q.lora_B.weight": torch.randn(8, 4),
        }
        result = convert_diffusers_if_needed(sd)

        # Both families must be present
        self.assertIn("lora_unet_aaa_block.lora_down.weight", result)
        self.assertIn("lora_unet_blocks_0_attn_q.lora_down.weight", result)
        # Original Diffusers keys should be gone
        self.assertNotIn("transformer.blocks.0.attn.q.lora_A.weight", result)


class TestLoraMergeDtypeStability(unittest.TestCase):
    """lora_merge_weights_to_tensor must preserve the base weight's dtype."""

    def test_bf16_base_fp16_lora_preserves_dtype(self):
        """bf16 base + fp16 LoRA weights must not promote to fp32."""
        from musubi_tuner.utils.lora_utils import lora_merge_weights_to_tensor

        base = torch.randn(8, 16, dtype=torch.bfloat16)
        lora_sd = {
            "lora_unet_test.lora_down.weight": torch.randn(4, 16, dtype=torch.float16),
            "lora_unet_test.lora_up.weight": torch.randn(8, 4, dtype=torch.float16),
            "lora_unet_test.alpha": torch.tensor(4.0),
        }
        keys = set(lora_sd.keys())
        result = lora_merge_weights_to_tensor(base, "lora_unet_test", lora_sd, keys, 1.0, torch.device("cpu"))
        self.assertEqual(result.dtype, torch.bfloat16, "output dtype must match base weight dtype")
        self.assertFalse(torch.equal(result, base), "merge must actually modify the weight")
        # Keys should be consumed
        self.assertNotIn("lora_unet_test.lora_down.weight", keys)

    def test_fp32_base_fp16_lora_preserves_dtype(self):
        """fp32 base + fp16 LoRA weights must stay fp32."""
        from musubi_tuner.utils.lora_utils import lora_merge_weights_to_tensor

        base = torch.randn(8, 16, dtype=torch.float32)
        lora_sd = {
            "lora_unet_test.lora_down.weight": torch.randn(4, 16, dtype=torch.float16),
            "lora_unet_test.lora_up.weight": torch.randn(8, 4, dtype=torch.float16),
        }
        keys = set(lora_sd.keys())
        result = lora_merge_weights_to_tensor(base, "lora_unet_test", lora_sd, keys, 1.0, torch.device("cpu"))
        self.assertEqual(result.dtype, torch.float32)
        self.assertFalse(torch.equal(result, base), "merge must actually modify the weight")

    def test_no_match_returns_unchanged(self):
        """When no LoRA keys match, return base weight unchanged."""
        from musubi_tuner.utils.lora_utils import lora_merge_weights_to_tensor

        base = torch.randn(8, 16, dtype=torch.bfloat16)
        lora_sd = {"lora_unet_other.lora_down.weight": torch.randn(4, 16)}
        keys = set(lora_sd.keys())
        result = lora_merge_weights_to_tensor(base, "lora_unet_test", lora_sd, keys, 1.0, torch.device("cpu"))
        self.assertTrue(torch.equal(result, base))


class TestWeightHookPrefixStripping(unittest.TestCase):
    """weight_hook_func must strip model-level prefixes for LoRA name matching.

    Tests call the shipped _make_lora_name_from_model_key() helper directly,
    so they break if the production code is changed or deleted.
    """

    def test_wan_prefix_stripped_for_lora_matching(self):
        """model.diffusion_model.* keys should match LoRA weights after prefix stripping."""
        from musubi_tuner.utils.lora_utils import _make_lora_name_from_model_key, lora_merge_weights_to_tensor

        lora_name = _make_lora_name_from_model_key("model.diffusion_model.blocks.0.attn.q.weight")
        self.assertEqual(lora_name, "lora_unet_blocks_0_attn_q")

        # Now verify the merge actually works with that name
        base = torch.randn(8, 16, dtype=torch.bfloat16)
        lora_sd = {
            "lora_unet_blocks_0_attn_q.lora_down.weight": torch.randn(4, 16, dtype=torch.bfloat16),
            "lora_unet_blocks_0_attn_q.lora_up.weight": torch.randn(8, 4, dtype=torch.bfloat16),
            "lora_unet_blocks_0_attn_q.alpha": torch.tensor(4.0),
        }
        keys = set(lora_sd.keys())
        result = lora_merge_weights_to_tensor(base, lora_name, lora_sd, keys, 1.0, torch.device("cpu"))
        # If the name matched, keys should be consumed
        self.assertEqual(len(keys), 0, "all LoRA keys should be consumed")
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_diffusion_model_prefix_stripped(self):
        """diffusion_model.* keys (without model. prefix) should also match."""
        from musubi_tuner.utils.lora_utils import _make_lora_name_from_model_key

        lora_name = _make_lora_name_from_model_key("diffusion_model.blocks.0.attn.q.weight")
        self.assertEqual(lora_name, "lora_unet_blocks_0_attn_q")

    def test_no_prefix_unchanged(self):
        """Keys without model prefixes should produce the same lora_name as before."""
        from musubi_tuner.utils.lora_utils import _make_lora_name_from_model_key

        lora_name = _make_lora_name_from_model_key("blocks.0.attn.q.weight")
        self.assertEqual(lora_name, "lora_unet_blocks_0_attn_q")


if __name__ == "__main__":
    unittest.main()
