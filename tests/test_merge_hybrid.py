import unittest

import torch
import torch.nn as nn

from musubi_tuner.utils.lora_utils import lora_merge_weights_to_tensor, merge_nonlora_to_model


class TestLoraMergeWeightsToTensor(unittest.TestCase):
    """Tests for the LoRA per-key-family merge function."""

    def _make_lora_sd(self, lora_name, out_dim, in_dim, rank=4):
        """Create a standard LoRA state dict."""
        return {
            f"{lora_name}.lora_down.weight": torch.randn(rank, in_dim),
            f"{lora_name}.lora_up.weight": torch.randn(out_dim, rank),
            f"{lora_name}.alpha": torch.tensor(float(rank)),
        }

    def test_linear_merge(self):
        """Linear LoRA weights are merged correctly."""
        lora_name = "module"
        model_weight = torch.zeros(8, 16)
        lora_sd = self._make_lora_sd(lora_name, 8, 16, rank=4)
        lora_keys = set(lora_sd.keys())

        result = lora_merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertEqual(result.shape, (8, 16))
        self.assertFalse(torch.all(result == 0), "Merged weight should be non-zero")

    def test_key_consumption(self):
        """Consumed keys are removed from the key set after merge."""
        lora_name = "module"
        lora_sd = self._make_lora_sd(lora_name, 8, 16, rank=4)
        lora_keys = set(lora_sd.keys())
        model_weight = torch.zeros(8, 16)

        lora_merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        for key in lora_sd:
            self.assertNotIn(key, lora_keys)

    def test_noop_no_keys(self):
        """Returns model_weight unchanged when no matching LoRA keys exist."""
        model_weight = torch.ones(8, 16)
        lora_sd = {"unrelated.weight": torch.zeros(4)}
        lora_keys = set(lora_sd.keys())

        result = lora_merge_weights_to_tensor(model_weight, "module", lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertTrue(torch.equal(result, model_weight))

    def test_result_returns_to_original_device(self):
        """Merged result is returned on the same device as the input model_weight."""
        lora_name = "module"
        model_weight = torch.zeros(8, 16, device="cpu")
        lora_sd = self._make_lora_sd(lora_name, 8, 16, rank=4)
        lora_keys = set(lora_sd.keys())

        result = lora_merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertEqual(result.device, model_weight.device)

    @unittest.skipUnless(getattr(torch, "float8_e4m3fn", None) is not None, "float8 not available")
    def test_fp8_cast(self):
        """FP8 model weights are temporarily upcast and restored."""
        lora_name = "fp8_module"
        model_weight = torch.zeros(8, 16, dtype=torch.float8_e4m3fn)
        lora_sd = self._make_lora_sd(lora_name, 8, 16, rank=4)
        lora_keys = set(lora_sd.keys())

        result = lora_merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertEqual(result.dtype, torch.float8_e4m3fn)

    def test_mixed_precision_preserves_non_fp8_dtype(self):
        """Non-FP8 mixed precision keeps original dtype (bf16 model, fp16 adapters)."""
        lora_name = "mixed_module"
        model_weight = torch.zeros(8, 16, dtype=torch.bfloat16)
        lora_sd = {
            f"{lora_name}.lora_down.weight": torch.randn(4, 16, dtype=torch.float16),
            f"{lora_name}.lora_up.weight": torch.randn(8, 4, dtype=torch.float16),
            f"{lora_name}.alpha": torch.tensor(4.0),
        }
        lora_keys = set(lora_sd.keys())

        result = lora_merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertEqual(result.dtype, torch.bfloat16)


class TestHybridMerge(unittest.TestCase):
    """Tests that merge_nonlora_to_model handles hybrid dicts (LoKr + LoRA)."""

    def _make_simple_model(self):
        """Create a simple model with two named linear layers."""
        model = nn.Module()
        model.block_a = nn.Module()
        model.block_a.linear = nn.Linear(16, 8, bias=False)
        model.block_b = nn.Module()
        model.block_b.linear = nn.Linear(16, 8, bias=False)
        # Zero out weights so we can detect merges
        with torch.no_grad():
            model.block_a.linear.weight.zero_()
            model.block_b.linear.weight.zero_()
        return model

    def test_hybrid_merges_both_families(self):
        """Hybrid dict with LoKr keys for one layer and LoRA keys for another."""
        model = self._make_simple_model()
        device = torch.device("cpu")

        # block_a uses LoKr: param_name = "block_a.linear.weight"
        # lora_name = "lora_unet_block_a_linear"
        lokr_name = "lora_unet_block_a_linear"
        weights_sd = {
            f"{lokr_name}.lokr_w1": torch.randn(2, 4),
            f"{lokr_name}.lokr_w2_a": torch.randn(4, 2),
            f"{lokr_name}.lokr_w2_b": torch.randn(2, 4),
            f"{lokr_name}.alpha": torch.tensor(2.0),
        }

        # block_b uses LoRA: param_name = "block_b.linear.weight"
        # lora_name = "lora_unet_block_b_linear"
        lora_name = "lora_unet_block_b_linear"
        weights_sd.update(
            {
                f"{lora_name}.lora_down.weight": torch.randn(4, 16),
                f"{lora_name}.lora_up.weight": torch.randn(8, 4),
                f"{lora_name}.alpha": torch.tensor(4.0),
            }
        )

        merged_count = merge_nonlora_to_model(model, weights_sd, 1.0, device)

        # Both layers should have been modified
        self.assertFalse(torch.all(model.block_a.linear.weight.data == 0), "LoKr layer should have been merged")
        self.assertFalse(torch.all(model.block_b.linear.weight.data == 0), "LoRA layer should have been merged")
        # All 7 weight keys (4 LoKr + 3 LoRA) should be consumed
        self.assertEqual(merged_count, 7)

    def test_pure_lora_via_merge_nonlora(self):
        """Pure LoRA dict is also handled by merge_nonlora_to_model."""
        model = self._make_simple_model()
        device = torch.device("cpu")

        lora_name = "lora_unet_block_a_linear"
        weights_sd = {
            f"{lora_name}.lora_down.weight": torch.randn(4, 16),
            f"{lora_name}.lora_up.weight": torch.randn(8, 4),
            f"{lora_name}.alpha": torch.tensor(4.0),
        }

        merged_count = merge_nonlora_to_model(model, weights_sd, 1.0, device)

        self.assertFalse(torch.all(model.block_a.linear.weight.data == 0))
        self.assertEqual(merged_count, 3)

    def test_merge_nonlora_preserves_param_dtype(self):
        """Per-key-family merge preserves parameter dtype for non-FP8 params."""
        model = self._make_simple_model()
        model.block_a.linear.weight.data = model.block_a.linear.weight.data.to(torch.bfloat16)
        device = torch.device("cpu")

        lora_name = "lora_unet_block_a_linear"
        weights_sd = {
            f"{lora_name}.lora_down.weight": torch.randn(4, 16, dtype=torch.float16),
            f"{lora_name}.lora_up.weight": torch.randn(8, 4, dtype=torch.float16),
            f"{lora_name}.alpha": torch.tensor(4.0),
        }

        merge_nonlora_to_model(model, weights_sd, 1.0, device)

        self.assertEqual(model.block_a.linear.weight.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
