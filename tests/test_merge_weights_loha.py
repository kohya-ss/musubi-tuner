import unittest

import torch

from musubi_tuner.networks.loha import merge_weights_to_tensor


class TestMergeWeightsLoHa(unittest.TestCase):
    def _make_loha_sd(self, lora_name, out_dim, in_dim, rank=4):
        """Create a LoHa state dict with random weights for testing."""
        return {
            f"{lora_name}.hada_w1_a": torch.randn(out_dim, rank),
            f"{lora_name}.hada_w1_b": torch.randn(rank, in_dim),
            f"{lora_name}.hada_w2_a": torch.randn(out_dim, rank),
            f"{lora_name}.hada_w2_b": torch.randn(rank, in_dim),
            f"{lora_name}.alpha": torch.tensor(float(rank)),
        }

    def test_linear_merge(self):
        """Linear weights (2D model_weight) are merged correctly."""
        lora_name = "module"
        out_dim, in_dim, rank = 8, 16, 4
        model_weight = torch.zeros(out_dim, in_dim)
        lora_sd = self._make_loha_sd(lora_name, out_dim, in_dim, rank)
        lora_keys = set(lora_sd.keys())

        result = merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertEqual(result.shape, (out_dim, in_dim))
        self.assertFalse(torch.all(result == 0), "Merged weight should be non-zero")

    def test_conv2d_merge(self):
        """Conv2d weights (4D model_weight) are merged with 2D->4D reshape."""
        lora_name = "conv_module"
        out_ch, in_ch, kh, kw = 8, 16, 3, 3
        flat_in = in_ch * kh * kw  # 144
        model_weight = torch.zeros(out_ch, in_ch, kh, kw)
        lora_sd = self._make_loha_sd(lora_name, out_ch, flat_in, rank=4)
        lora_keys = set(lora_sd.keys())

        result = merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertEqual(result.shape, (out_ch, in_ch, kh, kw))
        self.assertFalse(torch.all(result == 0))

    @unittest.skipUnless(getattr(torch, "float8_e4m3fn", None) is not None, "float8 not available")
    def test_fp8_cast(self):
        """FP8 model weights are temporarily upcast and restored."""
        lora_name = "fp8_module"
        out_dim, in_dim, rank = 8, 16, 4
        model_weight = torch.zeros(out_dim, in_dim, dtype=torch.float8_e4m3fn)
        lora_sd = self._make_loha_sd(lora_name, out_dim, in_dim, rank)
        lora_keys = set(lora_sd.keys())

        result = merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertEqual(result.dtype, torch.float8_e4m3fn)

    def test_key_consumption(self):
        """Consumed keys are removed from the key set after merge."""
        lora_name = "module"
        lora_sd = self._make_loha_sd(lora_name, 8, 16, 4)
        lora_keys = set(lora_sd.keys())
        model_weight = torch.zeros(8, 16)

        merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        # All LoHa keys for this module should be consumed
        for key in lora_sd:
            self.assertNotIn(key, lora_keys)

    def test_noop_no_keys(self):
        """Returns model_weight unchanged when no matching LoHa keys exist."""
        model_weight = torch.ones(8, 16)
        lora_sd = {"unrelated.weight": torch.zeros(4)}
        lora_keys = set(lora_sd.keys())

        result = merge_weights_to_tensor(model_weight, "module", lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertTrue(torch.equal(result, model_weight))
        self.assertEqual(lora_keys, {"unrelated.weight"})  # keys untouched


if __name__ == "__main__":
    unittest.main()
