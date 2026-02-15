import unittest

import torch

from musubi_tuner.networks.lokr import merge_weights_to_tensor


class TestMergeWeightsLoKr(unittest.TestCase):
    def _make_lokr_sd_lowrank(self, lora_name, w1_shape, w2_out, w2_in, rank=2):
        """Create a LoKr state dict in low-rank mode (w2_a + w2_b)."""
        return {
            f"{lora_name}.lokr_w1": torch.randn(*w1_shape),
            f"{lora_name}.lokr_w2_a": torch.randn(w2_out, rank),
            f"{lora_name}.lokr_w2_b": torch.randn(rank, w2_in),
            f"{lora_name}.alpha": torch.tensor(float(rank)),
        }

    def _make_lokr_sd_fullmatrix(self, lora_name, w1_shape, w2_shape):
        """Create a LoKr state dict in full-matrix mode (w2 directly)."""
        dim = max(w2_shape)
        return {
            f"{lora_name}.lokr_w1": torch.randn(*w1_shape),
            f"{lora_name}.lokr_w2": torch.randn(*w2_shape),
            f"{lora_name}.alpha": torch.tensor(float(dim)),
        }

    def test_lowrank_merge(self):
        """Low-rank mode: w2 = w2_a @ w2_b, then kron(w1, w2) * scale."""
        lora_name = "module"
        # model_weight: (8, 16) — factor 8=(2,4), 16=(4,4)
        model_weight = torch.zeros(8, 16)
        lora_sd = self._make_lokr_sd_lowrank(lora_name, w1_shape=(2, 4), w2_out=4, w2_in=4, rank=2)
        lora_keys = set(lora_sd.keys())

        result = merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertEqual(result.shape, (8, 16))
        self.assertFalse(torch.all(result == 0), "Merged weight should be non-zero")

    def test_fullmatrix_merge(self):
        """Full-matrix mode: w2 used directly, then kron(w1, w2) * scale."""
        lora_name = "module"
        model_weight = torch.zeros(8, 16)
        lora_sd = self._make_lokr_sd_fullmatrix(lora_name, w1_shape=(2, 4), w2_shape=(4, 4))
        lora_keys = set(lora_sd.keys())

        result = merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertEqual(result.shape, (8, 16))
        self.assertFalse(torch.all(result == 0))

    @unittest.skipUnless(getattr(torch, "float8_e4m3fn", None) is not None, "float8 not available")
    def test_fp8_cast(self):
        """FP8 model weights are temporarily upcast and restored."""
        lora_name = "fp8_module"
        model_weight = torch.zeros(8, 16, dtype=torch.float8_e4m3fn)
        lora_sd = self._make_lokr_sd_lowrank(lora_name, w1_shape=(2, 4), w2_out=4, w2_in=4, rank=2)
        lora_keys = set(lora_sd.keys())

        result = merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertEqual(result.dtype, torch.float8_e4m3fn)

    def test_key_consumption(self):
        """Consumed keys are removed from the key set after merge."""
        lora_name = "module"
        lora_sd = self._make_lokr_sd_lowrank(lora_name, w1_shape=(2, 4), w2_out=4, w2_in=4, rank=2)
        lora_keys = set(lora_sd.keys())
        model_weight = torch.zeros(8, 16)

        merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        for key in lora_sd:
            self.assertNotIn(key, lora_keys)

    def test_noop_no_keys(self):
        """Returns model_weight unchanged when no matching LoKr keys exist."""
        model_weight = torch.ones(8, 16)
        lora_sd = {"unrelated.weight": torch.zeros(4)}
        lora_keys = set(lora_sd.keys())

        result = merge_weights_to_tensor(model_weight, "module", lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertTrue(torch.equal(result, model_weight))
        self.assertEqual(lora_keys, {"unrelated.weight"})


if __name__ == "__main__":
    unittest.main()
