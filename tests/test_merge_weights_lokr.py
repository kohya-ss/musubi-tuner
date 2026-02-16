import unittest

import torch
import torch.nn as nn

from musubi_tuner.networks.lokr import LoKrInfModule, LoKrModule, merge_weights_to_tensor


class TestLoKrModuleConstruction(unittest.TestCase):
    """Smoke tests for LoKrModule instantiation (alpha buffer, parameter shapes)."""

    def test_construction_does_not_crash(self):
        """LoKrModule can be instantiated without alpha buffer collision."""
        linear = nn.Linear(16, 8)
        module = LoKrModule("test_module", linear, lora_dim=2, alpha=1.0, factor=-1)
        self.assertIsInstance(module.alpha, torch.Tensor)
        self.assertEqual(module.alpha.item(), 1.0)
        self.assertIsNotNone(module.lokr_w1)

    def test_construction_lowrank_mode(self):
        """Low-rank mode creates w2_a + w2_b when dim < max(factored shape)."""
        linear = nn.Linear(16, 8)
        module = LoKrModule("test_module", linear, lora_dim=2, alpha=1.0, factor=-1)
        self.assertIsNotNone(module.lokr_w2_a)
        self.assertIsNotNone(module.lokr_w2_b)
        self.assertIsNone(module.lokr_w2)

    def test_construction_fullmatrix_mode(self):
        """Full-matrix mode creates w2 when dim >= max(factored shape)."""
        linear = nn.Linear(16, 8)
        # Use very large dim to force full-matrix
        module = LoKrModule("test_module", linear, lora_dim=999, alpha=1.0, factor=-1)
        self.assertIsNotNone(module.lokr_w2)
        self.assertIsNone(module.lokr_w2_a)
        self.assertIsNone(module.lokr_w2_b)

    def test_conv2d_raises(self):
        """Conv2d modules raise ValueError (LoKr v1 is Linear-only)."""
        conv = nn.Conv2d(3, 16, 3)
        with self.assertRaises(ValueError):
            LoKrModule("test_conv", conv, lora_dim=2, alpha=1.0, factor=-1)


class TestLoKrInfModuleMergeDtype(unittest.TestCase):
    """Tests that LoKrInfModule.merge_to preserves original dtype."""

    def _build_sd(self, module):
        """Build per-module sd matching LoRANetwork.merge_to contract."""
        sd = {}
        for name, param in module.named_parameters():
            key = name.split(".", 1)[-1] if "." in name else name
            sd[key] = param.data.clone()
        # Include alpha buffer
        if hasattr(module, "alpha"):
            sd["alpha"] = module.alpha.clone()
        return sd

    def test_merge_preserves_bfloat16(self):
        """merge_to with dtype=None preserves bfloat16 model weights."""
        linear = nn.Linear(16, 8)
        linear.weight.data = linear.weight.data.to(torch.bfloat16)

        module = LoKrInfModule("test_module", linear, multiplier=1.0, lora_dim=2, alpha=1.0, factor=-1)
        sd = self._build_sd(module)

        module.merge_to(sd, dtype=None, device=torch.device("cpu"))
        self.assertEqual(linear.weight.dtype, torch.bfloat16)

    def test_merge_preserves_float16(self):
        """merge_to with dtype=None preserves float16 model weights."""
        linear = nn.Linear(16, 8)
        linear.weight.data = linear.weight.data.to(torch.float16)

        module = LoKrInfModule("test_module", linear, multiplier=1.0, lora_dim=2, alpha=1.0, factor=-1)
        sd = self._build_sd(module)

        module.merge_to(sd, dtype=None, device=torch.device("cpu"))
        self.assertEqual(linear.weight.dtype, torch.float16)

    def test_merge_uses_sd_not_internal_params(self):
        """merge_to reads from sd payload, not from module internal parameters."""
        linear = nn.Linear(16, 8)
        linear.weight.data = torch.zeros(8, 16)

        module = LoKrInfModule("test_module", linear, multiplier=1.0, lora_dim=2, alpha=1.0, factor=-1)

        # Build sd with known non-zero values (overriding init zeros in w2_b)
        sd = self._build_sd(module)
        sd["lokr_w1"] = torch.ones_like(sd["lokr_w1"])
        if "lokr_w2_a" in sd:
            sd["lokr_w2_a"] = torch.ones_like(sd["lokr_w2_a"])
            sd["lokr_w2_b"] = torch.ones_like(sd["lokr_w2_b"])
        if "lokr_w2" in sd:
            sd["lokr_w2"] = torch.ones_like(sd["lokr_w2"])

        # Now zero out internal params — if merge_to uses self, result will be zero
        with torch.no_grad():
            module.lokr_w1.zero_()
            if module.lokr_w2_a is not None:
                module.lokr_w2_a.zero_()
                module.lokr_w2_b.zero_()
            if module.lokr_w2 is not None:
                module.lokr_w2.zero_()

        module.merge_to(sd, dtype=None, device=torch.device("cpu"))

        # Weight should be non-zero because sd had non-zero values
        self.assertFalse(torch.all(linear.weight.data == 0), "merge_to should use sd, not zeroed internal params")


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

    def test_result_returns_to_original_device(self):
        """Merged result is returned on the same device as the input model_weight."""
        lora_name = "module"
        model_weight = torch.zeros(8, 16, device="cpu")
        lora_sd = self._make_lokr_sd_lowrank(lora_name, w1_shape=(2, 4), w2_out=4, w2_in=4, rank=2)
        lora_keys = set(lora_sd.keys())

        result = merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertEqual(result.device, model_weight.device)

    def test_mixed_precision_preserves_non_fp8_dtype(self):
        """Non-FP8 mixed precision keeps original dtype (bf16 model, fp16 adapters)."""
        lora_name = "mixed_module"
        model_weight = torch.zeros(8, 16, dtype=torch.bfloat16)
        lora_sd = {
            f"{lora_name}.lokr_w1": torch.randn(2, 4, dtype=torch.float16),
            f"{lora_name}.lokr_w2_a": torch.randn(4, 2, dtype=torch.float16),
            f"{lora_name}.lokr_w2_b": torch.randn(2, 4, dtype=torch.float16),
            f"{lora_name}.alpha": torch.tensor(2.0),
        }
        lora_keys = set(lora_sd.keys())

        result = merge_weights_to_tensor(model_weight, lora_name, lora_sd, lora_keys, 1.0, torch.device("cpu"))

        self.assertEqual(result.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
