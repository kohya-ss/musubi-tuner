"""Split-weight cache regressions independent of architecture-specific conversion hooks."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from musubi_tuner.utils.safetensors_utils import TensorWeightAdapter, WeightTransformHooks


def split_weight(key, tensor):
    if key == "fused.weight":
        return ["first.weight", "second.weight"], None if tensor is None else tensor.chunk(2)
    return None, None


class SplitWeightCacheTests(unittest.TestCase):
    def test_split_reads_once_and_releases_consumed_tensors(self):
        for order in (("first.weight", "second.weight"), ("second.weight", "first.weight")):
            with self.subTest(order=order):
                source = Mock()
                source.keys.return_value = ["fused.weight", "unchanged.weight"]
                device = torch.device("cpu")
                value = torch.arange(24, dtype=torch.float32).reshape(6, 4)
                source.get_tensor.return_value = value
                adapter = TensorWeightAdapter(WeightTransformHooks(split_hook=split_weight), source)
                self.assertEqual(set(adapter.keys()), {"first.weight", "second.weight", "unchanged.weight"})
                results = {}
                for key in order:
                    results[key] = adapter.get_tensor(key, device=device, dtype=torch.float32)
                    self.assertNotIn(key, adapter.tensor_cache)
                torch.testing.assert_close(torch.cat([results["first.weight"], results["second.weight"]]), value)
                source.get_tensor.assert_called_once_with("fused.weight", device=device, dtype=torch.float32)
                self.assertEqual(adapter.tensor_cache, {})
                torch.testing.assert_close(adapter.get_tensor("unchanged.weight"), value)
                self.assertEqual(source.get_tensor.call_count, 2)

    def test_original_key_can_also_be_a_split_output(self):
        source = Mock()
        source.keys.return_value = ["fused.weight"]
        source.get_tensor.return_value = torch.arange(8)

        def split(key, tensor):
            return [key, "other.weight"], None if tensor is None else tensor.chunk(2)

        adapter = TensorWeightAdapter(WeightTransformHooks(split_hook=split), source)
        first = adapter.get_tensor("fused.weight")
        second = adapter.get_tensor("other.weight")
        torch.testing.assert_close(torch.cat([first, second]), source.get_tensor.return_value)
        self.assertEqual(source.get_tensor.call_count, 1)
        self.assertEqual(adapter.tensor_cache, {})


if __name__ == "__main__":
    unittest.main()
