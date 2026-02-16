import os
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from musubi_tuner.networks.lokr import _resolve_factor
from musubi_tuner.utils.lora_utils import filter_lora_state_dict


class TestLoKrFactorRoundtrip(unittest.TestCase):
    def _make_lokr_sd(self, factor=4):
        return {
            "lokr_factor": torch.tensor(factor, dtype=torch.int64),
            "lora_unet_block.lokr_w1": torch.randn(2, 2),
            "lora_unet_block.lokr_w2_a": torch.randn(4, 2),
            "lora_unet_block.lokr_w2_b": torch.randn(2, 4),
            "lora_unet_block.alpha": torch.tensor(2.0),
        }

    def test_factor_persists_in_safetensors(self):
        """Factor saved as buffer survives save/load via safetensors."""
        state_dict = self._make_lokr_sd(factor=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.safetensors")
            save_file(state_dict, path)
            loaded = load_file(path)
            self.assertEqual(int(loaded["lokr_factor"].item()), 4)

    def test_factor_persists_in_pt(self):
        """Factor saved as buffer survives save/load via torch.save (.pt)."""
        state_dict = self._make_lokr_sd(factor=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            torch.save(state_dict, path)
            loaded = torch.load(path, weights_only=True)
            self.assertEqual(int(loaded["lokr_factor"].item()), 8)

    def test_metadata_mirror_in_safetensors(self):
        """ss_lokr_factor in safetensors metadata is readable."""
        state_dict = self._make_lokr_sd(factor=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.safetensors")
            save_file(state_dict, path, metadata={"ss_lokr_factor": "4"})
            with safe_open(path, framework="pt") as f:
                meta = f.metadata()
                self.assertEqual(meta["ss_lokr_factor"], "4")

    def test_factor_precedence_explicit_over_persisted(self):
        """Explicit factor kwarg takes precedence over persisted buffer value."""
        sd = self._make_lokr_sd(factor=4)
        factor, warning = _resolve_factor(sd, explicit_factor=8)
        self.assertEqual(factor, 8)
        self.assertTrue(warning)  # mismatch warning should be flagged

    def test_factor_precedence_persisted_when_no_explicit(self):
        """Persisted factor used when no explicit kwarg provided."""
        sd = self._make_lokr_sd(factor=4)
        factor, warning = _resolve_factor(sd, explicit_factor=None)
        self.assertEqual(factor, 4)
        self.assertFalse(warning)

    def test_factor_default_when_nothing_persisted(self):
        """Default factor=-1 when no explicit kwarg and no persisted value."""
        sd = self._make_lokr_sd(factor=4)
        del sd["lokr_factor"]
        factor, warning = _resolve_factor(sd, explicit_factor=None)
        self.assertEqual(factor, -1)
        self.assertFalse(warning)

    def test_factor_metadata_fallback_when_buffer_absent(self):
        """metadata_factor used when lokr_factor buffer is absent."""
        sd = self._make_lokr_sd(factor=4)
        del sd["lokr_factor"]
        factor, warning = _resolve_factor(sd, explicit_factor=None, metadata_factor=4)
        self.assertEqual(factor, 4)
        self.assertFalse(warning)

    def test_factor_buffer_takes_precedence_over_metadata(self):
        """Buffer value wins over metadata_factor."""
        sd = self._make_lokr_sd(factor=4)
        factor, warning = _resolve_factor(sd, explicit_factor=None, metadata_factor=8)
        self.assertEqual(factor, 4)  # buffer wins
        self.assertFalse(warning)

    def test_factor_fallback_from_metadata(self):
        """When lokr_factor buffer is missing, factor recovered from ss_lokr_factor metadata."""
        state_dict = self._make_lokr_sd(factor=4)
        del state_dict["lokr_factor"]  # simulate missing buffer
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.safetensors")
            save_file(state_dict, path, metadata={"ss_lokr_factor": "4"})
            # Verify buffer is missing from loaded tensors
            loaded = load_file(path)
            self.assertNotIn("lokr_factor", loaded)
            # But metadata is present for caller to read
            with safe_open(path, framework="pt") as f:
                meta = f.metadata()
                self.assertEqual(meta["ss_lokr_factor"], "4")

    def test_filter_preserves_lokr_factor(self):
        """filter_lora_state_dict with include pattern preserves non-dotted keys."""
        sd = self._make_lokr_sd(factor=4)
        filtered = filter_lora_state_dict(sd, include_pattern="lora_unet_block")
        self.assertIn("lokr_factor", filtered)


if __name__ == "__main__":
    unittest.main()
