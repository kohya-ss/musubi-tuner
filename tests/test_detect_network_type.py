import unittest

import torch

from musubi_tuner.utils.lora_utils import detect_network_type


class TestDetectNetworkType(unittest.TestCase):
    def test_lora_detection(self):
        sd = {"module.lora_down.weight": torch.zeros(4, 8), "module.lora_up.weight": torch.zeros(8, 4)}
        self.assertEqual(detect_network_type(sd), "lora")

    def test_loha_detection(self):
        sd = {"module.hada_w1_a": torch.zeros(8, 4), "module.hada_w1_b": torch.zeros(4, 8)}
        self.assertEqual(detect_network_type(sd), "loha")

    def test_lokr_detection(self):
        sd = {"module.lokr_w1": torch.zeros(4, 4), "module.lokr_w2_a": torch.zeros(8, 4)}
        self.assertEqual(detect_network_type(sd), "lokr")

    def test_hybrid_detection(self):
        """After QKV conversion: lokr_* keys + lora_* keys coexist."""
        sd = {
            "module_a.lokr_w1": torch.zeros(4, 4),
            "module_a.lokr_w2_a": torch.zeros(8, 4),
            "module_b.lora_down.weight": torch.zeros(4, 8),
            "module_b.lora_up.weight": torch.zeros(8, 4),
        }
        self.assertEqual(detect_network_type(sd), "hybrid")

    def test_unknown_detection(self):
        sd = {"some_random_key": torch.zeros(4)}
        self.assertEqual(detect_network_type(sd), "unknown")

    def test_empty_dict(self):
        self.assertEqual(detect_network_type({}), "unknown")

    def test_non_dotted_keys_ignored(self):
        """Network-level metadata keys (lokr_factor) don't affect type detection."""
        sd = {"lokr_factor": torch.tensor(4.0)}
        self.assertEqual(detect_network_type(sd), "unknown")

    def test_diffusers_lora_detection(self):
        """Diffusers-format LoRA keys (lora_A/lora_B) detected as 'lora'."""
        sd = {
            "diffusion_model.blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 8),
            "diffusion_model.blocks.0.attn.to_q.lora_B.weight": torch.zeros(8, 4),
        }
        self.assertEqual(detect_network_type(sd), "lora")

    def test_accepts_keys_iterable(self):
        """Can accept plain key strings (not just dict) for prepass efficiency."""
        keys = ["module.lora_down.weight", "module.lora_up.weight"]
        self.assertEqual(detect_network_type(keys), "lora")


if __name__ == "__main__":
    unittest.main()
