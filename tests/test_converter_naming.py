"""Tests for convert_lora.py naming parity across LoRA, LoHa, and LoKr key families."""

import unittest

import torch

from musubi_tuner.convert_lora import _normalize_module_name, convert_from_diffusers, convert_to_diffusers


class TestNormalizeModuleName(unittest.TestCase):
    """Tests for the shared _normalize_module_name helper."""

    PREFIX = "lora_unet_"

    def test_wan_cross_attn(self):
        """WAN cross_attn module names are normalized correctly."""
        name = "lora_unet_blocks_0_cross_attn_k_img"
        result = _normalize_module_name(name, self.PREFIX)
        self.assertEqual(result, "blocks.0.cross_attn.k_img")

    def test_wan_self_attn(self):
        """WAN self_attn module names are normalized correctly."""
        name = "lora_unet_blocks_0_self_attn_k_img"
        result = _normalize_module_name(name, self.PREFIX)
        self.assertEqual(result, "blocks.0.self_attn.k_img")

    def test_wan_v_img(self):
        """WAN v_img is preserved."""
        name = "lora_unet_blocks_0_cross_attn_v_img"
        result = _normalize_module_name(name, self.PREFIX)
        self.assertEqual(result, "blocks.0.cross_attn.v_img")

    def test_zimage_attention(self):
        """Z-Image attention module names are normalized correctly."""
        name = "lora_unet_layers_0_attention_to_q"
        result = _normalize_module_name(name, self.PREFIX)
        self.assertEqual(result, "layers.0.attention.to_q")

    def test_zimage_to_out(self):
        """Z-Image to_out is preserved."""
        name = "lora_unet_layers_0_attention_to_out_0"
        result = _normalize_module_name(name, self.PREFIX)
        self.assertEqual(result, "layers.0.attention.to_out.0")

    def test_zimage_feed_forward(self):
        """Z-Image feed_forward module names are normalized correctly."""
        name = "lora_unet_layers_0_feed_forward_net_0_proj"
        result = _normalize_module_name(name, self.PREFIX)
        self.assertEqual(result, "layers.0.feed_forward.net.0.proj")

    def test_hunyuan_double_blocks(self):
        """HunyuanVideo double_blocks module names are normalized correctly."""
        name = "lora_unet_double_blocks_0_img_attn_qkv"
        result = _normalize_module_name(name, self.PREFIX)
        self.assertEqual(result, "double_blocks.0.img_attn_qkv")

    def test_hunyuan_single_blocks(self):
        """HunyuanVideo single_blocks module names are normalized correctly."""
        name = "lora_unet_single_blocks_0_txt_attn_qkv"
        result = _normalize_module_name(name, self.PREFIX)
        self.assertEqual(result, "single_blocks.0.txt_attn_qkv")


class TestConverterNamingParity(unittest.TestCase):
    """Verifies LoRA, LoHa, and LoKr keys for the same module produce identical Diffusers module names."""

    PREFIX = "lora_unet_"

    def _make_lora_sd(self, lora_name, rank=4, in_dim=16, out_dim=8):
        return {
            f"{lora_name}.lora_down.weight": torch.randn(rank, in_dim),
            f"{lora_name}.lora_up.weight": torch.randn(out_dim, rank),
            f"{lora_name}.alpha": torch.tensor(float(rank)),
        }

    def _make_loha_sd(self, lora_name, rank=4, in_dim=16, out_dim=8):
        return {
            f"{lora_name}.hada_w1_a": torch.randn(out_dim, rank),
            f"{lora_name}.hada_w1_b": torch.randn(rank, in_dim),
            f"{lora_name}.hada_w2_a": torch.randn(out_dim, rank),
            f"{lora_name}.hada_w2_b": torch.randn(rank, in_dim),
            f"{lora_name}.alpha": torch.tensor(float(rank)),
        }

    def _make_lokr_sd(self, lora_name, w1_shape=(2, 4), w2_out=4, w2_in=4, rank=2):
        return {
            f"{lora_name}.lokr_w1": torch.randn(*w1_shape),
            f"{lora_name}.lokr_w2_a": torch.randn(w2_out, rank),
            f"{lora_name}.lokr_w2_b": torch.randn(rank, w2_in),
            f"{lora_name}.alpha": torch.tensor(float(rank)),
        }

    def _get_diffusers_module_names(self, sd):
        """Convert and extract unique Diffusers module name prefixes from output keys."""
        result = convert_to_diffusers(self.PREFIX, "diffusion_model", sd)
        module_names = set()
        for key in result:
            if "." not in key:
                continue
            # Strip "diffusion_model." prefix and parameter suffix
            parts = key.split(".")
            # Module name is everything between diffusers_prefix and param key
            if parts[0] == "diffusion_model":
                # Find where the param part starts (lora_A, lora_B, hada_*, lokr_*, alpha)
                param_starts = {"lora_A", "lora_B", "hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "lokr_w1", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "alpha"}
                for i, part in enumerate(parts[1:], 1):
                    if part in param_starts:
                        module_names.add(".".join(parts[1:i]))
                        break
        return module_names

    def _roundtrip_default_keys(self, sd):
        """Round-trip default -> diffusers -> default and return key set."""
        converted = convert_to_diffusers(self.PREFIX, "diffusion_model", sd)
        restored = convert_from_diffusers(self.PREFIX, converted)
        return set(restored.keys())

    def test_wan_lora_vs_loha_naming(self):
        """WAN: LoRA and LoHa keys for the same module produce the same Diffusers module name."""
        lora_name = "lora_unet_blocks_0_cross_attn_k_img"
        lora_sd = self._make_lora_sd(lora_name)
        loha_sd = self._make_loha_sd(lora_name)

        lora_names = self._get_diffusers_module_names(lora_sd)
        loha_names = self._get_diffusers_module_names(loha_sd)

        self.assertEqual(lora_names, loha_names, f"LoRA produced {lora_names}, LoHa produced {loha_names}")

    def test_wan_lora_vs_lokr_naming(self):
        """WAN: LoRA and LoKr keys for the same module produce the same Diffusers module name."""
        lora_name = "lora_unet_blocks_0_cross_attn_k_img"
        lora_sd = self._make_lora_sd(lora_name)
        lokr_sd = self._make_lokr_sd(lora_name)

        lora_names = self._get_diffusers_module_names(lora_sd)
        lokr_names = self._get_diffusers_module_names(lokr_sd)

        self.assertEqual(lora_names, lokr_names, f"LoRA produced {lora_names}, LoKr produced {lokr_names}")

    def test_zimage_lora_vs_loha_naming(self):
        """Z-Image: LoRA and LoHa keys for the same module produce the same Diffusers module name."""
        lora_name = "lora_unet_layers_0_attention_to_q"
        lora_sd = self._make_lora_sd(lora_name)
        loha_sd = self._make_loha_sd(lora_name)

        lora_names = self._get_diffusers_module_names(lora_sd)
        loha_names = self._get_diffusers_module_names(loha_sd)

        self.assertEqual(lora_names, loha_names, f"LoRA produced {lora_names}, LoHa produced {loha_names}")

    def test_hunyuan_lora_vs_lokr_naming(self):
        """HunyuanVideo: LoRA and LoKr keys for the same module produce the same Diffusers module name."""
        lora_name = "lora_unet_double_blocks_0_img_attn_qkv"
        lora_sd = self._make_lora_sd(lora_name)
        lokr_sd = self._make_lokr_sd(lora_name)

        lora_names = self._get_diffusers_module_names(lora_sd)
        lokr_names = self._get_diffusers_module_names(lokr_sd)

        self.assertEqual(lora_names, lokr_names, f"LoRA produced {lora_names}, LoKr produced {lokr_names}")

    def test_hybrid_dict_consistent_naming(self):
        """Hybrid dict: LoKr and LoRA keys coexisting produce correct names for both."""
        lokr_name = "lora_unet_blocks_0_cross_attn_k_img"
        lora_name = "lora_unet_blocks_0_cross_attn_v_img"

        sd = {}
        sd.update(self._make_lokr_sd(lokr_name))
        sd.update(self._make_lora_sd(lora_name))

        module_names = self._get_diffusers_module_names(sd)

        self.assertIn("blocks.0.cross_attn.k_img", module_names)
        self.assertIn("blocks.0.cross_attn.v_img", module_names)

    def test_roundtrip_lora_wan_keys_preserved(self):
        """LoRA WAN-style module names survive default->diffusers->default round-trip."""
        lora_name = "lora_unet_blocks_0_cross_attn_k_img"
        sd = self._make_lora_sd(lora_name)
        roundtrip_keys = self._roundtrip_default_keys(sd)

        self.assertIn(f"{lora_name}.lora_down.weight", roundtrip_keys)
        self.assertIn(f"{lora_name}.lora_up.weight", roundtrip_keys)
        self.assertIn(f"{lora_name}.alpha", roundtrip_keys)

    def test_roundtrip_loha_wan_keys_preserved(self):
        """LoHa WAN-style module names survive default->diffusers->default round-trip."""
        lora_name = "lora_unet_blocks_0_cross_attn_k_img"
        sd = self._make_loha_sd(lora_name)
        roundtrip_keys = self._roundtrip_default_keys(sd)

        self.assertIn(f"{lora_name}.hada_w1_a", roundtrip_keys)
        self.assertIn(f"{lora_name}.hada_w1_b", roundtrip_keys)
        self.assertIn(f"{lora_name}.hada_w2_a", roundtrip_keys)
        self.assertIn(f"{lora_name}.hada_w2_b", roundtrip_keys)
        self.assertIn(f"{lora_name}.alpha", roundtrip_keys)

    def test_roundtrip_lokr_hunyuan_keys_preserved(self):
        """LoKr Hunyuan-style module names survive default->diffusers->default round-trip."""
        lora_name = "lora_unet_double_blocks_0_img_attn_qkv"
        sd = self._make_lokr_sd(lora_name)
        roundtrip_keys = self._roundtrip_default_keys(sd)

        self.assertIn(f"{lora_name}.lokr_w1", roundtrip_keys)
        self.assertIn(f"{lora_name}.lokr_w2_a", roundtrip_keys)
        self.assertIn(f"{lora_name}.lokr_w2_b", roundtrip_keys)
        self.assertIn(f"{lora_name}.alpha", roundtrip_keys)


if __name__ == "__main__":
    unittest.main()
