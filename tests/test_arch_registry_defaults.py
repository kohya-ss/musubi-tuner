import unittest

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_FLUX_2_DEV,
    ARCHITECTURE_FLUX_KONTEXT,
    ARCHITECTURE_FRAMEPACK,
    ARCHITECTURE_HUNYUAN_VIDEO,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5,
    ARCHITECTURE_KANDINSKY5,
    ARCHITECTURE_QWEN_IMAGE,
    ARCHITECTURE_WAN,
    ARCHITECTURE_Z_IMAGE,
)
from musubi_tuner.networks.lora import HUNYUAN_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_flux import FLUX_KONTEXT_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_flux_2 import FLUX_2_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_framepack import FRAMEPACK_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_hv_1_5 import HV_1_5_IMAGE_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_kandinsky import KANDINSKY5_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_qwen_image import QWEN_IMAGE_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_wan import WAN_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_zimage import ZIMAGE_TARGET_REPLACE_MODULES
from musubi_tuner.networks.network_arch import ARCH_CONFIGS


class TestArchRegistryDefaults(unittest.TestCase):
    def test_target_modules_match_authoritative_source(self):
        """Registry target_modules must match the constants in lora_*.py files."""
        cases = [
            (ARCHITECTURE_WAN, WAN_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_FRAMEPACK, FRAMEPACK_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_FLUX_KONTEXT, FLUX_KONTEXT_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_FLUX_2_DEV, FLUX_2_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_HUNYUAN_VIDEO, HUNYUAN_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_HUNYUAN_VIDEO_1_5, HV_1_5_IMAGE_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_QWEN_IMAGE, QWEN_IMAGE_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_Z_IMAGE, ZIMAGE_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_KANDINSKY5, KANDINSKY5_TARGET_REPLACE_MODULES),
        ]
        for arch, expected in cases:
            with self.subTest(arch=arch):
                self.assertEqual(ARCH_CONFIGS[arch]["target_modules"], expected)

    def test_all_architectures_have_exclude_patterns(self):
        for arch, config in ARCH_CONFIGS.items():
            with self.subTest(arch=arch):
                self.assertIn("exclude_patterns", config)

    def test_qwen_has_exclude_mod_patterns(self):
        for arch in [ARCHITECTURE_QWEN_IMAGE]:
            self.assertIn("exclude_mod_patterns", ARCH_CONFIGS[arch])

    def test_kandinsky_has_include_patterns(self):
        self.assertIn("include_patterns", ARCH_CONFIGS[ARCHITECTURE_KANDINSKY5])
        self.assertTrue(len(ARCH_CONFIGS[ARCHITECTURE_KANDINSKY5]["include_patterns"]) > 0)

    def test_critical_exclude_patterns_present(self):
        """Each architecture must have its critical safety excludes to prevent training instability."""
        # Map of architecture -> substring that must appear in at least one exclude pattern
        critical_excludes = [
            (ARCHITECTURE_WAN, "norm", "WAN must exclude norm layers"),
            (ARCHITECTURE_WAN, "embedding", "WAN must exclude embedding layers"),
            (ARCHITECTURE_HUNYUAN_VIDEO, "modulation", "HunyuanVideo must exclude modulation layers"),
            (ARCHITECTURE_HUNYUAN_VIDEO, "img_mod", "HunyuanVideo must exclude img_mod layers"),
            (ARCHITECTURE_FLUX_KONTEXT, "modulation", "Flux Kontext must exclude modulation layers"),
            (ARCHITECTURE_FLUX_KONTEXT, "norm", "Flux Kontext must exclude norm layers"),
            (ARCHITECTURE_FLUX_2_DEV, "modulation", "Flux 2 must exclude modulation layers"),
            (ARCHITECTURE_Z_IMAGE, "modulation", "Z-Image must exclude modulation layers"),
            (ARCHITECTURE_Z_IMAGE, "refiner", "Z-Image must exclude refiner layers"),
            (ARCHITECTURE_KANDINSKY5, "modulation", "Kandinsky must exclude modulation layers"),
        ]
        for arch, substring, msg in critical_excludes:
            with self.subTest(arch=arch, pattern=substring):
                patterns = ARCH_CONFIGS[arch]["exclude_patterns"]
                has_match = any(substring in p for p in patterns)
                self.assertTrue(has_match, msg)


if __name__ == "__main__":
    unittest.main()
