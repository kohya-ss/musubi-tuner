"""Integration-style tests for unknown LoRA format rejection in on-the-fly loaders."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from musubi_tuner import fpack_generate_video
from musubi_tuner import hv_1_5_generate_video
from musubi_tuner import qwen_image_generate_image
from musubi_tuner import wan_generate_video
from musubi_tuner import zimage_generate_image


class TestUnknownFormatRejection(unittest.TestCase):
    def _unknown_sd(self):
        # No recognized key family substrings (lora_*, hada_*, lokr_*) -> detect_network_type == "unknown"
        return {"ia3_adapter_weight": torch.randn(1)}

    def test_wan_on_the_fly_rejects_unknown_format(self):
        args = SimpleNamespace(
            blocks_to_swap=0,
            prefer_lycoris=False,
            include_patterns=None,
            exclude_patterns=None,
        )
        with (
            patch.object(wan_generate_video, "load_file", return_value=self._unknown_sd()),
            patch.object(wan_generate_video, "load_wan_model") as mock_load_wan_model,
        ):
            with self.assertRaisesRegex(ValueError, r"Unrecognized weight format in bad\.safetensors"):
                wan_generate_video.load_dit_model(
                    args,
                    "dit.safetensors",
                    ["bad.safetensors"],
                    [1.0],
                    object(),
                    torch.device("cpu"),
                )
            mock_load_wan_model.assert_not_called()

    def test_qwen_on_the_fly_rejects_unknown_format(self):
        args = SimpleNamespace(
            blocks_to_swap=0,
            prefer_lycoris=False,
            lora_weight=["bad.safetensors"],
            include_patterns=None,
            exclude_patterns=None,
        )
        with (
            patch.object(qwen_image_generate_image, "load_file", return_value=self._unknown_sd()),
            patch.object(qwen_image_generate_image.qwen_image_model, "load_qwen_image_model") as mock_load_qwen_model,
        ):
            with self.assertRaisesRegex(ValueError, r"Unrecognized weight format in bad\.safetensors"):
                qwen_image_generate_image.load_dit_model(args, torch.device("cpu"))
            mock_load_qwen_model.assert_not_called()

    def test_zimage_on_the_fly_rejects_unknown_format(self):
        args = SimpleNamespace(
            blocks_to_swap=0,
            prefer_lycoris=False,
            lora_weight=["bad.safetensors"],
            include_patterns=None,
            exclude_patterns=None,
        )
        with (
            patch.object(zimage_generate_image, "load_file", return_value=self._unknown_sd()),
            patch.object(zimage_generate_image.zimage_model, "load_zimage_model") as mock_load_zimage_model,
        ):
            with self.assertRaisesRegex(ValueError, r"Unrecognized weight format in bad\.safetensors"):
                zimage_generate_image.load_dit_model(args, torch.device("cpu"))
            mock_load_zimage_model.assert_not_called()

    def test_hv_1_5_on_the_fly_rejects_unknown_format(self):
        args = SimpleNamespace(
            blocks_to_swap=0,
            prefer_lycoris=False,
            include_patterns=None,
            exclude_patterns=None,
        )
        with (
            patch.object(hv_1_5_generate_video, "load_file", return_value=self._unknown_sd()),
            patch.object(hv_1_5_generate_video, "load_hunyuan_video_1_5_model") as mock_load_hv15_model,
        ):
            with self.assertRaisesRegex(ValueError, r"Unrecognized weight format in bad\.safetensors"):
                hv_1_5_generate_video.load_dit_model(
                    args,
                    "dit.safetensors",
                    ["bad.safetensors"],
                    [1.0],
                    torch.device("cpu"),
                )
            mock_load_hv15_model.assert_not_called()

    def test_framepack_on_the_fly_rejects_unknown_format_before_conversion(self):
        args = SimpleNamespace(
            blocks_to_swap=0,
            prefer_lycoris=False,
            lora_weight=["bad.safetensors"],
            include_patterns=None,
            exclude_patterns=None,
        )
        with (
            patch.object(fpack_generate_video, "load_file", return_value=self._unknown_sd()),
            patch.object(fpack_generate_video, "convert_lora_for_framepack") as mock_convert_lora_for_framepack,
            patch.object(fpack_generate_video, "load_packed_model") as mock_load_packed_model,
        ):
            with self.assertRaisesRegex(ValueError, r"Unrecognized weight format in bad\.safetensors"):
                fpack_generate_video.load_dit_model(args, torch.device("cpu"))
            mock_convert_lora_for_framepack.assert_not_called()
            mock_load_packed_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
