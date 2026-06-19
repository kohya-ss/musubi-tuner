import unittest

from musubi_tuner.dataset.architectures import ARCHITECTURE_IDEOGRAM4
from musubi_tuner.utils import sai_model_spec


class SaiModelSpecTest(unittest.TestCase):
    def test_build_metadata_supports_ideogram4_lora(self):
        metadata = sai_model_spec.build_metadata(
            None,
            ARCHITECTURE_IDEOGRAM4,
            0,
            title="ideogram4_lora_test",
        )

        self.assertEqual(metadata["modelspec.architecture"], "Ideogram-4/lora")
        self.assertEqual(metadata["modelspec.implementation"], "https://huggingface.co/Comfy-Org/Ideogram-4")
        self.assertEqual(metadata["modelspec.resolution"], "1024x1024")


if __name__ == "__main__":
    unittest.main()
