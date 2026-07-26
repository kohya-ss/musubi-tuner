import unittest

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_IDEOGRAM4,
    ARCHITECTURE_MAGE_FLOW,
    ARCHITECTURE_MAGE_FLOW_EDIT,
)
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

    def test_build_metadata_supports_mage_flow_lora(self):
        metadata = sai_model_spec.build_metadata(
            None,
            ARCHITECTURE_MAGE_FLOW,
            0,
            title="mage_flow_lora_test",
        )

        self.assertEqual(metadata["modelspec.architecture"], "Mage-Flow/lora")
        self.assertEqual(metadata["modelspec.implementation"], "https://github.com/microsoft/Mage")

    def test_build_metadata_supports_mage_flow_edit_lora(self):
        metadata = sai_model_spec.build_metadata(
            None,
            ARCHITECTURE_MAGE_FLOW_EDIT,
            0,
            title="mage_flow_edit_lora_test",
        )

        self.assertEqual(metadata["modelspec.architecture"], "Mage-Flow-Edit/lora")
        self.assertEqual(metadata["modelspec.implementation"], "https://github.com/microsoft/Mage")


if __name__ == "__main__":
    unittest.main()
