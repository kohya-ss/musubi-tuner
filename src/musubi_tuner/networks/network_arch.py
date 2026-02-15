"""Architecture detection and configuration for network modules (LoHa, LoKr, etc.)."""

import logging

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_FLUX_2_DEV,
    ARCHITECTURE_FLUX_2_KLEIN_4B,
    ARCHITECTURE_FLUX_2_KLEIN_9B,
    ARCHITECTURE_FLUX_KONTEXT,
    ARCHITECTURE_FRAMEPACK,
    ARCHITECTURE_HUNYUAN_VIDEO,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5,
    ARCHITECTURE_KANDINSKY5,
    ARCHITECTURE_QWEN_IMAGE,
    ARCHITECTURE_QWEN_IMAGE_EDIT,
    ARCHITECTURE_QWEN_IMAGE_LAYERED,
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

logger = logging.getLogger(__name__)
# Note: logging.basicConfig removed to avoid conflicts with BlissfulLogger - configure at entry points

# Architecture registry: single source of truth for LoHa/LoKr defaults.
# target_modules are imported from lora_*.py (authoritative source).
# exclude_patterns / include_patterns match what each lora_*.py applies by default.
# exclude_mod_patterns is Qwen-specific: only appended when exclude_mod=True (default).
ARCH_CONFIGS = {
    ARCHITECTURE_WAN: {
        "target_modules": WAN_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*"],
    },
    ARCHITECTURE_HUNYUAN_VIDEO: {
        "target_modules": HUNYUAN_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(img_mod|txt_mod|modulation).*"],
    },
    ARCHITECTURE_HUNYUAN_VIDEO_1_5: {
        "target_modules": HV_1_5_IMAGE_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(_in).*"],
    },
    ARCHITECTURE_FRAMEPACK: {
        "target_modules": FRAMEPACK_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(norm).*"],
    },
    ARCHITECTURE_FLUX_KONTEXT: {
        "target_modules": FLUX_KONTEXT_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*", r".*(norm).*"],
    },
    ARCHITECTURE_FLUX_2_DEV: {
        "target_modules": FLUX_2_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*", r".*(norm).*"],
    },
    ARCHITECTURE_FLUX_2_KLEIN_4B: {
        "target_modules": FLUX_2_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*", r".*(norm).*"],
    },
    ARCHITECTURE_FLUX_2_KLEIN_9B: {
        "target_modules": FLUX_2_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*", r".*(norm).*"],
    },
    ARCHITECTURE_QWEN_IMAGE: {
        "target_modules": QWEN_IMAGE_TARGET_REPLACE_MODULES,
        "exclude_patterns": [],
        "exclude_mod_patterns": [r".*(_mod_).*"],
    },
    ARCHITECTURE_QWEN_IMAGE_EDIT: {
        "target_modules": QWEN_IMAGE_TARGET_REPLACE_MODULES,
        "exclude_patterns": [],
        "exclude_mod_patterns": [r".*(_mod_).*"],
    },
    ARCHITECTURE_QWEN_IMAGE_LAYERED: {
        "target_modules": QWEN_IMAGE_TARGET_REPLACE_MODULES,
        "exclude_patterns": [],
        "exclude_mod_patterns": [r".*(_mod_).*"],
    },
    ARCHITECTURE_Z_IMAGE: {
        "target_modules": ZIMAGE_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(_modulation|_refiner).*"],
    },
    ARCHITECTURE_KANDINSKY5: {
        "target_modules": KANDINSKY5_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*modulation.*"],
        "include_patterns": [
            r".*self_attention\.to_query.*",
            r".*self_attention\.to_key.*",
            r".*self_attention\.to_value.*",
            r".*self_attention\.out_layer.*",
            r".*cross_attention\.to_query.*",
            r".*cross_attention\.to_key.*",
            r".*cross_attention\.to_value.*",
            r".*cross_attention\.out_layer.*",
            r".*feed_forward\.in_layer.*",
            r".*feed_forward\.out_layer.*",
        ],
    },
}

SUPPORTED_ARCHITECTURES = list(ARCH_CONFIGS.keys())


def get_arch_config(architecture: str) -> dict:
    """Return config dict for given architecture. Raises ValueError if unsupported."""
    if architecture not in ARCH_CONFIGS:
        supported_list = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
        raise ValueError(f"Architecture '{architecture}' is not supported by LoHa/LoKr. Supported: {supported_list}")
    return ARCH_CONFIGS[architecture]
