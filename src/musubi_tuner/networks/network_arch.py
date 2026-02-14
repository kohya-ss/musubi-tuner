"""Architecture detection and configuration for network modules (LoHa, LoKr, etc.)."""

import logging

logger = logging.getLogger(__name__)


def detect_arch_config(unet):
    """Detect architecture from model structure.

    Returns: (target_replace_modules, default_exclude_patterns)
    """
    module_class_names = set()
    for module in unet.modules():
        module_class_names.add(type(module).__name__)

    # Order matters for disambiguation
    if "WanAttentionBlock" in module_class_names:
        from .lora_wan import WAN_TARGET_REPLACE_MODULES

        return WAN_TARGET_REPLACE_MODULES, [r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*"]

    if "QwenImageTransformerBlock" in module_class_names:
        from .lora_qwen_image import QWEN_IMAGE_TARGET_REPLACE_MODULES

        return QWEN_IMAGE_TARGET_REPLACE_MODULES, [r".*(_mod_).*"]

    if "ZImageTransformerBlock" in module_class_names:
        from .lora_zimage import ZIMAGE_TARGET_REPLACE_MODULES

        return ZIMAGE_TARGET_REPLACE_MODULES, [r".*(_modulation|_refiner).*"]

    if "HunyuanVideoTransformerBlock" in module_class_names:
        from .lora_framepack import FRAMEPACK_TARGET_REPLACE_MODULES

        return FRAMEPACK_TARGET_REPLACE_MODULES, [r".*(norm).*"]

    # Kandinsky5 is not supported in auto-detection (uses include_patterns, requires special handling)

    if "DoubleStreamBlock" in module_class_names:
        # FLUX Kontext and FLUX 2 share same target/exclude config
        from .lora_flux import FLUX_KONTEXT_TARGET_REPLACE_MODULES

        return FLUX_KONTEXT_TARGET_REPLACE_MODULES, [
            r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*",
            r".*(norm).*",
        ]

    if "MMSingleStreamBlock" in module_class_names:
        # HunyuanVideo (has both MMDoubleStreamBlock and MMSingleStreamBlock)
        from .lora import HUNYUAN_TARGET_REPLACE_MODULES

        return HUNYUAN_TARGET_REPLACE_MODULES, [r".*(img_mod|txt_mod|modulation).*"]

    if "MMDoubleStreamBlock" in module_class_names:
        # HunyuanVideo 1.5 (only MMDoubleStreamBlock, no MMSingleStreamBlock)
        from .lora_hv_1_5 import HV_1_5_IMAGE_TARGET_REPLACE_MODULES

        return HV_1_5_IMAGE_TARGET_REPLACE_MODULES, [r".*(_in).*"]

    raise ValueError(f"Cannot auto-detect architecture. Module classes found: {sorted(module_class_names)}")
