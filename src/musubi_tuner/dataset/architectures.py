# Architecture short names cannot contain underscore
ARCHITECTURE_HUNYUAN_VIDEO = "hv"
ARCHITECTURE_HUNYUAN_VIDEO_FULL = "hunyuan_video"
ARCHITECTURE_WAN = "wan"
ARCHITECTURE_WAN_FULL = "wan"
ARCHITECTURE_FRAMEPACK = "fp"
ARCHITECTURE_FRAMEPACK_FULL = "framepack"
ARCHITECTURE_FLUX_KONTEXT = "fk"
ARCHITECTURE_FLUX_KONTEXT_FULL = "flux_kontext"
ARCHITECTURE_FLUX_2_DEV = "f2d"
ARCHITECTURE_FLUX_2_DEV_FULL = "flux_2_dev"
ARCHITECTURE_FLUX_2_KLEIN_4B = "f2k4b"
ARCHITECTURE_FLUX_2_KLEIN_4B_FULL = "flux_2_klein_4b"
ARCHITECTURE_FLUX_2_KLEIN_9B = "f2k9b"
ARCHITECTURE_FLUX_2_KLEIN_9B_FULL = "flux_2_klein_9b"
ARCHITECTURE_QWEN_IMAGE = "qi"
ARCHITECTURE_QWEN_IMAGE_FULL = "qwen_image"
ARCHITECTURE_QWEN_IMAGE_EDIT = "qie"
ARCHITECTURE_QWEN_IMAGE_EDIT_FULL = "qwen_image_edit"
ARCHITECTURE_QWEN_IMAGE_LAYERED = "qil"
ARCHITECTURE_QWEN_IMAGE_LAYERED_FULL = "qwen_image_layered"
ARCHITECTURE_KANDINSKY5 = "k5"
ARCHITECTURE_KANDINSKY5_FULL = "kandinsky5"
ARCHITECTURE_HUNYUAN_VIDEO_1_5 = "hv15"
ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL = "hunyuan_video_1_5"
ARCHITECTURE_Z_IMAGE = "zi"
ARCHITECTURE_Z_IMAGE_FULL = "z_image"
ARCHITECTURE_HIDREAM_O1 = "ho1"
ARCHITECTURE_HIDREAM_O1_FULL = "hidream_o1_image"
ARCHITECTURE_IDEOGRAM4 = "i4"
ARCHITECTURE_IDEOGRAM4_FULL = "ideogram4"
ARCHITECTURE_KREA2 = "kr2"
ARCHITECTURE_KREA2_FULL = "krea2"
ARCHITECTURE_MINIMAX_H3 = "mmh3"
ARCHITECTURE_MINIMAX_H3_FULL = "minimax_h3"


def round_down_frame_count(frame_count: int, architecture: str, vae_frame_stride: int) -> int:
    if architecture == ARCHITECTURE_MINIMAX_H3:
        if frame_count < 5:
            raise ValueError("MiniMax-H3 requires at least 5 frames")
        return 5 + ((frame_count - 5) // 17) * 17
    return 1 + ((frame_count - 1) // vae_frame_stride) * vae_frame_stride
