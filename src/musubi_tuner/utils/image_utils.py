import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional

from musubi_tuner.dataset import image_video_dataset


# prepare image
def preprocess_image(
    image: Image, w: int, h: int, handle_alpha: bool = False
) -> Tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess the image for the model.
    Args:
        image (Image): The input image. RGB or RGBA format.
        w (int): The target bucket width.
        h (int): The target bucket height.
        handle_alpha (bool): Whether to handle alpha channel for tensor and numpy array.
    Returns:
        Tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]:
            - image_tensor: The preprocessed image tensor (NCHW format). -1.0 to 1.0.
            - image_np: The original image as a numpy array (HWC format). 0 to 255.
            - alpha: The alpha channel of the image if present in original size, otherwise None.
    """
    if image.mode == "RGBA":
        alpha = image.split()[-1]
    else:
        alpha = None
    if handle_alpha:
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")

    image_np = np.array(image)  # PIL to numpy, HWC

    image_np = image_video_dataset.resize_image_to_bucket(image_np, (w, h))  # TODO move this to this file
    image_tensor = torch.from_numpy(image_np).float() / 127.5 - 1.0  # -1 to 1.0, HWC
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW -> NCHW, N=1
    return image_tensor, image_np, alpha
