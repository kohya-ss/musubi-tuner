import numpy as np


def hf_clip_vision_encode(image, feature_extractor, image_encoder):
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8

    if hasattr(feature_extractor, "preprocess"):
        preprocessed = feature_extractor.preprocess(images=image, return_tensors="pt")
    else:
        preprocessed = feature_extractor(images=image, return_tensors="pt")

    if hasattr(preprocessed, "to"):
        preprocessed = preprocessed.to(device=image_encoder.device, dtype=image_encoder.dtype)
    else:
        preprocessed = {
            k: (v.to(device=image_encoder.device, dtype=image_encoder.dtype) if hasattr(v, "to") else v)
            for k, v in preprocessed.items()
        }
    image_encoder_output = image_encoder(**preprocessed)

    return image_encoder_output
