from typing import Union
from viam.media.video import ViamImage, RawImage
from PIL import Image
from viam.logging import getLogger
from viam.media.video import CameraMimeType
import numpy as np

LOGGER = getLogger(__name__)
SUPPORTED_IMAGE_TYPE = [
    CameraMimeType.JPEG,
    CameraMimeType.PNG,
    CameraMimeType.VIAM_RGBA,
]


def decode_image(image: Union[Image.Image, RawImage, ViamImage]) -> np.ndarray:
    """decode image to BGR numpy array

    Args:
        raw_image (Union[Image.Image, RawImage])

    Returns:
        np.ndarray: BGR numpy array
    """
    if isinstance(image, (RawImage, ViamImage)):
        if image.mime_type not in SUPPORTED_IMAGE_TYPE:
            LOGGER.error(
                f"Unsupported image type: {image.mime_type}. Supported types are {SUPPORTED_IMAGE_TYPE}."
            )
            raise ValueError(f"Unsupported image type: {image.mime_type}.")
        im = Image.open(image.data).convert(
            "RGB"
        )  # convert in RGB png openened in RGBA
        return np.array(im)

    res = image.convert("RGB")
    rgb = np.array(res)
    return rgb
