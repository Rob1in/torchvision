import io

import numpy as np
import torch
from PIL import Image
from viam.media.video import ViamImage

# def get_tensor_from_np_array(np_array: np.ndarray) -> torch.Tensor:
#     """
#     returns an RGB tensor
#     """
#     uint8_tensor = (
#         torch.from_numpy(np_array).permute(2, 0, 1).contiguous()
#     )  # -> to (C, H, W)
#     float32_tensor = uint8_tensor.to(dtype=torch.float32)
#     return uint8_tensor, float32_tensor


class ImageObject:
    def __init__(self, viam_image: ViamImage, pil_image: Image = None, device=None):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if pil_image is not None:
            self.pil_image = pil_image
        if viam_image is not None:
            self.viam_image = viam_image
            self.pil_image = Image.open(io.BytesIO(viam_image.data)).convert(
                "RGB"
            )  # -> in (H, W, C)
        self.np_array = np.array(self.pil_image, dtype=np.uint8)
        self.tensor = torch.from_numpy(self.np_array).permute(2, 0, 1).contiguous()
        self.float32_tensor = self.tensor.to(dtype=torch.float32)
        if device is not None:
            self.tensor = self.tensor.to(device)
            self.float32_tensor = self.float32_tensor.to(device)
