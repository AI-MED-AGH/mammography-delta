# Resize images with padding
from PIL import Image, ImageOps
import numpy as np

def resize_with_padding(img, target_size):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    result = ImageOps.pad(img,
                    (target_size, target_size),
                          method=Image.Resampling.LANCZOS,
                          color='black',
                          centering=(0.5, 0.5))

    result = np.array(result)
    return result
