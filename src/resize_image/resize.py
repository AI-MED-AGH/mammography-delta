from PIL import Image, ImageOps
import numpy as np


def resize_with_padding(img, target_size=224):
    """
    Smart resizing with padding strategy:
    1. If the image is LARGER than the target -> Downscale (maintaining aspect ratio) + Pad.
    2. If the image is SMALLER than the target -> NO SCALING. Paste it in the center of a black canvas.

    This approach avoids upscaling artifacts (interpolation) which destroy texture details 
    in small medical ROI images.

    Args:
        img (np.array or PIL.Image): Input image.
        target_size (int): The target width/height (square).

    Returns:
        np.array: Resized and padded image as a NumPy array.
    """

    # Convert NumPy array to PIL Image if necessary
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    w, h = img.size

    # Case 1: Image is larger than target (in at least one dimension) -> Downscale
    if w > target_size or h > target_size:
        # Resize while maintaining aspect ratio and adding black padding
        img = ImageOps.pad(img, (target_size, target_size),
                           method=Image.Resampling.LANCZOS,
                           color='black', centering=(0.5, 0.5))

    # Case 2: Image is smaller than target -> No scaling (Upscaling is forbidden)
    else:
        # Create a blank black canvas of the target size
        new_img = Image.new(img.mode, (target_size, target_size), color='black')

        # Calculate offsets to center the image on the canvas
        offset_x = (target_size - w) // 2
        offset_y = (target_size - h) // 2

        # Paste the original image onto the center (preserving original pixels)
        new_img.paste(img, (offset_x, offset_y))
        img = new_img

    # Convert back to NumPy array for further processing
    return np.array(img)