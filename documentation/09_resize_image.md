### `resize_with_padding`

**Description**

Standardizes input images to a fixed square size (default: 224x224 px) while preserving the original aspect ratio. This function implements a specific strategy to handle different image resolutions without introducing quality loss. It is advisable to use the function only when CNN is used. Otherwise some shape-related features might be distorted.

**Key Features**

* **Large Images:** Downscaled using high-quality Lanczos resampling with black padding to maintain the aspect ratio.
* **Small Images:** No scaling is applied. The original image is centered on a black canvas. This strictly avoids upscaling artifacts (interpolation), which is critical for preserving texture details in small Regions Of Interests.

**Parameters**

* `img` (np.array | PIL.Image): The input image to be processed.
* `target_size` (int): The target width and height of the output image (default: 224).

**Returns**

* `PIL.Image`: The processed, resized, and padded image.
