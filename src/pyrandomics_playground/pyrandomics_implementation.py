import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
import os
from typing import Dict, Tuple

def extract_shape_features_from_mask(mask_path: str, pixel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """
    Extracts 2D morphological (shape) features from a binary mask using PyRadiomics.

    This function is designed for scenarios where the original grayscale medical image
    is unavailable, and only the segmentation mask exists. It bypasses texture
    analysis and focuses solely on geometry by treating the mask as both the image
    and the ROI.

    Args:
        mask_path (str): The file path to the binary mask (e.g., .png, .jpg, .nii).
            The mask is expected to be binary (background=0, ROI>0).
        pixel_spacing (tuple, optional): The physical size of pixels in (x, y, z)
            order. Defaults to (1.0, 1.0, 1.0), which means the results (like Area)
            will be calculated in pixels. If known, provide actual spacing 
            (e.g., (0.07, 0.07, 1.0)) to obtain results in millimeters.

    Returns:
        Dict[str, float]: A dictionary of quantitative 2D morphological descriptors. 
        Keys follow the naming convention `original_shape2D_<FeatureName>`.
        
        Key features include:
        * 'original_shape2D_Elongation': Measure of shape extension (0.0 to 1.0). 
          Values closer to 1.0 indicate a circle/square; lower values indicate elongation.
        * 'original_shape2D_Sphericity': Measure of roundness (0.0 to 1.0). 
          1.0 is a perfect circle. Lower values suggest irregular or spiculated margins.
        * 'original_shape2D_PixelSurface': Area of the ROI. Unit depends on `pixel_spacing` 
          (mm² if spacing is provided, otherwise pixels count).
        * 'original_shape2D_Perimeter': Length of the ROI border.
        * 'original_shape2D_MajorAxisLength': The largest axis length of the enclosing ellipse.

    Raises:
        FileNotFoundError: If the provided mask_path does not exist.
        RuntimeError: If the mask is empty or extraction fails.
    """
    

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"The mask file was not found at: {mask_path}")

    try:
        mask_image = sitk.ReadImage(mask_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load image with SimpleITK: {e}")

    mask_arr = sitk.GetArrayFromImage(mask_image)
    mask_arr_binary = (mask_arr > 0).astype(np.uint8)
    
    if np.sum(mask_arr_binary) == 0:
        raise RuntimeError("The provided mask is empty (contains no ROI pixels).")

    mask_processed = sitk.GetImageFromArray(mask_arr_binary)

    # 4. Inject Metadata (Spacing)
    # Crucial for PNGs which lack physical dimensions.
    mask_processed.SetSpacing(pixel_spacing)
    mask_processed.SetOrigin((0, 0, 0)) # Reset origin for consistency

    # 5. Configure PyRadiomics
    settings = {
        'binWidth': 25,             # Irrelevant for shape, but required to silence warnings
        'interpolator': sitk.sitkNearestNeighbor, # Best for binary masks
        'resampledPixelSpacing': None,
        'force2D': True,            # MANDATORY for 2D images (like mammograms)
        'force2Ddimension': 0       # Usually 0 is correct for SimpleITK loaded PNGs, try 2 if fails
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    
    # Disable all feature classes (texture, firstorder, etc.)
    extractor.disableAllFeatures()
    # Enable only Shape2D
    extractor.enableFeatureClassByName('shape2D')

    # 6. Execute Extraction
    # We pass 'mask_processed' as BOTH the image and the mask.
    # Since we only calculate shape, the "image" content is ignored.
    result_vector = extractor.execute(mask_processed, mask_processed)

    # 7. Clean Result
    # Filter out metadata (keys starting with "diagnostics_")
    shape_features = {
        key: float(value) 
        for key, value in result_vector.items() 
        if not key.startswith("diagnostics_")
    }

    return shape_features

# ---(Example Usage) ---
if __name__ == '__main__':
    IMG_PATH = '../../images'
    names = ['2168', '1538', '1105', '1529', '1142', '1505', '2332']
    for i in names:
        extracted_info = extract_shape_features_from_mask(mask_path=f"{IMG_PATH}/{i}.png")
        print()
        print(extracted_info)