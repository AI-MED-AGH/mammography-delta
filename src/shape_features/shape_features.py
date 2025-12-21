import os
import numpy as np

# Image segmentation
from scipy import ndimage

# Image processing and feature extraction modules
import skimage.measure as skm
import skimage as ski


# Temporary import image function for testing purposes
def import_images(img_names: list, folder_path: str) -> dict:
    """
    Import all images from a folder path

    Args:
        img_names (list): A list of image names.

    Returns:
        dict: A dictionary containing np.array image object as value and an image name as a key.
    """

    images = {}
    for image in img_names:
        full_path = os.path.join(folder_path + image)
        try:
            name_key = os.path.basename(full_path).split('.')[0]
            images[name_key] = ski.io.imread(full_path)
        except Exception as e:
            print(e)
    return images


def extract_shape_features(img: np.array) -> dict:
    """
    Extract most important features from an image

    Args:
        img (np.array): A binary image (mask).

    Returns:
        dict: A dictionary containing most important features.
    """

    # Simple input validation
    if not isinstance(img, np.ndarray) or img.size == 0:
        print(f"Received {type(img)} instead of np.array or image is empty.")
        return None

    # Convert img_bool into logical type (True/False) required in binary_fill_holes
    # (OpenCV returns 0/255)
    img_bool = img.astype(bool)

    # Fill holes in the image and label all detected objects
    img_filled = ndimage.binary_fill_holes(img_bool)
    label_img = skm.label(img_filled, connectivity=img_filled.ndim)
    props = skm.regionprops(label_img)

    # If there are no objects detected in the image - return None
    if len(props) == 0:
        return None

    # Choose the largest object
    main_lesion = max(props, key=lambda x: x.area)

    # Epsilon in case of zero division
    eps = 1e-5

    # Function convex_image returns binary mask of the smallest convex polygon.
    convex_image = main_lesion.image_convex.astype(int)
    convex_props = skm.regionprops(skm.label(convex_image))

    # Declare secure variables in case of zero division while calculating features.
    perimeter_convex = convex_props[0].perimeter if len(convex_props) > 0 else eps
    safe_perimeter = main_lesion.perimeter if main_lesion.perimeter > 0 else eps
    safe_major_axis = main_lesion.axis_major_length if main_lesion.axis_major_length > 0 else eps
    safe_minor_axis = main_lesion.axis_minor_length if main_lesion.axis_minor_length > 0 else eps

    # If props variable is not empty, area will is not empty as well.
    safe_area = main_lesion.area

    # a set of 7 numbers that are invariant to scale, translation, and rotation.
    # If the tumor is rotated 90 degrees or is twice as large, the Hu Moments
    # will remain (almost) the same.
    hu_moments = skm.moments_hu(main_lesion.moments_central)

    # Description of features from official scikit-image documentation:
    # https://scikit-image.org/docs/0.25.x/api/skimage.measure.html#skimage.measure.regionprops

    features = {

        # -------------------------------------- Standard features from skimage --------------------------------------

        # Area of the region i.e. number of pixels of the region scaled by pixel-area.
        "Area": safe_area,

        # Area of the bounding box i.e. number of pixels of bounding box scaled by pixel-area.
        "Area Bounding Box": main_lesion.area_bbox,

        # Area of the convex hull image, which is the smallest convex polygon that encloses the region.
        "Area Convex": main_lesion.area_convex,

        # Area of the region with all the holes filled in.
        "Area Filled": main_lesion.area_filled,

        # The length of the major axis of the ellipse that has the same normalized second central moments as the region.
        "Axis Major Length": safe_major_axis,

        # The length of the minor axis of the ellipse that has the same normalized second central moments as the region.
        "Axis Minor Length": safe_minor_axis,

        # Centroid coordinate tuple (row, col).
        # Split centroid list into two separate variables.
        "Centroid X": main_lesion.centroid[1],
        "Centroid Y": main_lesion.centroid[0],

        # Eccentricity of the ellipse that has the same second-moments as the region.
        # The eccentricity is the ratio of the distance between focal points over the major axis length.
        # The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
        "Eccentricity": main_lesion.eccentricity,

        # Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols).
        "Extent": main_lesion.extent,

        # The diameter of a circle with the same area as the region.
        "Equivalent Diameter": main_lesion.equivalent_diameter,

        # Euler characteristic of the set of non-zero pixels.
        # Computed as number of connected components subtracted by number of holes.
        # 1 - no holes in object, 0 - holes detected in object.
        # IMPORTANT - IN PREPROCESSING WE FILLED ALL HOLES SO THIS FEATURE IS USELESS AT THE MOMENT
        # FIX IT LATER
        "Euler Number": main_lesion.euler_number,

        # Maximum Feret’s diameter computed as the longest distance between points
        # around a region’s convex hull contour as determined by find_contours
        "Feret Diameter Max": main_lesion.feret_diameter_max,

        # Perimeter of object which approximates the contour as a line through the centers of border pixels
        # using a 4-connectivity.
        "Perimeter": safe_perimeter,

        # Ratio of pixels in the region to pixels of the convex hull image.
        "Solidity": main_lesion.solidity,

        # Angle between the 0th axis (rows) and the major axis of the ellipse
        # that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.
        "Orientation": main_lesion.orientation,

        # --------------------------------------- Manually calculated features ---------------------------------------


        # Elongation measure. 1.0 = circle/square. High values (> 1.5) indicate an elongated (ellipsoidal) shape.
        "Aspect Ratio": safe_major_axis / safe_minor_axis,

        # Measures roundness + edge smoothness. 1.0 = perfect circle.
        # Highly sensitive to jagged edges (decreases significantly with edge noise).
        # Circularity is very low in case of mallignant tumor.
        # Mask smoothing function in preprocessing module reduces edge noise to increase Circularity
        # result in case of relatively round object. This may increase the discrimination between malignant
        # and benign lesions by increasing Circularity feature range.
        "Circularity": (4 * np.pi * safe_area) / (safe_perimeter ** 2),

        # Ratio of convex hull perimeter to actual perimeter.
        # Near 1.0 = smooth boundary. Low value = deep indentations or spicules (star-shaped).
        "Convexity": perimeter_convex / safe_perimeter,

        # Deviation from a circle. 1.0 = circle.
        # Higher values indicate a complex, rough border compared to the area.
        "Irregularity Index": safe_perimeter / (2 * np.sqrt(np.pi * safe_area + eps)),

        # Measures how well the area fits a circle, ignoring edge roughness.
        # Robust metric: tells if the overall silhouette is round, even if edges are jagged.
        "Roundness": (4 * safe_area) / (np.pi * (safe_major_axis ** 2)),

        # Inverse measure of compactness.
        # Min ~12.57 (circle). Higher values = less compact / more irregular boundary.
        "Shape Factor": (safe_perimeter ** 2) / safe_area
    }

    # Add Hu Moments to dictionary (split list into 7 separate features)
    # Raw Hu Moments can be 10^30 (too large for ML models).
    # We apply a log transform to bring them to a usable range (e.g., -30 to 30)
    for i, hu in enumerate(hu_moments):
        if hu == 0:
            features[f'Hu Moment {i + 1}'] = 0.0
        else:
            # Log transform preserving the sign
            features[f'Hu Moment {i + 1}'] = -1 * np.sign(hu) * np.log10(np.abs(hu))

    return features
