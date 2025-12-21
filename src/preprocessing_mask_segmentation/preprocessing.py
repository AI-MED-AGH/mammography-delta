# %%
import cv2
import numpy as np
import os


# %% [markdown]
# create function to transform image into numpy array
# Mask are binarizated but below are functions to do binarization anyway

# %%
def load_image(path: str) -> np.array:
    """
    Loads an image from the specified file path using OpenCV.

    Args:
        path (str): The file path to the image.

    Returns:
        np.array: The loaded image as a NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        ValueError: If the image could not be loaded (e.g., corrupted file).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Image could not be loaded.")
    return image


def show_image(image: np.array) -> None:
    """
    Displays an image in a window and waits for a key press to close it.

    Args:
        image (np.array): The image to display.
        title (str, optional): The title of the window. Defaults to "Image".
    """
    if not image.any():
        print("Image is None")
        return
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mask_binarization(image: np.array, grey_scale=128, max_val=255, type=cv2.THRESH_BINARY) -> np.array:
    """
    Applies thresholding to an image to create a binary mask.
    Automatically converts color images (BGR) to grayscale before processing.

    Args:
        image (np.array): Input image (color or grayscale).
        grey_scale (int, optional): The threshold value used to classify pixel values. Defaults to 128.
        max_val (int, optional): The value assigned to pixels exceeding the threshold. Defaults to 255.
        type (int, optional): The OpenCV thresholding type. Defaults to cv2.THRESH_BINARY.

    Returns:
        np.array: A binary mask where pixels are either 0 or max_val.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    th, im_th = cv2.threshold(image, grey_scale, max_val, type)
    return im_th


# %%
# test1=load_image(path="../../images/1001.png")
#
# binarizated = mask_binarization(test1)
# show_image(binarizated)


# %%
def get_largest_connected_component(image: np.array):
    """
    Extracts the largest connected component (object) from a binary image.

    Args:
        image (np.array): A binary image (mask).

    Returns:
        np.array: A binary mask containing only the largest connected component. 
                  Returns a black image if no components are found.
    """
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    if nb_components <= 1:
        return np.zeros_like(image)

    sizes = stats[1:, cv2.CC_STAT_AREA]
    max_label = np.argmax(sizes) + 1
    largest_component_mask = np.zeros(output.shape, dtype=np.uint8)
    largest_component_mask[output == max_label] = 255

    return largest_component_mask


def filter_by_area(binary_image: np.array, min_area: int) -> np.array:
    """
    Filters connected components in a binary image based on a minimum area threshold.

    Args:
        binary_image (np.array): A binary image (mask).
        min_area (int): The minimum area (in pixels) required to keep a component.

    Returns:
        np.array: A binary mask containing only components larger than min_area.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8, ltype=cv2.CV_32S
    )

    filtered_mask = np.zeros_like(binary_image)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            filtered_mask[labels == i] = 255

    return filtered_mask


def clean_mask(path: str, min_area: int = 0, only_largest: bool = False) -> np.array:
    """
    Main pipeline function to generate a cleaned binary mask from an image file.
    
    It orchestrates the loading, binarization, and filtering processes.

    Args:
        path (str): The file path to the source image.
        min_area (int, optional): The minimum pixel area to retain a component. 
                                  Used if only_largest is False. Defaults to 0.
        only_largest (bool, optional): If True, returns only the single largest connected component,
                                       ignoring min_area. Defaults to False.

    Returns:
        np.array: The final processed binary mask.
    """

    image = load_image(path)
    binary_mask = mask_binarization(image)

    if only_largest:
        return get_largest_connected_component(binary_mask)
    elif min_area > 0:
        return filter_by_area(binary_mask, min_area)
    else:
        return binary_mask


# %%
"""Example of 4 cleaned_masks their are really similar to original ones"""

# for i in range(1,5):
#     test1 = clean_mask(path=f"../../images/100{i}.png")
#     show_image(test1)
