import pandas as pd
import numpy as np
import cv2
import os
from skimage.measure import label, regionprops_table
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# --- KONFIGURACJA ---
CSV_PATH = 'labels.csv'
IMAGE_DIR = 'images'  # Upewnij się, że ścieżka jest poprawna


def get_clean_hybrid_data(csv_path, img_dir):
    df = pd.read_csv(csv_path)

    # KROK 1: CZYSZCZENIE DANYCH (Data Cleaning)
    # Usuwamy Assessment 1 (w mammografii to "Negative", a u nas ma etykietę MALIGNANT - ewidentny błąd danych)
    initial_len = len(df)
    df = df[df['assessment'] != 1].reset_index(drop=True)
    print(f"Usunięto {initial_len - len(df)} błędnych wpisów z Assessment=1.")

    properties = ['area', 'perimeter', 'solidity', 'extent', 'major_axis_length', 'minor_axis_length']
    data = []
    valid_indices = []

    print(f"Przetwarzanie {len(df)} obrazów...")

    for index, row in df.iterrows():
        file_path = os.path.join(img_dir, f"{row['id']}.png")
        if not os.path.exists(file_path): continue

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        # Lekki preprocessing (tylko zamykanie dziur)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        if cv2.countNonZero(mask) < 50: continue
        label_img = label(mask)
        if label_img.max() == 0: continue

        props = regionprops_table(label_img, properties=properties)
        props_df = pd.DataFrame(props)
        region = props_df.sort_values(by='area', ascending=False).iloc[0]

        # Inżynieria cech
        circularity = (4 * np.pi * region['area']) / (region['perimeter'] ** 2) if region['perimeter'] > 0 else 0
        norm_perimeter = region['perimeter'] / np.sqrt(region['area']) if region['area'] > 0 else 0
        aspect_ratio = region['minor_axis_length'] / region['major_axis_length'] if region[
                                                                                        'major_axis_length'] > 0 else 0

        features = {
            # Cechy obrazu (słabe, ale pomocne przy Assessment 4)
            'solidity': region['solidity'],
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'norm_perimeter': norm_perimeter,
            'extent': region['extent'],

            # Cechy kliniczne (SILNE)
            'assessment': row['assessment']
        }

        data.append(features)
        valid_indices.append(index)

    X = pd.DataFrame(data)
    df_filtered = df.loc[valid_indices].reset_index(drop=True)

    return X, df_filtered


# --- START ---
X, df_meta = get_clean_hybrid_data(CSV_PATH, IMAGE_DIR)
y = df_meta['pathology']
groups = df_meta['patient_id']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- WALIDACJA KRZYŻOWA (Cross-Validation) ---
# Zamiast jednego podziału, użyjmy CV, żeby wynik był bardziej wiarygodny do raportu
cv = StratifiedGroupKFold(n_splits=5)

clf = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=300,
    l2_regularization=5.0,  # Silna regularyzacja, żeby nie przeuczyć na obrazkach
    categorical_features=[X.columns.get_loc('assessment')],  # Wskazujemy, że to kategoria
    random_state=42
)

print("\nRozpoczynam 5-krotną walidację krzyżową (to potrwa chwilę)...")
y_pred = cross_val_predict(clf, X, y_encoded, groups=groups, cv=cv, n_jobs=-1)

# --- FINALNY RAPORT ---
print("\n" + "=" * 50)
print(" OSTATECZNY MODEL HYBRYDOWY (Final Deliverable)")
print("=" * 50)
print(f"Dokładność (Accuracy): {accuracy_score(y_encoded, y_pred):.4f}")
print(f"ROC AUC Score:         {roc_auc_score(y_encoded, y_pred):.4f}")
print("-" * 50)
print(classification_report(y_encoded, y_pred, target_names=le.classes_))
print("Macierz pomyłek:\n", confusion_matrix(y_encoded, y_pred))

# Ważność cech (trenujemy raz na całości, żeby pobrać wagi)
clf.fit(X, y_encoded)
from sklearn.inspection import permutation_importance

r = permutation_importance(clf, X, y_encoded, n_repeats=10, random_state=42)
importances = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
print("\nCo steruje modelem (Feature Importance):")
print(importances)






































import os
from typing import Optional, Union, Any

import numpy as np

# Image segmentation
from scipy import ndimage

# Image processing and feature extraction modules
import skimage.measure as skm
import skimage as ski

from skimage import morphology
import cv2
from typing import Optional, Union, Any

# Temporary import image function for testing purposes
def import_images(img_names: list, folder_path: str) -> dict:
    """
    Import all images from a folder path

    Args:
        img_names (list): A list of image names.
        folder_path (str): A folder path of the folder containing images.
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

def calculate_fractal_dimension(Z: np.array) -> float:
    """
    Calculates the Fractal Dimension (Minkowski–Bouligand dimension) using the Box-counting method.
    Measures the complexity/'roughness' of the border.

    Args:
        Z (np.array): Binary image (2D).

    Returns:
        float: The fractal dimension (slope of the log-log plot).
    """
    # Ensure binary and squeeze to 2D
    Z = (Z > 0)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (powers of 2 descending)
    sizes = 2**np.arange(n, 1, -1)

    # Box counting loop
    counts = []
    for size in sizes:
        # Reduce the image by summing blocks of size k*k
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], size), axis=0),
            np.arange(0, Z.shape[1], size), axis=1)

        # Count non-empty boxes
        counts.append(len(np.where((S > 0))[0]))

    # Fit a line to the log-log plot
    if len(counts) < 2 or counts[0] == 0:
        return 0.0

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

    # The fractal dimension is the negative slope
    return -coeffs[0]

def extract_shape_features(img: np.array) -> Optional[dict[Union[str, Any], Union[float, Any]]]:
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

    # Convert image into logical type (True/False) required in binary_fill_holes
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

    # ---------------- Advanced Topological Calculations ----------------

    # Use cropped image of the lesion for faster processing
    lesion_mask = main_lesion.image

    # 1. Skeletonization (Topology)
    # Reduces the object to a 1-pixel wide skeleton to measure "branching".
    skeleton = morphology.skeletonize(lesion_mask)
    skeleton_len = np.sum(skeleton)

    # 2. Fractal Dimension (Complexity)
    # Measures how "chaotic" the border is using Box Counting method.
    fd_value = calculate_fractal_dimension(lesion_mask)

    # 3. Convexity Defects (Deep Indentations)
    # Counts how many deep "valleys" exist in the contour.
    lesion_mask_uint8 = (lesion_mask.astype(np.uint8) * 255)
    lesion_mask_uint8 = cv2.copyMakeBorder(lesion_mask_uint8, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

    contours, _ = cv2.findContours(lesion_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_deep_defects = 0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull_indices = cv2.convexHull(cnt, returnPoints=False)

        if hull_indices is not None and len(hull_indices) > 3 and len(cnt) > 3:
            defects = cv2.convexityDefects(cnt, hull_indices)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    depth = d / 256.0  # Depth in pixels
                    # Count only defects deeper than 5% of the diameter to ignore noise
                    if depth > (main_lesion.equivalent_diameter * 0.05):
                        num_deep_defects += 1

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
        "Shape Factor": (safe_perimeter ** 2) / safe_area,

        # Ratio of object area to image size.
        "Relative Area": safe_area / img.size,

        # ---------------- Topological Features ----------------

        # Measures complexity/roughness of the border using Box Counting.
        # Higher values indicate more chaotic, cancer-like borders.
        "Fractal Dimension": fd_value,

        # Ratio of skeleton length to area.
        # High values indicate thin, branching structures (spicules).
        "Skeleton Ratio": skeleton_len / safe_area,

        # Number of significant indentations in the contour.
        # Indicates star-like shapes common in malignancy.
        "Deep Defects Count": num_deep_defects
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