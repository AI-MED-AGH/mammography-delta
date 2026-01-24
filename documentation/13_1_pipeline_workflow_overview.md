## Brief Pipeline Workflow Overview

This document outlines the high-level data flow within the `pipeline.ipynb` notebook. The process is divided into
logical processing stages, describing the transformation from raw images to final model evaluation.

### 1. Initialization and Configuration

* **Setup:** Importing necessary libraries (NumPy, Pandas, Scikit-learn, OpenCV).
* **Configuration:** Defining global constants (file paths, random seeds like `RANDOM_STATE` for reproducibility).
* **Modules:** Loading external helper modules (`preprocessing`, `features`, `utils`) to keep the notebook clean.

### 2. Image Loading and Preprocessing

* **Data Ingestion:** Iterating through raw image and mask files.
* **Mask Cleaning:** Applying the `clean_mask` function to:
* Binarize the image (0/255).
* Remove noise and artifacts.
* Smooth edges using morphological opening.

* **Standardization:** Selecting the Region of Interest (ROI) by isolating the largest connected component or resizing
  images if necessary.

### 3. Feature Extraction

* **Vectorization:** converting each binary mask into a numerical feature vector.
* **Geometric Features:** Calculating shape descriptors such as *Area*, *Perimeter*, *Circularity*, *Roundness*, and
  *Irregularity Index*.
* **Invariant Moments:** Computing *Hu Moments* to capture shape characteristics independent of scale, rotation, or
  translation.
* **Aggregation:** Storing the calculated features in a tabular structure (Pandas DataFrame).

### 4. Dataset Preparation

* **Labeling:** Merging extracted features with their corresponding class labels (e.g., *Benign* vs. *Malignant*).
* **Cleaning:** Handling missing data (if any) and removing erroneous samples to ensure dataset integrity.

### 5. Patient-Aware Data Split

* **Splitting:** Dividing the dataset into training and testing sets.
* **Isolation Strategy:** Utilizing `StratifiedGroupKFold` (or similar logic) to ensure that **all images from a single
  patient** are assigned exclusively to either the train or test set. This is a critical step to prevent data leakage.

### 6. Modeling and Training

* **Scaling:** Normalizing features (e.g., using `StandardScaler`) to ensure all variables contribute equally to the
  model.
* **Training:** Training machine learning classification models (e.g., SVM, Random Forest) on the training set.

### 7. Evaluation

* **Prediction:** Running the trained model on the separated test set.
* **Metrics:** Generating performance metrics including *Accuracy*, *Precision*, *Recall*, and *F1-Score*.
* **Visualization:** Creating confusion matrices to visually assess classification performance.

---
