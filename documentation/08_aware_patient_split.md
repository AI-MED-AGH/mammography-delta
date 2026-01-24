## Data Split Functions

This module provides a robust pipeline for splitting data based on patient identity and removing unnecessary columns
from features vector.

### `aware_patient_split`

**Description**
Performs a random train/test split of the dataset while strictly enforcing patient isolation. This ensures that all
records belonging to a specific `patient_id` end up in either the training set or the test set, but never both.

**Key Features**

* **Data Leakage Prevention:** Uses `GroupShuffleSplit` to group data by patient. This prevents the model from "
  memorizing" specific patients if they appear in both sets.
* **Data Cleaning:** Automatically removes specified columns (e.g., metadata like IDs) via `drop_cols` before returning
  the feature sets.
* **Persistence:** Saves the raw split DataFrames to CSV files (`train_split.csv`, `test_split.csv`) for
  reproducibility.

**Parameters**

* `features_vector` (pd.DataFrame): The full dataset containing features, target labels, and patient IDs.
* `drop_cols` (list): A list of column names to exclude from the training features (e.g., 'patient_id').

**Returns**

* `X_train`, `X_test`: Feature vectors for training and testing.
* `y_train`, `y_test`: Target labels for training and testing.
* `is_intersect_empty` (bool): A validation flag (True if no patient overlaps between sets).

---

### `aware_patient_split_stratified_kfold`

**Description**
An advanced splitting strategy that maintains both **patient isolation** and **class balance**. It ensures the ratio of
target classes (e.g., Benign vs. Malignant) remains consistent between training and test sets, which is critical for
imbalanced medical datasets.

**Key Features**

* **Stratified Group Split:** Uses `StratifiedGroupKFold` logic to generate a single train/test split.
* **Patient Aware:** Keeps unique patients separated to strictly avoid data leakage.
* **Stratification:** Preserves the distribution of the target variable (`pathology`) in both sets (approx. 45%
  Malignant in both).

**Parameters**

* `features_vector` (pd.DataFrame): The full dataset containing features, target labels, and patient IDs.
* `drop_cols` (list): A list of column names to exclude from the final feature sets.

**Returns**

* `X_train`, `X_test`: Feature matrices (stratified and grouped).
* `y_train`, `y_test`: Target labels.
* `is_intersect_empty` (bool): Validation flag ensuring strict patient separation.

---