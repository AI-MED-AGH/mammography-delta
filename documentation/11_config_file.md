## Project Configuration Documentation

This document describes the configuration flags available for the project pipeline.

### Feature Engineering & Data Loading

* **`import_features_instead_of_training`**
    * `0`: **Generate from scratch**. Creates the feature vector from raw data. This process is computationally
      expensive and time-consuming.
    * `1`: **Load from file**. Imports the feature vector from an existing `features.csv` file. Recommended for faster
      testing.

* **`resize_images`**
    * `0`: **Default (No Resize)**. Uses the original image size with a custom safe padding function. Resizing is
      typically reserved for CNN models, which are not currently implemented.
    * `1`: **Resize**. Forces image resizing.

### Model Training & Evaluation

* **`use_classic_aware_split`**
    * `0`: **Advanced Split With Stratification**. Uses a safe splitting function that prevents **data leakage** by
      ensuring a patient with a specific ID does not appear in both `X_train` and `X_test` simultaneously. Additionally,
      performs stratification and ensures that classes are balanced.
    * `1`: **Patient-Aware Split**. Uses a safe splitting function that prevents **data leakage** by ensuring
      a patient with a specific ID does not appear in both `X_train` and `X_test` simultaneously.


* **`perform_hyperparameter_tuning`**
    * `1`: **Default (Enabled)**. Performs hyperparameter optimization for every model and vector. This provides better
      results but is time-consuming.
    * `0`: **Disabled**. Skips hyperparameter tuning to save time.

### Dataset Filtering & Vector Selection

* **`selected assessment`**
    * A list defining which BI-RADS assessment rows are **kept** in the DataFrame (whitelist). Rows with assessments not
      listed here will be removed.
    * **Mapping:**
        * `0`: BI-RADS 0
        * `1`: BI-RADS 1
        * `2`: BI-RADS 2
        * `3`: BI-RADS 3
        * `4`: BI-RADS 4
        * `5`: BI-RADS 5

* **`train_and_evaluate`**
    * A list of feature vectors (strings) to be trained and evaluated. The strings must match the identifiers below
      exactly:
        * `"shape_only"`: Default shape vector (excludes BI-RADS).
        * `"hybrid"`: Hybrid model vector (Shape features + BI-RADS).
        * `"assessment_only"`: Model vector based on BI-RADS assessment only.

## Default Configuration File

```bash
{
  "import_features_instead_of_training": 0,

  "resize_images":  0,

  "use_classic_aware_split": 0,

  "perform_hyperparameter_tuning": 1,

  "selected_assessments": [0, 1, 2, 3, 4, 5],

  "train_and_evaluate": ["hybrid", "shape_only", "assessment_only"]
}
```