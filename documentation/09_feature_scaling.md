## Feature Scaling and Dataset Preparation

Feature Scaling section in the pipeline is responsible for normalizing the feature range across the dataset. We use *
*Standard Scaling** (z-score normalization) to ensure that features with larger magnitudes do not dominate the objective
function of the machine learning models (e.g., SVM, Logistic Regression).

To enable comprehensive model evaluation, the scaling and export process is executed in parallel for **three distinct
feature vectors**. This allows for A/B testing between pure image analysis, clinical assessment, and a hybrid approach.

### Scaling Methodology

We utilize `sklearn.preprocessing.StandardScaler`, which standardizes features by removing the mean and scaling to unit
variance:

* **Fit:** Calculated **only** on the training set to determine  (mean) and  (standard deviation).
* **Transform:** Applied to both the training and testing sets using the parameters derived from the training set. This
  strict separation prevents **data leakage** from the test set.

### Feature Vector Variants

The pipeline prepares three separate variations of the dataset to benchmark model performance:

1. **Standard Feature Vector (`X_scaled`)**

* **Content:** Contains **only** the quantitative shape and texture features extracted from the image masks.
* **Purpose:** To evaluate the pure performance of the Computer Vision/ML algorithm without human input.


2. **Hybrid Feature Vector (`X_hybrid_scaled`)**

* **Content:** A combination of the **Standard Feature Vector** AND the clinical **BI-RADS assessment**.
* **Purpose:** To test if adding AI-extracted features to the radiologist's assessment improves diagnostic accuracy
  compared to the assessment and features alone.


3. **Assessment-Only Vector (`X_assessment_only_scaled`)**

* **Content:** Contains **only** the clinical BI-RADS assessment.
* **Purpose:** Serves as a **baseline** (control group) to represent the current standard of care (radiologist's
  performance).

### Output

The final scaled datasets are exported to CSV files, ready for model training:

* **Train Sets:** `train_scaled.csv`, `train_hybrid_scaled.csv`, `train_assessment_scaled.csv`
* **Test Sets:** `test_scaled.csv`, `test_hybrid_scaled.csv`, `test_assessment_scaled.csv`
* **Target Labels:** Corresponding `y_train` and `y_test` files are saved for each set.

---