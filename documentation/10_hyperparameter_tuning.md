## Hyperparameter Tuning

This section in the pipeline is responsible for optimizing the configuration of machine learning models to maximize
their performance on  the specific feature vectors (Standard, Hybrid, Assessment-Only). Instead of relying on default 
parameters, we systematically search for the best combination of settings using **Grid Search**.

### Optimization Strategy

We utilize `GridSearchCV` from Scikit-Learn, which search over specified parameter values.

* **Optimization Metric:** The search is driven by **ROC AUC** (Receiver Operating Characteristic Area Under Curve).
  This metric was chosen over accuracy because it better reflects the model's ability to rank suspicious lesions higher 
* than benign ones, regardless of the classification threshold.
* **Internal Validation:** To prevent overfitting to the training set during optimization, the Grid Search employs *
  *Stratified 5-Fold Cross-Validation** (identical to the main evaluation strategy).
* **Parallelization:** The process uses all available CPU cores (`n_jobs=-1`) to accelerate computation.

### Model Search Spaces

The following algorithms and parameter grids are evaluated:

#### 1. Random Forest Classifier

* **Base Config:** `class_weight='balanced'` (to handle dataset imbalance).
* **Tuned Parameters:**
  * `n_estimators`: [100, 200, 300] (Number of trees).
  * `max_depth`: [None, 10, 20] (Tree complexity; limited to prevent overfitting).
  * `min_samples_split` / `min_samples_leaf`: Constraints to ensure leaf nodes represent a sufficient number of samples.

#### 2. Support Vector Machine (SVM) - RBF Kernel

* **Base Config:** `probability=True` (required for ROC AUC calculation), `class_weight='balanced'`.
* **Tuned Parameters:**
  * `C`: [0.1, 1, 10, 100] (Regularization parameter; controls the trade-off between smooth decision boundary and
    classifying training points correctly).
  * `gamma`: ['scale', 0.1, 0.01] (Kernel coefficient; defines how far the influence of a single training example
    reaches).

#### 3. Logistic Regression

* **Base Config:** `solver='liblinear'`, `class_weight='balanced'`.
* **Tuned Parameters:**
  * `C`: [0.01, 0.1, 1, 10, 100] (Inverse of regularization strength).
  * `penalty`: ['l1', 'l2'].
    * **L1 (Lasso):** Can drive weights to zero, effectively performing feature selection (removing useless shape
      descriptors).
    * **L2 (Ridge):** Penalizes large weights but keeps all features.

#### 4. K-Nearest Neighbors (KNN)

* **Tuned Parameters:**
  * `n_neighbors`: [3, 5, 7, 9, 11] (Odd numbers chosen to avoid voting ties).
  * `weights`: ['uniform', 'distance'] (Whether closer neighbors should have a greater influence on the vote).
  * `metric`: ['euclidean', 'manhattan'] (Distance calculation method).

### Outputs

1. **`best_models_per_vector` (Dictionary):**
   A nested dictionary storing the fully configured, optimized model instances for each vector type. This dictionary is
   passed to the subsequent **Cross-Validation** module.

```python
{
    'Shape-Only': {'Random Forest': best_rf_instance, 'SVM': best_svm_instance, ...},
    'Hybrid': {'Random Forest': ...},
    ...
}

```

2. **Tuning Report:**
   A summary DataFrame (`tuning_results_data`) containing:

* Best parameters found.
* Mean CV score (ROC AUC) achieved on the training set.
* Test AUC (sanity check on the hold-out test set).

---