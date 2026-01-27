## Cross-Validation & Model Evaluation

This module performs internal validation of machine learning models using **Stratified K-Fold Cross-Validation**. The goal is to estimate the expected performance and stability of the models on the training data before exposing
them to the final unseen test set.

### Validation Strategy: `StratifiedKFold`

We utilize `StratifiedKFold` with `n_splits=5`.

* **Stratification:** Ensures that the ratio of classes (Benign vs. Malignant) remains constant in every fold. This is
  critical for medical datasets where class imbalance might otherwise lead to biased validation scores.
* **Shuffling:** The data is shuffled (`shuffle=True`, `random_state=42`) before splitting to remove any bias resulting
  from the order of data collection.

### Logic Flow

The code executes a nested evaluation loop:

1. **Iterate Vectors:** Loops through all prepared datasets (Standard, Hybrid, Assessment-only).
2. **Iterate Models:** Loops through classifiers (SVM, Random Forest, KNN, etc.).
3. **Dynamic Model Selection:**

* **Priority:** Checks if a hyperparameter-tuned version of the model exists in `best_models_per_vector`.
* **Fallback:** If no tuned model is found, it defaults to the base configuration (`default_model`).

4. **Execution:** Runs `cross_validate` in parallel (`n_jobs=-1`) to compute multiple metrics simultaneously.

### Output Data Structure

The module aggregates results into a structured list (`cv_results_data`), capturing both the average performance and the
variance (stability) of the models.

| Field           | Description                                                                                                                            |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **Vector name** | The dataset variant used (e.g., 'Standard', 'Hybrid').                                                                                 |
| **Model**       | The algorithm name (e.g., 'SVM', 'Random Forest').                                                                                     |
| **Metric**      | The specific performance metric evaluated (Accuracy, Precision, Recall, F1, ROC-AUC).                                                  |
| **Mean**        | The average score across all 5 folds. High values indicate good performance.                                                           |
| **Std**         | The standard deviation across folds. **Low values** are preferred, as they indicate the model is stable and robust to data variations. |

### Error Handling

The process includes a `try-except` block within the inner loop. If a specific model fails to converge or errors out on
a specific vector, the pipeline logs the error and continues to the next model instead of crashing the entire
experiment.

---
