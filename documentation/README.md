# Technical Documentation: Mammography-Delta

This repository contains detailed explanations of the algorithms, data pipelines, and machine learning methodologies
used to classify breast lesions based
on mammographic imaging.

## Contents

- Library onboarding (overview of used Python libraries)
- Preprocessing and feature engineering notes
- Modeling and evaluation notes etc.

## Documentation Structure

| File / Section                                                               | Description                                                                                                   |
|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **[01_setup.md](01_setup.md)**                                               | Environment setup, Python versioning (3.9), and library installation guide.                                   |
| **[02_library_onboarding.md](02_library_onboarding.md)**                     | Tech stack overview (PyRadiomics, OpenCV, Scikit-Learn) and justification.                                    |
| **[03_data_analysis.md](03_data_analysis.md)**                               | Initial dataset overview, class distribution analysis, and raw data inspection.                               |
| **[04_mask_preprocessing.md](04_mask_preprocessing.md)**                     | Details on image cleaning, binarization, artifact removal, and edge smoothing algorithms.                     |
| **[05_shape_features.md](05_shape_features.md)**                             | Definitions of extracted quantitative features (Hu Moments, Circularity, etc.) and mathematical formulas.     |
| **[06_eda.md](06_eda.md)**                                                   | Exploratory Data Analysis including feature distributions, correlation matrices, and visual insights.         |
| **[07_feature_selection.md](07_feature_selection.md)**                       | Methodology for selecting the most relevant features and removing highly correlated variables.                |
| **[08_aware_patient_split.md](08_aware_patient_split.md)**                   | Explanation of the patient-aware splitting strategy (`StratifiedGroupKFold`) to prevent data leakage.         |
| **[09_feature_scaling.md](09_feature_scaling.md)**                           | Methodology for normalizing feature vectors and preparing variants (Standard vs. Hybrid vs. Assessment-only). |
| **[10_hyperparameter_tuning.md](10_hyperparameter_tuning.md)**               | Documentation of the optimization process used to find the best model parameters.                             |
| **[11_cross_validation.md](11_cross_validation.md)**                         | Internal validation strategy using Stratified K-Fold to estimate model stability and error.                   |
| **[12_evaluation.md](12_evaluation.md)**                                     | Final model assessment on the unseen test set, including confusion matrices and performance metrics.          |
| **[13_1_pipeline_workflow_overview.md](13_1_pipeline_workflow_overview.md)** | High-level data flow diagram (Mermaid) and step-by-step description of the entire processing pipeline.        |
| **[13_2_config_file.md](13_2_config_file.md)**                               | Documentation of global configuration constants, file paths, and random state settings.                       |
| **[13_3_resize_image.md](13_3_resize_image.md)**                             | Details on the "Smart Resizing" strategy (padding vs. downscaling) to preserve ROI details.                   |

---

## Quick Start

If you are setting up the project for the first time, please start with **[01_setup.md](01_setup.md)** to configure your
virtual environment and dependencies correctly.

---