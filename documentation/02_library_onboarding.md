## Library Onboarding – Project Tech Stack

### Libraries used in this project

| Library                   | Why it is used                                             | Where it is used in the project                         |
|---------------------------|------------------------------------------------------------|---------------------------------------------------------|
| **Python 3.9**            | Stable runtime compatible with PyRadiomics and NumPy < 2.0 | Entire project                                          |
| **NumPy (1.26.4)**        | Core numerical computations and array operations           | Masks, feature vectors, mathematical operations         |
| **Pandas (2.2.3)**        | Tabular data handling and CSV operations                   | `labels.csv`, `features.csv`, dataset construction, EDA |
| **SciPy (1.13.1)**        | Scientific computing utilities and statistics              | Feature analysis, optional geometry/statistics          |
| **Matplotlib (3.9.4)**    | Visualization and plotting                                 | EDA plots, correlation matrices, evaluation curves      |
| **Seaborn**               | High-level statistical data visualization                  | Advanced EDA, correlation heatmaps, distribution plots  |
| **scikit-image (0.24.0)** | Image and mask processing utilities                        | Mask preprocessing, region-based shape features         |
| **OpenCV (4.10.0.84)**    | Efficient image resizing and contour operations            | Optional preprocessing, geometry calculations           |
| **Pillow (11.0.0)**       | Image loading and saving                                   | Debug image export, format conversions                  |
| **SimpleITK (2.4.0)**     | Medical image I/O and bridge to PyRadiomics                | Reading images and masks for radiomics                  |
| **PyRadiomics (3.0.1)**   | Standardized radiomics feature extraction                  | Texture, shape, and intensity features (optional)       |
| **scikit-learn (1.5.2)**  | Classical machine learning framework                       | Splits, pipelines, scaling, models, metrics             |
| **joblib**                | Model and pipeline serialization                           | Saving trained models and pipelines                     |
| **pathlib**               | Cross-platform path handling                               | Managing data and output paths                          |
| **tqdm** *(optional)*     | Progress visualization                                     | Long-running feature extraction loops                   |

---

### Installation (see [First Time Setup](01_setup.md)) 
To install all dependencies with the correct versions, run:
```bash
pip install -r requirements.txt
```

---
