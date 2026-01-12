## Exploratory Data Analysis (EDA) 
EDA was performed to validate feature quality and predictive power.
A full-scale EDA was performed on all 28+ extracted features to determine data quality and guide the Feature Selection process.

### 1. Multicollinearity & Redundancy
Using Spearman rank correlation to detect redundant features.
* **Finding**: Extreme redundancy was detected within size-related features (Area, Perimeter, etc.) and between initial Hu Moments.
* **Decision**: We will implement a correlation-based filter to retain only one representative feature per highly correlated group ($\rho > 0.90$).

### 2. Predictive Power & Separability
Analyzing how morphological features differ between Benign and Malignant cases.
* **Finding**: `Circularity` and `Solidity` were confirmed as the strongest morphological indicators of malignancy.
* **Finding**: Lesion position (`Centroid X`, `Centroid Y`) and `Orientation` show no correlation with pathology, as expected in medical imaging. These will be considered for removal.

### 3. Outlier and Scale Report
Identifying extreme values using the IQR method to ensure data quality.
* **Finding**: Significant outliers exist in approx. 10% of the dataset for size metrics.
* **Strategy**: Robust Scaling (StandardScaler or RobustScaler) will be prioritized over simple Min-Max scaling to handle these extreme values without losing information.

### EDA Artifacts:
* **Full Correlation Matrix**: [correlation_matrix.csv](../data_analysis/eda/correlation_matrix.csv)
* **Outlier Statistics**: [outlier_report.csv](../data_analysis/eda/outlier_report.csv)
* **Visualizations**: Stored in `../data_analysis/eda/`.

