## Data Analysis

No missing values in CSV. 

All rows do not contain empty values.

Total records in CSV: **1664**

### Class Distribution

| Class         | Count | Distribution          | Percentage |
|:--------------|:-----:|:----------------------|:----------:|
| **BENIGN**    |  901  | `██████████░░░░░░░░░` |   54.15%   |
| **MALIGNANT** |  763  | `████████░░░░░░░░░░░` |   45.85%   |

### Assessment Distribution

| BI-RADS Score | Count | Distribution           | Percentage |
|:-------------:|:-----:|:-----------------------|:----------:|
|     **0**     |  162  | `██░░░░░░░░░░░░░░░░░░` |   9.74%    |
|     **1**     |   3   | `░░░░░░░░░░░░░░░░░░░░` |   0.18%    |
|     **2**     |  89   | `█░░░░░░░░░░░░░░░░░░░` |   5.35%    |
|     **3**     |  358  | `████░░░░░░░░░░░░░░░░` |   21.51%   |
|     **4**     |  689  | `████████░░░░░░░░░░░░` |   41.41%   |
|     **5**     |  363  | `████░░░░░░░░░░░░░░░░` |   21.81%   |

### Number of Patients Examined

Number of actual patients: 888

Number of patients examined once: **266**   

[List of patients (xlsx)](data_analysis/Patients_examined_once.xlsx)

[List of patients (csv)](data_analysis/Patients_examined_once.csv)

![img_1.png](data_analysis/img_1.png)

Number of patients examined more than once: **622**   

[List of patients (xlsx)](data_analysis/Patients_examined_more_than_once.xlsx)

[List of patients (csv)](data_analysis/Patients_examined_more_than_once.csv)

![img.png](data_analysis/img.png)

### Data Analysis Conclusions
No records were rejected. 
Assessment distribution imbalance problem will be solved 
separately, as well as the problem of patients with the same ID in the test 
and training sets (Data Leakage prevention).

We have decided not to reject any record, because
all masks and records in labels.csv file match
and all rows do not contain empty values.

Because of balanced class distribution we have decided
not to implement any class weight function. 

BI-RADS Score assessment distribution in very imbalanced.
When implementing Multiclass Classification model, class weights
function is crucial.
