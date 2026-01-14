import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 49  # Random State was selected to maintain balanced y mean in train and test sets
TEST_SIZE = 0.2
TRAIN_SPLIT_PATH = "../output/train_split.csv"
TEST_SPLIT_PATH = "../output/test_split.csv"

def aware_patient_split(features_vector, drop_cols) -> tuple:
    """
    Split data into train and test sets that is aware of patient's id.
    Rows with the same patient id only go to train or test set - never to both at the same time.

    Args:
        features_vector (pd.DataFrame): A DataFrame containing the features vector.
        drop_cols: (list): A list of columns that will be dropped.
    Returns:
       X_train, X_test, y_train, y_test, is_intersect_empty (tuple):
       Train and test DataFrame sets prepared for model training and True/False aware split test variable.
    """

    groups = features_vector['patient_id'].values
    gss = GroupShuffleSplit(n_splits=1,
                            test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)

    train_idx, test_idx = next(gss.split(features_vector, features_vector['pathology'], groups=groups))

    df_train = features_vector.iloc[train_idx]
    df_test = features_vector.iloc[test_idx]

    features_cols = [c for c in features_vector.columns if c not in drop_cols]

    X_train = df_train[features_cols]
    y_train = df_train['pathology']

    X_test = df_test[features_cols]
    y_test = df_test['pathology']

    df_train.to_csv(TRAIN_SPLIT_PATH, index=False)
    df_test.to_csv(TEST_SPLIT_PATH, index=False)

    is_intersect_empty = set(train_idx).isdisjoint(set(test_idx))

    return X_train, X_test, y_train, y_test, is_intersect_empty


def aware_patient_split_stratified_kfold(features_vector, drop_cols) -> tuple:
    """
    Split data into train and test sets that is aware of patient's id AND stratified.
    Rows with the same patient id only go to train or test set.
    The ratio of classes is preserved.
    It it version with balanced cases
    """
    
    groups = features_vector['patient_id'].values
    y = features_vector['pathology'] 
    

    n_splits = int(1 / TEST_SIZE)
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    
    train_idx, test_idx = next(sgkf.split(features_vector, y, groups=groups))

    df_train = features_vector.iloc[train_idx]
    df_test = features_vector.iloc[test_idx]

    features_cols = [c for c in features_vector.columns if c not in drop_cols]

    X_train = df_train[features_cols]
    y_train = df_train['pathology']

    X_test = df_test[features_cols]
    y_test = df_test['pathology']

    
    df_train.to_csv(TEST_SPLIT_PATH, index=False)
    df_test.to_csv(TRAIN_SPLIT_PATH, index=False)

    is_intersect_empty = set(train_idx).isdisjoint(set(test_idx))

    return X_train, X_test, y_train, y_test, is_intersect_empty

# print(f"Train features: {X_train.shape}")
# print(f"Test features:    {X_test.shape}")

# print("y mean in train:", y_train.mean())
# print("y mean in test: ", y_test.mean())

"""
Train features: (1323, 28)
Test features:    (341, 28)

y mean in train: 0.4580498866213152
y mean in test:  0.4604105571847507

FROM DATA_ANALYSIS:
- BENIGN:     901 - 54.15%
- MALIGNANT:  763 - 45.85%

CONCLUSION:
LABELS DISTRIBUTION BETWEEN TRAIN AND TEST SETS IS 
PRESERVED AT APPROX. 45% OF MALIGNANT IN EACH SET
"""
