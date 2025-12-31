import pandas as pd
import numpy as np

def drop_high_correlation_features(df, threshold=0.90, exclude=['pathology']):
    """
    Automated feature selection based on Spearman correlation.
    Keeps only the first feature of a highly correlated pair.
    """
    # Select only numeric features for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    if exclude:
        numeric_df = numeric_df.drop(columns=[col for col in exclude if col in numeric_df.columns])
    
    corr_matrix = numeric_df.corr(method='spearman').abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Dropped {len(to_drop)} redundant features due to high correlation (> {threshold}): {to_drop}")
    
    return df.drop(columns=to_drop), to_drop