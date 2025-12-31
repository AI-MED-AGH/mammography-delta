import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_correlation_matrix(df, columns, title="Feature Correlation Matrix (Spearman)", save_path=None, csv_path=None):
    """
    Calculates Spearman correlation and visualizes it with numerical coefficients.
    Saves the correlation matrix to CSV if requested.
    """
    corr_df = df[columns].corr(method='spearman')
    
    # Save matrix to CSV for documentation
    if csv_path:
        corr_df.to_csv(csv_path)
        
    corr_values = corr_df.values
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_values, cmap='coolwarm', vmin=-1, vmax=1)
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    
    for i in range(len(columns)):
        for j in range(len(columns)):
            val = corr_values[i, j]
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)
    
    ax.set_title(title, fontsize=14, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_class_distributions(df, features, target='pathology', save_path=None):
    """
    Visualizes feature distributions grouped by target class using boxplots 
    to estimate class separability.
    """
    n_features = len(features)
    rows = (n_features + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        # Pathology mapping: 0: Benign, 1: Malignant
        benign_data = df[df[target] == 0][col]
        malignant_data = df[df[target] == 1][col]
        
        axes[i].boxplot([benign_data, malignant_data], labels=['Benign', 'Malignant'], patch_artist=True)
        axes[i].set_title(f"Class Distribution: {col}", fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.3)
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_outliers_report(df, columns):
    """
    Identifies outliers using the Interquartile Range (IQR) method.
    """
    report = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        report[col] = {
            'count': len(outliers),
            'percentage': round((len(outliers) / len(df)) * 100, 2)
        }
    return pd.DataFrame(report).T