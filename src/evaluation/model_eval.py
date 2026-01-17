import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)


def plot_cv_comparison(cv_results_data, scoring, save_path):
    if not cv_results_data:
        print("No CV results found.")
        return

    results_df = pd.DataFrame(cv_results_data)

    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    metrics_list = list(scoring.keys())
    fig, axes = plt.subplots(len(metrics_list), 1, figsize=(14, 6 * len(metrics_list)))
    fig.suptitle('Models Cross-Validation Performance Comparison', fontsize=16, fontweight='bold')

    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    if len(metrics_list) == 1:
        axes = [axes]

    for idx, (metric, ax) in enumerate(zip(metrics_list, axes)):
        metric_data = results_df[results_df["Metric"] == metric].sort_values("Mean", ascending=False)

        x_pos = np.arange(len(metric_data["Model"]))
        bars = ax.bar(x_pos, metric_data["Mean"],
                      color=colors[idx % len(colors)],
                      edgecolor='black', linewidth=1.2, width=0.6)

        ax.errorbar(x_pos, metric_data["Mean"],
                    yerr=metric_data["Std"],
                    fmt='none', color='black', capsize=5)

        for bar, mean_val, std_val in zip(bars, metric_data["Mean"], metric_data["Std"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{mean_val:.3f}\n±{std_val:.3f}',
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_data["Model"], rotation=30, ha='right', fontsize=10)
        ax.set_title(f'METRIC: {metric.upper()}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()


def run_detailed_evaluation(model, X_test, y_test, feature_names, model_name="Best Model"):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))

    metrics_map = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    bars = axes[0].bar(metrics_map.keys(), metrics_map.values(), color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title(f"Key Metrics\nAccuracy: {acc:.2%}", fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2%', ax=axes[1], cmap='Blues', cbar=False)
    axes[1].set_title(f"Confusion Matrix (Normalized) - {model_name}")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_score(y_test, y_probs):.2f}')
    axes[2].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[2].set_title("ROC Curve")
    axes[2].legend(loc="lower right")

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_probs)
    axes[3].plot(rec_curve, prec_curve, color='green', lw=2)
    axes[3].set_title("Precision-Recall Curve")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"Feature importance not supported for {model_name}.")
        importances = []

    if len(importances) > 0:
        feat_importances = pd.Series(importances, index=feature_names)
        feat_importances.nlargest(10).sort_values().plot(kind='barh', color='teal')
        plt.title(f"Top Features driving {model_name} predictions")
        plt.xlabel("Relative Importance")
        plt.show()

    analysis_df = pd.DataFrame(X_test, columns=feature_names)
    analysis_df['Actual_Pathology'] = y_test.values
    analysis_df['Model_Prediction'] = y_pred

    fn_cases = analysis_df[(analysis_df['Actual_Pathology'] == 1) & (analysis_df['Model_Prediction'] == 0)]
    fp_cases = analysis_df[(analysis_df['Actual_Pathology'] == 0) & (analysis_df['Model_Prediction'] == 1)]

    return fn_cases, fp_cases
