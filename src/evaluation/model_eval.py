import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)


def plot_cv_comparison(cv_results_data, scoring, save_path):
    """
    Standardized visualization to compare performance across multiple models
    for different feature vectors. Generates separate plots for each vector.
    """
    if not cv_results_data:
        print("No CV results found.")
        return

    results_df = pd.DataFrame(cv_results_data)

    # Check if the vector column exists
    vector_col = "Vector name: "
    if vector_col not in results_df.columns:
        unique_vectors = ["All Data"]
        results_df[vector_col] = "All Data"
    else:
        unique_vectors = results_df[vector_col].unique()

    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange']

    for vec_name in unique_vectors:
        print(f"\n--- Generating CV Comparison Plots for: {vec_name} ---")
        vec_data = results_df[results_df[vector_col] == vec_name]

        metrics_list = list(scoring.keys())
        # Calculate figure size dynamically
        fig, axes = plt.subplots(len(metrics_list), 1, figsize=(14, 6 * len(metrics_list)))
        fig.suptitle(f'Cross-Validation Performance: {vec_name}', fontsize=16, fontweight='bold')

        if len(metrics_list) == 1:
            axes = [axes]

        for idx, (metric, ax) in enumerate(zip(metrics_list, axes)):
            metric_data = vec_data[vec_data["Metric"] == metric].sort_values("Mean", ascending=False)

            if metric_data.empty:
                continue

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

        final_save_path = save_path.replace(".png", f"_{vec_name.replace(' ', '_').replace(':', '')}.png")
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        plt.savefig(final_save_path, dpi=300)
        plt.show()


def run_detailed_evaluation(model, X_test, y_test, feature_names, model_name="Best Model"):
    """
    Deep-dive analysis for a single model: Metrics, Confusion Matrix,
    ROC Curve, PR Curve, Feature Importance.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.decision_function(X_test)

    # Calculate Key Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # --- ZMIANA UKŁADU NA 2x2 (Czytelniej) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()  # Pozwala odwoływać się jako axes[0], axes[1] itd.

    # 1. Overall Metrics (Bar Chart) - Top Left
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

    # 2. Confusion Matrix (Normalized) - Top Right
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2%', ax=axes[1], cmap='Blues', cbar=False)
    axes[1].set_title(f"Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    # 3. ROC Curve - Bottom Left
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_score(y_test, y_probs):.2f}')
    axes[2].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[2].set_title("ROC Curve")
    axes[2].legend(loc="lower right")

    # 4. Precision-Recall Curve - Bottom Right
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_probs)
    axes[3].plot(rec_curve, prec_curve, color='green', lw=2)
    axes[3].set_title("Precision-Recall Curve")

    plt.tight_layout()
    plt.show()

    # --- FEATURE IMPORTANCE (Naprawa pustego wykresu) ---
    importances = []

    # Próba pobrania ważności
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])

    # Rysujemy TYLKO jeśli mamy dane (eliminuje pusty wykres <Figure size...>)
    if len(importances) > 0:
        plt.figure(figsize=(10, 6))

        # Bezpieczna konwersja feature_names
        f_names = feature_names
        if not isinstance(f_names, list) and hasattr(f_names, 'tolist'):
            f_names = f_names.tolist()
        elif hasattr(f_names, 'columns'):  # Jeśli to DataFrame
            f_names = f_names.columns.tolist()

        feat_importances = pd.Series(importances, index=f_names)
        feat_importances.nlargest(10).sort_values().plot(kind='barh', color='teal')
        plt.title(f"Top Features driving {model_name} predictions")
        plt.xlabel("Relative Importance")
        plt.show()
    else:
        print(f"\n[INFO] Feature importance plot skipped for '{model_name}' (Not supported by this algorithm).")

    # Error Analysis
    analysis_df = pd.DataFrame(X_test, columns=feature_names)
    analysis_df['Actual_Pathology'] = y_test.values
    analysis_df['Model_Prediction'] = y_pred

    fn_cases = analysis_df[(analysis_df['Actual_Pathology'] == 1) & (analysis_df['Model_Prediction'] == 0)]
    fp_cases = analysis_df[(analysis_df['Actual_Pathology'] == 0) & (analysis_df['Model_Prediction'] == 1)]

    return fn_cases, fp_cases