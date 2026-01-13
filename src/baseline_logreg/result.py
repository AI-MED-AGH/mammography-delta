def evaluate_baseline_results(results_df):
    """
    Automatically evaluates whether baseline results indicate
    a meaningful predictive signal.
    """
    random_row = results_df[results_df["Model"] == "Random"].iloc[0]
    baseline_row = results_df[results_df["Model"] != "Random"].iloc[0]

    # ROC-AUC check
    if baseline_row["ROC-AUC"] < 0.6:
        print("ROC-AUC close to random, weak or no predictive signal.")

    # Comparison vs random
    better_than_random = (
        baseline_row["Accuracy"] > random_row["Accuracy"]
        and baseline_row["F1-score"] > random_row["F1-score"]
        and baseline_row["ROC-AUC"] > random_row["ROC-AUC"]
    )
    
    if better_than_random:
        print("Baseline model significantly outperforms random baseline.")
    else:
        print("Baseline does NOT outperform random baseline.")
