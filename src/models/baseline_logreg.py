from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def build_baseline_logreg(preprocessor):

    return Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])
