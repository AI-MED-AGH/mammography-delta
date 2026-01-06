from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Constants (modify after selecting the best hyperparameters for each model
RANDOM_STATE = 42


def train_lr(X_train, y_train):
    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_rf(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    return model


# ------------ Temporary Test Function ------------

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print(f"Model: {model}")
    print(
        f"""
    Accuracy: {accuracy:.3f}
    Recall: {recall:.3f}
    Precision: {precision:.3f}
    F1-Score: {f1:.3f}
    """)
    print(
        f"""
    Confusion Matrix: {matrix}
    """)
    print(
        f"""
    AUC: {auc:.3f}
    """)
