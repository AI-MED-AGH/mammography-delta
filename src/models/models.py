from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# Constants (modify after selecting the best hyperparameters for each model
# RANDOM_STATE = 42

def train_lr(X_train, y_train):
    model = LogisticRegression()
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


# Temporary Test Function

from sklearn.metrics import confusion_matrix


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    results = confusion_matrix(y_test, y_pred)
    return results
