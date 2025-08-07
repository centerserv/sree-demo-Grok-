import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def minimize_entropy(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)  # Add depth to reduce overfitting
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    accuracy = scores.mean()
    return accuracy
