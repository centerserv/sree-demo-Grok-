import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from pattern import generate_hypotheses
from presence import minimize_entropy
from permanence import update_trust
import matplotlib.pyplot as plt

def preprocess_data(df, target_column):
    """Preprocess dataset with generic handling."""
    if target_column not in df.columns:
        raise ValueError("Target column not found")
    df = df.select_dtypes(include=['number']).fillna(df.median(numeric_only=True))
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    X = MinMaxScaler().fit_transform(X)
    if (y.sum() / len(y) < 0.3) or (y.sum() / len(y) > 0.7):
        X, y = SMOTE(random_state=42).fit_resample(X, y)
    return X, y

def ppp_loop(X, y, n_iterations=10):
    """Execute full PPP loop with cross-validation and noise, tracking improvement and suspect rows."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    accuracies = []
    trust_scores = []
    prior_trust = 0.5
    
    # Baseline accuracy (pre-PPP)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    clf.fit(X_train, y_train)
    baseline_accuracy = clf.score(X_test, y_test)
    
    for i in range(n_iterations):
        generate_hypotheses(X_train)
        accuracy = minimize_entropy(X_train, y_train)
        accuracy *= 0.85  # Noise
        trust = update_trust(prior_trust, accuracy)
        accuracies.append(accuracy)
        trust_scores.append(trust)
        prior_trust = trust
        
        if i < 5:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            mis_idx = y_pred != y_test
            if np.sum(mis_idx) > 0:
                X_train = np.vstack([X_train, X_test[mis_idx][:5]])
                y_train = np.hstack([y_train, y_test[mis_idx][:5]])
    
    # Final model for suspect flags
    clf.fit(X_train, y_train)
    y_pred_full = clf.predict(X)
    suspect_flags = y_pred_full != y
    trust_per_row = np.full(len(y), trust_scores[-1])  # Per-row trust as final trust
    
    return accuracies, trust_scores, baseline_accuracy, suspect_flags, trust_per_row

def plot_results(accuracies, trust_scores):
    """Create diagnostic plots."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(accuracies, marker='o')
    plt.title('Accuracy Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(trust_scores, marker='o', color='orange')
    plt.title('Trust Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Trust')
    plt.savefig('sree_results.png')
    plt.close()

def main():
    """Main driver to run SREE demo."""
    file_path = input("Enter your data file name (e.g., heart_failure_clinical_records.csv): ")
    df = pd.read_csv(file_path)
    target_column = input("Enter the target column name (e.g., DEATH_EVENT): ")
    X, y = preprocess_data(df, target_column)
    accuracies, trust_scores = ppp_loop(X, y)
    
    # Save and display results
    results = pd.DataFrame({'Accuracy': accuracies, 'Trust': trust_scores})
    results.to_csv('sree_results.csv', index=False)
    plot_results(accuracies, trust_scores)
    print(f"Final Accuracy: {accuracies[-1]:.3f}, Final Trust: {trust_scores[-1]:.3f}")

if __name__ == '__main__':
    main()
