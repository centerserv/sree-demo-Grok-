import numpy as np
from sklearn.cluster import KMeans

def generate_hypotheses(X):
    """Generate initial patterns using KMeans clustering."""
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters