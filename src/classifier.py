"""
Nearest-Mean (Nearest-Centroid) Classifier.
Assigns each test sample to the class with the closest Euclidean centroid.
"""

import numpy as np


class NearestMeanClassifier:
    """A simple minimum-distance-to-mean classifier."""
    def __init__(self):
        self.centroids_ = {}
        self.classes_ = None

    def fit(self, X, y):
        """
        Compute class centroids in the feature space.
        
        Parameters:
        -----------
        X : np.ndarray of shape (N, D)
        y : np.ndarray of shape (N,)
        """
        self.classes_ = np.unique(y)
        for c in self.classes_:
            self.centroids_[c] = np.mean(X[y == c], axis=0)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : np.ndarray of shape (N, D)
        
        Returns:
        --------
        y_pred : np.ndarray of shape (N,)
        """
        predictions = []
        for sample in X:
            # Compute Euclidean distance to each centroid
            distances = [np.linalg.norm(sample - self.centroids_[c]) for c in self.classes_]
            nearest_class = self.classes_[np.argmin(distances)]
            predictions.append(nearest_class)
        return np.array(predictions)

    def score(self, X, y):
        """Return classification accuracy."""
        preds = self.predict(X)
        return np.mean(preds == y)