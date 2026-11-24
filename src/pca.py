"""
Principal Component Analysis (PCA) using Truncated SVD.
Based on Crista Arangala, Linear Algebra with Machine Learning and Data (2023), Section 3.4.
"""

import numpy as np


class MyPCA:
    """
    Principal Component Analysis (PCA) via Singular Value Decomposition (SVD).
    
    Parameters:
    -----------
    n_components : int
        Number of principal components to keep (default: 2).
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """
        Fit the model with data matrix X.
        
        Parameters:
        -----------
        X : np.ndarray of shape (N, D)
            Training data, where N is samples and D is features.
        """
        N = X.shape[0]
        # Step 1: Zero-center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: Perform SVD: X_c = U * Sigma * V^T
        U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Store principal directions (rows of V^T or columns of V)
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = Sigma[:self.n_components]

        # Step 3: Compute explained variance ratio: lambda_i = sigma_i^2 / (N - 1)
        variances = (Sigma ** 2) / (N - 1)
        total_variance = np.sum(variances)
        self.explained_variance_ratio_ = variances[:self.n_components] / total_variance

        return self

    def transform(self, X):
        """
        Apply dimensionality reduction on X.
        
        Parameters:
        -----------
        X : np.ndarray of shape (N, D)
        
        Returns:
        --------
        X_proj : np.ndarray of shape (N, n_components)
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        """Fit the model and apply dimensionality reduction."""
        self.fit(X)
        return self.transform(X)