"""
Linear Discriminant Analysis (LDA) for Multiclass Dimensionality Reduction.
Based on Crista Arangala, Linear Algebra with Machine Learning and Data (2023), Section 3.6.
Includes the textbook erratum fix for the Between-Class Scatter Matrix S_B.
"""

import numpy as np


class MyLDA:
    """
    Multiclass Linear Discriminant Analysis (LDA).
    Maximizes Fisher's criterion: J(w) = (w^T * S_B * w) / (w^T * S_W * w).
    
    Parameters:
    -----------
    n_components : int
        Number of discriminant directions to retain (at most n_classes - 1).
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.scalings_ = None # store optimal discriminant directions (eigenvectors)

    def fit(self, X, y):
        """
        Compute within-class scatter (S_W) and between-class scatter (S_B).
        
        Parameters:
        -----------
        X : np.ndarray of shape (N, D)
            Data matrix.
        y : np.ndarray of shape (N,)
            Target class labels.
        """
        classes = np.unique(y) # unique class labels
        D = X.shape[1] # number of features
        overall_mean = np.mean(X, axis=0) # matrix mean across all samples

        S_W = np.zeros((D, D))
        S_B = np.zeros((D, D))

        for c in classes:
            X_c = X[y == c] # data points in class c
            N_c = X_c.shape[0] # number of samples in class c
            mean_c = np.mean(X_c, axis=0) 

            # Within-class scatter matrix: S_W = sum_c sum_{x in c} (x - mu_c)(x - mu_c)^T
            diff_within = X_c - mean_c
            S_W += np.dot(diff_within.T, diff_within)

            # Between-class scatter matrix: S_B = sum_c N_c * (mu_c - mu)(mu_c - mu)^T
            # (Note: Fixed outer product erratum from textbook p. 105)
            mean_diff = (mean_c - overall_mean).reshape(-1, 1)
            S_B += N_c * np.dot(mean_diff, mean_diff.T)

        # Numerical stabilization (Tikhonov regularization) to avoid singularity when D > N
        epsilon = 1e-4
        S_W_reg = S_W + epsilon * np.eye(D)

        # Solve generalized eigenvalue problem: S_W^(-1) * S_B * w = lambda * w
        A = np.linalg.solve(S_W_reg, S_B) # compute S_W^(-1) * S_B
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Eigenvalues can have negligible imaginary artifacts due to numerical round-off
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        # Sort eigenvectors in descending order of their eigenvalues
        sort_idx = np.argsort(eigenvalues)[::-1] # sort indices for descending order
        self.scalings_ = eigenvectors[:, sort_idx[:self.n_components]] # take all the row, but only the first n_components columns (eigenvectors)

        return self

    def transform(self, X):
        """Project data matrix X onto the optimal discriminant axes."""
        return np.dot(X, self.scalings_)

    def fit_transform(self, X, y):
        """Fit LDA model and transform X."""
        self.fit(X, y)
        return self.transform(X)