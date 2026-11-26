"""
Data loading and preprocessing utilities.
"""

import numpy as np
from sklearn.datasets import load_digits # for loading the digits dataset


def load_digit_classes(classes=(0, 1, 2)):
    """
    Load specific digit classes from the digits dataset.
    
    Parameters:
    -----------
    classes : tuple or list
        The digit labels to extract (default: (0, 1, 2)).
        
    Returns:
    --------
    X : np.ndarray of shape (N, 64)
    y : np.ndarray of shape (N,)
    """
    digits = load_digits() # loading the digits dataset, consisting of digits.data (N, 64) -> data;and digits.target (N,) -> labels
    mask = np.isin(digits.target, classes) # filter for the specified classes
    return digits.data[mask], digits.target[mask]


def train_test_split_custom(X, y, train_ratio=0.8, random_state=42):
    """
    Split arrays into random train and test subsets using pure NumPy.
    
    Parameters:
    -----------
    X : np.ndarray of shape (N, D)
    y : np.ndarray of shape (N,)
    train_ratio : float (default: 0.8)
    random_state : int (default: 42)
    
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state) # set random seed for reproducibility
    N = len(X)
    shuffled_indices = np.random.permutation(N) # shuffle indices for random sampling
    split_point = int(N * train_ratio)
    
    train_idx = shuffled_indices[:split_point]
    test_idx = shuffled_indices[split_point:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]