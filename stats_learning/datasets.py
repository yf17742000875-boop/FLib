"""
Datasets Module
===============

【Purpose】
Provides utilities for loading and preprocessing popular datasets for
statistical learning and machine learning experiments.

【References】
- UCI Machine Learning Repository
- scikit-learn datasets
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris as sk_load_iris, fetch_california_housing, fetch_openml

def load_iris():
    """
    Load the Iris dataset.
    
    Returns:
        dict: Dictionary containing features and target
            - 'data': Feature matrix (150 samples, 4 features)
            - 'target': Target labels (0, 1, 2 for different iris species)
            - 'feature_names': List of feature names
            - 'target_names': List of target class names
            - 'df': pandas DataFrame with all data
    """
    iris = sk_load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    return {
        'data': iris.data,
        'target': iris.target,
        'feature_names': iris.feature_names,
        'target_names': iris.target_names,
        'df': df
    }

def load_boston():
    """
    Load the California housing dataset (replacement for Boston dataset).
    
    Returns:
        dict: Dictionary containing features and target
            - 'data': Feature matrix (20640 samples, 8 features)
            - 'target': Target values (house prices in $100,000s)
            - 'feature_names': List of feature names
            - 'df': pandas DataFrame with all data
    """
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target
    
    return {
        'data': housing.data,
        'target': housing.target,
        'feature_names': housing.feature_names,
        'df': df
    }

def load_titanic():
    """
    Load the Titanic dataset from OpenML.
    
    Returns:
        dict: Dictionary containing features and target
            - 'data': Feature matrix
            - 'target': Target values (1 = survived, 0 = did not survive)
            - 'feature_names': List of feature names
            - 'df': pandas DataFrame with all data
    """
    titanic = fetch_openml('titanic', version=1, as_frame=True)
    df = titanic.frame
    
    # Basic preprocessing
    df['survived'] = df['survived'].astype(int)
    
    return {
        'data': df.drop('survived', axis=1).values,
        'target': df['survived'].values,
        'feature_names': df.drop('survived', axis=1).columns.tolist(),
        'df': df
    }

def generate_synthetic_data(n_samples=100, noise=0.1, seed=None):
    """
    Generate synthetic regression data.
    
    Parameters:
        n_samples: Number of samples
        noise: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (X, y) where X is feature matrix and y is target vector
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.randn(n_samples, 1)
    y = 2 * X + 3 + noise * np.random.randn(n_samples, 1)
    
    return X, y.flatten()

def generate_classification_data(n_samples=100, n_features=2, n_classes=2, seed=None):
    """
    Generate synthetic classification data.
    
    Parameters:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (X, y) where X is feature matrix and y is target vector
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features)
    
    # Simple linear decision boundary
    weights = np.random.randn(n_features)
    logits = X @ weights
    prob = 1 / (1 + np.exp(-logits))
    y = (prob > 0.5).astype(int)
    
    if n_classes > 2:
        # For multi-class
        weights = np.random.randn(n_features, n_classes)
        logits = X @ weights
        y = np.argmax(logits, axis=1)
    
    return X, y

def generate_normal_data(mean=0, std=1, n_samples=1000, seed=None):
    """
    Generate data from a normal distribution.
    
    Parameters:
        mean: Mean of the distribution
        std: Standard deviation
        n_samples: Number of samples
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Array of samples from N(mean, std^2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.normal(mean, std, n_samples)

def generate_binomial_data(n_trials=10, p=0.5, n_samples=1000, seed=None):
    """
    Generate data from a binomial distribution.
    
    Parameters:
        n_trials: Number of trials per experiment
        p: Probability of success
        n_samples: Number of samples
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Array of samples from Binomial(n_trials, p)
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.binomial(n_trials, p, n_samples)

def generate_poisson_data(lam=5, n_samples=1000, seed=None):
    """
    Generate data from a Poisson distribution.
    
    Parameters:
        lam: Lambda parameter (mean)
        n_samples: Number of samples
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Array of samples from Poisson(lam)
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.poisson(lam, n_samples)

def preprocess_data(df, target_column=None, categorical_columns=None):
    """
    Preprocess a DataFrame: handle missing values, encode categorical variables.
    
    Parameters:
        df: pandas DataFrame
        target_column: Name of target column (optional)
        categorical_columns: List of categorical columns to encode
    
    Returns:
        pandas DataFrame: Preprocessed DataFrame
    """
    df = df.copy()
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    return df

def train_test_split(X, y, test_size=0.2, seed=None):
    """
    Split data into training and testing sets.
    
    Parameters:
        X: Feature matrix
        y: Target vector
        test_size: Fraction of data to use for testing
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = len(y)
    test_indices = np.random.choice(n_samples, int(n_samples * test_size), replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
