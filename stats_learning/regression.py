"""
Regression Analysis Module
=========================

【Purpose】
Provides implementations for linear and logistic regression with comprehensive
model evaluation metrics and visualization tools.

【Mathematical Foundation】
- Linear Regression: y = Xβ + ε
- Ordinary Least Squares: β = (X^T X)^(-1) X^T y
- Logistic Regression: p = 1 / (1 + exp(-Xβ))
- R²: 1 - SS_res / SS_tot
- MSE: (1/n) * Σ(y - ŷ)²

【References】
- D2L Chapter: https://zh.d2l.ai/chapter_linear-networks/linear-regression.html
- Related Concepts: Regression Analysis, Model Evaluation, Feature Selection
"""

import numpy as np
from .descriptive_stats import mean, variance
from .visualization import regression_line, residual_plot, confidence_interval_plot

class LinearRegression:
    """
    Linear Regression implementation using Ordinary Least Squares.

    Attributes:
        coef_: Coefficient vector
        intercept_: Intercept term
        X: Training features
        y: Training target
    """
    
    def __init__(self, fit_intercept=True):
        """
        Initialize LinearRegression.

        Parameters:
            fit_intercept: Whether to fit an intercept term (default: True)
        """
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.X = None
        self.y = None
    
    def fit(self, X, y):
        """
        Fit linear regression model.

        Parameters:
            X: Feature matrix (n_samples x n_features)
            y: Target vector (n_samples,)

        Returns:
            self: Fitted model
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(n_samples), X])
        
        XTX = X.T @ X
        XTy = X.T @ y
        
        try:
            beta = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XTX, XTy, rcond=None)[0]
        
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        
        self.X = X
        self.y = y
        
        return self
    
    def predict(self, X):
        """
        Predict target values.

        Parameters:
            X: Feature matrix (n_samples x n_features)

        Returns:
            np.ndarray: Predicted values
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        
        return X @ np.concatenate([[self.intercept_], self.coef_])
    
    def get_params(self):
        """
        Get model parameters.

        Returns:
            dict: Dictionary containing intercept and coefficients
        """
        return {
            'intercept': self.intercept_,
            'coefficients': self.coef_
        }
    
    def summary(self):
        """
        Generate summary statistics for the model.

        Returns:
            dict: Summary statistics including R², MSE, coefficients
        """
        if self.X is None or self.y is None:
            raise ValueError("Model has not been fitted")
        
        y_pred = self.predict(self.X[:, 1:] if self.fit_intercept else self.X)
        
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        ss_res = np.sum((self.y - y_pred) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot)
        mse = np.mean((self.y - y_pred) ** 2)
        mae = np.mean(np.abs(self.y - y_pred))
        
        return {
            'intercept': self.intercept_,
            'coefficients': self.coef_,
            'r_squared': r_squared,
            'mse': mse,
            'mae': mae,
            'n_samples': len(self.y),
            'n_features': len(self.coef_) if self.coef_ is not None else 0
        }

class LogisticRegression:
    """
    Logistic Regression implementation using gradient descent.

    Attributes:
        coef_: Coefficient vector
        intercept_: Intercept term
        X: Training features
        y: Training target
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, fit_intercept=True):
        """
        Initialize LogisticRegression.

        Parameters:
            learning_rate: Learning rate for gradient descent (default: 0.01)
            max_iter: Maximum iterations (default: 1000)
            fit_intercept: Whether to fit an intercept term (default: True)
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.X = None
        self.y = None
    
    def _sigmoid(self, z):
        """Compute sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """
        Fit logistic regression model using gradient descent.

        Parameters:
            X: Feature matrix (n_samples x n_features)
            y: Target vector (n_samples,) - binary values (0 or 1)

        Returns:
            self: Fitted model
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(n_samples), X])
            n_features += 1
        
        beta = np.zeros(n_features)
        
        for _ in range(self.max_iter):
            z = X @ beta
            p = self._sigmoid(z)
            gradient = (X.T @ (p - y)) / n_samples
            beta -= self.learning_rate * gradient
        
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        
        self.X = X
        self.y = y
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters:
            X: Feature matrix (n_samples x n_features)

        Returns:
            np.ndarray: Probabilities of class 1
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        
        z = X @ np.concatenate([[self.intercept_], self.coef_])
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.

        Parameters:
            X: Feature matrix (n_samples x n_features)
            threshold: Classification threshold (default: 0.5)

        Returns:
            np.ndarray: Predicted class labels
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def get_params(self):
        """
        Get model parameters.

        Returns:
            dict: Dictionary containing intercept and coefficients
        """
        return {
            'intercept': self.intercept_,
            'coefficients': self.coef_
        }
    
    def summary(self):
        """
        Generate summary statistics for the model.

        Returns:
            dict: Summary statistics including accuracy, precision, recall
        """
        if self.X is None or self.y is None:
            raise ValueError("Model has not been fitted")
        
        y_pred = self.predict(self.X[:, 1:] if self.fit_intercept else self.X)
        
        accuracy = np.mean(self.y == y_pred)
        precision = np.sum((y_pred == 1) & (self.y == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
        recall = np.sum((y_pred == 1) & (self.y == 1)) / np.sum(self.y == 1) if np.sum(self.y == 1) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'intercept': self.intercept_,
            'coefficients': self.coef_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_samples': len(self.y),
            'n_features': len(self.coef_) if self.coef_ is not None else 0
        }

def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error.

    Parameters:
        y_true: True values
        y_pred: Predicted values

    Returns:
        float: MSE value
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error.

    Parameters:
        y_true: True values
        y_pred: Predicted values

    Returns:
        float: MAE value
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    return np.mean(np.abs(y_true - y_pred))

def r_squared(y_true, y_pred):
    """
    Calculate R-squared (coefficient of determination).

    Parameters:
        y_true: True values
        y_pred: Predicted values

    Returns:
        float: R² value (0 to 1)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)

def adjusted_r_squared(y_true, y_pred, n_features):
    """
    Calculate Adjusted R-squared.

    Parameters:
        y_true: True values
        y_pred: Predicted values
        n_features: Number of features

    Returns:
        float: Adjusted R² value
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    n_samples = len(y_true)
    r2 = r_squared(y_true, y_pred)
    
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

def accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.

    Parameters:
        y_true: True class labels
        y_pred: Predicted class labels

    Returns:
        float: Accuracy (0 to 1)
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    """
    Calculate precision for binary classification.

    Parameters:
        y_true: True class labels (0 or 1)
        y_pred: Predicted class labels (0 or 1)

    Returns:
        float: Precision (0 to 1)
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred):
    """
    Calculate recall for binary classification.

    Parameters:
        y_true: True class labels (0 or 1)
        y_pred: Predicted class labels (0 or 1)

    Returns:
        float: Recall (0 to 1)
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred):
    """
    Calculate F1 score for binary classification.

    Parameters:
        y_true: True class labels (0 or 1)
        y_pred: Predicted class labels (0 or 1)

    Returns:
        float: F1 score (0 to 1)
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def plot_regression_results(X, y, y_pred, title='Regression Results', show_plots=True):
    """
    Plot regression results including regression line and residual plot.

    Parameters:
        X: Feature data
        y: True values
        y_pred: Predicted values
        title: Plot title
        show_plots: Whether to display plots

    Returns:
        None
    """
    if X.ndim > 1 and X.shape[1] == 1:
        regression_line(X.flatten(), y, y_pred, title=f'{title} - Regression Line', show=show_plots)
    
    if show_plots:
        residual_plot(y_pred, y - y_pred, title=f'{title} - Residual Plot')

def linear_regression_example():
    """
    Interactive example demonstrating linear regression.
    """
    print(f"{'='*60}")
    print(f"线性回归示例")
    print(f"{'='*60}")
    
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X + 3 + 0.5 * np.random.randn(100, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    print(f"\n模型参数:")
    print(f"  截距 (intercept): {model.intercept_:.4f}")
    print(f"  系数 (coefficient): {model.coef_[0]:.4f}")
    
    print(f"\n模型评估指标:")
    print(f"  R²: {r_squared(y, y_pred):.4f}")
    print(f"  MSE: {mean_squared_error(y, y_pred):.4f}")
    print(f"  MAE: {mean_absolute_error(y, y_pred):.4f}")
    
    print(f"\n{'='*60}")
    print(f"回归方程: y = {model.coef_[0]:.4f} * x + {model.intercept_:.4f}")
    print(f"{'='*60}")
    
    plot_regression_results(X, y, y_pred, title='Linear Regression Demo')
    
    return {
        'model': model,
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'r_squared': r_squared(y, y_pred),
        'mse': mean_squared_error(y, y_pred),
        'mae': mean_absolute_error(y, y_pred)
    }

def logistic_regression_example():
    """
    Interactive example demonstrating logistic regression.
    """
    print(f"{'='*60}")
    print(f"逻辑回归示例")
    print(f"{'='*60}")
    
    np.random.seed(42)
    X = np.random.randn(200, 2)
    weights = np.array([1.5, -1.0])
    z = X @ weights + 0.5
    probs = 1 / (1 + np.exp(-z))
    y = (probs > 0.5).astype(int)
    
    model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    print(f"\n模型参数:")
    print(f"  截距 (intercept): {model.intercept_:.4f}")
    print(f"  系数 (coefficients): {model.coef_}")
    
    print(f"\n模型评估指标:")
    print(f"  准确率 (Accuracy): {accuracy(y, y_pred):.4f}")
    print(f"  精确率 (Precision): {precision(y, y_pred):.4f}")
    print(f"  召回率 (Recall): {recall(y, y_pred):.4f}")
    print(f"  F1分数 (F1 Score): {f1_score(y, y_pred):.4f}")
    
    print(f"\n{'='*60}")
    print(f"决策边界: {model.coef_[0]:.4f}*x1 + {model.coef_[1]:.4f}*x2 + {model.intercept_:.4f} = 0")
    print(f"{'='*60}")
    
    return {
        'model': model,
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'accuracy': accuracy(y, y_pred),
        'precision': precision(y, y_pred),
        'recall': recall(y, y_pred),
        'f1': f1_score(y, y_pred)
    }

def print_regression_summary(summary):
    """
    Print regression model summary in readable format.

    Parameters:
        summary: Dictionary from model.summary()
    """
    print(f"{'='*60}")
    print(f"回归模型摘要")
    print(f"{'='*60}")
    
    print(f"\n模型参数:")
    print(f"  截距: {summary['intercept']:.4f}")
    print(f"  系数: {summary['coefficients']}")
    
    print(f"\n模型评估:")
    if 'r_squared' in summary:
        print(f"  R²: {summary['r_squared']:.4f}")
        print(f"  MSE: {summary['mse']:.4f}")
        print(f"  MAE: {summary['mae']:.4f}")
    else:
        print(f"  准确率: {summary['accuracy']:.4f}")
        print(f"  精确率: {summary['precision']:.4f}")
        print(f"  召回率: {summary['recall']:.4f}")
        print(f"  F1分数: {summary['f1']:.4f}")
    
    print(f"\n数据信息:")
    print(f"  样本数量: {summary['n_samples']}")
    print(f"  特征数量: {summary['n_features']}")
