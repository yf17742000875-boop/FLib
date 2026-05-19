"""
Machine Learning Statistics Learning Assistant
=============================================

This package provides comprehensive statistical learning tools for machine learning,
covering descriptive statistics, probability theory, inferential statistics,
regression analysis, and statistical learning theory.

Modules:
    descriptive_stats: Data distribution analysis and visualization
    probability: Probability distributions and Bayesian inference
    inferential_stats: Hypothesis testing and confidence intervals
    regression: Linear and logistic regression implementations
    statistical_learning: Bias-variance tradeoff, cross-validation, regularization
    datasets: Dataset loading and preprocessing utilities
    visualization: Plotting and visualization helper functions

Usage:
    import stats_learning as sl
    
    # Load dataset
    data = sl.datasets.load_iris()
    
    # Compute descriptive statistics
    stats = sl.descriptive_stats.compute_summary(data['sepal_length'])
    
    # Visualize data
    sl.visualization.histogram(data['sepal_length'], title='Sepal Length Distribution')
"""

__version__ = '1.0.0'
__author__ = 'ML Learning Assistant'

from . import descriptive_stats
from . import probability
from . import inferential_stats
from . import regression
from . import statistical_learning
from . import datasets
from . import visualization

__all__ = [
    'descriptive_stats',
    'probability',
    'inferential_stats',
    'regression',
    'statistical_learning',
    'datasets',
    'visualization'
]
