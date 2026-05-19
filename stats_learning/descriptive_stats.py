"""
Descriptive Statistics Module
=============================

【Purpose】
Provides functions for data distribution analysis and visualization.
Covers measures of central tendency, dispersion, and data exploration.

【Mathematical Foundation】
- Mean: μ = (1/n) * Σx_i
- Median: Middle value when data is sorted
- Mode: Most frequently occurring value
- Variance: σ² = (1/n) * Σ(x_i - μ)²
- Standard Deviation: σ = sqrt(σ²)
- Interquartile Range: IQR = Q3 - Q1

【References】
- D2L Chapter: https://zh.d2l.ai/chapter_linear-networks/linear-regression.html
- Related Concepts: Exploratory Data Analysis, Data Preprocessing
"""

import numpy as np
import pandas as pd
from collections import Counter
from .visualization import histogram, box_plot, scatter_plot

def mean(data):
    """
    Calculate the arithmetic mean.

    Parameters:
        data: Array-like numeric data

    Returns:
        float: The mean value

    Raises:
        TypeError: If data contains non-numeric values
    """
    data = np.asarray(data, dtype=np.float64)
    if np.isnan(data).any():
        data = data[~np.isnan(data)]
    return np.mean(data)

def median(data):
    """
    Calculate the median.

    Parameters:
        data: Array-like numeric data

    Returns:
        float: The median value
    """
    data = np.asarray(data, dtype=np.float64)
    if np.isnan(data).any():
        data = data[~np.isnan(data)]
    return np.median(data)

def mode(data):
    """
    Calculate the mode(s).

    Parameters:
        data: Array-like data (can be numeric or categorical)

    Returns:
        list: List of mode values
    """
    counter = Counter(data)
    max_count = max(counter.values())
    return [k for k, v in counter.items() if v == max_count]

def variance(data, ddof=0):
    """
    Calculate the variance.

    Parameters:
        data: Array-like numeric data
        ddof: Delta Degrees of Freedom (0 for population variance, 1 for sample)

    Returns:
        float: The variance
    """
    data = np.asarray(data, dtype=np.float64)
    if np.isnan(data).any():
        data = data[~np.isnan(data)]
    return np.var(data, ddof=ddof)

def standard_deviation(data, ddof=0):
    """
    Calculate the standard deviation.

    Parameters:
        data: Array-like numeric data
        ddof: Delta Degrees of Freedom (0 for population, 1 for sample)

    Returns:
        float: The standard deviation
    """
    data = np.asarray(data, dtype=np.float64)
    if np.isnan(data).any():
        data = data[~np.isnan(data)]
    return np.std(data, ddof=ddof)

def range(data):
    """
    Calculate the range (max - min).

    Parameters:
        data: Array-like numeric data

    Returns:
        float: The range
    """
    data = np.asarray(data, dtype=np.float64)
    if np.isnan(data).any():
        data = data[~np.isnan(data)]
    return np.max(data) - np.min(data)

def quartiles(data):
    """
    Calculate quartiles (Q1, Q2, Q3).

    Parameters:
        data: Array-like numeric data

    Returns:
        dict: Dictionary with quartile values
    """
    data = np.asarray(data, dtype=np.float64)
    if np.isnan(data).any():
        data = data[~np.isnan(data)]
    
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    
    return {
        'Q1': q1,
        'Q2': q2,
        'Q3': q3,
        'IQR': q3 - q1
    }

def iqr(data):
    """
    Calculate the Interquartile Range.

    Parameters:
        data: Array-like numeric data

    Returns:
        float: The IQR
    """
    q = quartiles(data)
    return q['IQR']

def residuals(y_true, y_pred):
    """
    Calculate residuals (y_true - y_pred).

    Parameters:
        y_true: Array of true values
        y_pred: Array of predicted values

    Returns:
        np.ndarray: Residual values
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    return y_true - y_pred

def skewness(data):
    """
    Calculate skewness.

    Parameters:
        data: Array-like numeric data

    Returns:
        float: Skewness value
    """
    data = np.asarray(data, dtype=np.float64)
    if np.isnan(data).any():
        data = data[~np.isnan(data)]
    
    n = len(data)
    mean_val = mean(data)
    std_val = standard_deviation(data)
    
    if std_val == 0:
        return 0.0
    
    return (n / ((n - 1) * (n - 2))) * np.sum(((data - mean_val) / std_val) ** 3)

def kurtosis(data):
    """
    Calculate kurtosis (excess kurtosis).

    Parameters:
        data: Array-like numeric data

    Returns:
        float: Kurtosis value
    """
    data = np.asarray(data, dtype=np.float64)
    if np.isnan(data).any():
        data = data[~np.isnan(data)]
    
    n = len(data)
    mean_val = mean(data)
    std_val = standard_deviation(data)
    
    if std_val == 0:
        return 0.0
    
    return ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean_val) / std_val) ** 4) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

def compute_summary(data):
    """
    Compute comprehensive summary statistics.

    Parameters:
        data: Array-like numeric data

    Returns:
        dict: Dictionary containing all summary statistics
    """
    data = np.asarray(data, dtype=np.float64)
    if np.isnan(data).any():
        data = data[~np.isnan(data)]
    
    q = quartiles(data)
    
    return {
        'count': len(data),
        'mean': mean(data),
        'median': median(data),
        'mode': mode(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': range(data),
        'variance': variance(data, ddof=1),
        'std': standard_deviation(data, ddof=1),
        'Q1': q['Q1'],
        'Q2': q['Q2'],
        'Q3': q['Q3'],
        'IQR': q['IQR'],
        'skewness': skewness(data),
        'kurtosis': kurtosis(data)
    }

def describe_dataframe(df):
    """
    Generate descriptive statistics for a DataFrame.

    Parameters:
        df: pandas DataFrame

    Returns:
        pandas DataFrame: Summary statistics for each numeric column
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    result = {}
    
    for col in numeric_cols:
        result[col] = compute_summary(df[col])
    
    return pd.DataFrame(result).T

def detect_outliers_iqr(data, threshold=1.5):
    """
    Detect outliers using IQR method.

    Parameters:
        data: Array-like numeric data
        threshold: IQR multiplier (default: 1.5)

    Returns:
        tuple: (lower_bound, upper_bound, outliers)
    """
    data = np.asarray(data, dtype=np.float64)
    if np.isnan(data).any():
        data = data[~np.isnan(data)]
    
    q = quartiles(data)
    lower_bound = q['Q1'] - threshold * q['IQR']
    upper_bound = q['Q3'] + threshold * q['IQR']
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    return lower_bound, upper_bound, outliers

def covariance(x, y):
    """
    Calculate covariance between two variables.

    Parameters:
        x: Array-like numeric data
        y: Array-like numeric data

    Returns:
        float: Covariance value
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)
    
    return np.sum((x - mean_x) * (y - mean_y)) / (n - 1)

def correlation(x, y):
    """
    Calculate Pearson correlation coefficient.

    Parameters:
        x: Array-like numeric data
        y: Array-like numeric data

    Returns:
        float: Correlation coefficient (-1 to 1)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    cov = covariance(x, y)
    std_x = standard_deviation(x, ddof=1)
    std_y = standard_deviation(y, ddof=1)
    
    if std_x == 0 or std_y == 0:
        return 0.0
    
    return cov / (std_x * std_y)

def correlation_matrix(df):
    """
    Compute correlation matrix for DataFrame.

    Parameters:
        df: pandas DataFrame

    Returns:
        pandas DataFrame: Correlation matrix
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    n = len(numeric_cols)
    corr_matrix = pd.DataFrame(np.eye(n), index=numeric_cols, columns=numeric_cols)
    
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:
                corr = correlation(df[col1], df[col2])
                corr_matrix.loc[col1, col2] = corr
                corr_matrix.loc[col2, col1] = corr
    
    return corr_matrix

def analyze_distribution(data, title='Data Distribution Analysis', show_plots=True):
    """
    Perform comprehensive distribution analysis with visualization.

    Parameters:
        data: Array-like numeric data
        title: Title for the analysis
        show_plots: Whether to display plots

    Returns:
        dict: Summary statistics
    """
    stats = compute_summary(data)
    
    print(f"{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"样本数量: {stats['count']}")
    print(f"均值: {stats['mean']:.4f}")
    print(f"中位数: {stats['median']:.4f}")
    print(f"众数: {stats['mode']}")
    print(f"最小值: {stats['min']:.4f}")
    print(f"最大值: {stats['max']:.4f}")
    print(f"极差: {stats['range']:.4f}")
    print(f"方差: {stats['variance']:.4f}")
    print(f"标准差: {stats['std']:.4f}")
    print(f"Q1 (25%): {stats['Q1']:.4f}")
    print(f"Q2 (50%/中位数): {stats['Q2']:.4f}")
    print(f"Q3 (75%): {stats['Q3']:.4f}")
    print(f"IQR: {stats['IQR']:.4f}")
    print(f"偏度: {stats['skewness']:.4f}")
    print(f"峰度: {stats['kurtosis']:.4f}")
    
    lower_bound, upper_bound, outliers = detect_outliers_iqr(data)
    print(f"异常值检测(IQR方法):")
    print(f"  下界: {lower_bound:.4f}")
    print(f"  上界: {upper_bound:.4f}")
    print(f"  异常值数量: {len(outliers)}")
    
    if show_plots:
        histogram(data, title=f'{title} - Histogram')
        box_plot(data, title=f'{title} - Box Plot')
    
    return stats

def analyze_relationship(x, y, title='Relationship Analysis', show_plot=True):
    """
    Analyze relationship between two variables.

    Parameters:
        x: Array-like numeric data
        y: Array-like numeric data
        title: Title for the analysis
        show_plot: Whether to display scatter plot

    Returns:
        dict: Statistics including covariance and correlation
    """
    cov = covariance(x, y)
    corr = correlation(x, y)
    
    print(f"{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"协方差: {cov:.4f}")
    print(f"相关系数: {corr:.4f}")
    
    if corr > 0.7:
        print("关系强度: 强正相关")
    elif corr > 0.3:
        print("关系强度: 中等正相关")
    elif corr > 0:
        print("关系强度: 弱正相关")
    elif corr == 0:
        print("关系强度: 无相关")
    elif corr > -0.3:
        print("关系强度: 弱负相关")
    elif corr > -0.7:
        print("关系强度: 中等负相关")
    else:
        print("关系强度: 强负相关")
    
    if show_plot:
        scatter_plot(x, y, title=title)
    
    return {
        'covariance': cov,
        'correlation': corr
    }

def summary_report(df):
    """
    Generate comprehensive summary report for a DataFrame.

    Parameters:
        df: pandas DataFrame

    Returns:
        None: Prints the report
    """
    print(f"{'='*60}")
    print(f"数据概览报告")
    print(f"{'='*60}")
    print(f"\n基本信息:")
    print(f"  行数: {df.shape[0]}")
    print(f"  列数: {df.shape[1]}")
    print(f"\n列类型分布:")
    print(df.dtypes.value_counts())
    
    print(f"\n{'='*60}")
    print(f"缺失值统计:")
    print(f"{'='*60}")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count} ({missing_percent[col]:.2f}%)")
    if missing.sum() == 0:
        print("  无缺失值")
    
    print(f"\n{'='*60}")
    print(f"数值列统计摘要:")
    print(f"{'='*60}")
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_cols.empty:
        print(describe_dataframe(df))
    else:
        print("  无数值列")
    
    print(f"\n{'='*60}")
    print(f"分类列统计:")
    print(f"{'='*60}")
    categorical_cols = df.select_dtypes(include=['object', 'category'])
    for col in categorical_cols:
        print(f"\n  {col}:")
        print(df[col].value_counts())
