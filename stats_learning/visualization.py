"""
Visualization Module
====================

【Purpose】
Provides visualization functions for statistical analysis and machine learning.
Supports histograms, box plots, scatter plots, probability distributions,
and regression visualization.

【References】
- D2L Chapter: https://zh.d2l.ai/chapter_linear-networks/index.html
- Related Concepts: Data Visualization, Exploratory Data Analysis
"""

import matplotlib.pyplot as plt
import numpy as np

def histogram(data, bins='auto', title='Histogram', xlabel='Value', ylabel='Frequency', 
              color='skyblue', edgecolor='black', show=True):
    """
    Create a histogram to visualize data distribution.

    Parameters:
        data: Array-like data to plot
        bins: Number of bins or bin specification (default: 'auto')
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Bar color
        edgecolor: Edge color of bars
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure object
        ax: matplotlib axes object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=bins, color=color, edgecolor=edgecolor)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis='y', alpha=0.75)
    
    if show:
        plt.show()
    
    return fig, ax

def box_plot(data, labels=None, title='Box Plot', xlabel='Category', ylabel='Value',
             show=True):
    """
    Create a box plot to visualize data distribution and outliers.

    Parameters:
        data: List of arrays or single array for multiple datasets
        labels: Labels for each dataset
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure object
        ax: matplotlib axes object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(data, (list, tuple)):
        ax.boxplot(data, labels=labels, patch_artist=True,
                   boxprops=dict(facecolor='skyblue'),
                   medianprops=dict(color='red'))
    else:
        ax.boxplot(data, patch_artist=True,
                   boxprops=dict(facecolor='skyblue'),
                   medianprops=dict(color='red'))
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis='y', alpha=0.75)
    
    if show:
        plt.show()
    
    return fig, ax

def scatter_plot(x, y, title='Scatter Plot', xlabel='X', ylabel='Y',
                 color='blue', alpha=0.6, show=True):
    """
    Create a scatter plot to visualize relationship between two variables.

    Parameters:
        x: X-axis data
        y: Y-axis data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Point color
        alpha: Point transparency
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure object
        ax: matplotlib axes object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color=color, alpha=alpha)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.75)
    
    if show:
        plt.show()
    
    return fig, ax

def probability_distribution(x, y, dist_name='Distribution', title=None,
                             xlabel='Value', ylabel='Probability',
                             color='blue', show=True):
    """
    Plot a probability distribution function.

    Parameters:
        x: X-axis values
        y: Probability values
        dist_name: Name of the distribution for legend
        title: Plot title (default: '{dist_name} Distribution')
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Line color
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure object
        ax: matplotlib axes object
    """
    if title is None:
        title = f'{dist_name} Distribution'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, color=color, linewidth=2, label=dist_name)
    ax.fill_between(x, y, alpha=0.3, color=color)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.75)
    
    if show:
        plt.show()
    
    return fig, ax

def regression_line(x, y, y_pred, title='Regression Line', xlabel='X', ylabel='Y',
                   data_color='blue', line_color='red', show=True):
    """
    Plot regression line with original data points.

    Parameters:
        x: Original X data
        y: Original Y data
        y_pred: Predicted Y values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        data_color: Color of data points
        line_color: Color of regression line
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure object
        ax: matplotlib axes object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color=data_color, alpha=0.6, label='Data Points')
    ax.plot(x, y_pred, color=line_color, linewidth=2, label='Regression Line')
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.75)
    
    if show:
        plt.show()
    
    return fig, ax

def residual_plot(x, residuals, title='Residual Plot', xlabel='Predicted Value',
                  ylabel='Residuals', show=True):
    """
    Plot residuals to check regression assumptions.

    Parameters:
        x: X values (usually predicted values)
        residuals: Residual values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure object
        ax: matplotlib axes object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, residuals, color='green', alpha=0.6)
    ax.axhline(y=0, color='red', linestyle='--', label='Zero Line')
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.75)
    
    if show:
        plt.show()
    
    return fig, ax

def confidence_interval_plot(x, y, y_pred, lower_bound, upper_bound,
                            title='Regression with Confidence Interval',
                            xlabel='X', ylabel='Y', show=True):
    """
    Plot regression line with confidence interval.

    Parameters:
        x: X values
        y: Original Y values
        y_pred: Predicted Y values
        lower_bound: Lower confidence bounds
        upper_bound: Upper confidence bounds
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure object
        ax: matplotlib axes object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color='blue', alpha=0.6, label='Data Points')
    ax.plot(x, y_pred, color='red', linewidth=2, label='Regression Line')
    ax.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.2,
                    label='95% Confidence Interval')
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.75)
    
    if show:
        plt.show()
    
    return fig, ax

def multiple_distributions(x_list, y_list, labels, title='Multiple Distributions',
                          xlabel='Value', ylabel='Probability', show=True):
    """
    Plot multiple probability distributions for comparison.

    Parameters:
        x_list: List of x arrays
        y_list: List of y arrays (probabilities)
        labels: List of labels for each distribution
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure object
        ax: matplotlib axes object
    """
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (x, y, label) in enumerate(zip(x_list, y_list, labels)):
        ax.plot(x, y, color=colors[i % len(colors)], linewidth=2, label=label)
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.75)
    
    if show:
        plt.show()
    
    return fig, ax

def bar_plot(x, y, title='Bar Plot', xlabel='Category', ylabel='Value',
             color='skyblue', edgecolor='black', show=True):
    """
    Create a bar plot.

    Parameters:
        x: Category labels
        y: Values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Bar color
        edgecolor: Edge color
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure object
        ax: matplotlib axes object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, y, color=color, edgecolor=edgecolor)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis='y', alpha=0.75)
    
    if show:
        plt.show()
    
    return fig, ax
