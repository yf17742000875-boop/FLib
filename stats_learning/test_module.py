"""
Test Module for Stats Learning Package
=====================================

This script tests the basic functionality of all modules.
Run from the parent directory using: python -m stats_learning.test_module
"""

import sys

def test_descriptive_stats():
    """Test descriptive statistics module."""
    print("Testing descriptive_stats...")
    from .descriptive_stats import mean, median, mode, variance, standard_deviation, compute_summary
    
    data = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9]
    
    assert mean(data) == 5.0
    assert median(data) == 5.0
    assert mode(data) == [5]
    assert variance(data, ddof=1) == 6.666666666666667
    assert standard_deviation(data, ddof=1) == 2.581988897471611
    
    stats = compute_summary(data)
    assert 'mean' in stats
    assert 'median' in stats
    assert 'std' in stats
    
    print("  ✓ descriptive_stats passed")

def test_probability():
    """Test probability module."""
    print("Testing probability...")
    from .probability import normal_pdf, binomial_pmf, poisson_pmf, bayes_theorem
    
    assert normal_pdf(0, 0, 1) > 0
    assert binomial_pmf(5, 10, 0.5) > 0
    assert poisson_pmf(5, 5) > 0
    
    result = bayes_theorem(0.95, 0.01, 0.059)
    assert 0.15 < result < 0.17
    
    print("  ✓ probability passed")

def test_inferential_stats():
    """Test inferential statistics module."""
    print("Testing inferential_stats...")
    from .inferential_stats import calculate_confidence_interval, t_test_one_sample
    
    data = [5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1, 4.9, 5.0, 5.1]
    
    ci = calculate_confidence_interval(data)
    assert 'lower_bound' in ci
    assert 'upper_bound' in ci
    
    result = t_test_one_sample(data, 5.0)
    assert 't_statistic' in result
    assert 'p_value' in result
    
    print("  ✓ inferential_stats passed")

def test_regression():
    """Test regression module."""
    print("Testing regression...")
    from .regression import LinearRegression, LogisticRegression, mean_squared_error, r_squared
    
    X = [[1], [2], [3], [4], [5]]
    y = [2, 4, 6, 8, 10]
    
    model = LinearRegression()
    model.fit(X, y)
    
    assert abs(model.intercept_) < 0.01
    assert abs(model.coef_[0] - 2.0) < 0.01
    
    y_pred = model.predict(X)
    assert mean_squared_error(y, y_pred) < 0.01
    assert r_squared(y, y_pred) > 0.99
    
    print("  ✓ regression passed")

def test_statistical_learning():
    """Test statistical learning module."""
    print("Testing statistical_learning...")
    from .statistical_learning import RegularizedLinearRegression, k_fold_cross_validation
    from .regression import LinearRegression
    
    X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    model = RegularizedLinearRegression(alpha=0.1, penalty='l2')
    model.fit(X, y)
    assert model.coef_ is not None
    
    result = k_fold_cross_validation(X, y, LinearRegression, k=5)
    assert 'mean_val_score' in result
    
    print("  ✓ statistical_learning passed")

def test_datasets():
    """Test datasets module."""
    print("Testing datasets...")
    from .datasets import load_iris, generate_synthetic_data, train_test_split
    
    iris = load_iris()
    assert iris['data'].shape[0] == 150
    assert iris['data'].shape[1] == 4
    
    X, y = generate_synthetic_data(n_samples=100, seed=42)
    assert X.shape == (100, 1)
    assert y.shape == (100,)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    assert len(y_train) == 80
    assert len(y_test) == 20
    
    print("  ✓ datasets passed")

def test_case_studies():
    """Test case studies module."""
    print("Testing case_studies...")
    from .case_studies import iris_dataset_case_study
    
    result = iris_dataset_case_study()
    assert 'dataset' in result
    
    print("  ✓ case_studies passed")

def main():
    print("="*60)
    print("Testing Stats Learning Package Modules")
    print("="*60)
    
    try:
        test_descriptive_stats()
        test_probability()
        test_inferential_stats()
        test_regression()
        test_statistical_learning()
        test_datasets()
        test_case_studies()
        
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
