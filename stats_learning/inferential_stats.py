"""
Inferential Statistics Module
=============================

【Purpose】
Provides functions for hypothesis testing, confidence intervals, and p-value calculations.
Covers t-tests, chi-square tests, ANOVA, and related statistical inference methods.

【Mathematical Foundation】
- Confidence Interval: x̄ ± t*(s/√n) or x̄ ± z*(σ/√n)
- t-test: t = (x̄ - μ) / (s/√n)
- Chi-square: χ² = Σ((O-E)²/E)
- ANOVA: F = (SSB/k-1) / (SSE/n-k)

【References】
- D2L Chapter: https://zh.d2l.ai/chapter_linear-networks/index.html
- Related Concepts: Hypothesis Testing, Statistical Inference, Confidence Intervals
"""

import numpy as np
import math
from scipy.special import gamma

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for population mean.

    Parameters:
        data: Array-like numeric data
        confidence: Confidence level (default: 0.95)

    Returns:
        dict: Lower bound, upper bound, and mean
    """
    data = np.asarray(data, dtype=np.float64)
    data = data[~np.isnan(data)]
    
    n = len(data)
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    
    if n < 30:
        t_critical = t_distribution_critical_value(confidence, n - 1)
        margin_of_error = t_critical * (std_val / np.sqrt(n))
    else:
        z_critical = normal_distribution_critical_value(confidence)
        margin_of_error = z_critical * (std_val / np.sqrt(n))
    
    return {
        'mean': mean_val,
        'lower_bound': mean_val - margin_of_error,
        'upper_bound': mean_val + margin_of_error,
        'confidence_level': confidence,
        'sample_size': n
    }

def normal_distribution_critical_value(confidence):
    """
    Calculate critical value from standard normal distribution.

    Parameters:
        confidence: Confidence level (e.g., 0.95)

    Returns:
        float: Critical z-value
    """
    alpha = 1 - confidence
    return inverse_normal_cdf(1 - alpha / 2)

def t_distribution_critical_value(confidence, df):
    """
    Calculate critical value from t-distribution.

    Parameters:
        confidence: Confidence level (e.g., 0.95)
        df: Degrees of freedom

    Returns:
        float: Critical t-value
    """
    alpha = 1 - confidence
    return inverse_t_cdf(1 - alpha / 2, df)

def inverse_normal_cdf(p):
    """
    Approximate inverse of standard normal CDF.

    Parameters:
        p: Probability (0 < p < 1)

    Returns:
        float: z such that P(Z <= z) = p
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be between 0 and 1")
    
    if p == 0.5:
        return 0.0
    
    # Approximation using Taylor series
    t = math.sqrt(-2 * math.log(p)) if p < 0.5 else math.sqrt(-2 * math.log(1 - p))
    
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d0 = 1.432788
    d1 = 0.189269
    d2 = 0.001308
    
    z = t - (c0 + c1 * t + c2 * t**2) / (1 + d0 * t + d1 * t**2 + d2 * t**3)
    
    return -z if p < 0.5 else z

def inverse_t_cdf(p, df):
    """
    Approximate inverse of t-distribution CDF.

    Parameters:
        p: Probability (0 < p < 1)
        df: Degrees of freedom

    Returns:
        float: t such that P(T <= t) = p with df degrees of freedom
    """
    if df <= 0:
        raise ValueError("df must be positive")
    if p <= 0 or p >= 1:
        raise ValueError("p must be between 0 and 1")
    
    if df > 100:
        return inverse_normal_cdf(p)
    
    # Simple approximation for t-distribution
    # Using the relationship between t-distribution and F-distribution
    if p < 0.5:
        return -inverse_t_cdf(1 - p, df)
    
    f_value = f_distribution_critical_value(p, 1, df)
    return math.sqrt(f_value)

def f_distribution_critical_value(p, df1, df2):
    """
    Approximate critical value from F-distribution.

    Parameters:
        p: Probability (0 < p < 1)
        df1: Degrees of freedom 1
        df2: Degrees of freedom 2

    Returns:
        float: F such that P(F <= f) = p
    """
    return (df2 / df1) * (1 / beta_inverse(1 - p, df1 / 2, df2 / 2))

def beta_inverse(p, alpha, beta):
    """
    Approximate inverse of beta distribution CDF.

    Parameters:
        p: Probability (0 < p < 1)
        alpha: Shape parameter 1
        beta: Shape parameter 2

    Returns:
        float: x such that P(B <= x) = p
    """
    if p < 0.5:
        return beta_inverse_impl(p, alpha, beta)
    else:
        return 1 - beta_inverse_impl(1 - p, beta, alpha)

def beta_inverse_impl(p, alpha, beta):
    """Internal implementation for beta inverse."""
    if p < 0.001:
        return ((alpha / (alpha + beta)) * (p * beta * gamma(alpha) * gamma(beta) / gamma(alpha + beta))) ** (1 / alpha)
    
    x = p
    for _ in range(10):
        f = beta_cdf(x, alpha, beta) - p
        f_prime = beta_pdf(x, alpha, beta)
        x -= f / f_prime
        if x <= 0:
            x = 1e-10
        if x >= 1:
            x = 1 - 1e-10
    
    return x

def beta_pdf(x, alpha, beta):
    """Beta distribution PDF."""
    if x <= 0 or x >= 1:
        return 0.0
    return x**(alpha - 1) * (1 - x)**(beta - 1) / (gamma(alpha) * gamma(beta) / gamma(alpha + beta))

def beta_cdf(x, alpha, beta):
    """Beta distribution CDF using incomplete beta function."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    
    return incomplete_beta(x, alpha, beta) / (gamma(alpha) * gamma(beta) / gamma(alpha + beta))

def incomplete_beta(x, alpha, beta):
    """Incomplete beta function approximation."""
    return gamma(alpha + beta) * x**alpha * (1 - x)**beta * hypergeometric_2f1(alpha, 1 - beta, alpha + 1, x) / ((alpha + beta) * gamma(alpha) * gamma(beta))

def hypergeometric_2f1(a, b, c, x):
    """Hypergeometric function approximation."""
    result = 1.0
    term = 1.0
    n = 0
    
    while abs(term) > 1e-15 and n < 100:
        term *= (a + n) * (b + n) / ((c + n) * (n + 1)) * x
        result += term
        n += 1
    
    return result

def t_test_one_sample(data, hypothesized_mean=0):
    """
    Perform one-sample t-test.

    Parameters:
        data: Array-like numeric data
        hypothesized_mean: Hypothesized population mean (default: 0)

    Returns:
        dict: Test results including t-statistic, p-value, and conclusion
    """
    data = np.asarray(data, dtype=np.float64)
    data = data[~np.isnan(data)]
    
    n = len(data)
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    se = std_val / np.sqrt(n)
    
    t_stat = (mean_val - hypothesized_mean) / se
    df = n - 1
    
    p_value = 2 * min(t_distribution_cdf(t_stat, df), 1 - t_distribution_cdf(t_stat, df))
    
    return {
        't_statistic': t_stat,
        'df': df,
        'p_value': p_value,
        'sample_mean': mean_val,
        'hypothesized_mean': hypothesized_mean,
        'sample_std': std_val,
        'sample_size': n
    }

def t_distribution_cdf(x, df):
    """
    Calculate CDF of t-distribution.

    Parameters:
        x: Value
        df: Degrees of freedom

    Returns:
        float: P(T <= x)
    """
    if df <= 0:
        raise ValueError("df must be positive")
    
    if df > 100:
        return normal_cdf_approx(x)
    
    # Using the relationship with F-distribution
    if x >= 0:
        p = 1 - 0.5 * f_distribution_cdf(1 / (1 + x**2 / df), df, 1)
    else:
        p = 0.5 * f_distribution_cdf(1 / (1 + x**2 / df), df, 1)
    
    return p

def normal_cdf_approx(x):
    """Approximate standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def f_distribution_cdf(x, df1, df2):
    """
    Approximate CDF of F-distribution.

    Parameters:
        x: Value
        df1: Degrees of freedom 1
        df2: Degrees of freedom 2

    Returns:
        float: P(F <= x)
    """
    if x <= 0:
        return 0.0
    
    return beta_cdf(df1 * x / (df2 + df1 * x), df1 / 2, df2 / 2)

def t_test_two_sample(data1, data2, equal_variances=True):
    """
    Perform two-sample t-test.

    Parameters:
        data1: First sample data
        data2: Second sample data
        equal_variances: Whether to assume equal variances (default: True)

    Returns:
        dict: Test results including t-statistic, p-value, and conclusion
    """
    data1 = np.asarray(data1, dtype=np.float64)
    data2 = np.asarray(data2, dtype=np.float64)
    
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]
    
    n1 = len(data1)
    n2 = len(data2)
    
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    
    std1 = np.std(data1, ddof=1)
    std2 = np.std(data2, ddof=1)
    
    if equal_variances:
        pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        se = np.sqrt(pooled_var * (1 / n1 + 1 / n2))
        df = n1 + n2 - 2
    else:
        se = np.sqrt(std1**2 / n1 + std2**2 / n2)
        df_numerator = (std1**2 / n1 + std2**2 / n2)**2
        df_denominator = (std1**4 / (n1**2 * (n1 - 1))) + (std2**4 / (n2**2 * (n2 - 1)))
        df = df_numerator / df_denominator
    
    t_stat = (mean1 - mean2) / se
    p_value = 2 * min(t_distribution_cdf(t_stat, df), 1 - t_distribution_cdf(t_stat, df))
    
    return {
        't_statistic': t_stat,
        'df': df,
        'p_value': p_value,
        'mean1': mean1,
        'mean2': mean2,
        'std1': std1,
        'std2': std2,
        'n1': n1,
        'n2': n2,
        'equal_variances': equal_variances
    }

def chi_square_goodness_of_fit(observed, expected):
    """
    Perform chi-square goodness of fit test.

    Parameters:
        observed: Observed frequencies
        expected: Expected frequencies

    Returns:
        dict: Test results including chi-square statistic, p-value, and conclusion
    """
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    
    if len(observed) != len(expected):
        raise ValueError("observed and expected must have same length")
    
    if np.any(expected <= 0):
        raise ValueError("expected frequencies must be positive")
    
    chi_sq_stat = np.sum((observed - expected)**2 / expected)
    df = len(observed) - 1
    
    p_value = 1 - chi_square_cdf(chi_sq_stat, df)
    
    return {
        'chi_square_statistic': chi_sq_stat,
        'df': df,
        'p_value': p_value,
        'observed': observed,
        'expected': expected
    }

def chi_square_cdf(x, df):
    """
    Calculate CDF of chi-square distribution.

    Parameters:
        x: Value
        df: Degrees of freedom

    Returns:
        float: P(χ² <= x)
    """
    if x <= 0:
        return 0.0
    
    return gamma_lower(df / 2, x / 2) / gamma(df / 2)

def gamma_lower(a, x):
    """Lower incomplete gamma function."""
    if x <= 0:
        return 0.0
    
    result = 0.0
    term = x**a * np.exp(-x) / gamma(a)
    result += term
    
    n = 0
    while abs(term) > 1e-15 and n < 100:
        n += 1
        term *= x / (a + n)
        result += term
    
    return result

def chi_square_independence(observed_matrix):
    """
    Perform chi-square test of independence.

    Parameters:
        observed_matrix: 2D array of observed frequencies

    Returns:
        dict: Test results including chi-square statistic, p-value, and conclusion
    """
    observed = np.asarray(observed_matrix, dtype=np.float64)
    
    if observed.ndim != 2:
        raise ValueError("observed_matrix must be 2D")
    
    row_totals = np.sum(observed, axis=1)
    col_totals = np.sum(observed, axis=0)
    grand_total = np.sum(observed)
    
    expected = np.outer(row_totals, col_totals) / grand_total
    
    chi_sq_stat = np.sum((observed - expected)**2 / expected)
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    
    p_value = 1 - chi_square_cdf(chi_sq_stat, df)
    
    return {
        'chi_square_statistic': chi_sq_stat,
        'df': df,
        'p_value': p_value,
        'observed': observed,
        'expected': expected
    }

def anova_one_way(*groups):
    """
    Perform one-way ANOVA.

    Parameters:
        *groups: Multiple arrays of data (one per group)

    Returns:
        dict: ANOVA results including F-statistic, p-value, and summary table
    """
    groups = [np.asarray(g, dtype=np.float64) for g in groups]
    groups = [g[~np.isnan(g)] for g in groups]
    
    n_groups = len(groups)
    n_total = sum(len(g) for g in groups)
    
    overall_mean = np.mean(np.concatenate(groups))
    
    ssb = sum(len(g) * (np.mean(g) - overall_mean)**2 for g in groups)
    sse = sum(np.sum((g - np.mean(g))**2) for g in groups)
    
    df_between = n_groups - 1
    df_within = n_total - n_groups
    
    msb = ssb / df_between
    mse = sse / df_within
    
    f_stat = msb / mse
    p_value = 1 - f_distribution_cdf(f_stat, df_between, df_within)
    
    group_means = [np.mean(g) for g in groups]
    group_std = [np.std(g, ddof=1) for g in groups]
    group_sizes = [len(g) for g in groups]
    
    return {
        'f_statistic': f_stat,
        'df_between': df_between,
        'df_within': df_within,
        'p_value': p_value,
        'ssb': ssb,
        'sse': sse,
        'msb': msb,
        'mse': mse,
        'group_means': group_means,
        'group_std': group_std,
        'group_sizes': group_sizes,
        'overall_mean': overall_mean
    }

def interpret_p_value(p_value, alpha=0.05):
    """
    Interpret p-value for hypothesis testing.

    Parameters:
        p_value: Calculated p-value
        alpha: Significance level (default: 0.05)

    Returns:
        str: Interpretation message
    """
    if p_value < 0.001:
        return f"p值 = {p_value:.4f}，强烈拒绝原假设（p < 0.001）"
    elif p_value < 0.01:
        return f"p值 = {p_value:.4f}，显著拒绝原假设（p < 0.01）"
    elif p_value < alpha:
        return f"p值 = {p_value:.4f}，拒绝原假设（p < {alpha}）"
    else:
        return f"p值 = {p_value:.4f}，不拒绝原假设（p >= {alpha}）"

def print_confidence_interval(result):
    """
    Print confidence interval result in readable format.

    Parameters:
        result: Dictionary from calculate_confidence_interval
    """
    print(f"{'='*60}")
    print(f"置信区间计算结果")
    print(f"{'='*60}")
    print(f"样本数量: {result['sample_size']}")
    print(f"样本均值: {result['mean']:.4f}")
    print(f"置信水平: {result['confidence_level']*100:.0f}%")
    print(f"置信区间: [{result['lower_bound']:.4f}, {result['upper_bound']:.4f}]")
    print(f"边际误差: {(result['upper_bound'] - result['mean']):.4f}")

def print_t_test_result(result, test_type='one-sample'):
    """
    Print t-test result in readable format.

    Parameters:
        result: Dictionary from t_test_one_sample or t_test_two_sample
        test_type: 'one-sample' or 'two-sample'
    """
    print(f"{'='*60}")
    if test_type == 'one-sample':
        print(f"单样本t检验结果")
    else:
        print(f"双样本t检验结果")
    print(f"{'='*60}")
    
    if test_type == 'one-sample':
        print(f"样本数量: {result['sample_size']}")
        print(f"样本均值: {result['sample_mean']:.4f}")
        print(f"样本标准差: {result['sample_std']:.4f}")
        print(f"假设均值: {result['hypothesized_mean']:.4f}")
    else:
        print(f"样本1数量: {result['n1']}")
        print(f"样本1均值: {result['mean1']:.4f}")
        print(f"样本1标准差: {result['std1']:.4f}")
        print(f"样本2数量: {result['n2']}")
        print(f"样本2均值: {result['mean2']:.4f}")
        print(f"样本2标准差: {result['std2']:.4f}")
        print(f"方差齐性假设: {'是' if result['equal_variances'] else '否'}")
    
    print(f"\nt统计量: {result['t_statistic']:.4f}")
    print(f"自由度: {result['df']:.1f}")
    print(f"p值: {result['p_value']:.4f}")
    print(f"\n{interpret_p_value(result['p_value'])}")

def print_chi_square_result(result, test_type='goodness_of_fit'):
    """
    Print chi-square test result in readable format.

    Parameters:
        result: Dictionary from chi_square_goodness_of_fit or chi_square_independence
        test_type: 'goodness_of_fit' or 'independence'
    """
    print(f"{'='*60}")
    if test_type == 'goodness_of_fit':
        print(f"卡方拟合优度检验结果")
    else:
        print(f"卡方独立性检验结果")
    print(f"{'='*60}")
    
    print(f"卡方统计量: {result['chi_square_statistic']:.4f}")
    print(f"自由度: {result['df']}")
    print(f"p值: {result['p_value']:.4f}")
    print(f"\n{interpret_p_value(result['p_value'])}")
    
    if test_type == 'goodness_of_fit':
        print(f"\n观测频数: {result['observed']}")
        print(f"期望频数: {result['expected']}")

def print_anova_result(result):
    """
    Print ANOVA result in readable format.

    Parameters:
        result: Dictionary from anova_one_way
    """
    print(f"{'='*60}")
    print(f"单因素方差分析结果")
    print(f"{'='*60}")
    
    print(f"\n方差分析表:")
    print(f"{'来源':<10} {'SS':>10} {'df':>5} {'MS':>10} {'F':>8} {'p值':>10}")
    print(f"{'='*60}")
    print(f"{'组间':<10} {result['ssb']:>10.4f} {result['df_between']:>5} {result['msb']:>10.4f} {result['f_statistic']:>8.4f} {result['p_value']:>10.4f}")
    print(f"{'组内':<10} {result['sse']:>10.4f} {result['df_within']:>5} {result['mse']:>10.4f}")
    print(f"{'总计':<10} {result['ssb'] + result['sse']:>10.4f} {result['df_between'] + result['df_within']:>5}")
    
    print(f"\n各组统计量:")
    for i, (mean_val, std_val, size) in enumerate(zip(result['group_means'], result['group_std'], result['group_sizes'])):
        print(f"  组{i+1}: n={size}, 均值={mean_val:.4f}, 标准差={std_val:.4f}")
    
    print(f"\n{interpret_p_value(result['p_value'])}")

def hypothesis_testing_workflow():
    """
    Interactive workflow demonstrating hypothesis testing concepts.
    """
    print(f"{'='*60}")
    print(f"假设检验学习流程")
    print(f"{'='*60}")
    
    print(f"\n假设检验的基本步骤:")
    print(f"1. 提出原假设(H₀)和备择假设(H₁)")
    print(f"2. 选择显著性水平(α)")
    print(f"3. 计算检验统计量")
    print(f"4. 计算p值")
    print(f"5. 做出决策")
    
    print(f"\n{'='*60}")
    print(f"示例: 单样本t检验")
    print(f"{'='*60}")
    print(f"\n场景: 某工厂生产的零件直径服从正态分布，声称平均直径为5mm。")
    print(f"现抽取20个样本，测得直径如下（单位：mm）:")
    
    sample_data = [5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1, 4.9, 5.0, 5.1,
                   5.0, 4.8, 5.2, 4.9, 5.1, 5.0, 4.9, 5.0, 5.1, 5.2]
    
    print(f"\n样本数据: {sample_data}")
    
    result = t_test_one_sample(sample_data, hypothesized_mean=5.0)
    print_t_test_result(result, test_type='one-sample')
    
    print(f"\n{'='*60}")
    print(f"置信区间示例")
    print(f"{'='*60}")
    ci_result = calculate_confidence_interval(sample_data, confidence=0.95)
    print_confidence_interval(ci_result)
    
    print(f"\n{'='*60}")
    print(f"ANOVA示例")
    print(f"{'='*60}")
    print(f"\n场景: 比较三种不同肥料对作物产量的影响")
    
    group1 = [20, 22, 18, 21, 19, 23]  # 肥料A
    group2 = [25, 23, 24, 26, 22, 24]  # 肥料B
    group3 = [18, 19, 20, 17, 21, 18]  # 肥料C
    
    anova_result = anova_one_way(group1, group2, group3)
    print_anova_result(anova_result)
    
    return {
        't_test': result,
        'confidence_interval': ci_result,
        'anova': anova_result
    }
