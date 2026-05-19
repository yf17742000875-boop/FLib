"""
Main Entry Point for Stats Learning Package
==========================================

Run this script to access interactive demonstrations and case studies.
"""

def main():
    import sys
    
    print(f"{'='*70}")
    print(f"机器学习统计学学习辅助代码")
    print(f"Machine Learning Statistics Learning Assistant")
    print(f"{'='*70}")
    
    menu = """
    
请选择要学习的模块:

1. 描述性统计 (Descriptive Statistics)
   - 数据分布分析
   - 数据可视化
   - 相关性分析

2. 概率论基础 (Probability)
   - 概率分布
   - 贝叶斯定理
   - 马尔科夫链

3. 推断统计 (Inferential Statistics)
   - 假设检验
   - 置信区间
   - ANOVA

4. 回归分析 (Regression Analysis)
   - 线性回归
   - 逻辑回归
   - 模型评估

5. 统计学习理论 (Statistical Learning Theory)
   - 偏差-方差权衡
   - 交叉验证
   - 正则化

6. 真实数据集案例分析 (Case Studies)
   - Iris数据集
   - Boston房价数据集
   - Titanic数据集

7. 退出 (Exit)

请输入选项编号: """
    
    while True:
        try:
            choice = int(input(menu))
            
            if choice == 1:
                run_descriptive_stats_demo()
            elif choice == 2:
                run_probability_demo()
            elif choice == 3:
                run_inferential_stats_demo()
            elif choice == 4:
                run_regression_demo()
            elif choice == 5:
                run_statistical_learning_demo()
            elif choice == 6:
                run_case_studies_demo()
            elif choice == 7:
                print("\n感谢使用机器学习统计学学习辅助代码!")
                print("Goodbye!")
                sys.exit(0)
            else:
                print("\n无效选项，请输入1-7之间的数字")
                
        except ValueError:
            print("\n请输入有效数字")

def run_descriptive_stats_demo():
    """Run descriptive statistics demonstration."""
    print(f"\n{'='*60}")
    print(f"描述性统计模块演示")
    print(f"{'='*60}")
    
    from .descriptive_stats import (
        compute_summary, analyze_distribution, analyze_relationship,
        summary_report, correlation_matrix
    )
    from .datasets import generate_normal_data
    import numpy as np
    
    data = generate_normal_data(mean=50, std=10, n_samples=1000, seed=42)
    
    print(f"\n1. 数据分布分析:")
    stats = compute_summary(data)
    print(f"均值: {stats['mean']:.4f}")
    print(f"中位数: {stats['median']:.4f}")
    print(f"标准差: {stats['std']:.4f}")
    print(f"偏度: {stats['skewness']:.4f}")
    print(f"峰度: {stats['kurtosis']:.4f}")
    
    print(f"\n2. 交互式分布分析(显示图表):")
    analyze_distribution(data, title='示例正态分布数据', show_plots=True)
    
    print(f"\n3. 相关性分析示例:")
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5
    analyze_relationship(x, y, title='正相关示例', show_plot=True)

def run_probability_demo():
    """Run probability demonstration."""
    print(f"\n{'='*60}")
    print(f"概率论基础模块演示")
    print(f"{'='*60}")
    
    from .probability import (
        plot_normal_distribution, plot_binomial_distribution,
        plot_poisson_distribution, compare_normal_distributions,
        bayesian_inference_example, markov_chain_example
    )
    
    print(f"\n1. 正态分布可视化:")
    plot_normal_distribution(mean=0, std=1, show=True)
    
    print(f"\n2. 比较不同参数的正态分布:")
    compare_normal_distributions([(0, 1), (0, 2), (2, 1)], show=True)
    
    print(f"\n3. 二项分布可视化:")
    plot_binomial_distribution(n=20, p=0.5, show=True)
    
    print(f"\n4. 泊松分布可视化:")
    plot_poisson_distribution(lam=5, show=True)
    
    print(f"\n5. 贝叶斯定理示例:")
    bayesian_inference_example()
    
    print(f"\n6. 马尔科夫链示例:")
    markov_chain_example()

def run_inferential_stats_demo():
    """Run inferential statistics demonstration."""
    print(f"\n{'='*60}")
    print(f"推断统计模块演示")
    print(f"{'='*60}")
    
    from .inferential_stats import (
        calculate_confidence_interval, print_confidence_interval,
        t_test_one_sample, print_t_test_result,
        anova_one_way, print_anova_result,
        hypothesis_testing_workflow
    )
    
    hypothesis_testing_workflow()

def run_regression_demo():
    """Run regression demonstration."""
    print(f"\n{'='*60}")
    print(f"回归分析模块演示")
    print(f"{'='*60}")
    
    from .regression import (
        linear_regression_example, logistic_regression_example,
        print_regression_summary
    )
    
    print(f"\n1. 线性回归示例:")
    linear_regression_example()
    
    print(f"\n2. 逻辑回归示例:")
    logistic_regression_example()

def run_statistical_learning_demo():
    """Run statistical learning theory demonstration."""
    print(f"\n{'='*60}")
    print(f"统计学习理论模块演示")
    print(f"{'='*60}")
    
    from .statistical_learning import (
        bias_variance_decomposition,
        cross_validation_demo,
        regularization_demo,
        model_selection_demo,
        learning_curves_demo
    )
    
    print(f"\n1. 偏差-方差权衡演示:")
    bias_variance_decomposition()
    
    print(f"\n2. 交叉验证演示:")
    cross_validation_demo()
    
    print(f"\n3. 正则化演示:")
    regularization_demo()
    
    print(f"\n4. 模型选择演示:")
    model_selection_demo()
    
    print(f"\n5. 学习曲线演示:")
    learning_curves_demo()

def run_case_studies_demo():
    """Run case studies demonstration."""
    print(f"\n{'='*60}")
    print(f"真实数据集案例分析")
    print(f"{'='*60}")
    
    from .case_studies import run_all_case_studies
    run_all_case_studies()

if __name__ == "__main__":
    main()
