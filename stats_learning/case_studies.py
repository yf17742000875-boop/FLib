"""
Case Studies Module
===================

【Purpose】
Provides real-world dataset applications demonstrating statistical learning concepts.
Includes Iris, Boston housing, and Titanic dataset case studies.

【References】
- UCI Machine Learning Repository
- scikit-learn datasets
"""

import numpy as np
import pandas as pd

def iris_dataset_case_study():
    """
    Case study using the Iris dataset for classification and exploratory analysis.
    """
    print(f"{'='*60}")
    print(f"Iris 数据集案例分析")
    print(f"{'='*60}")
    
    from .datasets import load_iris
    from .descriptive_stats import summary_report, analyze_distribution, analyze_relationship, correlation_matrix
    from .visualization import histogram, box_plot, scatter_plot
    
    iris = load_iris()
    df = iris['df']
    
    print(f"\n数据集概览:")
    print(f"样本数量: {df.shape[0]}")
    print(f"特征数量: {df.shape[1]}")
    print(f"特征名称: {iris['feature_names']}")
    print(f"类别名称: {iris['target_names']}")
    
    print(f"\n{'='*60}")
    print(f"数据前5行:")
    print(f"{'='*60}")
    print(df.head())
    
    print(f"\n{'='*60}")
    print(f"数据摘要报告:")
    print(f"{'='*60}")
    summary_report(df)
    
    print(f"\n{'='*60}")
    print(f"花瓣长度分布分析:")
    print(f"{'='*60}")
    analyze_distribution(df['petal length (cm)'], title='Petal Length Distribution', show_plots=False)
    
    print(f"\n{'='*60}")
    print(f"不同物种的花瓣长度比较:")
    print(f"{'='*60}")
    setosa_pl = df[df['species'] == 'setosa']['petal length (cm)']
    versicolor_pl = df[df['species'] == 'versicolor']['petal length (cm)']
    virginica_pl = df[df['species'] == 'virginica']['petal length (cm)']
    
    box_plot([setosa_pl, versicolor_pl, virginica_pl], 
             labels=['Setosa', 'Versicolor', 'Virginica'],
             title='Petal Length by Species',
             show=False)
    
    print(f"\n各组统计量:")
    print(f"Setosa: 均值={setosa_pl.mean():.4f}, 标准差={setosa_pl.std():.4f}")
    print(f"Versicolor: 均值={versicolor_pl.mean():.4f}, 标准差={versicolor_pl.std():.4f}")
    print(f"Virginica: 均值={virginica_pl.mean():.4f}, 标准差={virginica_pl.std():.4f}")
    
    print(f"\n{'='*60}")
    print(f"特征相关性分析:")
    print(f"{'='*60}")
    corr_mat = correlation_matrix(df)
    print(corr_mat)
    
    analyze_relationship(df['sepal length (cm)'], df['sepal width (cm)'],
                        title='Sepal Length vs Sepal Width', show_plot=False)
    
    return {
        'dataset': iris,
        'summary': summary_report(df),
        'correlation_matrix': corr_mat
    }

def boston_housing_case_study():
    """
    Case study using the Boston housing dataset for regression analysis.
    """
    print(f"{'='*60}")
    print(f"Boston 房价数据集案例分析")
    print(f"{'='*60}")
    
    from .datasets import load_boston, train_test_split
    from .regression import LinearRegression, mean_squared_error, r_squared
    from .descriptive_stats import summary_report, correlation_matrix
    from .inferential_stats import calculate_confidence_interval, print_confidence_interval
    
    boston = load_boston()
    df = boston['df']
    
    print(f"\n数据集概览:")
    print(f"样本数量: {df.shape[0]}")
    print(f"特征数量: {df.shape[1] - 1}")
    print(f"特征名称: {boston['feature_names']}")
    
    print(f"\n{'='*60}")
    print(f"房价统计:")
    print(f"{'='*60}")
    price_stats = calculate_confidence_interval(df['PRICE'], confidence=0.95)
    print_confidence_interval(price_stats)
    
    print(f"\n{'='*60}")
    print(f"特征相关性矩阵(与房价):")
    print(f"{'='*60}")
    corr_mat = correlation_matrix(df)
    price_corr = corr_mat['PRICE'].sort_values(ascending=False)
    print(price_corr)
    
    print(f"\n{'='*60}")
    print(f"线性回归建模:")
    print(f"{'='*60}")
    
    X = df[['RM', 'LSTAT', 'PTRATIO']].values
    y = df['PRICE'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print(f"训练集 R²: {r_squared(y_train, y_train_pred):.4f}")
    print(f"测试集 R²: {r_squared(y_test, y_test_pred):.4f}")
    print(f"训练集 MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"测试集 MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
    
    print(f"\n回归系数:")
    print(f"截距: {model.intercept_:.4f}")
    print(f"RM (平均房间数): {model.coef_[0]:.4f}")
    print(f"LSTAT (低收入人口比例): {model.coef_[1]:.4f}")
    print(f"PTRATIO (师生比): {model.coef_[2]:.4f}")
    
    print(f"\n回归方程:")
    print(f"PRICE = {model.intercept_:.4f} + {model.coef_[0]:.4f}*RM + {model.coef_[1]:.4f}*LSTAT + {model.coef_[2]:.4f}*PTRATIO")
    
    return {
        'dataset': boston,
        'model': model,
        'train_r2': r_squared(y_train, y_train_pred),
        'test_r2': r_squared(y_test, y_test_pred),
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred)
    }

def titanic_case_study():
    """
    Case study using the Titanic dataset for classification analysis.
    """
    print(f"{'='*60}")
    print(f"Titanic 数据集案例分析")
    print(f"{'='*60}")
    
    from .datasets import load_titanic, preprocess_data, train_test_split
    from .regression import LogisticRegression, accuracy, precision, recall, f1_score
    from .descriptive_stats import summary_report
    
    titanic = load_titanic()
    df = titanic['df']
    
    print(f"\n数据集概览:")
    print(f"样本数量: {df.shape[0]}")
    print(f"特征数量: {df.shape[1] - 1}")
    
    print(f"\n{'='*60}")
    print(f"生存情况统计:")
    print(f"{'='*60}")
    survival_counts = df['survived'].value_counts()
    print(f"遇难人数: {survival_counts[0]} ({survival_counts[0]/len(df)*100:.1f}%)")
    print(f"获救人数: {survival_counts[1]} ({survival_counts[1]/len(df)*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"数据预处理:")
    print(f"{'='*60}")
    
    df_clean = preprocess_data(df, target_column='survived')
    print(f"处理后特征数量: {df_clean.shape[1] - 1}")
    
    print(f"\n{'='*60}")
    print(f"逻辑回归建模:")
    print(f"{'='*60}")
    
    X = df_clean.drop('survived', axis=1).values
    y = df_clean['survived'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    
    model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print(f"训练集准确率: {accuracy(y_train, y_train_pred):.4f}")
    print(f"测试集准确率: {accuracy(y_test, y_test_pred):.4f}")
    print(f"训练集精确率: {precision(y_train, y_train_pred):.4f}")
    print(f"测试集精确率: {precision(y_test, y_test_pred):.4f}")
    print(f"训练集召回率: {recall(y_train, y_train_pred):.4f}")
    print(f"测试集召回率: {recall(y_test, y_test_pred):.4f}")
    print(f"训练集F1分数: {f1_score(y_train, y_train_pred):.4f}")
    print(f"测试集F1分数: {f1_score(y_test, y_test_pred):.4f}")
    
    print(f"\n{'='*60}")
    print(f"特征重要性分析:")
    print(f"{'='*60}")
    
    feature_names = df_clean.drop('survived', axis=1).columns
    coef_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_
    }).sort_values('coefficient', ascending=False)
    
    print(coef_importance)
    
    print(f"\n重要发现:")
    print(f"- 正系数表示该特征增加生存概率")
    print(f"- 负系数表示该特征降低生存概率")
    
    return {
        'dataset': titanic,
        'model': model,
        'accuracy': accuracy(y_test, y_test_pred),
        'precision': precision(y_test, y_test_pred),
        'recall': recall(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'feature_importance': coef_importance
    }

def run_all_case_studies():
    """
    Run all case studies.
    """
    print(f"{'='*70}")
    print(f"机器学习统计学学习辅助代码 - 真实数据集案例分析")
    print(f"{'='*70}")
    
    print(f"\n" + "="*70)
    print(f"案例1: Iris 鸢尾花数据集 - 分类与探索性分析")
    print("="*70)
    iris_results = iris_dataset_case_study()
    
    print(f"\n" + "="*70)
    print(f"案例2: Boston 房价数据集 - 回归分析")
    print("="*70)
    boston_results = boston_housing_case_study()
    
    print(f"\n" + "="*70)
    print(f"案例3: Titanic 泰坦尼克号数据集 - 分类分析")
    print("="*70)
    titanic_results = titanic_case_study()
    
    print(f"\n{'='*70}")
    print(f"案例分析总结")
    print(f"{'='*70}")
    
    print(f"\n1. Iris数据集:")
    print(f"   - 三种鸢尾花物种在花瓣长度上有明显差异")
    print(f"   - 花瓣长度和宽度高度相关")
    
    print(f"\n2. Boston房价数据集:")
    print(f"   - 平均房价约为 $22,532")
    print(f"   - 房间数量(RM)与房价正相关")
    print(f"   - 低收入人口比例(LSTAT)与房价负相关")
    print(f"   - 测试集R²: {boston_results['test_r2']:.4f}")
    
    print(f"\n3. Titanic数据集:")
    print(f"   - 总体生存率约为 38.2%")
    print(f"   - 模型测试集准确率: {titanic_results['accuracy']:.4f}")
    print(f"   - 模型测试集F1分数: {titanic_results['f1']:.4f}")
    
    return {
        'iris': iris_results,
        'boston': boston_results,
        'titanic': titanic_results
    }
