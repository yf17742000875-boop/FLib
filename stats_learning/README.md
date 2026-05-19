# 机器学习统计学学习辅助代码

## 概述

本项目提供一套系统化的机器学习统计学学习辅助代码，涵盖核心统计学概念及其在机器学习中的应用。代码基于 Python 语言，使用 NumPy、Pandas、Matplotlib 等主流库实现。

## 功能模块

### 1. 描述性统计 (descriptive_stats)
- 数据分布分析：均值、中位数、众数、标准差、方差、残差、四分位数
- 数据可视化：直方图、箱线图、散点图
- 相关性分析：协方差、相关系数、相关矩阵

### 2. 概率论基础 (probability)
- 概率分布：正态分布、二项分布、泊松分布、均匀分布
- 概率计算：PDF、CDF
- 贝叶斯定理：条件概率、后验概率计算
- 马尔科夫链：状态转移、平稳分布

### 3. 推断统计 (inferential_stats)
- 假设检验：单样本 t 检验、双样本 t 检验
- 卡方检验：拟合优度检验、独立性检验
- ANOVA：单因素方差分析
- 置信区间：均值置信区间计算
- p 值计算与解释

### 4. 回归分析 (regression)
- 线性回归：最小二乘法实现
- 逻辑回归：梯度下降实现
- 模型评估：R²、MSE、MAE、准确率、精确率、召回率、F1 分数
- 残差分析与可视化

### 5. 统计学习理论 (statistical_learning)
- 偏差-方差权衡：交互式演示
- 交叉验证：k-fold 交叉验证实现
- 正则化：L1 (Lasso)、L2 (Ridge)、Elastic Net
- 模型选择：学习曲线、模型复杂度分析

### 6. 数据集与可视化 (datasets & visualization)
- 真实数据集：Iris、Boston、Titanic
- 合成数据集生成
- 绘图函数：直方图、箱线图、散点图、概率分布图

## 安装要求

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

## 使用方法

### 方法一：运行交互式主程序

```bash
python -m stats_learning
```

然后按照菜单选择要学习的模块。

### 方法二：导入模块使用

```python
import stats_learning as sl

# 加载数据集
iris = sl.datasets.load_iris()

# 计算描述性统计
stats = sl.descriptive_stats.compute_summary(iris['data'][:, 0])
print(stats)

# 可视化
sl.visualization.histogram(iris['data'][:, 0], title='Sepal Length')

# 线性回归
X, y = sl.datasets.generate_synthetic_data()
model = sl.regression.LinearRegression()
model.fit(X, y)
print(f"系数: {model.coef_}")
print(f"截距: {model.intercept_}")
```

## 案例分析

### Iris 鸢尾花数据集
- 探索性数据分析
- 物种分类特征比较
- 特征相关性分析

### Boston 房价数据集
- 房价预测回归模型
- 特征重要性分析
- 模型评估

### Titanic 泰坦尼克号数据集
- 生存预测分类模型
- 特征工程
- 分类指标评估

## 文件结构

```
stats_learning/
├── __init__.py          # 包初始化
├── __main__.py          # 主入口
├── descriptive_stats.py # 描述性统计模块
├── probability.py       # 概率论基础模块
├── inferential_stats.py # 推断统计模块
├── regression.py        # 回归分析模块
├── statistical_learning.py # 统计学习理论模块
├── datasets.py          # 数据集模块
├── visualization.py     # 可视化模块
├── case_studies.py      # 案例分析模块
└── README.md            # 使用说明
```

## 学习路径建议

1. **入门**：从描述性统计开始，学习数据探索和可视化
2. **基础**：掌握概率论基础，理解概率分布
3. **进阶**：学习推断统计，掌握假设检验方法
4. **应用**：实践回归分析，建立预测模型
5. **深化**：理解统计学习理论，优化模型泛化能力

## 参考资料

- Dive into Deep Learning: https://zh.d2l.ai
- UCI Machine Learning Repository
- scikit-learn documentation

## 许可证

本项目仅供学习和研究使用。
