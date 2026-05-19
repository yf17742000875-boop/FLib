import math
import time
import numpy as np
import torch
from d2l import torch as d2l

"""
==============================================================================
3.1.6 练习解答
==============================================================================
"""

# ==============================================================================
# 练习1：最小化平方和常数b的解析解
# ==============================================================================
"""
【题目】
假设我们有一些数据x₁,...,xₙ ∈ ℝ。目标是找到常数b，使得最小化 Σ(xᵢ - b)²

【解答】

1. 最优值b的解析解：

解题思路：
- 对目标函数关于b求导，令导数为0，解出b

关键步骤：
- 目标函数：L(b) = Σᵢ(xᵢ - b)²
- 求导：dL/db = Σᵢ 2(xᵢ - b)(-1) = -2Σᵢ(x - b)
- 令导数为0：-2Σᵢ(xᵢ - b) = 0
- 展开：Σᵢxᵢ - nb = 0
- 解得：b = (1/n)Σᵢxᵢ = x̄（样本均值）

使用的知识点：
- 微积分求极值（求导+令导数为0）
- 二次函数是凸函数，驻点即为全局最小值

"""

def exercise_1_solution():
    """练习1：最小化平方和的解析解"""
    # 生成示例数据
    np.random.seed(42)
    x = np.random.randn(100)
    
    # 解析解：b = 样本均值
    b_analytical = np.mean(x)
    print(f"数据样本均值: {b_analytical:.4f}")
    
    # 验证：计算不同b值对应的损失
    b_test = np.linspace(b_analytical - 1, b_analytical + 1, 100)
    losses = [np.sum((x - b)**2) for b in b_test]
    
    print(f"最优b = {b_analytical:.4f} 时，损失 = {np.sum((x - b_analytical)**2):.4f}")
    print(f"b = {b_analytical + 0.1:.4f} 时，损失 = {np.sum((x - (b_analytical + 0.1))**2):.4f}")
    print("损失确实是最小的（因为均值是最优解）")
    
    return b_analytical

"""
2. 与正态分布的关系：

解题思路：
- 如果假设数据来自正态分布 N(μ, σ²)，那么最大似然估计是什么？

关键步骤：
- 正态分布概率密度：p(x|μ,σ²) = (1/√(2πσ²))exp(-(x-μ)²/(2σ²))
- 对数似然：log L(μ) = Σᵢ[-½log(2πσ²) - (xᵢ-μ)²/(2σ²)]
- 最大化对数似然等价于最小化 Σᵢ(xᵢ-μ)²
- 所以μ的MLE = (1/n)Σᵢxᵢ = b的最优解

使用的知识点：
- 最大似然估计（MLE）
- 正态分布的性质
- 最小二乘法与最大似然估计的等价性

结论：
最小化平方和等价于假设数据服从正态分布时的最大似然估计。
这就是为什么线性回归通常假设噪声服从正态分布——因为这样最小化平方误差
就等价于最大化似然函数。
"""


# ==============================================================================
# 练习2：线性回归的解析解推导
# ==============================================================================
"""
【题目】
推导使用平方误差的线性回归优化问题的解析解（忽略偏置b）

【解答】
"""

def exercise_2_solution():
    """练习2：线性回归解析解推导"""
    print("=" * 60)
    print("练习2：线性回归的解析解")
    print("=" * 60)
    
    """
    1. 用矩阵和向量表示优化问题：
    
    解题思路：
    - 将所有样本表示为矩阵形式
    - 将目标函数写成矩阵运算形式
    
    关键步骤：
    - 数据矩阵 X ∈ ℝⁿˣᵈ，每行是一个样本
    - 权重向量 w ∈ ℝᵈ
    - 目标向量 y ∈ ℝⁿ
    - 预测值：ŷ = Xw
    - 损失函数：L(w) = (1/2)||y - Xw||² = (1/2)(y - Xw)ᵀ(y - Xw)
    
    使用的知识点：
    - 矩阵乘法
    - 向量范数（L2范数）
    """
    
    # 生成示例数据
    np.random.seed(42)
    n, d = 100, 3
    X = np.random.randn(n, d)
    true_w = np.array([2, -1, 0.5])
    y = X @ true_w + 0.1 * np.random.randn(n)
    
    print(f"数据矩阵X形状: {X.shape}")
    print(f"真实权重w: {true_w}")
    
    """
    2. 计算损失对w的梯度：
    
    解题思路：
    - 展开损失函数
    - 对w求导
    
    关键步骤：
    - L(w) = (1/2)(y - Xw)ᵀ(y - Xw)
    - 展开：L(w) = (1/2)(yᵀy - yᵀXw - wᵀXᵀy + wXᵀXw)
    - 注意：yᵀXw = wᵀXᵀy（都是标量，转置不变）
    - 简化：L(w) = (1/2)(yᵀy - 2wᵀXᵀy + wᵀXᵀXw)
    - 求导：∇L(w) = (1/2)(0 - 2Xᵀy + 2XᵀXw) = XXw - Xᵀy
    
    使用的知识点：
    - 矩阵微积分
    - 标量对向量求导规则
    """
    
    """
    3. 通过令梯度为0求解：
    
    解题思路：
    - 令∇L(w) = 0，解线性方程组
    
    关键步骤：
    - XᵀXw - Xᵀy = 0
    - XᵀXw = Xᵀy
    - w = (XᵀX)⁻¹Xy（假设XᵀX可逆）
    
    使用的知识点：
    - 线性方程组求解
    - 矩阵求逆
    - 正规方程（Normal Equation）
    """
    
    # 用解析解计算w
    w_analytical = np.linalg.solve(X.T @ X, X.T @ y)
    print(f"\n解析解 w = {w_analytical}")
    print(f"真实权重 w = {true_w}")
    print(f"误差 = {np.abs(w_analytical - true_w)}")
    
    """
    4. 什么时候随机梯度下降更好？
    
    解题思路：
    - 比较解析解和SGD的计算复杂度、适用场景
    
    关键对比：
    
    | 方面 | 解析解 (XᵀX)⁻¹Xᵀy | 随机梯度下降 (SGD) |
    |------|-------------------|-------------------|
    | 时间复杂度 | O(nd² + d³) | O(kn)，k为迭代次数 |
    | 空间复杂度 | O(nd + d²) | O(nd) |
    | 数值稳定性 | XX可能病态 | 更稳定 |
    | 大规模数据 | 慢（需全部数据） | 快（逐个样本） |
    | 在线学习 | 不支持 | 支持 |
    | 内存需求 | 需存储所有数据 | 可分批处理 |
    
    什么时候SGD更好：
    1. 样本数n非常大（百万级）时，(XᵀX)⁻¹计算太慢
    2. 特征数d很大时，d³矩阵求逆开销大
    3. XᵀX接近奇异（特征共线性）时，解析解数值不稳定
    4. 需要在线学习（数据流式到达）时
    5. 内存有限，无法加载全部数据时
    
    什么时候解析解更好：
    1. 数据量较小（n < 10000）
    2. 需要精确解而非近似解
    3. 计算资源充足
    """
    
    # 演示SGD求解
    lr = 0.01
    epochs = 100
    w_sgd = np.zeros(d)
    
    for epoch in range(epochs):
        for i in range(n):
            xi = X[i]
            yi = y[i]
            gradient = xi * (xi @ w_sgd - yi)
            w_sgd -= lr * gradient
    
    print(f"\nSGD解 w = {w_sgd}")
    print(f"真实权重 w = {true_w}")
    print(f"SGD误差 = {np.abs(w_sgd - true_w)}")


# ==============================================================================
# 练习3：指数分布噪声下的线性回归
# ==============================================================================
"""
【题目】
假设附加噪声ε的模型是指数分布（实际上是Laplace分布）：
p(ε) = (1/2)exp(-|ε|)

【解答】
"""

def exercise_3_solution():
    """练习3：Laplace噪声下的线性回归"""
    print("\n" + "=" * 60)
    print("练习3：指数分布噪声下的线性回归")
    print("=" * 60)
    
    """
    1. 写出负对数似然：
    
    解题思路：
    - 写出似然函数
    - 取负对数
    
    关键步骤：
    - 模型：y = wᵀx + ε，其中ε ~ Laplace(0, 1)
    - 条件概率：p(y|x;w) = (1/2)exp(-|y - wx|)
    - 似然函数：L(w) = Πᵢ(1/2)exp(-|yᵢ - wᵀxᵢ|)
    - 对数似然：log L(w) = Σᵢ[-log(2) - |yᵢ - wᵀxᵢ|]
    - 负对数似然：-log L(w) = n·log(2) + Σᵢ|y - wᵀxᵢ|
    
    简化后（忽略常数项）：
    L(w) = Σᵢ|yᵢ - wxᵢ|
    
    使用的知识点：
    - 最大似然估计
    - Laplace分布的性质
    - 对数运算
    """
    
    """
    2. 解析解：
    
    解题思路：
    - 对损失函数求导
    - 注意绝对值函数的导数是符号函数
    
    关键步骤：
    - L(w) = Σ|yᵢ - wxᵢ|
    - L/∂w = Σᵢ sign(yᵢ - wxᵢ)·(-x) = -Σᵢ sign(y - wᵀxᵢ)·xᵢ
    - 其中 sign(z) = +1 (z>0), -1 (z<0), [-1,1] (z=0)
    - 令梯度为0：Σᵢ sign(y - wᵀx)·xᵢ = 0
    
    这个方程没有闭式解！
    - 因为sign函数是非线性的
    - 与最小二乘法不同，这里不能用矩阵运算直接求解
    
    使用的知识点：
    - 次梯度（subgradient）
    - 中位数回归（Median Regression）
    """
    
    # 生成示例数据
    np.random.seed(42)
    n, d = 100, 2
    X = np.random.randn(n, d)
    true_w = np.array([2, -1])
    # Laplace噪声
    laplace_noise = np.random.laplace(0, 1, n)
    y = X @ true_w + laplace_noise
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"真实权重: {true_w}")
    
    """
    3. 随机梯度下降算法及可能的问题：
    
    解题思路：
    - 实现SGD更新
    - 分析驻点附近的问题
    - 提出解决方案
    
    可能出错的地方：
    - 当 yᵢ ≈ wᵀxᵢ 时，sign函数在0附近振荡
    - 次梯度在0处不唯一，导致更新不稳定
    - 学习率过大时，参数在最优值附近来回震荡
    
    提示分析：
    - 当参数接近驻点时，|yᵢ - wxᵢ| ≈ 0
    - 此时 sign(0) 在[-1, +1]之间
    - 不同样本可能给出矛盾的方向
    - 导致参数在驻点附近震荡，无法收敛
    
    解决方案：
    1. 使用学习率衰减（Learning Rate Decay）
    2. 使用Huber损失代替绝对值损失（平滑化）
    3. 使用次梯度方法，当接近0时使用0作为梯度
    4. 使用更复杂的优化器（如AdaGrad, Adam）
    """
    
    # SGD实现
    lr = 0.1
    epochs = 50
    w_sgd = np.zeros(d)
    
    for epoch in range(epochs):
        lr_decay = lr / (1 + epoch * 0.01)  # 学习率衰减
        for i in range(n):
            xi = X[i]
            yi = y[i]
            residual = yi - xi @ w_sgd
            
            # 次梯度（绝对值函数的导数）
            if residual > 0:
                gradient = -xi
            elif residual < 0:
                gradient = xi
            else:
                gradient = np.zeros(d)  # 驻点处使用0梯度
            
            w_sgd -= lr_decay * gradient
    
    print(f"\nSGD解: {w_sgd}")
    print(f"真实权重: {true_w}")
    print(f"误差: {np.abs(w_sgd - true_w)}")
    
    # 对比最小二乘法
    w_ls = np.linalg.solve(X.T @ X, X.T @ y)
    print(f"\n最小二乘解: {w_ls}")
    print(f"真实权重: {true_w}")
    print(f"最小二乘误差: {np.abs(w_ls - true_w)}")
    
    """
    解决方案代码：Huber损失（平滑绝对值）
    
    Huber损失结合了MSE和MAE的优点：
    - 小误差时使用平方损失（平滑，易优化）
    - 大误差时使用绝对值损失（对异常值鲁棒）
    """
    
    def huber_loss(residual, delta=1.0):
        """Huber损失函数"""
        if np.abs(residual) <= delta:
            return 0.5 * residual ** 2
        else:
            return delta * (np.abs(residual) - 0.5 * delta)
    
    def huber_loss_gradient(residual, delta=1.0):
        """Huber损失的梯度"""
        if np.abs(residual) <= delta:
            return residual  # 二次区域，梯度线性
        elif residual > delta:
            return delta  # 线性区域，梯度恒定
        else:
            return -delta  # 线性区域，梯度恒定
    
    # 使用Huber损失的SGD
    w_huber = np.zeros(d)
    lr = 0.1
    
    for epoch in range(epochs):
        lr_decay = lr / (1 + epoch * 0.01)
        for i in range(n):
            xi = X[i]
            yi = y[i]
            residual = yi - xi @ w_huber
            
            gradient = huber_loss_gradient(residual, delta=1.0) * (-xi)
            w_huber -= lr_decay * gradient
    
    print(f"\nHuber损失SGD解: {w_huber}")
    print(f"真实权重: {true_w}")
    print(f"Huber误差: {np.abs(w_huber - true_w)}")


# ==============================================================================
# 运行所有练习
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("D2L 第3章 线性回归 - 练习解答")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("练习1：最小化平方和")
    print("=" * 60)
    exercise_1_solution()
    
    exercise_2_solution()
    
    exercise_3_solution()
    
    print("\n" + "=" * 60)
    print("练习总结")
    print("=" * 60)
    print("""
【练习1总结】
- 最优b是样本均值
- 与正态分布MLE等价
- 最小二乘 = 正态分布假设下的最大似然

【练习2总结】
- 解析解：w = (XX)⁻¹Xy
- SGD优势：大规模数据、在线学习、数值稳定
- 解析解优势：精确解、小数据高效

【练习3总结】
- Laplace噪声对应绝对值损失（L1损失）
- 没有闭式解析解（sign函数非线性）
- SGD在驻点附近振荡，需用学习率衰减或Huber损失
    """)
