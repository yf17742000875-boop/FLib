"""
D2L 第2.4节：微积分（Calculus）

【目的】
微积分是深度学习的数学基础，本章节系统介绍导数、偏导数、链式法则等核心概念，
为理解自动微分和优化算法奠定理论基础。

【数学基础】
微积分主要研究函数的变化率和累积量：
- 微分（Differentiation）：研究函数的瞬时变化率
- 积分（Integration）：研究函数在区间上的累积

【与自动微分的衔接】
本章内容是理解自动微分的关键基础：
- 导数概念 → 自动微分的基本目标
- 偏导数 → 多元函数梯度计算
- 链式法则 → 反向传播算法的核心原理

【参考文献】
- D2L Chapter: https://zh.d2l.ai/chapter_preliminaries/calculus.html
- Related Concepts: 导数、偏导数、链式法则、梯度、极值

===============================
1. 核心概念定义与数学原理
===============================
"""

import numpy as np
import matplotlib.pyplot as plt


def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams["figure.figsize"] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1) or (
            isinstance(X, list) and not hasattr(X[0], "__len__")
        )

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def derivative_definition_example():
    """
    2.4.1 导数的定义

    【核心概念】
    导数（Derivative）表示函数在某一点的瞬时变化率：
    
        f'(x) = lim(h→0) [f(x+h) - f(x)] / h
    
    这个极限定义了函数 f 在点 x 处的切线斜率。

    【几何意义】
    - 导数 f'(x) 是曲线 y=f(x) 在点 (x, f(x)) 处的切线斜率
    - 正导数表示函数递增，负导数表示函数递减
    - 导数为零的点可能是极值点

    【与自动微分的衔接】
    自动微分正是基于此定义，通过计算图追踪操作，在反向传播时
    自动应用链式法则计算梯度，避免了数值微分的误差累积问题。
    """
    print("========2.4.1 导数的定义========")

    def f(x):
        return 3 * x ** 2 - 4 * x

    def numerical_lim(f, x, h):
        """数值计算导数的近似值（有限差分法）"""
        return (f(x + h) - f(x)) / h

    h = 0.1
    for i in range(5):
        print(f'h={h:.5f}, f\'(1)≈{numerical_lim(f, 1, h):.5f}')
        h *= 0.1

    # 解析解：f'(x) = 6x - 4，f'(1) = 2
    print(f"解析解 f'(1) = 6*1 - 4 = 2")

    # 绘制函数和切线
    x = np.arange(0, 3, 0.1)
    plot(x, [f(x), 4 * x - 4], 'x', 'f(x)', 
         legend=['f(x) = 3x² - 4x', 'Tangent line at x=1'],
         figsize=(6, 4))
    plt.title('Function and Tangent Line')
    plt.savefig('derivative_tangent.png')
    print("\n图像已保存为 derivative_tangent.png")


def basic_derivatives():
    """
    2.4.2 基本导数公式与法则

    【基本初等函数导数】
    - d/dx [c] = 0（常数的导数为0）
    - d/dx [x^n] = n*x^(n-1)（幂函数法则）
    - d/dx [sin(x)] = cos(x)
    - d/dx [cos(x)] = -sin(x)
    - d/dx [e^x] = e^x
    - d/dx [ln(x)] = 1/x

    【导数运算法则】
    1. 和差法则：(f ± g)' = f' ± g'
    2. 乘积法则：(f * g)' = f'*g + f*g'
    3. 商法则：(f/g)' = (f'*g - f*g') / g²
    4. 复合函数法则（链式法则）：(f∘g)'(x) = f'(g(x)) * g'(x)

    【链式法则的重要性】
    链式法则是深度学习反向传播的数学基础。在自动微分中，
    每个操作节点的梯度都是通过链式法则从后续节点累积而来。
    """
    print("\n========2.4.2 基本导数公式与法则========")

    def power_rule_example():
        """幂函数求导示例"""
        def f(x):
            return x ** 3
        
        def f_prime(x):
            return 3 * x ** 2  # 幂函数法则：d/dx x^3 = 3x^2
        
        x = 2.0
        print(f"f(x) = x³")
        print(f"f({x}) = {f(x)}")
        print(f"f'({x}) = {f_prime(x)}")

    def chain_rule_example():
        """链式法则示例"""
        def outer(u):
            return np.sin(u)
        
        def inner(x):
            return 2 * x + 1
        
        def compose(x):
            return outer(inner(x))
        
        # 复合函数求导：d/dx [sin(2x+1)] = cos(2x+1) * 2
        x = 1.0
        u = inner(x)
        df_du = np.cos(u)  # d/du sin(u) = cos(u)
        du_dx = 2          # d/dx (2x+1) = 2
        df_dx = df_du * du_dx  # 链式法则
        
        print(f"\n复合函数 f(x) = sin(2x+1)")
        print(f"f({x}) = {compose(x):.4f}")
        print(f"f'({x}) = cos(2*{x}+1) * 2 = {df_dx:.4f}")

    power_rule_example()
    chain_rule_example()


def partial_derivatives():
    """
    2.4.3 偏导数与梯度

    【偏导数定义】
    对于多元函数 f(x₁, x₂, ..., xₙ)，偏导数 ∂f/∂xᵢ 表示在其他变量固定时，
    f 关于 xᵢ 的变化率：
    
        ∂f/∂xᵢ = lim(h→0) [f(x₁,...,xᵢ+h,...,xₙ) - f(x₁,...,xᵢ,...,xₙ)] / h

    【梯度向量】
    梯度（Gradient）是由所有偏导数组成的向量：
    
        ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]^T

    梯度方向是函数值增长最快的方向，负梯度方向是下降最快的方向。

    【与自动微分的衔接】
    在深度学习中，损失函数 L(θ₁, θ₂, ..., θₙ) 是关于模型参数的多元函数，
    自动微分系统自动计算 ∇L，这正是梯度下降优化的核心。
    """
    print("\n========2.4.3 偏导数与梯度========")

    def f(x, y):
        """二元函数示例：f(x, y) = x² + xy + y²"""
        return x ** 2 + x * y + y ** 2

    def numerical_partial_x(f, x, y, h=1e-5):
        """数值计算 ∂f/∂x"""
        return (f(x + h, y) - f(x, y)) / h

    def numerical_partial_y(f, x, y, h=1e-5):
        """数值计算 ∂f/∂y"""
        return (f(x, y + h) - f(x, y)) / h

    x, y = 2.0, 3.0
    
    # 数值计算偏导数
    grad_x_num = numerical_partial_x(f, x, y)
    grad_y_num = numerical_partial_y(f, x, y)
    
    # 解析解：∂f/∂x = 2x + y，∂f/∂y = x + 2y
    grad_x_analytic = 2 * x + y
    grad_y_analytic = x + 2 * y
    
    print(f"函数 f(x, y) = x² + xy + y²")
    print(f"在点 ({x}, {y}) 处：")
    print(f"  ∂f/∂x（数值）= {grad_x_num:.6f}")
    print(f"  ∂f/∂x（解析）= {grad_x_analytic}")
    print(f"  ∂f/∂y（数值）= {grad_y_num:.6f}")
    print(f"  ∂f/∂y（解析）= {grad_y_analytic}")
    print(f"  梯度 ∇f = [{grad_x_analytic}, {grad_y_analytic}]")


def chain_rule_multivariable():
    """
    2.4.4 多元复合函数的链式法则

    【链式法则一般形式】
    设 z = f(u₁, u₂, ..., uₘ)，且 uᵢ = gᵢ(x₁, x₂, ..., xₙ)，则：
    
        ∂z/∂xⱼ = Σ(∂z/∂uᵢ * ∂uᵢ/∂xⱼ)  （i从1到m）

    【深度学习中的应用】
    在神经网络中，每个层的输出是前一层的函数，通过链式法则，
    我们可以逐层反向传播梯度。这正是PyTorch等框架自动微分的核心原理。

    【计算图表示】
    计算图将复合函数分解为基本操作节点，反向传播时：
    1. 从输出层开始
    2. 逐层应用链式法则
    3. 将梯度传递到输入层
    """
    print("\n========2.4.4 多元复合函数的链式法则========")

    def example_chain_rule():
        """
        示例：z = sin(u)，u = x² + y²
        求 ∂z/∂x 和 ∂z/∂y
        
        链式法则：
        ∂z/∂x = ∂z/∂u * ∂u/∂x = cos(u) * 2x
        ∂z/∂y = ∂z/∂u * ∂u/∂y = cos(u) * 2y
        """
        x, y = 1.0, 2.0
        u = x ** 2 + y ** 2
        z = np.sin(u)
        
        dz_du = np.cos(u)
        du_dx = 2 * x
        du_dy = 2 * y
        
        dz_dx = dz_du * du_dx
        dz_dy = dz_du * du_dy
        
        print(f"复合函数：z = sin(x² + y²)")
        print(f"在点 ({x}, {y}) 处：")
        print(f"  u = {u}")
        print(f"  z = sin({u}) = {z:.4f}")
        print(f"  ∂z/∂x = cos({u}) * 2*{x} = {dz_dx:.4f}")
        print(f"  ∂z/∂y = cos({u}) * 2*{y} = {dz_dy:.4f}")

    example_chain_rule()


def applications():
    """
    2.4.5 典型应用场景

    【场景1：优化问题】
    寻找函数的极值点，梯度为零的点是候选极值点。

    【场景2：切线与线性近似】
    f(x) ≈ f(a) + f'(a)(x-a)，用于局部线性化。

    【场景3：牛顿法求根】
    xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)

    【场景4：深度学习】
    - 梯度下降优化
    - 反向传播算法
    - 损失函数优化

    【场景5：物理建模】
    速度是位移的导数，加速度是速度的导数。
    """
    print("\n========2.4.5 典型应用场景========")

    def newtons_method():
        """牛顿法求方程根"""
        def f(x):
            return x ** 3 - x - 1  # 求 f(x) = 0 的根
        
        def f_prime(x):
            return 3 * x ** 2 - 1  # f'(x) = 3x² - 1
        
        x = 1.0
        for i in range(5):
            x_new = x - f(x) / f_prime(x)
            print(f"迭代 {i+1}: x = {x:.6f}, f(x) = {f(x):.6f}")
            x = x_new
        
        print(f"方程 x³ - x - 1 = 0 的根近似为: {x:.6f}")

    def linear_approximation():
        """线性近似示例"""
        def f(x):
            return np.sqrt(x)
        
        def f_prime(x):
            return 1 / (2 * np.sqrt(x))
        
        a = 4.0  # 在 x=4 处进行线性近似
        x = 4.1
        
        # 线性近似：f(x) ≈ f(a) + f'(a)(x-a)
        approx = f(a) + f_prime(a) * (x - a)
        exact = f(x)
        
        print(f"\n在 x={a} 处对 f(x)=√x 进行线性近似")
        print(f"f({x}) 的精确值 = {exact:.6f}")
        print(f"f({x}) 的近似值 = {approx:.6f}")
        print(f"误差 = {abs(exact - approx):.6f}")

    newtons_method()
    linear_approximation()


def common_errors():
    """
    2.4.6 常见问题与易错点

    【问题1：混淆导数与偏导数】
    - 导数：一元函数的变化率
    - 偏导数：多元函数中某一变量的变化率（其他变量固定）

    【问题2：数值微分的精度问题】
    - h太小：浮点精度损失
    - h太大：截断误差增大
    - 解决方案：使用中心差分或自动微分

    【问题3：链式法则应用错误】
    - 遗漏中间变量的导数
    - 符号错误
    - 解决方案：绘制计算图辅助分析

    【问题4：梯度消失与爆炸】
    - 梯度消失：链式相乘后梯度趋近于0
    - 梯度爆炸：链式相乘后梯度趋近于无穷
    - 解决方案：梯度裁剪、归一化、合适的初始化

    【问题5：导数不存在的情况】
    - 函数不连续
    - 尖点（如绝对值函数在x=0处）
    """
    print("\n========2.4.6 常见问题与易错点========")

    def numerical_precision_issue():
        """数值微分的精度问题演示"""
        def f(x):
            return np.exp(x)
        
        x = 1.0
        
        # 尝试不同的h值
        for h in [1e-1, 1e-5, 1e-10, 1e-15]:
            approx = (f(x + h) - f(x)) / h
            exact = np.exp(x)
            error = abs(approx - exact)
            print(f"h={h:.0e}: 近似值={approx:.10f}, 精确值={exact:.10f}, 误差={error:.2e}")

    numerical_precision_issue()


def exercises():
    """
    2.4.7 练习题与知识巩固

    【练习1】计算函数 f(x) = x³ - 3x + 1 在 x=2 处的导数
    【练习2】计算函数 f(x, y) = x²y + xy² 在点 (1, 2) 处的梯度
    【练习3】使用链式法则计算复合函数 f(g(x)) 的导数，其中 f(u)=sin(u), g(x)=x²
    【练习4】用数值方法验证练习3的结果
    【练习5】绘制函数 f(x) = x³ - x 及其导数的图像
    """
    print("\n========2.4.7 练习题解答========")

    def exercise1():
        """练习1：f(x) = x³ - 3x + 1，求 f'(2)"""
        def f(x):
            return x ** 3 - 3 * x + 1
        
        def f_prime(x):
            return 3 * x ** 2 - 3  # 解析解
        
        x = 2.0
        print(f"练习1：f(x) = x³ - 3x + 1")
        print(f"f'(x) = 3x² - 3")
        print(f"f'({x}) = {f_prime(x)}")

    def exercise2():
        """练习2：f(x, y) = x²y + xy²，求在(1, 2)处的梯度"""
        def f(x, y):
            return x ** 2 * y + x * y ** 2
        
        x, y = 1.0, 2.0
        # ∂f/∂x = 2xy + y²
        # ∂f/∂y = x² + 2xy
        grad_x = 2 * x * y + y ** 2
        grad_y = x ** 2 + 2 * x * y
        
        print(f"\n练习2：f(x, y) = x²y + xy²")
        print(f"∂f/∂x = 2xy + y² = {grad_x}")
        print(f"∂f/∂y = x² + 2xy = {grad_y}")
        print(f"梯度 ∇f({x}, {y}) = [{grad_x}, {grad_y}]")

    def exercise3():
        """练习3：链式法则应用"""
        def f(u):
            return np.sin(u)
        
        def g(x):
            return x ** 2
        
        x = 1.0
        u = g(x)
        
        # df/dx = df/du * du/dx = cos(u) * 2x
        df_du = np.cos(u)
        du_dx = 2 * x
        df_dx = df_du * du_dx
        
        print(f"\n练习3：f(g(x)) = sin(x²)")
        print(f"df/dx = cos(x²) * 2x")
        print(f"在 x={x} 处，df/dx = {df_dx:.4f}")

    def exercise4():
        """练习4：数值验证"""
        def f(x):
            return np.sin(x ** 2)
        
        x = 1.0
        h = 1e-5
        
        # 数值微分
        approx = (f(x + h) - f(x)) / h
        
        # 解析解
        exact = np.cos(x ** 2) * 2 * x
        
        print(f"\n练习4：数值验证")
        print(f"数值近似值 = {approx:.6f}")
        print(f"解析精确值 = {exact:.6f}")
        print(f"相对误差 = {abs(approx - exact) / abs(exact):.2e}")

    def exercise5():
        """练习5：绘制函数及其导数"""
        def f(x):
            return x ** 3 - x
        
        def f_prime(x):
            return 3 * x ** 2 - 1
        
        x = np.linspace(-2, 2, 100)
        
        plot(x, [f(x), f_prime(x)], 'x', 'y',
             legend=['f(x) = x³ - x', "f'(x) = 3x² - 1"],
             figsize=(6, 4))
        plt.title('Function and its Derivative')
        plt.savefig('function_derivative.png')
        print("\n练习5：图像已保存为 function_derivative.png")

    exercise1()
    exercise2()
    exercise3()
    exercise4()
    exercise5()


def main():
    """主函数：运行所有示例"""
    print("D2L 2.4 微积分（数学基础）")
    print("="*50)
    
    derivative_definition_example()
    basic_derivatives()
    partial_derivatives()
    chain_rule_multivariable()
    applications()
    common_errors()
    exercises()


if __name__ == "__main__":
    main()


"""
【章节总结】

1. 导数是函数的瞬时变化率，是理解深度学习优化的基础
2. 偏导数扩展了导数概念到多元函数，梯度是偏导数向量
3. 链式法则是自动微分和反向传播的核心数学原理
4. 数值微分存在精度问题，自动微分提供了精确高效的解决方案
5. 微积分知识是理解后续自动微分章节的关键前提

【与自动微分章节的衔接要点】
- 导数概念 → 自动微分的计算目标
- 偏导数 → 梯度向量的组成元素
- 链式法则 → 反向传播的核心算法
- 数值微分 vs 自动微分 → 精度与效率的对比
"""