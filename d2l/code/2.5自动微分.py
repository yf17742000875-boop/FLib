"""
D2L 第2.5节：自动微分（Automatic Differentiation）

【目的】
自动微分是深度学习框架的核心技术，它将微积分中的导数、偏导数和链式法则
自动化实现，使开发者能够高效计算复杂函数的梯度，无需手动推导公式。

【与微积分章节的衔接】
本章是微积分章节的自然延伸和工程实现：
- 导数（Derivative）→ 自动微分的计算目标
- 偏导数（Partial Derivative）→ 多元函数梯度计算
- 链式法则（Chain Rule）→ 反向传播算法的核心原理

【核心概念】
自动微分是一种介于数值微分和符号微分之间的计算技术：
- 数值微分：通过有限差分近似，存在精度和稳定性问题
- 符号微分：通过代数推导，可能产生复杂表达式
- 自动微分：通过追踪计算图，在前向传播时记录操作，在反向传播时
  应用链式法则自动计算梯度，兼具精度和效率

【前向模式 vs 反向模式】
- 前向模式（Forward Mode）：从输入到输出，同时计算函数值和导数值
  复杂度：O(n) 次前向传播，适合输入维度 n 远小于输出维度 m 的场景
- 反向模式（Reverse Mode）：先前向传播记录计算图，再反向传播计算梯度
  复杂度：O(1) 次前向 + O(1) 次反向，适合输入维度远大于输出维度的场景
  PyTorch 默认使用反向模式（reverse-mode AD）

【计算图原理】
计算图是一个有向无环图（DAG），节点表示张量或操作：
- 叶子节点（Leaf Nodes）：用户输入的变量（requires_grad=True）
- 中间节点（Internal Nodes）：操作的结果
- 反向传播：从根节点（通常是标量损失）沿图逆向传播梯度

【梯度累积机制】
PyTorch 的梯度默认会累积（accumulate），而非覆盖。这意味着：
- 多次调用 backward() 会将梯度累加
- 训练循环中需要手动清零梯度（optimizer.zero_grad() 或 tensor.grad.zero_()）

【参考文献】
- D2L Chapter: https://zh.d2l.ai/chapter_preliminaries/autograd.html
- Related Concepts: 导数、偏导数、链式法则、梯度、计算图、反向传播
"""

import torch


def simple_example():
    """
    2.5.1 简单示例：计算 y = 2 * x^T x 的梯度
    
    【数学原理】
        设 x = [x₀, x₁, x₂, x₃]^T
        y = 2 * (x₀² + x₁² + x₂² + x₃²)
        ∂y/∂xᵢ = 4 * xᵢ  （对每个分量）
    
    【与微积分的衔接】
        这是多元函数求偏导数的典型例子：
        - y 是关于 x 的多元函数
        - x.grad 存储的是梯度向量 ∇y = [∂y/∂x₀, ∂y/∂x₁, ∂y/∂x₂, ∂y/∂x₃]
        - 自动微分自动计算每个偏导数，无需手动推导
    
    【关键操作】
        1. torch.arange(4.0, requires_grad=True) - 创建可求导张量
        2. torch.dot(x, x) - 向量点积，等价于 x^T x
        3. y.backward() - 触发反向传播，计算梯度
        4. x.grad - 存储计算得到的梯度
    
    【预期输出】
        x: tensor([0., 1., 2., 3.], requires_grad=True)
        y: 28.0  (2*(0+1+4+9) = 2*14 = 28)
        x.grad: tensor([0., 4., 8., 12.])  (4*x)
    """
    print("========2.5.1 简单例子========")

    # 创建形状为(4,)的浮点张量，requires_grad=True表示需要追踪梯度
    # 参数说明：
    #   start=0（默认）, end=4, step=1（默认）
    #   requires_grad: bool类型，控制是否计算梯度，默认为False
    #   dtype: 默认float32，此处显式为float64（通过4.0）
    x = torch.arange(4.0, requires_grad=True)
    
    # 计算 y = 2 * x^T x
    # torch.dot 要求两个张量都是1D的，返回标量
    # 计算图：x -> dot(x,x) -> multiply(2) -> y
    y = 2 * torch.dot(x, x)
    
    # 触发反向传播，从y开始沿计算图反向计算梯度
    # 对于标量输出，无需指定gradient参数
    y.backward()

    print("x:", x)
    print("y:", y.item())
    print("x.grad:", x.grad)
    print("验证 x.grad == 4 * x:", torch.allclose(x.grad, 4 * x))
    
    # 梯度累积演示：不清零直接再次计算会累加
    x.grad.zero_()  # 手动清零梯度
    y = torch.dot(x, x)  # y = x^T x，此时dy/dx = 2x
    y.backward()
    print("x.grad:", x.grad)  # 输出 tensor([0., 2., 4., 6.])


def backward_for_vector_output():
    """
    2.5.2 非标量变量的反向传播
    
    技术要点：
        PyTorch的backward()默认只对标量输出有效。
        对于向量或矩阵输出，需要先聚合为标量（如sum），或指定gradient参数。
    
    数学原理：
        y = x * x = [x₀², x₁², x₂², x₃²]^T
        若 z = sum(y) = Σxᵢ²
        则 ∂z/∂xᵢ = 2xᵢ
    
    常见错误提示：
        若直接调用 y.backward()（y为向量），会报错：
        "RuntimeError: grad can be implicitly created only for scalar outputs"
        解决方案：使用 y.sum().backward() 或 y.backward(torch.ones_like(y))
    """
    print("========2.5.2 非标量反向传播========")

    x = torch.arange(4.0, requires_grad=True)
    y = x * x  # 逐元素相乘，输出形状与x相同，为向量

    # 对向量输出求梯度时，通常先转换为标量
    # y.sum() 将向量聚合为标量，等价于求所有元素的梯度之和
    y.sum().backward()

    print("x:", x)
    print("y = x*x:", y)
    print("x.grad:", x.grad)
    print("验证 x.grad == 2 * x:", torch.allclose(x.grad, 2 * x))


def detach_example():
    """
    2.5.3 分离计算（Detach）
    
    detach() 方法作用：
        1. 返回一个与原张量共享数据但不追踪梯度的新张量
        2. 从计算图中分离该张量，阻断梯度回传路径
        3. 常用于冻结部分模型参数（如迁移学习）或创建常量
    
    应用场景：
        - 迁移学习中冻结特征提取器
        - 计算梯度时排除某些计算分支
        - 提高推理性能（无需追踪梯度）
    
    本示例计算流程：
        x -> y = x*x -> u = detach(y) -> z = u*x -> sum(z)
        梯度回传路径：sum(z) -> z -> x（u被视为常量，不回传梯度到y）
    
    数学分析：
        u = y.detach() 时，u的值等于y，但梯度不回传
        z = u * x，其中u为常量
        ∂z/∂x = u = y = x*x
    """
    print("========2.5.3 分离计算 detach========")

    x = torch.arange(4.0, requires_grad=True)
    y = x * x                    # y = x²
    u = y.detach()               # u 从计算图中分离，成为常量
    z = u * x                    # z = u * x，u不参与梯度计算

    z.sum().backward()           # 计算 ∂(sum(z))/∂x
    print("x.grad (来自 z = u*x):", x.grad)
    print("x.grad == u:", torch.allclose(x.grad, u))  # 预期为True

    x.grad.zero_()               # 清零梯度
    y.sum().backward()           # 重新计算y的梯度
    print("对 y 再反向传播后 x.grad:", x.grad)
    print("验证 x.grad == 2*x:", torch.allclose(x.grad, 2 * x))  # dy/dx = 2x


def f(a: torch.Tensor) -> torch.Tensor:
    """
    2.5.4 Python 控制流的梯度计算
    
    自动微分的强大之处：
        PyTorch能够处理任意Python控制流（if/while/for等），
        无需手动构建静态计算图。这是动态计算图的核心优势。
    
    函数逻辑：
        b = 2*a
        while ||b|| < 1000: b = 2*b  (范数小于1000时继续倍增)
        if sum(b) > 0: c = b
        else: c = 100*b
        return c
    
    参数说明：
        a: torch.Tensor - 输入标量张量，需设置requires_grad=True
        返回: torch.Tensor - 输出张量，保留梯度追踪信息
    
    梯度分析：
        由于控制流由输入值决定，每次执行的计算图可能不同，
        但PyTorch会在运行时动态构建并记录计算路径，
        反向传播时只沿实际执行的路径回传梯度。
        
        设最终 c = k * a（k为某个常数，由while循环次数和if条件决定），
        则 dc/da = k = c/a
    """
    b = a * 2
    while b.norm() < 1000:       # 动态while循环
        b = b * 2
    if b.sum() > 0:              # 动态条件分支
        c = b
    else:
        c = 100 * b
    return c


def control_flow_grad_example():
    """
    2.5.4 控制流梯度计算示例

    演示要点：
        1. 随机输入 a，观察不同输入下的梯度
        2. 验证梯度公式：da/dc = c/a（因为c是a的线性变换）
        3. 展示动态图的灵活性

    性能优化提示：
        动态图每次前向传播都重新构建计算图，在循环中使用时可能产生开销。
        对于固定结构的计算，可以考虑使用 torch.jit.trace 或 torch.jit.script
        进行编译优化。
    """
    print("========2.5.4 控制流梯度计算========")

    # 创建随机标量张量，requires_grad=True
    # size=() 表示标量，等价于 torch.randn(requires_grad=True)
    a = torch.randn(size=(), requires_grad=True)
    d = f(a)                     # 执行带控制流的计算
    d.sum().backward()                 # 自动处理动态计算图的梯度

    print("a:", a)
    print("d:", d)
    print("a.grad:", a.grad)
    # 验证：由于d是a的线性函数（d = k*a），梯度应该等于 d/a（逐元素）
    print("验证 a.grad == d/a:", torch.allclose(a.grad, d / a))


def no_grad_example():
    """
    2.5.5 torch.no_grad() 上下文管理器示例

    核心作用：
        在不需要计算梯度的区域禁用梯度追踪，节省内存并加速计算

    应用场景：
        1. 推理阶段（inference）
        2. 计算验证指标（如准确率、损失）
        3. 冻结部分网络参数的前向传播

    技术细节：
        - torch.no_grad() 不会改变张量的 requires_grad 属性
        - 退出上下文后，梯度追踪自动恢复
        - 嵌套使用时，只有所有 no_grad 上下文退出后才恢复追踪
    """
    print("========2.5.5 torch.no_grad() 示例========")

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # 在no_grad上下文中，梯度不会被追踪
    with torch.no_grad():
        y = x * 2
        print("在no_grad中 y:", y)
        print("y.requires_grad:", y.requires_grad)  # False
    
    # 退出上下文后，恢复梯度追踪
    z = x * 3
    print("退出no_grad后 z:", z)
    print("z.requires_grad:", z.requires_grad)  # True

    # 验证梯度计算不受影响
    z.sum().backward()
    print("x.grad:", x.grad)  # 应为 [3., 3., 3.]，因为 z = x * 3，每个元素对x的梯度都是3
    print("验证 x.grad == [3., 3., 3.]:", torch.allclose(x.grad, torch.tensor([3.0, 3.0, 3.0])))


def multiple_inputs_grad_example():
    """
    2.5.6 多输入梯度计算示例

    数学原理：
        设 f(x, y) = x^2 + y^2 + xy
        ∂f/∂x = 2x + y
        ∂f/∂y = 2y + x

    关键概念：
        - backward() 可以同时计算多个输入的梯度
        - 每个输入张量需要设置 requires_grad=True
        - 梯度分别存储在各自的 .grad 属性中
    """
    print("========2.5.6 多输入梯度计算========")

    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    # f(x, y) = x^2 + y^2 + xy
    f = x**2 + y**2 + x * y
    
    f.backward()

    print("x:", x)
    print("y:", y)
    print("f:", f.item())
    print("x.grad:", x.grad)  # 预期: 2*2 + 3 = 7
    print("y.grad:", y.grad)  # 预期: 2*3 + 2 = 8

    # 手动验证梯度
    expected_grad_x = 2*x + y
    expected_grad_y = 2*y + x
    print("验证 x.grad == 2x + y:", torch.allclose(x.grad, expected_grad_x))
    print("验证 y.grad == 2y + x:", torch.allclose(y.grad, expected_grad_y))


def grad_clipping_example():
    """
    2.5.7 梯度裁剪示例

    梯度裁剪作用：
        防止梯度爆炸（gradient explosion），确保梯度范数在可控范围内

    常见方法：
        1. torch.nn.utils.clip_grad_norm_ - 按范数裁剪
        2. torch.nn.utils.clip_grad_value_ - 按值裁剪

    应用场景：
        - RNN/LSTM训练中常见梯度爆炸问题
        - 深层网络训练
        - 使用较大学习率时
    """
    print("========2.5.7 梯度裁剪示例========")

    # 创建一个模拟的参数向量
    params = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    
    # 模拟梯度爆炸场景（人为制造大梯度）
    y = (params * 100).sum()
    y.backward()

    print("裁剪前 params.grad:", params.grad)
    print("裁剪前梯度范数:", params.grad.norm())

    # 方法1: 按范数裁剪（常用）
    torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
    print("按范数裁剪后 params.grad:", params.grad)
    print("按范数裁剪后梯度范数:", params.grad.norm())

    # 重置梯度，演示按值裁剪
    params.grad.zero_()
    y = (params * 100).sum()
    y.backward()

    # 方法2: 按值裁剪
    torch.nn.utils.clip_grad_value_(params, clip_value=1.0)
    print("按值裁剪后 params.grad:", params.grad)


def main():
    """
    主函数：运行所有示例

    使用说明：
        取消注释对应的函数调用即可运行相应示例。
        建议按顺序学习：simple_example -> backward_for_vector_output -> 
                       detach_example -> control_flow_grad_example ->
                       no_grad_example -> multiple_inputs_grad_example ->
                       grad_clipping_example

    学习路径建议：
        1. 理解基本梯度计算（simple_example）
        2. 掌握非标量输出处理（backward_for_vector_output）
        3. 学会梯度阻断技术（detach_example）
        4. 深入动态图机制（control_flow_grad_example）
        5. 优化推理性能（no_grad_example）
        6. 多输入梯度计算（multiple_inputs_grad_example）
        7. 梯度裁剪技巧（grad_clipping_example）
    """
    print("D2L 2.5 自动微分（PyTorch 教学版）")
    
    # 运行所有示例
    simple_example()
    backward_for_vector_output()
    detach_example()
    control_flow_grad_example()
    no_grad_example()
    multiple_inputs_grad_example()
    grad_clipping_example()

if __name__ == "__main__":
    main()

"""
【常见错误及解决方法】

1. RuntimeError: grad can be implicitly created only for scalar outputs
   - 原因：对非标量张量直接调用backward()
   - 解决：先聚合为标量（如y.sum().backward()）或指定gradient参数

2. AttributeError: 'NoneType' object has no attribute 'zero_'
   - 原因：访问未调用backward()的张量的grad属性
   - 解决：确保先调用backward()，或检查requires_grad是否为True

3. 梯度值异常（NaN/Inf）
   - 原因：数值不稳定（如除以零、指数爆炸）
   - 解决：使用梯度裁剪（torch.nn.utils.clip_grad_norm_）、调整学习率、检查数据范围

4. 梯度累积错误
   - 原因：忘记在训练循环中清零梯度
   - 解决：在每个batch前调用optimizer.zero_grad()或tensor.grad.zero_()

5. detach()后数据不一致
   - 原因：修改detach后的张量会影响原张量（共享底层数据）
   - 解决：需要独立副本时使用detach().clone()

【性能优化建议】

1. 减少不必要的梯度追踪
   - 使用torch.no_grad()上下文管理器包裹无需梯度的代码
   - 推理时设置model.eval()，自动关闭梯度追踪

2. 合理设置requires_grad
   - 仅对需要优化的参数设置requires_grad=True
   - 冻结层使用detach()或设置requires_grad=False

3. 混合精度训练
   - 使用torch.cuda.amp自动混合精度，减少内存占用和计算时间

4. 避免在循环中创建张量
   - 预分配张量内存，减少内存碎片化

5. 使用JIT编译
   - 对于重复执行的计算，使用torch.jit.trace/script编译加速
"""


# ============================================
#            2.5.6 练习解答
# ============================================

"""
练习1解答：为什么计算二阶导数比一阶导数的开销要更大？

【理论分析】
1. 计算图构建开销：
   - 一阶导数：只需在前向传播时构建一次计算图
   - 二阶导数：需要先计算一阶导数，然后对一阶导数再求导，需要构建两次计算图

2. 内存占用：
   - 一阶导数：只需存储原始张量和梯度
   - 二阶导数：需要存储原始张量、一阶导数、二阶导数，内存翻倍

3. 计算复杂度：
   - 假设计算图有n个节点，每个节点有k个操作
   - 一阶导数：O(n*k)次操作
   - 二阶导数：需要对每个梯度再求一次导数，复杂度为O((n*k)^2)

4. 实际实现：
   - PyTorch中计算二阶导数需要使用create_graph=True参数
   - 这会导致计算图被保留，内存占用显著增加

【示例说明】
对于函数 f(x) = x^2，计算二阶导数：
1. 一阶导数：df/dx = 2x
2. 二阶导数：d²f/dx² = d(2x)/dx = 2
需要执行两次backward()，且第二次需要保留计算图
"""


def exercise2_multiple_backward():
    """
    练习2：在运行反向传播函数之后，立即再次运行它，看看会发生什么
    
    预期结果：
        1. 第二次调用backward()时，梯度会累积到已有梯度上
        2. 如果不清零梯度，梯度值会翻倍
        3. 这是PyTorch的默认行为，用于支持梯度累积训练
    
    关键要点：
        - PyTorch的backward()默认累加梯度而非覆盖
        - 这允许在多个mini-batch上累积梯度
        - 在训练循环中需要手动调用zero_grad()清零
    """
    print("\n" + "="*40)
    print("练习2：连续两次调用backward()")
    print("="*40)

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.sum()
    
    print("第一次 backward():")
    y.backward()
    print(f"  x.grad = {x.grad}")  # 预期: [1., 1., 1.]
    
    print("\n第二次 backward()（不清零）:")
    y.backward()
    print(f"  x.grad = {x.grad}")  # 预期: [2., 2., 2.]（梯度累加）
    
    print("\n分析：")
    print("  - 第二次backward()将梯度累加到了第一次的结果上")
    print("  - 这是PyTorch的默认行为，支持梯度累积训练")
    print("  - 在实际训练中，需要在每个batch前调用x.grad.zero_()清零")


def exercise3_vector_input():
    """
    练习3：在控制流的例子中，将变量a更改为随机向量或矩阵，会发生什么？
    
    分析：
        1. 控制流条件（如b.norm() < 1000）仍然有效，因为norm返回标量
        2. 返回值c的形状与输入a相同
        3. 梯度计算仍然正常工作，PyTorch会自动处理逐元素操作
        4. 验证：dc/da = c/a 仍然成立（因为c = k*a，k为常数）
    """
    print("\n" + "="*40)
    print("练习3：控制流示例 - 向量输入")
    print("="*40)

    # 测试1：向量输入
    print("\n测试1：向量输入")
    a_vector = torch.tensor([1.0, 2.0], requires_grad=True)
    d_vector = f(a_vector)
    d_vector.sum().backward()
    
    print(f"  a = {a_vector}")
    print(f"  d = {d_vector}")
    print(f"  a.grad = {a_vector.grad}")
    print(f"  验证 a.grad == d/a: {torch.allclose(a_vector.grad, d_vector / a_vector)}")
    
    # 测试2：矩阵输入
    print("\n测试2：矩阵输入")
    a_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    d_matrix = f(a_matrix)
    d_matrix.sum().backward()
    
    print(f"  a =\n{a_matrix}")
    print(f"  d =\n{d_matrix}")
    print(f"  a.grad =\n{a_matrix.grad}")
    print(f"  验证 a.grad == d/a: {torch.allclose(a_matrix.grad, d_matrix / a_matrix)}")
    
    print("\n分析：")
    print("  - 控制流对向量/矩阵输入同样有效")
    print("  - 条件判断（如norm、sum）返回标量，保持控制流逻辑")
    print("  - 梯度计算正确，每个元素的梯度等于 d[i,j] / a[i,j]")


def exercise4_new_control_flow():
    """
    练习4：重新设计一个求控制流梯度的例子
    
    新示例设计：
        函数逻辑：根据输入值的大小选择不同的计算路径
        1. 如果输入的绝对值小于1，使用二次函数
        2. 如果输入在[1, 5)之间，使用三次函数
        3. 如果输入大于等于5，使用指数函数
        4. 增加一个随机分支，展示动态控制流
    
    数学分析：
        f(x) = { x^2,          |x| < 1
               { x^3 + 2x,     1 <= |x| < 5
               { exp(x),       |x| >= 5
        
        由于函数分段定义，梯度也分段：
        f'(x) = { 2x,          |x| < 1
                { 3x^2 + 2,    1 <= |x| < 5
                { exp(x),       |x| >= 5
    """
    print("\n" + "="*40)
    print("练习4：新的控制流梯度示例")
    print("="*40)
    
    def piecewise_function(x: torch.Tensor) -> torch.Tensor:
        """分段函数示例"""
        result = torch.empty_like(x)
        
        # 分段1：|x| < 1，使用二次函数
        mask1 = torch.abs(x) < 1
        result[mask1] = x[mask1] ** 2
        
        # 分段2：1 <= |x| < 5，使用三次函数
        mask2 = (torch.abs(x) >= 1) & (torch.abs(x) < 5)
        result[mask2] = x[mask2] ** 3 + 2 * x[mask2]
        
        # 分段3：|x| >= 5，使用指数函数
        mask3 = torch.abs(x) >= 5
        result[mask3] = torch.exp(x[mask3])
        
        return result
    
    # 测试不同范围的输入
    test_values = torch.tensor([-6.0, -3.0, -0.5, 0.0, 0.5, 2.0, 6.0], requires_grad=True)
    output = piecewise_function(test_values)
    output.sum().backward()
    
    print(f"输入 x: {test_values}")
    print(f"输出 f(x): {output}")
    print(f"梯度 df/dx: {test_values.grad}")
    
    # 手动验证梯度
    expected_grad = torch.empty_like(test_values)
    for i, val in enumerate(test_values):
        if abs(val) < 1:
            expected_grad[i] = 2 * val
        elif abs(val) < 5:
            expected_grad[i] = 3 * val**2 + 2
        else:
            expected_grad[i] = torch.exp(val)
    
    print(f"预期梯度: {expected_grad}")
    print(f"验证梯度正确: {torch.allclose(test_values.grad, expected_grad)}")
    
    print("\n分析：")
    print("  - PyTorch能够正确处理条件分支的梯度")
    print("  - 每个分支的梯度独立计算")
    print("  - 动态图在运行时记录实际执行的路径")


def exercise5_plot_sin_grad():
    """
    练习5：绘制 f(x) = sin(x) 和 df/dx 的图像
    
    要求：不使用 f'(x) = cos(x) 的解析解，而是通过自动微分计算
    
    实现步骤：
        1. 创建x的取值范围
        2. 对每个x计算sin(x)
        3. 使用自动微分计算梯度
        4. 绘制函数和梯度曲线
    """
    print("\n" + "="*40)
    print("练习5：绘制sin(x)及其导数")
    print("="*40)
    
    # 创建x值（需要requires_grad=True）
    x = torch.linspace(-2 * torch.pi, 2 * torch.pi, 100, requires_grad=True)
    y = torch.sin(x)
    
    # 计算梯度（通过backward）
    y.sum().backward()
    grad_y = x.grad  # dy/dx
    
    # 提取数值用于绘图
    x_np = x.detach().numpy()
    y_np = y.detach().numpy()
    grad_y_np = grad_y.detach().numpy()
    
    print("计算完成！")
    print(f"x范围: [{x_np[0]:.2f}, {x_np[-1]:.2f}]")
    print(f"sin(x)最大值: {y_np.max():.4f}, 最小值: {y_np.min():.4f}")
    print(f"梯度最大值: {grad_y_np.max():.4f}, 最小值: {grad_y_np.min():.4f}")
    
    # 尝试导入matplotlib绘图（如果可用）
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(x_np, y_np, label='sin(x)', color='blue', linewidth=2)
        plt.plot(x_np, grad_y_np, label='dy/dx (自动微分)', color='red', linestyle='--', linewidth=2)
        plt.plot(x_np, torch.cos(torch.tensor(x_np)).numpy(), label='cos(x) (解析解)', color='green', linestyle=':', linewidth=2)
        
        plt.title('sin(x) 及其导数')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.savefig('sin_derivative.png')
        print("\n图像已保存为 sin_derivative.png")
        
    except ImportError:
        print("\n提示：未安装matplotlib，无法绘图")
        print("安装命令: pip install matplotlib")
        print("\n数值验证：")
        # 验证梯度是否接近cos(x)
        cos_x = torch.cos(x)
        print(f"梯度与cos(x)的最大误差: {torch.max(torch.abs(grad_y - cos_x)):.6f}")
        print(f"验证梯度 ≈ cos(x): {torch.allclose(grad_y, cos_x, atol=1e-5)}")


def run_exercises():
    """运行所有练习"""
    print("\n" + "="*60)
    print("D2L 2.5.6 练习题解答")
    print("="*60)
    
    exercise2_multiple_backward()
    exercise3_vector_input()
    exercise4_new_control_flow()
    exercise5_plot_sin_grad()


# 在主函数中添加练习调用
if __name__ == "__main__":
    run_exercises()
