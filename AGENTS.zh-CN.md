# AGENTS.zh-CN.md

## 语言约定

本文件是给用户阅读的中文版说明。

- 仓库中的唯一标准 agent 说明文件是 `AGENTS.md` 英文版。
- agent 在读取项目约定时，应始终只查看并遵循 `AGENTS.md`。
- 如果中英文内容出现差异，以 `AGENTS.md` 为准。
- 本文件仅用于帮助用户快速阅读和理解仓库约定。

## 仓库目的

本仓库用于以动手实践的方式学习深度学习，主要有两个并行目标：

1. 系统学习《动手学深度学习（D2L）》中的基础知识：https://zh.d2l.ai/chapter_introduction/index.html
2. 把这些基础知识和真实项目结合起来，包括 FoundationPose、DINOv3 以及其他视觉 / 多模态仓库的使用、阅读、调试、适配和修改

在本仓库中工作的 agent，应优先服务于"学习清晰度"，其次才是工程严谨性。这里的代码和笔记不只是为了跑通结果，更是为了理解结果为什么会出现。

## 主要工作风格

- 优先写清晰、可读的代码，而不是追求炫技或过度压缩。
- 只要实现稍微复杂，就应在注释或附近的 Markdown 中说明思路，尤其是在原因不明显时。
- 始终把理论和实践连接起来：
  - 新增模型、损失、优化技巧或训练流程时，尽量说明它与 D2L 中哪个概念相关
  - 修改外部仓库流程时，说明改了什么、为什么改
- 倾向于小步、可验证的修改，而不是大范围重构。
- 在探索阶段，不要过早引入不必要的抽象。

## 仓库优先支持的工作

agent 应优先支持以下几类任务：

1. D2L 学习笔记与最小复现
2. 基于 PyTorch 的实验与训练脚本
3. 阅读和修改真实世界项目仓库
4. 排查环境、依赖、数据集、训练 / 推理流程问题
5. 搭建"教材概念"与"实际系统"之间的桥梁

## 建议目录结构

如果需要新增内容，优先组织到类似以下目录：

- `notes/`
  - 按章节整理的 D2L 笔记
  - 概念总结
  - 调试记录
- `experiments/`
  - 小型、可独立运行的复现脚本
  - 消融或对比实验脚本
- `projects/`
  - 实际项目集成
  - 基于上游仓库做的适配版本
- `datasets/`
  - 只放轻量元数据、脚本或说明
  - 除非明确需要，否则不要提交大型原始数据集
- `tools/`
  - 环境搭建、格式转换、评估、可视化、性能分析脚本
- `reports/`
  - 实验结论、失败分析、阅读总结

除非用户明确要求，不要为追求结构统一而强行重组已有内容；但新内容优先遵循这一结构。

## 处理 D2L 学习内容的方式

处理 D2L 相关内容时：

- 示例代码要尽量小而精，以教学为目标。
- 优先从零实现核心思想，再考虑封装到更大的框架里。
- 如果是在复现某个 D2L 小节，应尽量把以下信息对应清楚：
  - 章节 / 小节名
  - 关键数学思想
  - 实现文件
  - 观察到的结果
- 如果某个概念又在真实项目里出现，应显式写出这种联系。

常见且有价值的联系包括：

- 线性代数与张量操作 -> PyTorch 仓库中的 shape 调试
- 卷积 / 池化 -> backbone 特征提取行为
- 注意力机制 -> transformer 编码器和多模态骨干网络
- 优化方法 -> 学习率调度、warmup、weight decay、梯度裁剪
- 计算机视觉章节 -> 检测、分割、位姿估计、表征学习

## AI 组件开发模板框架

本节定义了一个标准化的模板框架，用于开发各类 AI 组件，包括神经网络、优化算法、概率模型、自动微分系统等。

### 框架结构

框架由五个核心部分组成，所有组件都应实现：

| 部分 | 目的 | 关键要素 |
|------|------|----------|
| **模块定义** | 描述组件的目的和范围 | 名称、描述、数学基础 |
| **接口规范** | 定义输入/输出和参数 | 类型提示、参数验证、错误处理 |
| **核心实现** | 实现主要功能 | 算法、数据结构、计算逻辑 |
| **测试策略** | 验证正确性和性能 | 单元测试、边界情况、数值稳定性 |
| **使用示例** | 演示实际应用 | 代码示例、预期输出、使用场景 |

### 模板规范

#### 1. 模块定义部分
```python
"""
{组件名称} 模块
=============

【目的】
简要描述该组件的功能和应用场景。

【数学基础】
该组件所基于的关键数学概念和公式。

【参考资料】
- D2L章节：{链接}
- 相关概念：{概念1}, {概念2}, {概念3}
"""
```

#### 2. 接口规范部分
```python
def function_name(param1: Type, param2: Type, ...) -> ReturnType:
    """
    函数功能的简要描述。

    参数：
        param1: 描述，包括类型和有效范围
        param2: 描述，包括类型和有效范围
        ...

    返回：
        返回值的描述和格式

    异常：
        TypeError: 参数类型不正确时
        ValueError: 参数超出有效范围时
    """
```

#### 3. 核心实现部分
```python
class ComponentClass:
    """
    实现该组件的主类。

    属性：
        attr1: 描述
        attr2: 描述
        ...

    方法：
        method1(): 简要描述
        method2(): 简要描述
        ...
    """
    
    def __init__(self, param1, param2, ...):
        """使用给定参数初始化组件。"""
        # 参数验证
        # 初始化逻辑
    
    def core_method(self, inputs):
        """
        主要计算方法。
        
        数学原理：
        算法描述及其数学基础。
        """
        # 带注释的实现代码
```

#### 4. 测试策略部分
```python
def test_component_functionality():
    """测试组件的核心功能。"""
    # 测试用例1：基本功能
    # 测试用例2：边界情况
    # 测试用例3：数值稳定性
    # 测试用例4：参数验证
```

#### 5. 使用示例部分
```python
def demo_component():
    """演示组件的典型用法。"""
    # 示例1：基本用法
    # 示例2：高级用法
    # 示例3：与其他组件的集成
```

### 示例1：线性神经网络组件

```python
"""
线性神经网络模块
===============

【目的】
实现神经网络中的线性层（全连接层）。
用作深度学习模型的基本构建块。

【数学基础】
输出：y = x @ W + b
其中：
- x: 输入张量，形状为 (batch_size, input_dim)
- W: 权重矩阵，形状为 (input_dim, output_dim)
- b: 偏置向量，形状为 (output_dim,)
- @: 矩阵乘法

【参考资料】
- D2L章节：https://zh.d2l.ai/chapter_linear-networks/index.html
- 相关概念：线性回归、前馈网络
"""

import numpy as np

class LinearLayer:
    """
    线性层实现。

    属性：
        weight: 权重矩阵 (input_dim x output_dim)
        bias: 偏置向量 (output_dim,)
        input_dim: 输入特征维度
        output_dim: 输出特征维度
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        初始化线性层。

        参数：
            input_dim: 输入特征数量
            output_dim: 输出特征数量

        异常：
            ValueError: 如果维度非正
        """
        if input_dim <= 0:
            raise ValueError(f"input_dim必须为正数，得到 {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim必须为正数，得到 {output_dim}")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros(output_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播计算。

        参数：
            x: 输入张量，形状为 (batch_size, input_dim)

        返回：
            输出张量，形状为 (batch_size, output_dim)

        异常：
            TypeError: 如果输入不是numpy数组
            ValueError: 如果输入形状不正确
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"输入必须是numpy数组，得到 {type(x)}")
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"输入形状必须是 (batch_size, {self.input_dim})，得到 {x.shape}")
        
        return x @ self.weight + self.bias
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> dict:
        """
        反向传播计算（梯度计算）。

        参数：
            x: 前向传播的输入张量
            grad_output: 损失函数对输出的梯度

        返回：
            包含梯度的字典：{'weight', 'bias', 'input'}
        """
        batch_size = x.shape[0]
        
        grad_weight = x.T @ grad_output / batch_size
        grad_bias = np.mean(grad_output, axis=0)
        grad_input = grad_output @ self.weight.T
        
        return {
            'weight': grad_weight,
            'bias': grad_bias,
            'input': grad_input
        }

# 测试
def test_linear_layer():
    """测试LinearLayer功能。"""
    layer = LinearLayer(input_dim=3, output_dim=2)
    
    # 测试前向传播
    x = np.random.randn(5, 3)
    output = layer.forward(x)
    assert output.shape == (5, 2), f"期望 (5, 2)，得到 {output.shape}"
    
    # 测试反向传播
    grad_output = np.random.randn(5, 2)
    grads = layer.backward(x, grad_output)
    assert grads['weight'].shape == (3, 2)
    assert grads['bias'].shape == (2,)
    assert grads['input'].shape == (5, 3)
    
    # 测试参数验证
    try:
        LinearLayer(input_dim=0, output_dim=5)
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass
    
    print("LinearLayer测试通过！")

# 使用演示
def demo_linear_layer():
    """演示LinearLayer的使用。"""
    # 创建线性层
    layer = LinearLayer(input_dim=10, output_dim=5)
    
    # 生成随机输入
    x = np.random.randn(32, 10)  # 32个样本，每个10个特征
    
    # 前向传播
    output = layer.forward(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 计算梯度（模拟反向传播）
    grad_output = np.random.randn(32, 5)
    grads = layer.backward(x, grad_output)
    print(f"梯度形状:")
    print(f"  weight: {grads['weight'].shape}")
    print(f"  bias: {grads['bias'].shape}")
    print(f"  input: {grads['input'].shape}")
```

### 示例2：优化算法组件

```python
"""
梯度下降优化器模块
==================

【目的】
实现梯度下降优化算法，用于训练机器学习模型。
通过沿梯度方向迭代更新参数来最小化损失函数。

【数学基础】
参数更新规则：θ = θ - lr * ∇L(θ)
其中：
- θ: 模型参数
- lr: 学习率（步长）
- ∇L(θ): 损失函数对参数的梯度

【参考资料】
- D2L章节：https://zh.d2l.ai/chapter_optimization/index.html
- 相关概念：随机梯度下降、动量、Adam
"""

import numpy as np
from typing import Dict, Callable

class GradientDescent:
    """
    带动量支持的梯度下降优化器。

    属性：
        lr: 学习率
        momentum: 动量因子（标准GD为0）
        velocity: 动量速度缓冲
    """
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        """
        初始化优化器。

        参数：
            lr: 学习率（默认：0.01）
            momentum: 动量因子 [0, 1)（默认：0.0）

        异常：
            ValueError: 如果lr非正或momentum超出范围
        """
        if lr <= 0:
            raise ValueError(f"lr必须为正数，得到 {lr}")
        if not (0 <= momentum < 1):
            raise ValueError(f"momentum必须在 [0, 1) 范围内，得到 {momentum}")
        
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
    
    def step(self, params: Dict[str, np.ndarray], 
             grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        执行一次优化步骤。

        参数：
            params: 参数数组的字典
            grads: 梯度数组的字典（与params键相同）

        返回：
            更新后的参数字典

        异常：
            KeyError: 如果params和grads的键不匹配
        """
        if params.keys() != grads.keys():
            raise KeyError("params和grads必须有相同的键")
        
        updated_params = {}
        
        for key in params:
            grad = grads[key]
            
            # 如果速度缓冲不存在则初始化
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            # 应用动量
            self.velocity[key] = self.momentum * self.velocity[key] + grad
            
            # 更新参数
            updated_params[key] = params[key] - self.lr * self.velocity[key]
        
        return updated_params
    
    def zero_grad(self):
        """重置速度缓冲。"""
        self.velocity = {}

# 测试
def test_gradient_descent():
    """测试GradientDescent优化器。"""
    optimizer = GradientDescent(lr=0.1, momentum=0.9)
    
    # 测试基本步骤
    params = {'w': np.array([1.0, 2.0]), 'b': np.array([0.5])}
    grads = {'w': np.array([0.1, 0.2]), 'b': np.array([0.05])}
    
    updated = optimizer.step(params, grads)
    assert updated['w'].shape == (2,)
    assert updated['b'].shape == (1,)
    
    # 测试动量累积
    updated2 = optimizer.step(params, grads)
    # 有动量时，速度应该累积
    assert not np.allclose(updated['w'], updated2['w'])
    
    # 测试参数验证
    try:
        GradientDescent(lr=-0.1)
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass
    
    try:
        GradientDescent(momentum=1.0)
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass
    
    print("GradientDescent测试通过！")

# 使用演示
def demo_gradient_descent():
    """演示GradientDescent的使用。"""
    # 定义简单的二次损失函数
    def loss(w):
        return (w - 3) ** 2  # 最小值在 w = 3
    
    def grad_loss(w):
        return 2 * (w - 3)  # 导数
    
    # 初始化优化器
    optimizer = GradientDescent(lr=0.1, momentum=0.9)
    
    # 初始化参数
    params = {'w': np.array([0.0])}
    
    # 训练循环
    print("优化步骤：")
    for i in range(10):
        grads = {'w': grad_loss(params['w'])}
        params = optimizer.step(params, grads)
        current_loss = loss(params['w'])
        print(f"步骤 {i+1}: w = {params['w'][0]:.4f}, loss = {current_loss:.4f}")
    
    print(f"\n最终 w = {params['w'][0]:.4f}（预期约为 3.0）")
```

### 组件开发最佳实践

1. **模块化**：每个组件应有单一、明确的目的
2. **类型提示**：使用Python类型提示提高文档质量和IDE支持
3. **参数验证**：始终验证输入并提供清晰的错误消息
4. **文档**：提供带有数学解释的docstrings
5. **测试**：包含涵盖边界情况的全面单元测试
6. **数值稳定性**：处理除以零等边界情况
7. **一致性**：始终遵循命名约定和代码风格

### 框架扩展性

该模板框架可扩展支持：
- **神经网络层**：卷积层、循环层、注意力机制
- **优化器**：Adam、RMSprop、Adagrad、学习率调度
- **概率模型**：贝叶斯网络、隐马尔可夫模型、变分推断
- **损失函数**：交叉熵、MSE、自定义损失函数
- **评估指标**：准确率、精确率、召回率、F1分数

## 处理真实项目仓库的方式

当使用或修改 FoundationPose、DINOv3 或类似实践仓库时：

- 若无充分理由，保持上游行为不变。
- 尽量把本地适配逻辑和上游原始代码分开。
- 每个重要修改都应记录：
  - 改了哪个文件或模块
  - 这是兼容性修改、调试修改、实验修改还是性能修改
  - 预期会如何影响输出、速度、显存或可复现性
- 在深改第三方内部实现前，优先考虑写一层薄封装、适配器、配置覆盖或实验脚本。
- 如果确实需要深度修改，改动要尽量小，并且说明理由。

## 修改外部库时的规则

- 不要悄悄改变模型语义。
- 不要在没有说明理由的情况下移除断言、安全检查或预处理步骤。
- 在修改训练或推理流程前，先识别清楚：
  - 输入与预期 shape
  - 设备和 dtype 假设
  - checkpoint 加载路径
  - 配置依赖
  - 输出契约
- 如果是在修 bug，应记录：
  - 现象
  - 根因
  - 验证方法

## 代码质量要求

- 如果目标仓库没有明确使用其他栈，优先采用 Python 和 PyTorch 的惯用写法。
- 脚本优先做到可以独立运行，参数明确。
- 对以下内容补充简洁的 docstring 或注释：
  - 张量 shape 预期
  - 不直观的坐标系
  - 预处理和归一化假设
  - 损失函数组合方式
  - 评估逻辑
- 避免硬编码机器相关路径。优先使用配置文件、环境变量或清楚标记的本地占位符。
- Notebook 可以作为补充，但关键逻辑尽量落在普通源码文件中。

## 实验规范

做实验时，应尽量让过程可理解、可复现：

- 记录配置、数据集版本、checkpoint 来源和主要指标。
- 保存简短结论，而不是只保留原始日志。
- 不仅记录成功实验，也要记录失败和意外结果。
- 做方法对比时，尽量一次只改一个主要变量。

有价值的实验元数据包括：

- 目标
- 假设
- 环境
- 命令
- 指标
- 定性观察
- 下一步动作

## 调试要求

调试时：

- 先缩小范围：环境、导入、数据、模型、损失、优化、评估还是可视化。
- 尽早检查 shape、dtype、device 放置和坐标系约定。
- 对视觉项目，尽可能查看有代表性的输入和输出。
- 优先写针对性的日志和最小复现脚本，而不是基于猜测做大面积修改。
- 如果问题可能由版本不匹配导致，要明确指出具体是哪一个包、API 或 checkpoint 假设出了问题。

## Agent 的沟通风格

在这个仓库里工作的 agent 应：

- 明确陈述假设
- 简短而具体地说明权衡
- 总结"学到了什么"，而不只是"改了什么"
- 保持简洁，但技术上有实质内容

如果被要求做代码评审，优先级应是：

1. 正确性
2. 训练 / 推理行为
3. 可复现性
4. 可维护性
5. 性能

## 安全默认值

- 优先采用非破坏性修改。
- 除非用户明确要求，不要删除用户的实验、笔记、checkpoint 或数据集。
- 对大型生成产物默认视为可丢弃，除非用户说明它们需要被版本化。
- 除非任务确实需要，否则避免进行网络下载或大规模环境改动。

## Agent 的优质输出应该做到什么

在这个仓库里，好的贡献通常至少满足以下之一：

- 让一个 D2L 概念更容易理解
- 把一个理论概念落成可运行代码
- 让一个实践仓库更容易运行或调试
- 用具体技术推理解释一个真实 bug
- 提高一个实验的可复现性

## 不确定时的选择原则

如果存在多个可行方案，优先选择更符合以下原则的那个：

1. 更能帮助理解底层概念
2. 实现更容易检查和追踪
3. 对外部代码的不可逆改动更少
4. 更有利于未来实验之间的比较
