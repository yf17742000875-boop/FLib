"""
3.4 softmax回归
================

【章节来源】
- D2L: https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html

【本节目标】
理解如何把线性模型从“预测一个连续数值”的回归问题，扩展到“预测属于哪一类”的多分类问题。
softmax回归虽然名字里有“回归”，但它解决的是分类问题，是后续神经网络分类模型的基础。


一、本节内容总结
----------------

1. 回归问题与分类问题的区别

线性回归回答“多少”：
- 房价是多少？
- 住院几天？
- 球队赢几场？

分类问题回答“哪一个”：
- 邮件是不是垃圾邮件？
- 图片中是猫、狗还是鸡？
- 输入样本属于哪个类别？

分类标签不能简单当作连续数值来回归。例如“猫=1、狗=2、鸡=3”会暗示狗在猫和鸡之间，
但类别编号本身没有这种大小和距离关系。因此，多分类问题通常用 one-hot 编码表示标签：
- 猫: [1, 0, 0]
- 狗: [0, 1, 0]
- 鸡: [0, 0, 1]


2. softmax回归的模型结构

假设输入特征有 d 维，类别数有 q 个。softmax回归先为每个类别计算一个未归一化分数，
这些分数通常叫 logits：

    o_j = x_1 w_1j + x_2 w_2j + ... + x_d w_dj + b_j

向量形式为：

    o = x W + b

其中：
- x 的形状是 (d,)
- W 的形状是 (d, q)
- b 的形状是 (q,)
- o 的形状是 (q,)

对一个小批量 X 来说：

    O = X W + b

如果 X 的形状是 (n, d)，那么 O 的形状是 (n, q)。
每一行对应一个样本，每一列对应一个类别的 logit。

因为每个输出类别的 logit 都依赖全部输入特征，所以 softmax回归的输出层是全连接层。
从神经网络角度看，它是一个单层神经网络。


3. 为什么需要 softmax

logits o_j 可以是任意实数，不能直接解释为概率。分类模型的输出最好满足：
- 每个类别概率非负；
- 所有类别概率之和为 1；
- 分数越高的类别，概率也越高。

softmax函数把 logits 转换为概率分布：

    y_hat_j = exp(o_j) / sum_k exp(o_k)

性质：
- y_hat_j > 0
- sum_j y_hat_j = 1
- argmax_j o_j = argmax_j y_hat_j

最后一个性质很重要：softmax不会改变类别排序，所以预测类别时可以直接取最大概率类别。

实践中要注意数值稳定性。直接计算 exp(o_j) 可能溢出，常见做法是先减去最大 logit：

    softmax(o)_j = exp(o_j - max(o)) / sum_k exp(o_k - max(o))

减去同一个常数不会改变softmax结果。


4. 损失函数：交叉熵

如果真实标签 y 是 one-hot 向量，预测概率是 y_hat，则交叉熵损失为：

    l(y, y_hat) = - sum_j y_j log(y_hat_j)

因为 one-hot 向量只有真实类别那一项为 1，所以它等价于：

    l(y, y_hat) = - log(模型分配给真实类别的概率)

如果真实类别概率越高，损失越小；如果模型给真实类别的概率接近 0，损失会非常大。

把 softmax 代入交叉熵，可以得到对 logits 更方便的形式：

    l(y, softmax(o)) = - sum_j y_j o_j + log(sum_k exp(o_k))

这个形式也解释了为什么深度学习框架通常把 softmax 和 cross entropy 合并实现：
它更稳定，也避免重复计算。


5. 最大似然视角

分类模型给出条件概率 P(y | x)。给定训练集，最大化所有样本真实标签的概率：

    P(Y | X) = product_i P(y_i | x_i)

等价于最大化对数似然：

    sum_i log P(y_i | x_i)

也等价于最小化负对数似然：

    - sum_i log P(y_i | x_i)

这正是交叉熵损失。因此，softmax回归训练可以理解为：
调整参数 W 和 b，让模型给真实类别分配尽可能高的概率。


6. 信息论视角

交叉熵可以衡量两个概率分布之间的差异，也可以理解为：
如果用模型预测分布来编码真实数据，需要多少信息量。

真实分布和模型分布越接近，交叉熵越小。
在 one-hot 标签下，交叉熵就是惩罚“真实类别概率不够高”。


7. 预测与评估

训练后，模型对一个样本输出每个类别的概率。预测时通常选择概率最大的类别：

    y_pred = argmax_j y_hat_j

分类任务常用准确率 accuracy 评估：

    accuracy = 正确预测数量 / 总预测数量

准确率直观，但在类别极度不平衡时可能不够全面，实践中还会结合精确率、召回率、F1等指标。


二、和D2L前面内容的联系
----------------------

1. 与线性回归的联系

线性回归：

    y_hat = Xw + b

softmax回归：

    O = XW + b
    Y_hat = softmax(O)

两者都先做线性变换。区别在于：
- 线性回归输出连续值；
- softmax回归输出类别概率分布。


2. 与后续深度学习模型的联系

后续图像分类、Transformer分类、多模态分类模型中，最后一层经常仍然是：

    特征表示 -> 线性分类头 -> logits -> softmax / cross entropy

因此，softmax回归可以看作复杂分类网络的“最后一层原型”。


三、课后练习
------------

练习1：我们可以更深入地探讨指数族与softmax之间的联系。

1.1 计算softmax交叉熵损失 l(y, y_hat) 的二阶导数。

答案：

令：

    p = softmax(o)
    p_i = exp(o_i) / sum_k exp(o_k)

交叉熵损失为：

    l(y, p) = - sum_i y_i log(p_i)

将 softmax 代入：

    l(y, softmax(o))
      = - sum_i y_i o_i + log(sum_k exp(o_k))

其中 y 是 one-hot 标签，或更一般地，是满足 sum_i y_i = 1 的目标概率分布。

先求一阶导数：

    d l / d o_i = p_i - y_i

也就是说，logit 的梯度等于：

    预测概率 - 真实标签

再求二阶导数：

    d^2 l / d o_i d o_j = d p_i / d o_j

softmax 的导数为：

    d p_i / d o_j = p_i (delta_ij - p_j)

其中 delta_ij 是Kronecker delta：
- i = j 时，delta_ij = 1
- i != j 时，delta_ij = 0

所以二阶导数矩阵，也就是 Hessian 矩阵为：

    H_ij = p_i (delta_ij - p_j)

写成矩阵形式：

    H = diag(p) - p p^T

展开看：
- 对角线元素：H_ii = p_i (1 - p_i)
- 非对角线元素：H_ij = - p_i p_j, i != j

这个 Hessian 与真实标签 y 无关，只由模型当前预测分布 p 决定。
它是半正定矩阵，因为它对应一个类别分布的协方差矩阵。


1.2 计算 softmax(o) 给出的分布方差，并与上面计算的二阶导数匹配。

答案：

把类别看成一个 one-hot 随机向量 Z：

    Z = e_i 的概率为 p_i

其中 e_i 是第 i 个标准基向量。例如三分类中：

    e_1 = [1, 0, 0]
    e_2 = [0, 1, 0]
    e_3 = [0, 0, 1]

先计算期望：

    E[Z] = p

再计算二阶矩：

    E[Z Z^T] = diag(p)

因此协方差矩阵为：

    Var(Z) = E[Z Z^T] - E[Z] E[Z]^T
           = diag(p) - p p^T

这与 1.1 中 softmax交叉熵关于 logits 的 Hessian 完全相同：

    Hessian = diag(p) - p p^T = Var(Z)

这说明：
- softmax交叉熵的曲率由模型预测分布的方差决定；
- 当模型很不确定时，多个 p_i 都较大，协方差和曲率更明显；
- 当模型几乎确定某一类时，p 接近 one-hot，协方差趋近于 0，曲率也变小。

从指数族角度看，softmax对应分类分布的自然参数化：
- logits o 是自然参数；
- log(sum_k exp(o_k)) 是 log-partition function；
- 它的一阶导数给出均值 p；
- 它的二阶导数给出方差/协方差 diag(p) - p p^T。


练习2：假设三个类发生的概率相等，即概率向量是 (1/3, 1/3, 1/3)。

2.1 如果我们尝试为它设计二进制代码，有什么问题？

答案：

如果每次只编码一个观测值，三个类别需要三个不同码字。二进制定长编码的码字数量必须是 2 的幂：

    1 bit 只能表示 2 个类别
    2 bit 可以表示 4 个类别

所以三个类别不能被整数 bit 数“刚好”表示。若使用 2 bit 定长编码，例如：

    A -> 00
    B -> 01
    C -> 10

则 11 被浪费。平均每个类别需要 2 bit。

但三等概率分布的理论熵为：

    H = -3 * (1/3) log2(1/3) = log2(3) ≈ 1.585 bit

也就是说，单个样本用定长二进制编码会浪费：

    2 - log2(3) ≈ 0.415 bit

问题的本质是：二进制编码天然按 2 的幂划分，而 3 个等概率类别不是 2 的幂。


2.2 请设计一个更好的代码。提示：如果编码两个独立观测结果会发生什么？如果联合编码 n 个观测值怎么办？

答案：

更好的办法是联合编码多个独立观测值，而不是逐个样本定长编码。

如果编码两个独立观测结果，总共有：

    3^2 = 9

种组合。定长二进制码需要：

    ceil(log2(9)) = 4 bit

平均每个观测值需要：

    4 / 2 = 2 bit

这仍然不比单独编码更好，因为 9 离 16 还比较远。

但如果联合编码 n 个观测值，总共有：

    3^n

种序列。需要的定长二进制位数是：

    ceil(log2(3^n))

平均每个观测值需要：

    ceil(n log2(3)) / n

当 n 越来越大时：

    ceil(n log2(3)) / n -> log2(3) ≈ 1.585 bit

因此，联合编码长序列可以逐渐接近信息论最优编码长度。

一个直观例子：

    n = 10 时，3^10 = 59049，ceil(log2(59049)) = 16
    平均每个观测值需要 16 / 10 = 1.6 bit

这已经很接近理论下界 log2(3) ≈ 1.585 bit。

实践中的算术编码、范围编码等方法正是利用这种思想，让平均码长接近熵。


练习3：softmax是对上面介绍的映射的误称。真正的softmax被定义为：

    RealSoftMax(a, b) = log(exp(a) + exp(b))

3.1 证明 RealSoftMax(a, b) > max(a, b)。

答案：

设：

    m = max(a, b)

因为 exp(a) 和 exp(b) 都为正，并且至少有一项等于 exp(m)，所以：

    exp(a) + exp(b) > exp(m)

两边取 log：

    log(exp(a) + exp(b)) > log(exp(m)) = m

因此：

    RealSoftMax(a, b) > max(a, b)


3.2 证明 lambda^{-1} RealSoftMax(lambda a, lambda b) > max(a, b)，其中 lambda > 0。

答案：

令：

    m = max(a, b)

则：

    max(lambda a, lambda b) = lambda max(a, b) = lambda m

根据 3.1：

    RealSoftMax(lambda a, lambda b) > lambda m

由于 lambda > 0，两边同时除以 lambda：

    lambda^{-1} RealSoftMax(lambda a, lambda b) > m = max(a, b)


3.3 证明当 lambda -> infinity 时，
    lambda^{-1} RealSoftMax(lambda a, lambda b) -> max(a, b)。

答案：

仍令：

    m = max(a, b)

则：

    lambda^{-1} log(exp(lambda a) + exp(lambda b))
    = lambda^{-1} log(exp(lambda m) * [exp(lambda(a-m)) + exp(lambda(b-m))])
    = m + lambda^{-1} log(exp(lambda(a-m)) + exp(lambda(b-m)))

因为 a - m <= 0，b - m <= 0，并且至少有一个等于 0，所以括号中的和满足：

    1 <= exp(lambda(a-m)) + exp(lambda(b-m)) <= 2

因此：

    0 <= lambda^{-1} log(exp(lambda(a-m)) + exp(lambda(b-m))) <= lambda^{-1} log 2

当 lambda -> infinity 时：

    lambda^{-1} log 2 -> 0

由夹逼定理可得：

    lambda^{-1} RealSoftMax(lambda a, lambda b) -> m = max(a, b)

这说明 RealSoftMax 是 max 的光滑近似；lambda 越大，近似越接近硬最大值。


3.4 soft-min 会是什么样子？

答案：

soft-min 可以用 softmax 的相反数构造：

    SoftMin_lambda(a, b)
      = - lambda^{-1} log(exp(-lambda a) + exp(-lambda b))

当 lambda > 0 且 lambda -> infinity 时：

    SoftMin_lambda(a, b) -> min(a, b)

它是 min(a, b) 的光滑近似，并且满足：

    SoftMin_lambda(a, b) < min(a, b)

原因与 softmax 类似，只是把“最大”转成了“最小”。


3.5 将其扩展到两个以上的数字。

答案：

对 n 个数 x_1, x_2, ..., x_n，RealSoftMax 的扩展是 log-sum-exp：

    RealSoftMax(x_1, ..., x_n)
      = log(sum_i exp(x_i))

带温度参数 lambda 的 max 光滑近似为：

    SoftMaxApprox_lambda(x_1, ..., x_n)
      = lambda^{-1} log(sum_i exp(lambda x_i))

它满足：

    SoftMaxApprox_lambda(x_1, ..., x_n) > max_i x_i

并且：

    SoftMaxApprox_lambda(x_1, ..., x_n) -> max_i x_i
    as lambda -> infinity

对应的 soft-min 是：

    SoftMinApprox_lambda(x_1, ..., x_n)
      = - lambda^{-1} log(sum_i exp(-lambda x_i))

并且：

    SoftMinApprox_lambda(x_1, ..., x_n) -> min_i x_i
    as lambda -> infinity
"""

from __future__ import annotations

import torch


def stable_softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    数值稳定版softmax。

    Parameters:
        logits: 任意形状的logits张量，最后一维表示类别分数。

    Returns:
        与logits同形状的概率张量，最后一维之和为1。
    """
    shifted_logits = logits - logits.max(dim=-1, keepdim=True).values
    exp_logits = torch.exp(shifted_logits)
    return exp_logits / exp_logits.sum(dim=-1, keepdim=True)


def cross_entropy_from_logits(logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
    """
    从logits直接计算交叉熵，避免先softmax再log带来的数值不稳定。

    Parameters:
        logits: 形状为(num_classes,)的logits。
        target_prob: 形状为(num_classes,)的目标分布，one-hot标签也属于这种形式。

    Returns:
        标量交叉熵损失。
    """
    if logits.ndim != 1:
        raise ValueError(f"logits must be a 1-D tensor, got shape {tuple(logits.shape)}")
    if target_prob.shape != logits.shape:
        raise ValueError(
            f"target_prob shape must match logits shape, got {tuple(target_prob.shape)} "
            f"and {tuple(logits.shape)}"
        )

    log_partition = torch.logsumexp(logits, dim=0)
    return -(target_prob * logits).sum() + log_partition


def softmax_cross_entropy_hessian(logits: torch.Tensor) -> torch.Tensor:
    """
    softmax交叉熵关于logits的二阶导数。

    Mathematical Principle:
        若 p = softmax(logits)，则 Hessian = diag(p) - p p^T。
        这也等于类别one-hot随机向量的协方差矩阵。
    """
    if logits.ndim != 1:
        raise ValueError(f"logits must be a 1-D tensor, got shape {tuple(logits.shape)}")

    probabilities = stable_softmax(logits)
    return torch.diag(probabilities) - torch.outer(probabilities, probabilities)


def demo_hessian_matches_categorical_variance() -> None:
    """用PyTorch自动微分验证课后练习中的二阶导数结论。"""
    logits = torch.tensor([1.0, 2.0, -0.5], requires_grad=True)
    target = torch.tensor([0.0, 1.0, 0.0])

    auto_hessian = torch.autograd.functional.hessian(
        lambda current_logits: cross_entropy_from_logits(current_logits, target),
        logits,
    )
    formula_hessian = softmax_cross_entropy_hessian(logits.detach())

    print("softmax probabilities:")
    print(stable_softmax(logits.detach()))
    print("\nHessian from autograd:")
    print(auto_hessian)
    print("\nHessian from diag(p) - p p^T:")
    print(formula_hessian)
    print("\nDo they match?")
    print(torch.allclose(auto_hessian, formula_hessian, atol=1e-6))


if __name__ == "__main__":
    demo_hessian_matches_categorical_variance()
