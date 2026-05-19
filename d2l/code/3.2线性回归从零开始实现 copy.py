import random
import torch
from d2l import torch as d2l

# 批量大小：每次训练使用的样本数量
batch_size = 10

def synthetic_data(w, b, num_examples):
    """
    生成模拟的训练数据
    公式: y = X*w + b + 噪声
    参数:
        w: 真实的权重向量
        b: 真实的偏置值
        num_examples: 要生成的样本数量
    返回:
        X: 特征数据 (输入)
        y: 标签数据 (真实输出)
    """
    # 生成服从正态分布(均值0, 标准差1)的输入特征
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 根据线性公式计算真实输出: y = X*w + b
    y = torch.matmul(X, w) + b
    # 添加少量噪声(均值0, 标准差0.01)，模拟真实数据中的误差
    y += torch.normal(0, 0.01, y.shape)
    # 将y重塑为列向量形状 (num_examples, 1)
    return X, y.reshape((-1, 1))

# 设置真实的权重和偏置，用于生成模拟数据
true_w = torch.tensor([2, -3.4])    # 两个特征的权重
true_b = 4.2                         # 偏置项

# 生成1000个样本用于训练
features, labels = synthetic_data(true_w, true_b, 1000)

# 可视化第二个特征与标签的关系（散点图）
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)

def data_iter(batch_size, features, labels):
    """
    迭代器：每次返回一个小批量的样本
    作用：将数据分批，随机打乱顺序后逐个返回
    """
    num_examples = len(features)
    # 创建索引列表 [0, 1, 2, ..., num_examples-1]
    indices = list(range(num_examples))
    # 随机打乱索引，使样本顺序随机
    random.shuffle(indices)
    
    # 按批量大小依次取出样本
    for i in range(0, num_examples, batch_size):
        # 当前批次的索引范围（最后一个批次可能小于batch_size）
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        # 根据索引取出对应的特征和标签
        yield features[batch_indices], labels[batch_indices]

# 初始化模型参数（权重和偏置）
# 权重从正态分布中随机初始化（标准差0.01），requires_grad=True表示需要计算梯度
# w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
# 也可以初始化为全0（但收敛速度会变慢）
w = torch.zeros((2,1), requires_grad=True)
# 偏置初始化为0，也需要计算梯度
b = torch.zeros(1, requires_grad=True)

def linear_regression(X, w, b):
    """
    线性回归模型：y = X*w + b
    参数:
        X: 输入特征
        w: 权重
        b: 偏置
    返回: 模型的预测值
    """
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """
    均方损失函数：衡量预测值与真实值的差距
    公式: (y_hat - y)^2 / 2
    参数:
        y_hat: 模型预测值
        y: 真实标签
    返回: 每个样本的损失值
    """
    # 使用reshape确保y的形状与y_hat一致，避免广播导致错误
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """
    随机梯度下降优化器
    作用：根据计算出的梯度更新模型参数
    公式: param = param - lr * grad / batch_size
    参数:
        params: 需要更新的参数列表 [w, b]
        lr: 学习率（控制每次更新的步长）
        batch_size: 批量大小（用于归一化梯度）
    """
    # 不计算梯度更新过程中的梯度（只需要更新，不需要再次求导）
    with torch.no_grad():
        for param in params:
            # 根据梯度更新参数
            param -= lr * param.grad / batch_size
            # 清空梯度，为下一轮计算做准备
            param.grad.zero_()

# 超参数设置
lr = 0.03          # 学习率：控制参数更新的步长
num_epochs = 3     # 训练轮数：整个数据集遍历的次数
net = linear_regression  # 模型
loss = squared_loss      # 损失函数

# ========== 开始训练 ==========
for epoch in range(num_epochs):
    # 每一轮：遍历所有小批量数据
    for X, y in data_iter(batch_size, features, labels):
        # 计算当前批量数据的损失
        l = loss(net(X, w, b), y)
        # l的形状是(batch_size, 1)，需要先求和变为标量
        # 然后调用backward()自动计算关于[w, b]的梯度
        l.sum().backward()
        # 使用计算好的梯度更新参数
        sgd([w, b], lr, batch_size)
    
    # 每轮结束后，计算并打印整个数据集的平均损失
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# ========== 训练完成，评估结果 ==========
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')


"========================================================================================"

"""
3.2.9.1 如果权重初始化为0，训练过程如何变化

答：权重初始化为0时，训练仍然可以正常进行，但会有以下变化：
(1) 初始预测值全部为0（或仅等于偏置b），模型开始时没有区分能力
(2) 收敛速度变慢，需要更多轮训练才能接近真实参数
(3) 对于线性回归这种简单模型，零初始化是可行的，最终仍能收敛
(4) 但对于深度神经网络，零初始化会导致所有神经元学习相同的内容，
    所以神经网络通常使用随机初始化

代码验证：将第31行改为 w = torch.zeros((2,1), requires_grad=True) 即可测试
"""

"""
3.2.9.2 依据电压和电流的关系建立模型，自动微分是否可以用来学习模型参数

答：可以。根据欧姆定律 V = I * R（电压=电流×电阻）：
(1) 将电阻R作为需要学习的参数，初始化一个猜测值
(2) 用模型计算预测电压 V_pred = I * R
(3) 计算预测电压与真实电压的损失
(4) 调用 loss.backward() 自动计算损失关于R的梯度
(5) 使用梯度下降更新R的值
(6) 重复上述步骤直到收敛，就能学习到正确的电阻值
这种方法对任何数学关系（包括非线性关系）都适用
"""

"""
3.2.9.3 基于普朗克定律使用光谱能量密度能确定物体温度吗？

答：可以。普朗克定律描述了黑体辐射的能量分布与温度的关系：
(1) 不同温度的物体会发出不同波长的光，形成特定的光谱分布
(2) 测量物体在不同波长下的辐射能量
(3) 将温度T作为需要学习的参数
(4) 用普朗克公式计算预测的光谱能量密度
(5) 通过最小化预测值与测量值的差距，就能确定物体的真实温度
实际应用：天文学家通过恒星的光谱确定其表面温度（如太阳约5800K）
"""

"""
3.2.9.4 计算梯度的二阶导数可能遇到什么问题，如何解决

答：主要问题有：
(1) 计算量大：二阶导数（海森矩阵）的计算复杂度随参数数量平方增长
(2) 内存占用大：存储完整的二阶导数矩阵需要大量内存
(3) 数值不稳定：二阶导数对计算误差很敏感

解决方案：
(1) 使用对角近似：只计算对角线元素，大幅减少计算量
(2) 使用拟牛顿法（如L-BFGS）：不直接计算，而是通过迭代近似
(3) 使用自适应优化器（如Adam）：只用一阶导数，但通过自适应学习率获得类似效果
(4) 使用PyTorch等框架提供的工具简化实现
"""

"""
3.2.9.5 为什么在loss_squared中使用reshape函数

答：使用reshape是为了确保预测值y_hat和真实值y的形状一致：
(1) y_hat的形状通常是(batch_size, 1)，是一个二维列向量
(2) 真实标签y的形状可能是(batch_size,)，是一维数组
(3) 如果直接相减，PyTorch的广播机制会导致错误结果（生成batch_size×batch_size的矩阵）
(4) 使用reshape(y_hat.shape)将y转换为与y_hat相同的二维形状
(5) 这样可以保证逐元素相减，得到正确的损失值

这是一种防御性编程，使代码能够处理不同形状的输入
"""

"""
3.2.9.6 使用不同的学习率，观察损失函数下降的快慢

答：学习率控制参数更新的步长：
(1) 学习率过大（如 lr=1.0）：更新步子太大，损失可能震荡甚至不收敛
(2) 学习率过小（如 lr=0.001）：更新步子太小，损失下降很慢，需要更多轮训练
(3) 学习率适中（如 lr=0.03）：损失稳定下降，收敛速度合适

建议实验：修改第50行的lr值，分别尝试0.001、0.01、0.1、1.0，
观察损失下降的速度和稳定性，找到最佳学习率
"""

"""
3.2.9.7 如果样本个数不能被批量大小整除，data_iter函数会如何处理

答：函数会正确处理这种情况：
(1) 关键代码：min(i + batch_size, num_examples)
(2) 当到达最后一批时，如果剩余样本不足batch_size，则取剩余的所有样本
(3) 例如：1007个样本，batch_size=10，前100批每批10个，最后1批7个
(4) 优点：不丢弃任何样本，充分利用所有数据
(5) 由于每次迭代前会随机打乱顺序，最后一个批次的组成每次都不同

这种设计保证了所有样本都能被用于训练
"""
