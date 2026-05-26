import random
import torch
from torch.utils import data    # PyTorch官方数据加载工具
from d2l import torch as d2l   # D2L库，提供可视化等工具


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
true_w = torch.tensor([2, -3.4])    # 两个特征的真实权重
true_b = 4.2                         # 真实偏置项

# 生成1000个训练样本
features, labels = synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    """
    创建数据迭代器（使用PyTorch官方工具）
    参数:
        data_arrays: 包含特征和标签的元组
        batch_size: 每次返回的样本数量
        is_train: 是否打乱数据顺序（训练时通常需要打乱）
    返回:
        DataLoader对象，可迭代获取小批量数据
    """
    # 将特征和标签打包成Dataset
    dataset = data.TensorDataset(*data_arrays)
    # 创建DataLoader，自动处理批量和打乱
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 超参数设置
batch_size = 10      # 批量大小：每次训练使用的样本数量
num_epochs = 3       # 训练轮数：整个数据集遍历的次数
lr = 0.03            # 学习率：控制参数更新的步长

# 创建数据迭代器
data_iter = load_array((features, labels), batch_size)


# 定义神经网络模型
# Sequential: 按顺序堆叠网络层
# Linear: 线性层，实现 y = X*w + b
net = torch.nn.Sequential(
    torch.nn.Linear(len(features[0]), 1)  # 输入维度: 特征数，输出维度: 1
)


# 初始化模型参数（权重和偏置）
net[0].weight.data.normal_(0, 0.01)  # 权重初始化为正态分布(均值0, 标准差0.01)
net[0].bias.data.fill_(0)            # 偏置初始化为0


# 定义损失函数和优化器
# MSELoss: 均方误差损失 (y_hat - y)^2
# reduction='sum': 将所有样本的损失相加
loss = torch.nn.MSELoss(reduction='sum')

# SGD: 随机梯度下降优化器
# net.parameters(): 获取模型的所有可训练参数
# lr=lr: 设置学习率
trainer = torch.optim.SGD(net.parameters(), lr=lr)


# ========== 开始训练 ==========
for epoch in range(num_epochs):
    # 遍历所有小批量数据
    for X, y in data_iter:
        # 计算当前批量的预测损失
        l = loss(net(X), y)
        
        # 清空上一轮计算的梯度（防止累加）
        trainer.zero_grad()
        
        # 自动计算梯度（反向传播）
        l.backward()
        
        # 根据梯度更新参数
        trainer.step()
    
    # 每轮结束后，计算整个数据集的损失并打印
    l = loss(net(features), labels)
    print(f"epoch {epoch}, loss: {l.item():.4f}")


# ========== 训练完成，评估结果 ==========
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)


"========================================================================================"


"""
3.3.9.1 如果将小批量的总损失替换为小批量损失的平均值，需要如何更改学习率

答：需要将学习率乘以 batch_size（批量大小）。

原因：
(1) 当前代码使用 reduction='sum'，损失是所有样本损失的总和
(2) 如果改为 reduction='mean'（平均值），损失会变为原来的 1/batch_size
(3) 梯度会随之变小（也是原来的 1/batch_size）
(4) 为了保持相同的参数更新幅度，需要将学习率乘以 batch_size

修改方法：
- 将第50行改为: loss = torch.nn.MSELoss(reduction='mean')
- 将第39行学习率改为: lr = 0.03 * 10 = 0.3

这样可以保持参数更新的步长不变，训练效果相近
"""


"""
3.3.9.2 查看深度学习框架文档，它们提供了哪些损失函数和初始化方法？用Huber损失代替原损失

答：

1. PyTorch提供的常见损失函数：
   - MSELoss: 均方误差（回归问题）
   - CrossEntropyLoss: 交叉熵损失（分类问题）
   - L1Loss: L1损失（绝对值误差）
   - SmoothL1Loss: 平滑L1损失（Huber损失）
   - BCEWithLogitsLoss: 二元交叉熵（带sigmoid）

2. PyTorch提供的初始化方法：
   - normal_(): 正态分布初始化
   - uniform_(): 均匀分布初始化
   - constant_(): 常数初始化
   - xavier_uniform_(): Xavier均匀初始化（用于激活函数为tanh/sigmoid）
   - kaiming_normal_(): He正态初始化（用于ReLU激活函数）

3. 使用Huber损失替换原损失：
   Huber损失结合了MSE和MAE的优点，对异常值不敏感
   公式: 当|y_hat-y| <= delta时用MSE，否则用MAE
   
   修改方法：
   将第50行改为:
   loss = torch.nn.SmoothL1Loss(reduction='sum')
   
   SmoothL1Loss就是PyTorch中的Huber损失，默认delta=1.0
"""


"""
3.3.9.3 如何访问线性回归的梯度？

答：可以通过以下方法访问：

1. 访问权重的梯度：
   net[0].weight.grad  # 返回权重的梯度张量
   
2. 访问偏置的梯度：
   net[0].bias.grad    # 返回偏置的梯度张量

3. 完整示例：
   # 在训练循环中，计算梯度后访问
   for X, y in data_iter:
       l = loss(net(X), y)
       trainer.zero_grad()
       l.backward()      # 计算梯度
       
       # 此时可以访问梯度
       print('权重梯度:', net[0].weight.grad)
       print('偏置梯度:', net[0].bias.grad)
       
       trainer.step()    # 更新参数

注意：
- 梯度只在调用backward()后存在
- 调用trainer.step()或trainer.zero_grad()后会清空梯度
- grad属性是一个张量，形状与对应的参数相同
"""
