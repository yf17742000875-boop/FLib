import random
import torch
from torch.utils import data
from d2l import torch as d2l


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

true_w = torch.tensor([2, -3.4])    
true_b = 4.2                        

features, labels = synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
num_epochs = 3
lr = 0.03
data_iter = load_array((features, labels), batch_size)

net = torch.nn.Sequential(
    torch.nn.Linear(len(features[0]), 1)
)


net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = torch.nn.MSELoss(reduction='sum')
trainer = torch.optim.SGD(net.parameters(), lr=lr)


for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f"epoch {epoch}, loss: {l.item():.4f}")


w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)



"""
3.3.9.1 如果将小批量的总损失替换为小批量损失的平均值，需要如何更改学习率
"""