import torch
from torch import nn
from torch.nn import functional as F

X = torch.rand(2, 20)

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
net(X)



class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hide1 = nn.Linear(20, 256)  # 隐藏层
        self.hide2 = nn.Linear(256, 128)  # 隐藏层
        self.out = nn.Linear(128, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        x = F.relu(self.hide1(X))
        x = F.relu(self.hide2(x))
        return self.out(x)


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)



class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


net = FixedHiddenMLP()
net(X)


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)


# 问题1：平行块 —— 输入同时经过两个块，输出拼接
class ParallelBlock(nn.Module):
    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, X):
        out1 = self.net1(X)
        out2 = self.net2(X)
        # 在最后一个维度（特征维度）上拼接
        return torch.cat((out1, out2), dim=-1)


# 示例：输入 (2,20)，net1 输出 (2,10)，net2 输出 (2,5)，拼接后 (2,15)
parallel = ParallelBlock(nn.Linear(20, 10), nn.Linear(20, 5))
print("平行块输出形状:", parallel(X).shape)  # torch.Size([2, 15])


# 问题2：生成同一块类的多个实例，串联构成更大的网络
def build_repeated_net(block_factory, num_instances):
    """生成同一块的多个独立实例，串联构成更大的网络。

    参数:
        block_factory: 一个可调用对象（如 lambda），每次调用返回一个新的块实例
        num_instances: 生成多少个实例
    """
    layers = [block_factory() for _ in range(num_instances)]
    return nn.Sequential(*layers)


# 示例：生成 3 个相同的 (Linear+ReLU) 块，串联
repeated_net = build_repeated_net(
    lambda: nn.Sequential(nn.Linear(20, 20), nn.ReLU()),
    num_instances=3
)
