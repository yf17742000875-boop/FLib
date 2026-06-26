import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

Y = net(torch.rand(4, 8))
Y.mean()


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight) + self.bias
        return F.relu(linear)


linear = MyLinear(5, 3)
print(linear.weight)
linear(torch.rand(2, 5))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))


# 练习1：张量降维层 y_k = sum_{i,j} W_{ijk} * x_i * x_j
class BilinearReduction(nn.Module):
    """y_k = sum_{i,j} W_{ijk} * x_i * x_j, 即 x^T W_k x 的二次型降维"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features,))

    def forward(self, X):
        # X: (batch, in_features), einsum: kij,bi,bj -> bk
        return torch.einsum('kij,bi,bj->bk', self.weight, X, X) + self.bias


bilinear = BilinearReduction(5, 3)
print(bilinear(torch.rand(2, 5)).shape)  # torch.Size([2, 3])


# 练习2：返回傅立叶系数前半部分的层
class FourierHalf(nn.Module):
    """返回输入数据傅立叶变换系数的前半部分"""
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # X: (batch, dim), rfft 沿最后一维做变换，天然只返回前半部分
        return torch.fft.rfft(X, dim=-1).real


fourier = FourierHalf()
print(fourier(torch.rand(2, 8)).shape)  # torch.Size([2, 5])  8 -> floor(8/2)+1 = 5


