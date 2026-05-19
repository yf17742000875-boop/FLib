# 注意：torch 需要单独安装，使用 pip install torch 进行安装
import torch
'''深度学习存储和操作数据的主要接口是张量（n维数组）。它提供了各种功能，包括基本数学运算、广播、索引、切片、内存节省和转换其他Python对象。'''

x = torch.arange(12)
print(x)
print("x.shape:", x.shape)
print("x.numel():", x.numel())

print("x.reshape(3, 4):", x.reshape(3, 4))
print("torch.zeros((2, 3, 4)):", torch.zeros((2, 3, 4)))
print("torch.randn(3, 4):", torch.randn(3, 4))

x = torch.tensor([1, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print("x + y:", x + y)
print("x - y:", x - y)
print("x * y:", x * y)
print("x / y:", x / y)
print("x ** y:", x ** y)

print("torch.exp(x):", torch.exp(x))

x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("torch.cat((x, y), dim=0):", torch.cat((x, y), dim=0))
print("torch.cat((x, y), dim=1):", torch.cat((x, y), dim=1))

print("x == y:", x == y)
print("x.sum():", x.sum())

a = torch.arange(9).reshape(3, 3)
b = torch.arange(3).reshape(1, 3)
print("a:", a)
print("b:", b)
print("a + b:", a + b)


print("x[-1]:", x[-1])
print("x[1:3]:", x[1:3])
x[1:2] = 9
print("x:", x)

z = torch.zeros_like(x)
print("z:", z)
z[:] = 9
print("z:", z)


before = id(x)
print('id(z)', id(z))
z[:] = x+y
print('id(z)', id(z))

before = id(x)
x += y
print("id(x) == before:", id(x) == before)

a = x.numpy()
b = torch.tensor(a)
print("type(a):", type(a))
print("type(b):", type(b))

a = torch.tensor([3.5])
print("a, a.item(), float(a), int(a):", a, a.item(), float(a), int(a))
