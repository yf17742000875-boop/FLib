import torch
from torch import nn

# ============================================================
# 5.3.1 实例化网络
# ============================================================

# PyTorch 中，nn.Sequential 在定义 nn.Linear 时必须指定输入维度，
# 因此不存在真正的"延后初始化"。但 PyTorch 提供了 Lazy 版本：
#   nn.LazyLinear  —— 在第一次 forward 时自动推断 in_features

def get_net():
    net = nn.Sequential(
        nn.LazyLinear(256),  # 无需指定 in_features
        nn.ReLU(),
        nn.LazyLinear(10)
    )
    return net

net = get_net()
print(net)
print("---")

# 此时参数存在但未初始化（shape 未知），访问 shape 会报错
print("初始化前参数列表（UninitializedParameter）：")
for name, param in net.named_parameters():
    if isinstance(param, nn.UninitializedParameter):
        print(f"  {name}: 未初始化（需要传入数据后才能确定形状）")
    else:
        print(f"  {name}: {param.shape}")

# 直接访问 shape 会报错：Can't access the shape of an uninitialized parameter
# print(net[0].weight.shape)  # RuntimeError

# ----------------------------------------------------------
# 将数据传入网络，触发参数推断和初始化
X = torch.rand(size=(2, 20))
print("\n传入数据后：")
net(X)

for name, param in net.named_parameters():
    print(f"  {name}: {param.shape}")

# ----------------------------------------------------------
# 查看 state_dict
print("\nstate_dict:")
for k, v in net.state_dict().items():
    print(f"  {k}: {v.shape}")

# ============================================================
# 小结与说明
# ============================================================
"""
PyTorch 与 MXNet 的区别：

1. nn.Linear 必须在构造时指定 in_features，无法延后推断。
2. nn.LazyLinear 提供了延后初始化能力，第一次 forward 时自动推断 in_features。
3. MXNet 中参数始终存在但形状为 -1（未知），PyTorch 中延后层在推断前连参数都不存在。
4. 延后初始化对 CNN 特别有用：图片分辨率影响后续层形状，不必手动计算。
"""
