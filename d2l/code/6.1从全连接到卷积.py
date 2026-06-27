import torch
from torch import nn

print("=" * 60)
print("6.1 从全连接到卷积 — 代码演示")
print("=" * 60)

# ============================================================
# 1. 全连接层的参数爆炸问题
# ============================================================
print("\n1. 全连接层的参数爆炸")

# 模拟一张 100x100 的单通道图像
H, W = 100, 100
in_features = H * W       # 10,000 个输入
out_features = 1000       # 1000 个隐藏神经元

fc = nn.Linear(in_features, out_features)
fc_params = sum(p.numel() for p in fc.parameters())
print(f"   输入: {H}x{W} = {in_features} 个像素")
print(f"   隐藏神经元: {out_features}")
print(f"   全连接参数量: {fc_params:,}")          # 10,000,000 + 1,000 偏置

# 参数量的通用公式
print(f"   公式: 输入 × 输出 + 输出(偏置) = {in_features}×{out_features}+{out_features}")

# ============================================================
# 2. 卷积：对比全连接
# ============================================================
print("\n2. 卷积层 — 参数大幅减少")

# 同样 100x100 输入，3×3 卷积核，输出 64 通道
conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
conv_params = sum(p.numel() for p in conv.parameters())
print(f"   输入: 1×{H}×{W}")
print(f"   卷积核: 3×3, 输出通道: 64")
print(f"   卷积参数量: {conv_params:,}")          # 3*3*1*64 + 64 = 640
print(f"   参数量对比: 卷积 / 全连接 = {conv_params / fc_params:.6f}")
print(f"   减少了 {(1 - conv_params/fc_params)*100:.2f}% 的参数！")

# ============================================================
# 3. 卷积输出尺寸只和几何关系有关，和权重值无关
# ============================================================
print("\n3. 输出尺寸由几何关系决定，不由权重决定")

x = torch.randn(1, 1, 32, 32)  # 32×32 单通道图

# 同样的 3×3 核，因步幅/填充不同，输出尺寸不同
for kernel_size, stride, padding, label in [
    (3, 1, 0, "默认: k=3, s=1, p=0"),
    (3, 2, 0, "步幅加倍: k=3, s=2, p=0"),
    (3, 1, 1, "填充1: k=3, s=1, p=1"),
    (3, 2, 1, "混合: k=3, s=2, p=1"),
]:
    conv = nn.Conv2d(1, 8, kernel_size, stride=stride, padding=padding)
    out = conv(x)
    # 输出尺寸公式: (W - K + 2P) / S + 1
    expected = (32 - kernel_size + 2 * padding) // stride + 1
    print(f"   {label} → 输出: {out.shape[2]}×{out.shape[3]}  (公式计算: {expected})")
    print(f"      卷积核权重形状: {conv.weight.shape}, 参数值不影响输出尺寸")

# ============================================================
# 4. 互相关运算（深度学习中的"卷积"）
# ============================================================
print('\n4. 互相关运算（不翻转核，这才是深度学习的"卷积"）')


def corr2d(X, K):
    """二维互相关运算 —— 不翻转核"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# 示例：一张 5×5 的图，一个 3×3 的卷积核
X = torch.arange(25, dtype=torch.float32).reshape(5, 5)
K = torch.tensor([[1., 0, -1],
                  [1., 0, -1],
                  [1., 0, -1]])  # 垂直边缘检测器（右侧亮=正，左侧暗=负）

Y = corr2d(X, K)
print(f"   输入 X (5×5):\n{X}")
print(f"   卷积核 K (3×3, 垂直边缘):\n{K}")
print(f"   输出 Y (3×3):\n{Y}")

# 用 PyTorch 内置函数验证
Y_torch = torch.nn.functional.conv2d(
    X.reshape(1, 1, 5, 5),
    K.reshape(1, 1, 3, 3)
)
print(f"   PyTorch 内置 conv2d 结果:\n{Y_torch.squeeze()}")

# ============================================================
# 5. 权重共享：同一个核扫全图
# ============================================================
print("\n5. 权重共享——同一个核在整张图上复用")

# 创建一个 Conv2d 层，手动设置权重来观察
conv = nn.Conv2d(1, 1, 3, bias=False)
# 手动赋予一个可识别的核权重
conv.weight.data = torch.tensor([[[[1., 2., 3.],
                                    [4., 5., 6.],
                                    [7., 8., 9.]]]])

# 在三个不同位置各取 3×3 区域，用同一核计算
x = torch.randn(1, 1, 10, 10)
print(f"   卷积核权重:\n{conv.weight.data.squeeze()}")
print(f"   这个核将在 (10-3+1)×(10-3+1) = 8×8 = 64 个位置被复用")
print(f"   每个位置都用同一套权重，不因位置而变化——这就是平移不变性")

out = conv(x)
print(f"   输入: 10×10, 输出: {out.shape[2]}×{out.shape[3]}")

# ============================================================
# 6. 多通道卷积
# ============================================================
print("\n6. 多通道卷积（彩色图 RGB）")

# 彩色图：3×32×32，64 个输出通道
conv_rgb = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
print(f"   输入形状: (batch, 3, 32, 32)")
print(f"   卷积核权重形状: {conv_rgb.weight.shape}")
print(f"     - 维度 0: 64 个输出通道")
print(f"     - 维度 1: 3 个输入通道 (RGB)")
print(f"     - 维度 2,3: 3×3 空间窗口")
print(f"   参数量: 3×3×3×64 + 64(偏置) = {sum(p.numel() for p in conv_rgb.parameters()):,}")

# 前向传播
x_rgb = torch.randn(2, 3, 32, 32)  # batch=2
out = conv_rgb(x_rgb)
print(f"   输出形状: {out.shape}  ← 空间尺寸缩小到30，通道数变为64")

# ============================================================
# 7. Δ=0：1×1 卷积 = 逐像素全连接
# ============================================================
print("\n7. Δ=0 即 1×1 卷积 = 每个像素位置独立做全连接")

# 1×1 卷积
conv_1x1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
print(f"   1×1 卷积核权重形状: {conv_1x1.weight.shape}")  # [16, 3, 1, 1]
print(f"   参数量: {sum(p.numel() for p in conv_1x1.parameters()):,}")

# 等价于：在每个像素位置 (i,j)，对 3 个输入通道做全连接，输出 16 个通道
x = torch.randn(1, 3, 32, 32)
out_conv = conv_1x1(x)

# 手动实现逐像素全连接（验证等价性）
weight = conv_1x1.weight.data.squeeze()  # [16, 3]
bias = conv_1x1.bias.data               # [16]
out_manual = torch.zeros(1, 16, 32, 32)
for i in range(32):
    for j in range(32):
        pixel = x[0, :, i, j]                    # [3]
        out_manual[0, :, i, j] = weight @ pixel + bias  # 3→16 全连接

diff = (out_conv - out_manual).abs().max().item()
print(f"   1×1 卷积输出 vs 逐像素全连接输出:")
print(f"   最大误差: {diff:.2e}  (等价！)")
print(f"   这就是 Δ=0 时，卷积退化为逐位置全连接的证明")

# ============================================================
# 8. 卷积 vs 全连接：参数量数值对比
# ============================================================
print("\n8. 参数量数值对比（32×32×3 彩色图 → 30×30×64 输出）")

# 卷积方式
conv = nn.Conv2d(3, 64, 3)
conv_p = sum(p.numel() for p in conv.parameters())

# 全连接等价方式（需要输入 3072，输出 30×30×64 = 57600）
fc = nn.Linear(32 * 32 * 3, 30 * 30 * 64)
fc_p = sum(p.numel() for p in fc.parameters())

print(f"   卷积方式参数: {conv_p:,}")
print(f"   全连接方式参数: {fc_p:,}")
print(f"   比值: {conv_p / fc_p * 100:.3f}%（卷积仅为全连接的万分之三）")

# ============================================================
# 9. 权重更新：梯度累加演示
# ============================================================
print("\n9. 卷积的梯度：同一权重多个位置梯度的累加")

conv = nn.Conv2d(1, 1, 3)
x = torch.randn(1, 1, 10, 10)
target = torch.randn(1, 1, 8, 8)

# 前向传播
out = conv(x)
loss = ((out - target) ** 2).sum()

# 反向传播
conv.zero_grad()
loss.backward()

# 卷积核的梯度包含了 8×8=64 个滑动位置的梯度累加
print(f"   卷积核形状: {conv.weight.shape}")
print(f"   滑动位置数: 8×8 = 64")
print(f"   梯度形状: {conv.weight.grad.shape}")
print(f"   梯度是 64 个位置上误差的累加，并非单个位置的误差")

# 全连接对比：一个权重只对应一条连接
fc = nn.Linear(9, 1)
x_fc = torch.randn(9)
target_fc = torch.randn(1)
out_fc = fc(x_fc)
loss_fc = ((out_fc - target_fc) ** 2).sum()
fc.zero_grad()
loss_fc.backward()
print(f"\n   全连接权重形状: {fc.weight.shape}")
print(f"   全连接梯度形状: {fc.weight.grad.shape}")
print(f"   全连接每个权重只对应一条连接，不需要累加")

# ============================================================
# 10. 总结表
# ============================================================
print("\n" + "=" * 60)
print("核心总结")
print("=" * 60)

print("""
  ┌────────────────────┬───────────────────┬──────────────────────┐
  │                    │     全连接层       │       卷积层          │
  ├────────────────────┼───────────────────┼──────────────────────┤
  │ 输入结构           │ 展平为一维         │ 保留空间三维          │
  │ 权重维度           │ in_features × out │ K_H × K_W × C_in × C_out│
  │ 连接方式           │ 全连接(全局)      │ 稀疏连接(局部)         │
  │ 权重共享           │ 否(每连接独有)    │ 是(滑动复用)           │
  │ 输出尺寸决定因素   │ 权重的列数         │ 输入+核+步幅+填充     │
  │ 梯度               │ 一条连接一份梯度   │ 所有位置累加          │
  │ 更新频率           │ 每 batch 一次     │ 每 batch 一次(相同)    │
  │ 设计原则           │ 无先验            │ 平移不变性+局部性      │
  └────────────────────┴───────────────────┴──────────────────────┘
""")
