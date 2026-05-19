import os
import torch
import pandas as pd

# 创建一个用于示例的小数据集文件。
os.makedirs(os.path.join('.', 'data'), exist_ok=True)
data_file = os.path.join('.', 'data', 'house_tiny.csv')
with open(data_file, 'w', encoding='utf-8') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


data = pd.read_csv(data_file)
print("data:", data)

# 前两列是特征，最后一列 Price 是要预测的标签。
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

# 数值型缺失值通常可以用该列均值填充。
# 注意：inputs 里包含 Alley 这样的字符串列，不能直接对整个 DataFrame 求 mean。
numeric_cols = inputs.select_dtypes(include='number').columns
inputs[numeric_cols] = inputs[numeric_cols].fillna(inputs[numeric_cols].mean())
print("inputs:", inputs)

# 类别型特征无法直接转为张量，先用独热编码转成 0/1 数值列。
# dummy_na=True 会把缺失值也当作一个合法类别，生成单独的一列。
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=float)
print("inputs:", inputs)

# PyTorch 训练需要张量输入，因此将 pandas/numpy 数据转换为 float32 张量。
x = torch.tensor(inputs.to_numpy(), dtype=torch.float32)
y = torch.tensor(outputs.to_numpy(), dtype=torch.float32)
print("x:", x)
print("y:", y)