import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print("A:", A)

print("A.T:", A.T)

print("A.matmul(A.T):", A.matmul(A.T))

print("A.matmul(A.T).shape:", A.matmul(A.T).shape)

print("A.matmul(A.T).shape:", A.matmul(A.T).shape)

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("x.norm():", torch.norm(x))
