import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print("A:", A)


x = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("x.norm():", torch.norm(x))
