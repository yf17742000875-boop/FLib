import torch


def demo_linear_algebra():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])

    print("x + y =", x + y)
    print("x dot y =", torch.dot(x, y))

    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    print("A * B =")
    print(torch.matmul(A, B))

    print("L2 norm of x =", torch.norm(x))


if __name__ == "__main__":
    demo_linear_algebra()
