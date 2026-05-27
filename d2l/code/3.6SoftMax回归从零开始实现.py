import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import argparse
from pathlib import Path
from IPython import display
from d2l import torch as d2l

def load_data_fashion_mnist(batch_size, resize=None, num_workers=4):
    """下载 Fashion-MNIST 数据集，然后用可配置 worker 数加载。

    多线程/多进程读取发生在 DataLoader 中：主进程负责训练循环，worker 进程
    并行调用 Dataset.__getitem__ 读取和转换样本，再组合成一个 batch。
    Windows 下使用 num_workers > 0 时，训练入口必须放在
    ``if __name__ == '__main__':`` 保护内，本文件已经这样组织。
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)

    num_workers = 4
    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers}
    if num_workers > 0:
        # 让 worker 在多个 epoch 间复用，减少反复创建进程的开销。
        loader_kwargs["persistent_workers"] = True

    return (
        data.DataLoader(mnist_train, shuffle=True, **loader_kwargs),
        data.DataLoader(mnist_test, shuffle=False, **loader_kwargs)
    )

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum().detach()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        try:
            d2l.use_svg_display()
        except NotImplementedError:
            # 普通 Python 脚本没有 notebook 的 inline GUI hook；训练仍可继续。
            pass
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    return animator, train_loss, train_acc, test_acc

def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    axes = d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    return axes[0].figure

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="D2L 3.6 softmax regression from scratch")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker 数量；Windows 下必须从 main 入口运行")
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="保存训练曲线和预测结果图片的目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    train_iter, test_iter = load_data_fashion_mnist(
        batch_size, resize=args.resize, num_workers=args.num_workers)
    print(f"DataLoader num_workers = {args.num_workers}")
    
    num_inputs = 784
    num_outputs = 10
    
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))
    
    X = torch.normal(0, 1, (2, 5))
    X_prob = softmax(X)
    print(X_prob, X_prob.sum(1))
    
    def net(X):
        return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
    
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    print(y_hat[[0, 1], y])
    print(cross_entropy(y_hat, y))
    print(accuracy(y_hat, y) / len(y))
    
    print(evaluate_accuracy(net, test_iter))
    
    lr = 0.1
    def updater(batch_size):
        return d2l.sgd([W, b], lr, batch_size)
    
    num_epochs = 10
    animator, train_loss, train_acc, test_acc = train_ch3(
        net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    curve_path = output_dir / "softmax_training_curve.png"
    animator.fig.savefig(curve_path, dpi=160, bbox_inches="tight")
    
    prediction_fig = predict_ch3(net, test_iter)
    prediction_path = output_dir / "softmax_predictions.png"
    prediction_fig.savefig(prediction_path, dpi=160, bbox_inches="tight")

    print(f"训练曲线已保存: {curve_path}")
    print(f"预测结果已保存: {prediction_path}")
    print(f"最终指标: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
