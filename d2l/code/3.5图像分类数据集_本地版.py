import torch
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image

class ProcessDataset(data.Dataset):
    """本地 process 图像分类数据集。

    Dataset 只需要回答两个问题：
    1. 一共有多少个样本？ -> __len__
    2. 给我第 idx 个样本是什么？ -> __getitem__
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = ['Cross', 'Cube']
        
        data_dir = os.path.join(root_dir, 'process')
        
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"类别目录不存在: {class_dir}")
            
            # 初始化阶段只建立“索引表”：图片路径 -> 数字标签。
            # 不在这里读入所有图片，否则数据集稍大就会占用很多内存。
            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith('.png'):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(label)
        
        if len(self.image_paths) == 0:
            raise ValueError("未找到任何图像文件")
    
    def __len__(self):
        # DataLoader 会用这个长度知道一轮 epoch 需要取多少个索引。
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 真正读图发生在这里：DataLoader 每需要一个样本，就调用一次 dataset[idx]。
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            raise RuntimeError(f"读取图像失败 {img_path}: {e}")
        
        label = self.labels[idx]
        
        if self.transform:
            # transform 把 PIL 图像处理成模型可用的张量。
            # ToTensor 后形状通常是 [C, H, W]，灰度图 C=1，数值范围变成 [0, 1]。
            image = self.transform(image)
        
        return image, label

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy().squeeze(), cmap='gray')
        else:
            ax.imshow(img, cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def load_data_process(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, '../dataset')
    root_dir = os.path.abspath(root_dir)
    
    dataset = ProcessDataset(
        root_dir=root_dir,
        transform=trans
    )
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # DataLoader 不关心图片怎么读取；它只负责：
    # 1. 生成索引顺序（shuffle=True 时会打乱）
    # 2. 反复调用 Dataset.__getitem__
    # 3. 把多个样本自动堆叠成一个 batch
    train_iter = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0
    )
    test_iter = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0
    )
    
    return train_iter, test_iter, dataset.classes

if __name__ == '__main__':
    try:
        train_iter, test_iter, classes = load_data_process(batch_size=18, resize=64)
        base_dataset = train_iter.dataset.dataset
        
        print(f"类别: {classes}")
        print(f"完整数据集样本数: {len(base_dataset)}")
        print(f"训练集样本数: {len(train_iter.dataset)}")
        print(f"测试集样本数: {len(test_iter.dataset)}")
        
        one_image, one_label = base_dataset[0]
        print("\nDataset 取单样本:")
        print(f"dataset[0] -> 图像形状: {one_image.shape}, 标签: {one_label} ({classes[one_label]})")
        
        X, y = next(iter(train_iter))
        print("\nDataLoader 取一个 batch:")
        print(f"next(iter(train_iter)) -> 图像形状: {X.shape}, 标签形状: {y.shape}")
        print("理解形状: 图像 batch 是 [batch_size, channels, height, width]，标签 batch 是 [batch_size]")
        
        titles = [classes[int(label)] for label in y]
        show_images(X, 2, 9, titles=titles)
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_dataset_samples.png')
        plt.savefig(output_path)
        print(f"\n图像已保存为 {output_path}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
