# 输出规范 (Output Specification)

本框架"输出"有两层：

1. **Dataset 层**：`__getitem__(idx)` 的返回结构
2. **DataLoader 层**：`next(iter(loader))` 一个 batch 的 dict 结构

这两层是下游 train/eval/infer 真正消费的契约，必须**全项目统一**。
违反本契约的实现会导致下游代码静默拿到错误形状或错位标签。

---

## 1. Dataset 单样本契约

唯一返回类型：**`dict[str, Any]`**（不返回 tuple，便于增量加字段时不破坏调用方）。

```python
sample = dataset[i]
sample == {
    "image":     torch.FloatTensor,   # shape (3, H, W), dtype float32, value in standardized range
    "label":     int,                 # 0 <= label < num_classes
    "class_name": str,                # 与 classes.json 一致
    "path":      str,                 # 相对 prepared_root 的 POSIX 路径，例 "Aluminum/img_00000.png"
    "index":     int,                 # 在该 split 内的稳定索引，==i
}
```

### 1.1 字段强约束

| 字段 | 类型 | 形状 / 取值 | 备注 |
| --- | --- | --- | --- |
| `image` | `torch.Tensor` | `(3, H, W)`，`H==W==image_size` | float32；做完 ToTensor + Normalize 后 |
| `label` | `int`（Python 原生） | `[0, K)` | **不要返回 0-d tensor**，否则 default_collate 行为有歧义 |
| `class_name` | `str` | 必须 ∈ `classes.json.classes` | 仅用于日志 / debug，不参与 loss |
| `path` | `str` | 相对 prepared 根的 POSIX 字符串 | 永远用 `/`，Windows 也一样 |
| `index` | `int` | `0 <= index < len(dataset)` | 便于 collate 后回溯原样本 |

### 1.2 transforms 应用顺序（固定）

`prepare_data.py` 不做几何变换，只做格式归一化。所有几何 / 颜色变换都在 Dataset 内：

```
PIL.open(path).convert("RGB")
  → Resize((image_size, image_size))             # train/val/test 都做
  → [train only] RandomHorizontalFlip(p=0.5)
  → [train only] ColorJitter(brightness=0.1, contrast=0.1)
  → ToTensor()
  → Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
```

- 这是 **DINOv3 / torchvision 默认**的归一化参数，与本仓库已有实验一致。
- 若用户传 `custom_transform=...`，则上述链条整段被替换；框架不再保证 image_size。

---

## 2. DataLoader batch 契约

```python
batch = next(iter(loader))
batch == {
    "image":      torch.FloatTensor,   # (B, 3, H, W)
    "label":      torch.LongTensor,    # (B,) dtype=int64
    "class_name": list[str],           # len == B
    "path":       list[str],           # len == B
    "index":      torch.LongTensor,    # (B,)
}
```

- 框架提供 `make_dataloader()` 工厂；默认使用 `torch.utils.data.default_collate`
  即可得到上述结构。Python `int` 会自然 collate 成 LongTensor，`str` 会被收集成 `list[str]`。
- **不允许**自定义 collate 改变上述键名 / 形状，否则下游 `model(batch["image"])` 会断。
- Loader 默认参数：

  | 参数 | train | val / test |
  | --- | --- | --- |
  | `batch_size` | 用户传入 | 同 train 或更小 |
  | `shuffle` | `True` | `False` |
  | `drop_last` | `True` | `False` |
  | `num_workers` | `min(4, os.cpu_count() // 2)` | 同 train |
  | `pin_memory` | `True` if cuda available else `False` | 同 train |
  | `persistent_workers` | `True` if num_workers > 0 else `False` | 同 train |

---

## 3. 形状速查表（image_size=224, batch_size=32, K=2）

| 阶段 | 张量 | shape | dtype |
| --- | --- | --- | --- |
| `Image.open` | PIL.Image | (H_raw, W_raw, 3) | uint8 |
| `Resize` | PIL.Image | (224, 224, 3) | uint8 |
| `ToTensor` | torch | (3, 224, 224) | float32, [0, 1] |
| `Normalize` | torch | (3, 224, 224) | float32, ~N(0, 1) |
| `Dataset.__getitem__` | dict | `image:(3,224,224)`, `label:int` | — |
| `DataLoader` 一个 batch | dict | `image:(32,3,224,224)`, `label:(32,)` | float32 / int64 |
| `model(batch["image"])` | torch | (32, K) | float32 |
| `F.cross_entropy(logits, batch["label"])` | scalar | `()` | float32 |

调试时把这张表打印一遍即可判断哪一环出了问题。

---

## 4. 错误模式与排查指引

| 现象 | 最可能原因 |
| --- | --- |
| `RuntimeError: stack expects each tensor to be equal size` | 漏了 `Resize` 或 `image_size` 不一致 |
| `IndexError: Target ... is out of bounds` | `splits.json` 与 `classes.json` 不同步；或下游模型 `output_dim` 小于 `len(classes)` |
| 验证集 acc 异常高 | `val_fraction=0` 而调用方未察觉 / split 泄漏到 train |
| Loader 卡死 | Windows 下 `num_workers>0` 且未在 `if __name__ == '__main__':` 内启动；默认 `persistent_workers=True` 会放大现象 |
| GPU 利用率长期 < 30% | `num_workers` 过小或 `pin_memory=False`；优先看 `len(loader)` 是否过小（建议 >= 100 步/epoch） |

---

## 5. 与下游的硬契约

下游脚本应永远按下面这种方式拿数据，**禁止解构 tuple**：

```python
for batch in loader:
    images = batch["image"].to(device, non_blocking=True)
    targets = batch["label"].to(device, non_blocking=True)
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
```

这样新增字段（例如未来扩展 `sample_weight`）不会破坏既有训练脚本。
