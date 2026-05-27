# 划分策略 (Split Policy)

`prepare` 阶段对 train/val/test 的切分必须满足：

1. **按类分层** (stratified)：每个类的样本按相同比例进 train / val / test，
   保证小类不会被整体丢进某一边。
2. **可复现**：相同 `(task_root, seed, val_fraction, test_fraction)` 永远产出
   位逐字节一致的 `splits.json`。
3. **小样本兜底**：哪怕某类只剩 2 张，也要至少各 1 张在 train 与 val。
4. **不破坏既有 split**：若输入已带 `train/val/test`，沿用而不重洗。

本文件定义实现这四点的算法与边界条件，模板代码必须严格按此实现。

---

## 1. 输入

```python
samples_by_class: dict[str, list[str]]   # class_name -> 该类下的相对路径列表
val_fraction:    float                   # [0, 1)
test_fraction:   float                   # [0, 1)
seed:            int
```

约束：`0 <= val_fraction + test_fraction < 1`。

---

## 2. 算法（伪代码）

```python
g = torch.Generator().manual_seed(seed)
splits = {"train": [], "val": [], "test": []}

for class_name in sorted(samples_by_class):       # 字典序，避免依赖文件系统顺序
    items = sorted(samples_by_class[class_name])  # 同上
    n = len(items)
    perm = torch.randperm(n, generator=g).tolist()
    shuffled = [items[i] for i in perm]

    n_test = _quota(n, test_fraction)
    n_val  = _quota(n, val_fraction)
    n_train = n - n_val - n_test

    # 兜底：n>=2 时保证 train >=1 且 val >=1（test 不强制）
    if n >= 2:
        if n_val == 0: n_val = 1
        if n_train == 0: n_train = 1
        # 若被兜底挤压，先牺牲 test，再牺牲 val
        while n_train + n_val + n_test > n:
            if n_test > 0: n_test -= 1
            else: n_val -= 1

    splits["test"].extend (shuffled[:n_test])
    splits["val"].extend  (shuffled[n_test : n_test + n_val])
    splits["train"].extend(shuffled[n_test + n_val:])
```

```python
def _quota(n: int, fraction: float) -> int:
    if fraction <= 0: return 0
    if fraction >= 1: return n
    return int(round(n * fraction))
```

> **为什么 `randperm` 而不是 `random.shuffle`？**
> `torch.Generator` 的种子在跨平台 (Win / Linux / macOS) 与跨 Python 小版本下
> 比 stdlib `random` 更稳定；这跟 dinov3_industrial 现有实现保持一致。

---

## 3. 三类典型场景的产出

### 3.1 标准场景（K=2, 每类 402 张, val=0.2, test=0）

| 类 | total | train | val | test |
| --- | --- | --- | --- | --- |
| Aluminum | 402 | 322 | 80 | 0 |
| Copper   | 402 | 322 | 80 | 0 |
| **sum**  | 804 | 644 | 160 | 0 |

### 3.2 加 test 的场景（val=0.2, test=0.1）

| 类 | total | train | val | test |
| --- | --- | --- | --- | --- |
| Aluminum | 402 | 281 | 80 | 41 |
| Copper   | 402 | 281 | 80 | 41 |

### 3.3 极小样本（某类 n=2, val=0.2）

`_quota(2, 0.2) = 0` → 触发兜底 `n_val=1`，结果 `train=1, val=1, test=0`。
若 n=1，类被 R3.2 提前丢弃，根本不进入本算法。

---

## 4. 与既有 split 共存

当 `pre_split_detected=true`（见 auto-fix R4.1）：

- 直接读取 `<task_root>/train/<class>/*` 等并按类聚合，作为 `splits["train"]`。
- 不再调用 §2 的算法。
- 若顶层只有 `train/` 与 `test/`（缺 val），调用 §2 但仅对 `train/` 内样本切 val
  （`pre_split_detected="partial"`）。

---

## 5. 反模式（明确禁止）

| 反模式 | 风险 |
| --- | --- |
| 在 Dataset.__init__ 里再洗一次 | 训练/验证之间互相污染；多卡训练每张卡看到不同 split |
| `random.shuffle(samples)` | Python 版本间结果不稳定；跨机不可复现 |
| 不按类分层、整体洗 | 小类可能整片掉进 val，模型连训练样本都没见过 |
| 把 val 当 test 用并据此 early stop | 过拟合 val；要么 (a) 不 early stop，要么 (b) 同时切出真 test |
| 用文件 mtime / inode 顺序确定划分 | 跨机不可复现，OneDrive / git checkout 后顺序乱掉 |

---

## 6. 调用方扩展点

- **k-fold**：不要改 `splits.json`。另开 `splits.kfold.json` 描述 K 个 fold 的 train/val。
- **自定义 split**：调用方可直接提供 `splits.json`，绕过 §2 的算法；prepare 会校验
  §2 的"分层 / 不重叠 / 文件存在"三条不变量，不通过即拒绝。
