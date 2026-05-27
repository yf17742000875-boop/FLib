---
name: dataset-dataloader-builder
description: Build a standardized PyTorch Dataset and DataLoader pipeline for single-label RGB image classification. Use when the user has a folder of images that needs to become a reproducible (prepared/, splits.json, classes.json) training-ready layout, or when the user asks to "make a dataset/dataloader", "标准化数据流水线", "整理训练数据", "split into train/val", or wants a Dataset / DataLoader that conforms to the repo's dict-based contract. Out of scope: detection / segmentation / multi-modal / multi-task / non-PyTorch frameworks.
---

# Dataset & DataLoader Builder

## 何时使用本 skill

- 用户给出一个原始图像目录（ImageFolder 风格或接近），希望生成可复现的训练数据。
- 用户已有 `inputs/<task>/<class>/...` 但下游训练代码不存在，需要标准化的 Dataset / DataLoader。
- 用户的输入"不太规范"（大小写不一、夹带 corrupt 图、缺 val 划分等），需要自动归一化。
- 用户问"数据怎么传给模型？" / "DataLoader 应该返回什么？"。

**不要**在以下场景使用本 skill（直接拒绝并说明原因）：

- 检测 / 分割 / 关键点 / 姿态：标签不是单个 int，本 skill 的 prepared 契约不覆盖。
- 多任务（同一张图带多个独立 label）：参考 `dinov3/experiments/dinov3_industrial/` 已有实现，
  本 skill **不**重复该工作。
- 自监督 / 视频 / 多模态：超出"单标签 + RGB"范围。
- 非 PyTorch 框架。

## 工作流（顺序固定）

1. **读规范**。开工前必须读这几份 reference（agent 自身阅读，不展示给用户）：
   - `references/input-spec.md`        — 用户原始输入必须 / 允许长什么样
   - `references/prepared-layout.md`   — prepared/ 目录的硬契约
   - `references/output-spec.md`       — Dataset / DataLoader 的返回结构
   - `references/auto-fix-rules.md`    — 哪些不规范可以自动修
   - `references/split-policy.md`      — train/val/test 划分算法
2. **核对用户输入**。运行（或让用户运行）`templates/validate_template.py`，
   告诉用户：是合规 / 需 auto-fix / 直接拒绝。给出"修复后会变成什么"的预览，
   再决定是否物理写出 prepared。
3. **决定落地位置**。默认 `outputs/<task_name>/prepared/`；若用户已有自己的输出根，
   就放到 `<user_root>/<task_name>/prepared/`。**永远不要**写到 `task_root` 本身。
4. **拷贝并改写模板四件套**到目标项目里：
   - `validate.py` (来自 `templates/validate_template.py`) — 输入合规性检查
   - `prepare.py` (来自 `templates/prepare_template.py`)   — auto-fix + 落地 prepared/
   - `dataset.py` (来自 `templates/dataset_template.py`)   — 标准 Dataset
   - `dataloader.py` (来自 `templates/dataloader_template.py`) — 标准 DataLoader 工厂
   每份模板顶部的 `# TODO:` 注释要解决干净再交付。
5. **跑一次 prepare + 取一个 batch 自检**。在 README 里以代码块的形式给出：
   ```bash
   python prepare.py --task-root <...>            # 产出 prepared/
   python -c "from dataset import ImageClassificationDataset; ..."
   ```
6. **总结**。最后向用户给出：
   - prepared/ 路径与产物清单（含 manifest 中的 per_class / per_split 计数）
   - `conversion_report.json` 里所有 `dropped / renamed / merged_classes` 的逐条解读
   - 下游训练脚本如何 import Dataset / DataLoader 的 1 行示例

## 决策表（常见输入 → 应做什么）

| 用户输入情况 | 立刻做的事 |
| --- | --- |
| `task_root/<class>/*.png`，类目录都齐 | 直接走标准 prepare；无需 auto-fix |
| `task_root/{train,val,test}/<class>/*.png` | `pre_split_detected=true`，沿用既有划分 |
| `task_root/images/*.png` + `labels.csv` | 走 R5 (manifest 路径)；先按 csv 归类再标准化 |
| 含损坏图、混 RGBA、夹文件等 | 走 R2，丢弃并写 `conversion_report.json` |
| 单类只有 1 张 | 触发 R3.2，丢弃该类；若所有类都 ≤1 张，**拒绝**并解释 |
| 用户希望 "不动原文件" | 用 `--linkonly`（见 prepared-layout §1） |
| 用户希望 "失败即停" | 用 `--strict` 而不是默认的 `fix` |

## 输出对下游代码的硬契约

下游训练 / 推理代码**只能这样消费数据**（违反则该代码有 bug，不是 dataset 问题）：

```python
from dataset import ImageClassificationDataset
from dataloader import make_dataloader

ds = ImageClassificationDataset(prepared_root="outputs/material/prepared", split="train")
loader = make_dataloader(ds, batch_size=32, split="train")

for batch in loader:                # batch 是 dict，不是 tuple
    images = batch["image"]         # (B, 3, H, W) float32
    labels = batch["label"]         # (B,)         int64
    # ... model(images), loss(logits, labels)
```

`batch` 的完整键集合见 `references/output-spec.md`。新增字段必须保持向后兼容。

## 质量自检（交付前 agent 自己跑一遍）

完成 prepare 后，强制做以下验证；任何一项不过，回退并告诉用户原因。

1. `set(splits.train) ∩ set(splits.val) ∩ set(splits.test) == ∅`
2. `set(splits.train ∪ splits.val ∪ splits.test) == set(images/<class>/*.png from disk)`
3. 每个 split 对每个类的覆盖率 > 0（除非该类样本数太少被算法兜底）
4. `Dataset[0]["image"].shape == (3, image_size, image_size)`
5. `next(iter(DataLoader))["image"].shape == (batch_size, 3, image_size, image_size)`
6. `next(iter(DataLoader))["label"].dtype == torch.int64`

## 与本仓库已有代码的关系

- `dinov3/experiments/dinov3_industrial/data.py` 是**多任务**版本，本 skill 不要去替换它；
  反过来若有人想新做"只做 material 单任务"的实验，**应该**用本 skill 产出的代码。
- `d2l/code/3.5图像分类数据集.py` 是 D2L 教程级用法（FashionMNIST），本 skill 是工业级版本，
  在概念上对齐但增加了 prepare / auto-fix / 契约校验。
- 任何**新建**的图像分类实验目录都应优先调用本 skill 而不是从零写 Dataset。

## 约束（agent 行为边界）

- 不修改用户提供的 `task_root`。
- prepare 默认物理拷贝；用 `--linkonly` 才用 symlink。
- 不联网下载；不调用任何 `torchvision.datasets.<X>(download=True)`。
- 不写 GPU 相关代码（DataLoader 不带 device 逻辑；device 放到训练脚本里）。
- 不引入除 `torch`, `torchvision`, `PIL`, `numpy` 之外的依赖。
