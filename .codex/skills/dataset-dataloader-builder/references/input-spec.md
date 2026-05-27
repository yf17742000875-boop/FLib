# 输入规范 (Input Specification)

本规范定义"原始数据"在交给本框架前应当满足的目录布局与文件契约。
本规范**只覆盖单标签图像分类（RGB）场景**，多任务 / 多模态 / 检测 / 分割不在范围内。

任何不满足本规范的输入，框架默认进入 **auto-fix 模式**（见 `auto-fix-rules.md`）；
若调用方显式要求 `--strict`，则直接抛错并打印整改清单。

---

## 1. 基本约定

| 维度 | 约定 |
| --- | --- |
| 任务类型 | 单标签图像分类 (1 image -> 1 class) |
| 数据模态 | RGB 静态图像 |
| 数据来源 | 本地磁盘（不下载、不联网） |
| 字符集 | 目录与文件名必须是 ASCII；非 ASCII 在 prepare 阶段会被重命名为 hash 后缀 |
| 大小写 | 类名大小写敏感保留，但 auto-fix 会合并仅大小写不同的类（见 auto-fix-rules） |
| 允许扩展名 | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp` |
| 颜色空间 | 任意 PIL 可解码的格式，最终在 Dataset 内统一 `convert("RGB")` |
| 单类最少样本 | 严格模式 >= 10；auto-fix 模式 >= 2（否则该类被丢弃并记录） |

---

## 2. 标准目录布局（推荐 / 严格模式必须）

唯一标准布局是 **ImageFolder 风格**：

```
<task_root>/
├── <class_name_1>/
│   ├── img_0001.png
│   ├── img_0002.png
│   └── ...
├── <class_name_2>/
│   ├── ...
└── <class_name_K>/
    └── ...
```

- 一个 `<task_root>` 对应**一个独立的分类任务**。
- 子目录名即类名 (`class_name`)，类名集合按字典序生成 `class_to_idx`。
- 类目录下只允许图像文件，不允许嵌套子目录（嵌套子目录会在 auto-fix 中被展平）。
- 不要求文件名格式，但**同一类目录内文件名必须唯一**。

> 单一 `<task_root>` 调用方式举例（仓库内真实数据）：
>
> - `dinov3/inputs/material/` 是一个 task root，类是 `Aluminum/`, `Copper/`
> - `dinov3/inputs/process/`  是另一个 task root，类是 `Cross/`, `Cube/`
>
> 若想"一张图同时学 material 和 process"，那是**多任务**场景，超出本 skill
> 范围；按"两个独立单标签任务"分别调用本框架即可。

---

## 3. 已知可接受的非标准布局（auto-fix 会归一化）

下列布局允许直接喂给框架，会被自动转换到 §2 的标准形态，并产出
`conversion_report.json` 记录每一次重命名 / 移动 / 丢弃。

### 3.1 已带 split 的布局

```
<task_root>/
├── train/
│   ├── <class_a>/...
│   └── <class_b>/...
├── val/
│   ├── <class_a>/...
│   └── <class_b>/...
└── test/                # 可选
    └── ...
```

- 当顶层目录恰好命中 `{train, val, test}` 或 `{train, valid, test}` 或
  `{training, validation, testing}`（不区分大小写）时，视为"已划分"。
- auto-fix **不会重新洗牌**，而是把已有的 split 写入 `splits.json`，
  并在 `conversion_report.json` 里标 `pre_split_detected: true`。
- 缺失 `val/` 时会从 `train/` 中按类分层切出（见 `split-policy.md`）。

### 3.2 单目录扁平 + manifest 文件

```
<task_root>/
├── images/
│   ├── any_name_001.png
│   └── ...
└── labels.csv           # 列：filename,label  (label 是字符串)
```

- 必须恰好包含一个 `images/` 与一个 `labels.{csv,tsv,json,jsonl}`。
- `labels.csv` 表头必须有 `filename` 与 `label` 两列；额外列被忽略。
- `labels.json` 形如 `{"any_name_001.png": "ClassA", ...}`。
- auto-fix 会重组为 §2 的 ImageFolder 布局（物理拷贝或 symlink，见 prepared-layout）。

### 3.3 嵌套子类 / 多层目录

```
<task_root>/<superclass>/<subclass>/*.png
```

- auto-fix 默认行为：将每一条叶子路径 `<sc>/<ssc>` 拼成 `<sc>__<ssc>` 作为新类名。
- 若调用方显式传 `--collapse-depth=1`，则只保留第一层 `<sc>` 作为类。
- 若不希望合并、希望直接报错，使用 `--strict`。

---

## 4. 明确**不**支持的输入（直接报错，不 auto-fix）

| 情形 | 处理 |
| --- | --- |
| `<task_root>` 不是目录 | `FileNotFoundError` |
| `<task_root>` 下没有任何子目录 | `ValueError: no class folder found` |
| 所有类都仅有 1 张图 | `ValueError: dataset too small to split` |
| 同一文件出现在多个类目录下（内容 SHA1 相同） | `ValueError: label conflict, see conflicts.json` |
| 图像 PIL 解码失败超过 5% | `RuntimeError: too many corrupt images` |

---

## 5. 调用方与框架之间的契约

调用方在调用本框架前需提供：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `task_root` | `str \| Path` | 是 | 见 §2/§3 |
| `task_name` | `str` | 否 | 用于日志与 prepared 目录命名，默认取 `task_root.name` |
| `image_size` | `int` | 否 | 默认 224 |
| `val_fraction` | `float` | 否 | `[0, 1)`，默认 0.2，已带 split 时被忽略 |
| `test_fraction` | `float` | 否 | `[0, 1)`，默认 0.0；与 `val_fraction` 之和 < 1 |
| `seed` | `int` | 否 | 默认 42 |
| `mode` | `'strict' \| 'fix'` | 否 | 默认 `fix` |

框架对调用方的承诺：

1. 在 `prepared/` 下生成可被 `output-spec.md` 描述的标准产物；
2. 任何破坏性操作（重命名、丢弃、合并类）都必须写入 `conversion_report.json`；
3. 不修改原始 `task_root`；
4. 相同 `(task_root, seed, val_fraction, test_fraction)` 必须产出位逐字节一致的 `splits.json`。
