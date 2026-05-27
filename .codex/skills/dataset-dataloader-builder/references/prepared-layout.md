# 中间态规范 (Prepared Layout)

`prepare` 阶段的唯一职责是把任意合法输入（见 `input-spec.md`）落地为
**位逐字节可复现的标准产物**，供训练 / 推理脚本零参数加载。

prepared 目录是本框架与下游一切代码（train / eval / infer / score / app）之间
真正的"硬契约"。Dataset / DataLoader 实现只读 prepared 目录，绝不回头去看原始输入。

---

## 1. 目录布局

```
<prepared_root>/                              # 例：outputs/<task_name>/prepared/
├── manifest.json            必有  数据集元信息（类名、计数、来源、seed 等）
├── splits.json              必有  train/val/test 文件名列表
├── classes.json             必有  classes + class_to_idx + 反向映射
├── conversion_report.json   必有  auto-fix 期间所有改写操作
├── images/                  必有  规范化后的图像
│   ├── <class_name_1>/
│   │   ├── img_00000.png
│   │   ├── img_00001.png
│   │   └── ...
│   ├── <class_name_2>/
│   └── ...
└── samples.jsonl            可选  调试用 flat 索引：每行 {id, path, class, split}
```

> **物理 vs 引用**：默认 `prepare` 把图像拷贝进 `images/`（最稳，CI / 跨机最友好）。
> 若 `--linkonly`，则 `images/<class>/img_*.png` 是相对 symlink 指向原始路径；
> 此时 prepared 目录不可独立打包，但磁盘开销为 0。

---

## 2. 文件格式契约

### 2.1 `classes.json`

```json
{
  "classes": ["Aluminum", "Copper"],
  "class_to_idx": {"Aluminum": 0, "Copper": 1},
  "idx_to_class": {"0": "Aluminum", "1": "Copper"}
}
```

- `classes` 按字典序，**永远是 `class_to_idx` 的真理源**。
- 下游模型保存权重时必须把 `classes.json` 一起带走（`output_dim` 与 `classes` 必须匹配）。

### 2.2 `splits.json`

```json
{
  "schema_version": 1,
  "seed": 42,
  "val_fraction": 0.2,
  "test_fraction": 0.0,
  "pre_split_detected": false,
  "train": ["Aluminum/img_00000.png", "Aluminum/img_00003.png", "Copper/img_00001.png"],
  "val":   ["Aluminum/img_00007.png", "Copper/img_00005.png"],
  "test":  []
}
```

- 路径**相对 `<prepared_root>/images/`**，使用正斜杠（Windows 也用 `/`）。
- 每个文件名在 `train/val/test` 中**至多出现一次**。
- 文件内顺序必须由 `(class_name, original_index, seed)` 决定，不依赖文件系统遍历顺序。

### 2.3 `classes.json` 与 `splits.json` 的派生约束

```
set(splits.train ∪ splits.val ∪ splits.test) ⊆ set(images/<class>/<file>.png for class in classes)
```

`prepare` 完成后必须显式校验这一不变量；不变量被破坏即视为 prepare 失败。

### 2.4 `manifest.json`

```json
{
  "schema_version": 1,
  "task_name": "material",
  "task_type": "image_classification_single_label",
  "modality": "rgb",
  "source": {
    "root": "D:/Work/ML/dinov3/inputs/material",
    "scanned_at": "2026-05-26T03:51:00Z",
    "sha1_of_listing": "ab12..."
  },
  "image": {
    "expected_channels": 3,
    "target_size": [224, 224],
    "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
  },
  "counts": {
    "total": 804,
    "per_class": {"Aluminum": 402, "Copper": 402},
    "per_split": {"train": 644, "val": 160, "test": 0}
  },
  "auto_fix": {
    "applied": true,
    "n_renamed": 0,
    "n_dropped": 0,
    "n_merged_classes": 0,
    "n_corrupt_skipped": 0
  }
}
```

`schema_version` 用于未来扩展（多任务、多模态会增加字段，但旧字段必须保留）。

### 2.5 `conversion_report.json`

记录每一项破坏性 / 重写操作，便于审计和回滚。结构：

```json
{
  "renamed": [{"from": "<orig>", "to": "<images/Aluminum/img_00000.png>"}],
  "dropped": [{"path": "<orig>", "reason": "corrupt_image"}],
  "merged_classes": [{"into": "Aluminum", "from": ["aluminum", "ALUMINUM"]}],
  "label_conflicts": [{"sha1": "abc...", "appears_in": ["Aluminum", "Copper"]}],
  "warnings": ["images/Copper has 2 files with non-ascii name, hash-renamed"]
}
```

即便 `applied=false`（严格模式），也应写出 `conversion_report.json`
（仅含 `warnings` / `would_*` 字段），让用户看到框架的判断。

---

## 3. 命名规则

- 类目录名：原始类名 → ASCII slug → 大小写归一化（见 auto-fix-rules）。
- 文件名：`img_{global_index:05d}.png`，`global_index` 按 `(class_name 字典序, 原始文件名字典序)` 全局递增；**所有图像统一存为 PNG，无损解码后写出**，原扩展名不保留。

> 为什么强制 PNG？  
> 1) 训练时再做 `decode -> tensor` 时 PIL 不需要判分支；  
> 2) prepared 目录可哈希校验；  
> 3) 不同源图（jpg/webp）解码差异在 prepare 阶段一次性吃掉，下游可复现。  
> 代价：磁盘占用上升约 1.5~3 倍。若不接受，使用 `--linkonly` 关闭物理化。

---

## 4. 与下游的接口

下游训练 / 推理脚本只能这样初始化数据集：

```python
ds_train = ImageClassificationDataset(
    prepared_root="outputs/material/prepared",
    split="train",
    image_size=224,            # 必须 == manifest.image.target_size[0]
)
```

- Dataset 启动时**必须**读 `classes.json` + `splits.json` + `manifest.json`，
  并断言 `image_size` 与 manifest 一致。不一致即抛错，禁止"静默 resize 到别的尺寸"。
- 任何下游代码若想另存自己的 split（例如 k-fold），**应在 prepared/ 旁边新建
  `splits.kfold.json`**，而不是改写 `splits.json`。
