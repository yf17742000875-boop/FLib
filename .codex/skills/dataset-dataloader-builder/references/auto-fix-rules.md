# 自动修复规则 (Auto-Fix Rules)

`mode=fix`（默认）下，框架会在不修改原始 `task_root` 的前提下，
把若干"常见不规范"在 prepare 阶段归一化为标准布局。

设计原则：

1. **可追溯**：每一条自动修复都写入 `conversion_report.json`。
2. **不破坏语义**：宁可丢一张图，也不要悄悄把它换到另一个类。
3. **可复现**：所有修复结果对 `(task_root, seed)` 必须确定。
4. **失败安全**：任何一条 fix 失败都不应让 prepare 进程崩溃，应记录到
   `conversion_report.warnings` 并继续。

下表是优先级从高到低的修复规则；遇到冲突时按优先级早的为准。

---

## R1. 文件 / 类名归一化（路径级）

| 子规则 | 触发条件 | 修复动作 |
| --- | --- | --- |
| R1.1 类目录大小写合并 | `{"aluminum", "Aluminum", "ALUMINUM"}` 同时存在 | 合并为字典序最小的非纯小写形式，记 `merged_classes` |
| R1.2 类名首尾空格 | `" Copper"` | `.strip()` 后再判 R1.1 |
| R1.3 类名含非 ASCII | `"铜"` | 转拼音失败则保留原名，但目录名改为 `class_{sha1(name)[:8]}`；`class_name` 字段保留原文用于显示 |
| R1.4 文件名非 ASCII | 任意 | 用 `img_{global_index:05d}.png` 替换；原始名记入 `renamed.from` |
| R1.5 文件名重复（含扩展名归一化后） | `a.JPG` 与 `a.jpg` | 二者均落到 `img_*.png`，记一条 `warnings` |
| R1.6 嵌套子目录 | `<class>/<subdir>/*.png` | 默认展平：把 `subdir` 拼接到类名（`<class>__<subdir>`），见 input-spec §3.3 |

---

## R2. 内容级清洗（解码 / 哈希）

| 子规则 | 触发条件 | 修复动作 |
| --- | --- | --- |
| R2.1 解码失败 | `PIL.Image.open(...).load()` 抛错 | 丢弃，记 `dropped.reason="corrupt_image"` |
| R2.2 单像素 / 极小图 | `min(W, H) < 8` | 丢弃，记 `dropped.reason="too_small"` |
| R2.3 同 sha1 在多类中出现 | 同一字节内容在 `ClassA/` 与 `ClassB/` 各出现 | **不修复**，记入 `label_conflicts` 并丢弃所有副本；若超过总数 1%，prepare 整体失败 |
| R2.4 alpha / 灰度 / CMYK | mode 非 RGB | 显式 `.convert("RGB")` 后保存为 PNG；记 `warnings`，不丢弃 |
| R2.5 EXIF 旋转 | `getexif().Orientation != 1` | 调用 `ImageOps.exif_transpose` 后保存；下游 transform 不需要再处理方向 |

---

## R3. 类级别清洗

| 子规则 | 触发条件 | 修复动作 |
| --- | --- | --- |
| R3.1 单类 0 张（清洗后） | 经过 R2 后某类剩 0 张 | 直接丢弃该类；`classes.json` 不再包含它；记 `warnings` |
| R3.2 单类 1 张 | 经过 R2 后某类剩 1 张 | 该类整体丢弃；记 `dropped.reason="class_too_small"` |
| R3.3 类极不均衡 | `max(per_class) / min(per_class) > 50` | 不修复，仅写 `warnings.append("severe_imbalance")` 提醒下游加权重 |

---

## R4. split 兜底（与 split-policy.md 联动）

| 子规则 | 触发条件 | 修复动作 |
| --- | --- | --- |
| R4.1 检测到顶层 train/val/test | 见 input-spec §3.1 | 使用既有 split，记 `pre_split_detected=true`，不重新洗牌 |
| R4.2 既有 split 中 val 为空 | `train/` 非空但 `val/` 缺失 | 从 train 按类分层切 `val_fraction`；写 `pre_split_detected=partial` |
| R4.3 用户传 `val_fraction` 但已有 split | 同时检测到 R4.1 与 `val_fraction>0` | **优先保留既有 split**，把用户的参数写入 `manifest.split.requested_val_fraction_ignored=...` |

---

## R5. manifest 文件冲突

| 子规则 | 触发条件 | 修复动作 |
| --- | --- | --- |
| R5.1 同时存在 ImageFolder 与 manifest 文件（labels.csv） | 两种布局共存 | 以 `labels.csv` 为准；目录树仅用于查找文件位置；记 `warnings` |
| R5.2 manifest 行的 label 不在任何已知类中 | csv label 未出现在目录名 | 新建该类；不丢弃 |
| R5.3 manifest 行的 filename 找不到对应文件 | csv 引用了不存在的图 | 丢弃该行，记 `dropped.reason="manifest_orphan"` |

---

## 严格模式 (`mode=strict`) 的差异

- 仅运行 R1.1 / R1.2 的检测、不修复；命中即抛 `ValueError`。
- R2.1 / R2.2 / R2.4 / R2.5 仍执行清洗（解码层面的修正属于"读图"而非"改语义"）。
- R3.x / R4.2 / R5.x 不允许：发现即抛错，把所有冲突一次性打印（不要打到一半就 raise）。
- 仍写 `conversion_report.json`，但里面只有 `would_*` 字段表示"在 fix 模式下会怎么改"。

---

## 修复优先级与终止条件

`prepare` 主循环：

```
for rule in [R1, R2, R3, R4, R5]:
    rule.apply(state) | rule.detect(state)  # fix 或 strict
    if state.has_fatal: raise ...
state.validate_invariants()                  # 见 prepared-layout §2.3
state.write_outputs()
```

任何一条规则被触发都不能影响后续规则的输入语义——这就是为什么 R1（路径级）
永远先于 R2（内容级）执行；R3（类级别）必须看完所有图之后再做。
