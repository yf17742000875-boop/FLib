"""dataloader_template.py — 标准 DataLoader 工厂.

把这份模板拷贝到目标项目即可使用。**禁止**：

- 自定义 collate 改变 batch 的键名 / 形状
- 在这里做 .to(device)（device 是训练脚本的责任）
- 在 worker_init_fn 里改全局状态（除了 numpy/torch seed）

batch 结构见 `references/output-spec.md` §2。
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, Dataset


def _default_num_workers() -> int:
    """Conservative default that behaves well on Windows and CI."""
    cpu = os.cpu_count() or 1
    return min(4, max(0, cpu // 2))


def _seed_worker(worker_id: int) -> None:
    """Make per-worker augmentations deterministic given the loader's seed."""
    base = torch.initial_seed() % 2**32
    import random
    random.seed(base + worker_id)
    try:
        import numpy as np
        np.random.seed(base + worker_id)
    except ImportError:
        pass


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    split: str = "train",
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    drop_last: bool | None = None,
    shuffle: bool | None = None,
    persistent_workers: bool | None = None,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader with split-aware defaults.

    Parameters
    ----------
    dataset : Dataset
        Should be an ``ImageClassificationDataset`` (or any dataset whose
        ``__getitem__`` already returns the dict described in
        ``references/output-spec.md``).
    split : {"train", "val", "test"}
        Drives the defaults for ``shuffle`` and ``drop_last``. Any explicit
        argument overrides the default.
    seed : int
        Seed for the per-worker RNG (so augmentations are reproducible).
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train/val/test, got {split!r}")

    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train"
    if num_workers is None:
        num_workers = _default_num_workers()
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
    )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Smoke test for make_dataloader().")
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    # Local import so this template runs standalone next to dataset.py
    from dataset import ImageClassificationDataset

    ds = ImageClassificationDataset(args.prepared_root, split=args.split)
    loader = make_dataloader(ds, batch_size=args.batch_size, split=args.split, num_workers=0)
    batch = next(iter(loader))
    print({k: (v.shape if isinstance(v, torch.Tensor) else (type(v).__name__, len(v))) for k, v in batch.items()})
    print(f"image dtype = {batch['image'].dtype}, label dtype = {batch['label'].dtype}")
