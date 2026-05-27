"""dataset_template.py — 标准 PyTorch Dataset，只读 prepared/.

把这份模板拷贝到目标项目即可直接使用。**禁止**在本文件里：

- 读 task_root / 原始 inputs（那是 prepare 的工作）
- 做任何"动态划分"——splits.json 一切以 prepare 阶段为准
- 把字段从 dict 改成 tuple——会让下游 train 代码隐式断

返回结构见 `references/output-spec.md` §1。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int, split: str) -> T.Compose:
    """Order is fixed (see output-spec.md §1.2). Train adds light augment."""
    ops: list = [T.Resize((image_size, image_size))]
    if split == "train":
        ops.append(T.RandomHorizontalFlip(p=0.5))
        ops.append(T.ColorJitter(brightness=0.1, contrast=0.1))
    ops.append(T.ToTensor())
    ops.append(T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD))
    return T.Compose(ops)


class ImageClassificationDataset(Dataset):
    """Single-label RGB image classification dataset backed by a prepared/ dir.

    Parameters
    ----------
    prepared_root : str | Path
        Path produced by ``prepare.py``. Must contain
        ``classes.json`` / ``splits.json`` / ``manifest.json`` / ``images/``.
    split : {"train", "val", "test"}
        Selects which slice of ``splits.json`` to expose.
    image_size : int, optional
        Sanity-checked against ``manifest.image.target_size``. Default 224.
    transform : callable, optional
        Custom transform. If provided, replaces the entire default chain;
        the framework no longer guarantees ``image_size`` consistency.
    """

    def __init__(
        self,
        prepared_root: str | Path,
        split: str = "train",
        image_size: int = 224,
        transform=None,
    ) -> None:
        self.prepared_root = Path(prepared_root).resolve()
        self.split = split

        manifest = _read_json(self.prepared_root / "manifest.json")
        classes_payload = _read_json(self.prepared_root / "classes.json")
        splits_payload = _read_json(self.prepared_root / "splits.json")

        target = tuple(manifest["image"]["target_size"])
        if transform is None and (image_size, image_size) != target:
            raise ValueError(
                f"image_size={image_size} disagrees with manifest target_size={target}. "
                f"Re-run prepare.py with --image-size {image_size} or pass a custom transform."
            )

        if split not in splits_payload:
            raise KeyError(f"split={split!r} not found in splits.json (have {list(splits_payload)})")
        rel_paths: list[str] = splits_payload[split]
        if not isinstance(rel_paths, list):
            raise TypeError(f"splits.json[{split!r}] must be a list, got {type(rel_paths)}")

        self.classes: list[str] = classes_payload["classes"]
        self.class_to_idx: dict[str, int] = classes_payload["class_to_idx"]
        self.image_size = image_size
        self.transform = transform if transform is not None else build_transform(image_size, split)

        # Resolve and validate samples up-front (fail loud, not at __getitem__ time)
        images_root = self.prepared_root / "images"
        self.samples: list[tuple[str, int, str]] = []  # (rel_path, label, class_name)
        for rel in rel_paths:
            cls = rel.split("/", 1)[0]
            if cls not in self.class_to_idx:
                raise KeyError(f"class {cls!r} from splits.json not in classes.json")
            abs_path = images_root / rel
            if not abs_path.is_file():
                raise FileNotFoundError(f"missing image referenced by splits.json: {abs_path}")
            self.samples.append((rel, self.class_to_idx[cls], cls))

        if not self.samples:
            raise ValueError(f"split={split!r} is empty under {self.prepared_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        rel, label, cls = self.samples[index]
        abs_path = self.prepared_root / "images" / rel
        with Image.open(abs_path) as im:
            im = im.convert("RGB")
            tensor = self.transform(im)
        return {
            "image": tensor,
            "label": int(label),         # plain int -> collates to LongTensor
            "class_name": cls,
            "path": rel,                 # POSIX-style relative path
            "index": int(index),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(prepared_root={self.prepared_root}, "
            f"split={self.split!r}, n={len(self)}, classes={self.classes})"
        )


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing required prepared/ file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test for ImageClassificationDataset.")
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    ds = ImageClassificationDataset(args.prepared_root, split=args.split)
    print(ds)
    sample = ds[0]
    print({k: (v.shape if isinstance(v, torch.Tensor) else v) for k, v in sample.items()})
