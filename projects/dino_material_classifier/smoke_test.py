"""Local smoke test that does not require DINOv3 weights.

It creates a tiny synthetic dataset in a temporary directory and trains
the small fallback backbone for one epoch. The goal is to validate the
data/model/checkpoint/inference plumbing before using real industrial
images and DINOv3 weights.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
from PIL import Image, ImageDraw

from .infer import build_model_from_checkpoint, predict_paths
from .train import parse_args, train_from_args
from .training import resolve_device


def _make_image(path: Path, color: tuple[int, int, int], texture: str) -> None:
    image = Image.new("RGB", (96, 96), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, 78, 78), fill=color)
    if texture == "polished":
        draw.line((25, 30, 72, 24), fill=(255, 255, 255), width=4)
    else:
        for x in range(22, 78, 8):
            draw.line((x, 20, x + 6, 78), fill=(80, 80, 80), width=1)
    image.save(path)


def run_smoke_test() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_dir = root / "images"
        image_dir.mkdir()
        rows: list[dict[str, object]] = []
        materials = {
            "aluminum": (170, 175, 180),
            "brass": (190, 145, 45),
        }
        processes = ["polished", "sandblasted"]
        split_cycle = ["train", "train", "val", "test"]
        index = 0
        for material, color in materials.items():
            for process in processes:
                for repeat in range(4):
                    image_path = image_dir / f"{material}_{process}_{repeat}.png"
                    _make_image(image_path, color, process)
                    rows.append(
                        {
                            "image_path": str(image_path),
                            "material": material,
                            "process": process,
                            "sample_id": image_path.stem,
                            "group_id": f"{material}_{process}_{repeat}",
                            "split": split_cycle[index % len(split_cycle)],
                            "bbox": "18,18,78,78",
                        }
                    )
                    index += 1

        manifest = root / "manifest.csv"
        output_dir = root / "run"
        pd.DataFrame(rows).to_csv(manifest, index=False)

        args = parse_args(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--backbone-kind",
                "tiny",
                "--epochs",
                "1",
                "--head-epochs",
                "1",
                "--batch-size",
                "4",
                "--image-size",
                "64",
            ]
        )
        train_from_args(args)

        checkpoint = output_dir / "best.pt"
        device = resolve_device("cpu")
        model, schema, config = build_model_from_checkpoint(checkpoint, device=device)
        predictions = predict_paths(
            model,
            schema,
            [Path(rows[0]["image_path"])],
            image_size=int(config.get("image_size", 64)),
            device=device,
        )
        assert predictions and predictions[0]["material_top1"] in schema.material_labels
        print("Smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()

