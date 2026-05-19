"""Batch and single-image inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image

from .backbones import build_backbone
from .checkpoint import load_checkpoint
from .data import build_transforms
from .labels import LabelSchema
from .model import MaterialProcessClassifier
from .training import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run material/process inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", default=None, help="Single image path.")
    parser.add_argument("--manifest", default=None, help="CSV with image_path column for batch inference.")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output", default=None, help="Optional output CSV path for batch inference.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--backbone-kind", default=None, choices=[None, "dinov3", "tiny"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--repo-or-dir", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--source", default=None, choices=["local"])
    return parser.parse_args()


def _load_image_paths(image: str | None, manifest: str | None, image_root: str | None) -> list[Path]:
    paths: list[Path] = []
    if image:
        paths.append(Path(image))
    if manifest:
        df = pd.read_csv(manifest)
        if "image_path" not in df.columns:
            raise ValueError("Inference manifest must include an image_path column.")
        for item in df["image_path"].tolist():
            path = Path(str(item))
            if not path.is_absolute() and image_root is not None:
                path = Path(image_root) / path
            paths.append(path)
    if not paths:
        raise ValueError("Pass --image or --manifest.")
    return paths


def _topk(probabilities: torch.Tensor, labels: list[str], top_k: int) -> list[dict[str, object]]:
    k = min(top_k, probabilities.numel())
    values, indices = probabilities.topk(k)
    return [
        {"label": labels[int(index)], "score": float(value)}
        for value, index in zip(values.detach().cpu(), indices.detach().cpu(), strict=True)
    ]


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    override_backbone_kind: str | None = None,
    override_model_name: str | None = None,
    override_repo_or_dir: str | None = None,
    override_weights: str | None = None,
    override_source: str | None = None,
) -> tuple[MaterialProcessClassifier, LabelSchema, dict[str, object]]:
    payload = torch.load(Path(checkpoint_path), map_location=device)
    config = dict(payload.get("config", {}))
    label_schema = LabelSchema(
        material_labels=list(payload["label_schema"]["material_labels"]),
        process_labels=list(payload["label_schema"]["process_labels"]),
    )

    backbone = build_backbone(
        kind=override_backbone_kind or str(config.get("backbone_kind", "dinov3")),
        model_name=override_model_name or str(config.get("model_name", "dinov3_vitb16")),
        repo_or_dir=override_repo_or_dir or str(config.get("repo_or_dir", "facebookresearch/dinov3")),
        weights=override_weights if override_weights is not None else config.get("weights"),
        source=override_source if override_source is not None else config.get("source"),
    )
    model = MaterialProcessClassifier(
        backbone=backbone,
        num_materials=label_schema.num_materials,
        num_processes=label_schema.num_processes,
        hidden_dim=int(config.get("hidden_dim", 512)),
        dropout=float(config.get("dropout", 0.2)),
    )
    model.to(device)
    # Initialize LazyLinear with the stored classifier shapes before loading.
    state = payload["model_state"]
    lazy_weight = state.get("fusion.0.weight")
    if isinstance(lazy_weight, torch.Tensor):
        dummy_feature = torch.zeros(1, lazy_weight.shape[1], device=device)
        _ = model.fusion(dummy_feature)
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    model.eval()
    return model, label_schema, config


@torch.no_grad()
def predict_paths(
    model: MaterialProcessClassifier,
    label_schema: LabelSchema,
    image_paths: Iterable[Path],
    image_size: int = 224,
    normalization: str = "imagenet",
    top_k: int = 3,
    device: torch.device | None = None,
) -> list[dict[str, object]]:
    device = device or resolve_device()
    transform = build_transforms(image_size=image_size, train=False, normalization=normalization)
    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        output = model(tensor)
        material = _topk(output.material_probs[0], label_schema.material_labels, top_k)
        process = _topk(output.process_probs[0], label_schema.process_labels, top_k)
        rows.append(
            {
                "image_path": str(image_path),
                "material_top1": material[0]["label"],
                "material_conf": material[0]["score"],
                "process_top1": process[0]["label"],
                "process_conf": process[0]["score"],
                "material_topk": json.dumps(material, ensure_ascii=False),
                "process_topk": json.dumps(process, ensure_ascii=False),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, label_schema, config = build_model_from_checkpoint(
        args.checkpoint,
        device=device,
        override_backbone_kind=args.backbone_kind,
        override_model_name=args.model_name,
        override_repo_or_dir=args.repo_or_dir,
        override_weights=args.weights,
        override_source=args.source,
    )
    image_paths = _load_image_paths(args.image, args.manifest, args.image_root)
    rows = predict_paths(
        model,
        label_schema,
        image_paths,
        image_size=int(config.get("image_size", 224)),
        normalization=str(config.get("normalization", "imagenet")),
        top_k=args.top_k,
        device=device,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
    else:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
