"""Command-line training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .backbones import build_backbone, freeze_backbone, unfreeze_last_trainable_block
from .checkpoint import load_checkpoint, save_checkpoint
from .data import create_dataloaders
from .model import MaterialProcessClassifier
from .training import evaluate, flatten_metrics, resolve_device, save_history, train_one_epoch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DINOv3 material/process classifier.")
    parser.add_argument("--manifest", required=True, help="CSV manifest path.")
    parser.add_argument("--image-root", default=None, help="Optional root for relative image paths.")
    parser.add_argument("--output-dir", default="projects/dino_material_classifier/runs/exp001")
    parser.add_argument("--backbone-kind", default="dinov3", choices=["dinov3", "tiny"])
    parser.add_argument("--model-name", default="dinov3_vitb16")
    parser.add_argument("--repo-or-dir", default="facebookresearch/dinov3")
    parser.add_argument("--weights", default=None, help="DINOv3 checkpoint path or URL.")
    parser.add_argument("--source", default=None, choices=["local"], help="Use 'local' when repo-or-dir points to a cloned DINOv3 repo.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--normalization", default="imagenet", choices=["imagenet", "dinov3_lvd", "dinov3_sat"])
    parser.add_argument("--roi-mode", default="bbox", choices=["full", "bbox", "mask"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--head-epochs", type=int, default=3, help="Epochs with frozen backbone.")
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=1)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--material-loss-weight", type=float, default=1.0)
    parser.add_argument("--process-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: MaterialProcessClassifier, lr_head: float, lr_backbone: float, weight_decay: float) -> torch.optim.Optimizer:
    head_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    parameter_groups: list[dict[str, Any]] = []
    if head_params:
        parameter_groups.append({"params": head_params, "lr": lr_head})
    if backbone_params:
        parameter_groups.append({"params": backbone_params, "lr": lr_backbone})
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


@torch.no_grad()
def initialize_lazy_layers(model: MaterialProcessClassifier, train_loader: torch.utils.data.DataLoader, device: torch.device) -> None:
    """Run one batch so LazyLinear can infer the DINO feature dimension."""

    try:
        batch = next(iter(train_loader))
    except StopIteration as exc:
        raise ValueError("Training split is empty; cannot initialize the classifier head.") from exc
    model.eval()
    _ = model(batch["image"].to(device))


def make_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items() if key != "resume"}


def summarize_eval(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "loss": float(metrics.get("loss", 0.0)),
        "material_acc": float(metrics["material"]["accuracy"]),
        "material_macro_f1": float(metrics["material"]["macro_f1"]),
        "process_acc": float(metrics["process"]["accuracy"]),
        "process_macro_f1": float(metrics["process"]["macro_f1"]),
    }


def train_from_args(args: argparse.Namespace) -> tuple[MaterialProcessClassifier, dict[str, Any]]:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(args)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    train_loader, val_loader, test_loader, label_schema, _records = create_dataloaders(
        manifest_path=args.manifest,
        image_root=args.image_root,
        image_size=args.image_size,
        roi_mode=args.roi_mode,
        normalization=args.normalization,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    label_schema.save(output_dir / "label_schema.json")

    backbone = build_backbone(
        kind=args.backbone_kind,
        model_name=args.model_name,
        repo_or_dir=args.repo_or_dir,
        weights=args.weights,
        source=args.source,
    )
    model = MaterialProcessClassifier(
        backbone=backbone,
        num_materials=label_schema.num_materials,
        num_processes=label_schema.num_processes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    device = resolve_device(args.device)
    model.to(device)
    initialize_lazy_layers(model, train_loader, device)

    material_criterion = nn.CrossEntropyLoss()
    process_criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_score = -1.0

    if args.resume:
        resume_epoch = int(torch.load(args.resume, map_location="cpu").get("epoch", 0))
        start_epoch = resume_epoch + 1
        if start_epoch > args.head_epochs:
            unfreeze_last_trainable_block(model.backbone, last_n_blocks=args.unfreeze_last_n_blocks)
        else:
            freeze_backbone(model.backbone)
        optimizer = build_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay)
        payload = load_checkpoint(args.resume, model=model, optimizer=None, map_location=device)
        if payload.get("optimizer_state") is not None:
            try:
                optimizer.load_state_dict(payload["optimizer_state"])
            except ValueError:
                print("Skipping optimizer state because the trainable parameter groups changed after resume.")
        best_metrics = payload.get("metrics", {})
        best_score = float(best_metrics.get("selection_score", -1.0))
    else:
        freeze_backbone(model.backbone)
        optimizer = build_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay)

    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.head_epochs + 1:
            unfreeze_last_trainable_block(model.backbone, last_n_blocks=args.unfreeze_last_n_blocks)
            optimizer = build_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            material_criterion,
            process_criterion,
            device=device,
            material_loss_weight=args.material_loss_weight,
            process_loss_weight=args.process_loss_weight,
            max_grad_norm=args.max_grad_norm,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            material_criterion,
            process_criterion,
            device=device,
            label_schema=label_schema,
            output_dir=output_dir / "eval",
            split_name="val",
            material_loss_weight=args.material_loss_weight,
            process_loss_weight=args.process_loss_weight,
        )

        val_summary = summarize_eval(val_metrics)
        selection_score = 0.5 * (val_summary["material_macro_f1"] + val_summary["process_macro_f1"])
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_summary.items()})
        row["selection_score"] = selection_score
        history.append(row)
        save_history(history, output_dir / "history.csv")

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_material_f1={val_summary['material_macro_f1']:.4f} | "
            f"val_process_f1={val_summary['process_macro_f1']:.4f}"
        )

        epoch_metrics = {"selection_score": selection_score, **row}
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, label_schema, config, epoch_metrics)
        if selection_score >= best_score:
            best_score = selection_score
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, label_schema, config, epoch_metrics)

    test_metrics = evaluate(
        model,
        test_loader,
        material_criterion,
        process_criterion,
        device=device,
        label_schema=label_schema,
        output_dir=output_dir / "eval",
        split_name="test",
        material_loss_weight=args.material_loss_weight,
        process_loss_weight=args.process_loss_weight,
    )
    test_summary = summarize_eval(test_metrics)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test_summary, handle, ensure_ascii=False, indent=2)
    print(f"Test metrics: {test_summary}")
    return model, {"test": test_summary, "history": history}


def main() -> None:
    args = parse_args()
    train_from_args(args)


if __name__ == "__main__":
    main()
