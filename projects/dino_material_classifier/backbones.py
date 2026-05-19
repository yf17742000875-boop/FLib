"""Backbone wrappers.

The classifier needs a stable feature interface, but the upstream DINOv3
hub models can expose slightly different output structures. This module
normalizes them into a single dataclass with a global embedding and an
optional patch-token tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class BackboneOutput:
    embedding: torch.Tensor
    patch_tokens: torch.Tensor | None = None
    feature_map: torch.Tensor | None = None
    raw: Any | None = None


class TinyBackbone(nn.Module):
    """Small local backbone for smoke tests and fast iteration."""

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward_features(self, images: torch.Tensor) -> BackboneOutput:
        feature_map = self.stem(images)
        embedding = self.pool(feature_map).flatten(1)
        patch_tokens = feature_map.flatten(2).transpose(1, 2)
        return BackboneOutput(
            embedding=embedding,
            patch_tokens=patch_tokens,
            feature_map=feature_map,
            raw=feature_map,
        )

    def forward(self, images: torch.Tensor) -> BackboneOutput:
        return self.forward_features(images)


class DINOv3Backbone(nn.Module):
    """Thin adapter around the official DINOv3 hub models."""

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        repo_or_dir: str = "facebookresearch/dinov3",
        weights: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.repo_or_dir = repo_or_dir
        self.weights = weights
        self.source = source
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        load_kwargs: dict[str, Any] = {"weights": self.weights}
        if self.source == "local" or Path(self.repo_or_dir).exists():
            load_kwargs["source"] = "local"
            return torch.hub.load(self.repo_or_dir, self.model_name, **load_kwargs)

        try:
            return torch.hub.load(self.repo_or_dir, self.model_name, trust_repo=True, **load_kwargs)
        except TypeError:
            return torch.hub.load(self.repo_or_dir, self.model_name, **load_kwargs)

    def forward_features(self, images: torch.Tensor) -> BackboneOutput:
        if hasattr(self.model, "forward_features"):
            raw = self.model.forward_features(images)
        else:
            raw = self.model(images)
        return self._normalize_features(raw)

    def _normalize_features(self, raw: Any) -> BackboneOutput:
        if isinstance(raw, dict):
            embedding = raw.get("x_norm_clstoken")
            if embedding is None:
                embedding = raw.get("pooler_output")
            if embedding is None:
                tensor_like = raw.get("last_hidden_state")
                if isinstance(tensor_like, torch.Tensor) and tensor_like.ndim == 3:
                    embedding = tensor_like[:, 0, :]
            patch_tokens = raw.get("x_norm_patchtokens")
            if patch_tokens is None:
                tensor_like = raw.get("last_hidden_state")
                if isinstance(tensor_like, torch.Tensor) and tensor_like.ndim == 3:
                    patch_tokens = tensor_like[:, 1:, :]
            feature_map = raw.get("x_prenorm")
            if embedding is None:
                raise RuntimeError(f"Unsupported DINOv3 output keys: {list(raw.keys())}")
            return BackboneOutput(
                embedding=embedding,
                patch_tokens=patch_tokens,
                feature_map=feature_map,
                raw=raw,
            )

        if hasattr(raw, "pooler_output"):
            embedding = raw.pooler_output
            patch_tokens = getattr(raw, "last_hidden_state", None)
            if isinstance(patch_tokens, torch.Tensor) and patch_tokens.ndim == 3:
                patch_tokens = patch_tokens[:, 1:, :]
            return BackboneOutput(embedding=embedding, patch_tokens=patch_tokens, raw=raw)

        if isinstance(raw, torch.Tensor):
            if raw.ndim == 4:
                feature_map = raw
                embedding = torch.nn.functional.adaptive_avg_pool2d(raw, 1).flatten(1)
                patch_tokens = raw.flatten(2).transpose(1, 2)
                return BackboneOutput(embedding=embedding, patch_tokens=patch_tokens, feature_map=feature_map, raw=raw)
            if raw.ndim == 3:
                return BackboneOutput(embedding=raw[:, 0, :], patch_tokens=raw[:, 1:, :], raw=raw)
            if raw.ndim == 2:
                return BackboneOutput(embedding=raw, raw=raw)

        raise TypeError(f"Unsupported DINOv3 output type: {type(raw)!r}")

    def forward(self, images: torch.Tensor) -> BackboneOutput:
        return self.forward_features(images)


def build_backbone(
    kind: str = "dinov3",
    model_name: str = "dinov3_vitb16",
    repo_or_dir: str = "facebookresearch/dinov3",
    weights: str | None = None,
    source: str | None = None,
) -> nn.Module:
    """Factory for the classifier backbone."""

    kind = kind.lower().strip()
    if kind == "dinov3":
        return DINOv3Backbone(model_name=model_name, repo_or_dir=repo_or_dir, weights=weights, source=source)
    if kind == "tiny":
        return TinyBackbone()
    raise ValueError(f"Unknown backbone kind: {kind!r}")


def freeze_backbone(backbone: nn.Module) -> None:
    for parameter in backbone.parameters():
        parameter.requires_grad = False


def _trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters()]


def unfreeze_last_trainable_block(backbone: nn.Module, last_n_blocks: int = 1) -> None:
    """Unfreeze the last transformer block or stage when available.

    If the backbone exposes an unfamiliar structure, we fall back to
    unfreezing the whole module. That keeps the training loop simple and
    makes the code work with DINOv3 and the local tiny backbone.
    """

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    if hasattr(backbone, "model"):
        model = backbone.model
    else:
        model = backbone

    if hasattr(model, "blocks"):
        blocks = list(model.blocks)
        for block in blocks[-last_n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        if hasattr(model, "norm"):
            for parameter in model.norm.parameters():
                parameter.requires_grad = True
        return

    if hasattr(model, "stages"):
        stages = list(model.stages)
        for stage in stages[-1:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True
        return

    for parameter in backbone.parameters():
        parameter.requires_grad = True

