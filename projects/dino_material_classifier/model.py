"""Classifier head for material and process prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .backbones import BackboneOutput


@dataclass
class ClassificationOutput:
    material_logits: torch.Tensor
    process_logits: torch.Tensor
    material_probs: torch.Tensor
    process_probs: torch.Tensor
    embedding: torch.Tensor
    patch_tokens: torch.Tensor | None
    fused_features: torch.Tensor


class MaterialProcessClassifier(nn.Module):
    """Two-head classifier on top of a vision backbone.

    The backbone provides a global image embedding and optional patch
    tokens. We fuse them with a small lazy projection head so the model
    can work with DINOv3 or with a local fallback backbone without any
    hard-coded feature dimension.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_materials: int,
        num_processes: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        use_patch_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_patch_tokens = use_patch_tokens
        self.fusion = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.material_head = nn.Linear(hidden_dim, num_materials)
        self.process_head = nn.Linear(hidden_dim, num_processes)

    @staticmethod
    def _summarize_patch_tokens(patch_tokens: torch.Tensor | None) -> torch.Tensor | None:
        if patch_tokens is None:
            return None
        if patch_tokens.ndim == 4:
            patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        if patch_tokens.ndim != 3 or patch_tokens.size(1) == 0:
            return None
        mean_pool = patch_tokens.mean(dim=1)
        max_pool = patch_tokens.amax(dim=1)
        return 0.5 * (mean_pool + max_pool)

    def forward(self, images: torch.Tensor) -> ClassificationOutput:
        backbone_output = self.backbone(images)
        if not isinstance(backbone_output, BackboneOutput):
            raise TypeError(
                "The backbone must return a BackboneOutput instance. "
                f"Got {type(backbone_output)!r} instead."
            )

        global_embedding = backbone_output.embedding
        patch_summary = self._summarize_patch_tokens(backbone_output.patch_tokens) if self.use_patch_tokens else None
        fused_input = global_embedding if patch_summary is None else torch.cat([global_embedding, patch_summary], dim=-1)
        fused_features = self.fusion(fused_input)

        material_logits = self.material_head(fused_features)
        process_logits = self.process_head(fused_features)

        return ClassificationOutput(
            material_logits=material_logits,
            process_logits=process_logits,
            material_probs=F.softmax(material_logits, dim=-1),
            process_probs=F.softmax(process_logits, dim=-1),
            embedding=global_embedding,
            patch_tokens=backbone_output.patch_tokens,
            fused_features=fused_features,
        )

