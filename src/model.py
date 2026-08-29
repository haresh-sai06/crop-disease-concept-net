"""Concept Bottleneck Network with transfer-learning backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .constants import NUM_CLASSES, NUM_CONCEPTS


class ConceptBottleneckNet(nn.Module):
    """
    Transfer-learning pipeline:
    ImageNet-pretrained backbone -> concept bottleneck -> disease classifier.

    Concepts act as an interpretable middle layer (e.g., rust patches, brown spots)
    before the final crop-disease label.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_concepts: int = NUM_CONCEPTS,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_concepts = num_concepts

        weights = "DEFAULT" if pretrained else None
        if backbone == "resnet18":
            base = models.resnet18(weights=weights)
            feature_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone == "mobilenet_v2":
            base = models.mobilenet_v2(weights=weights)
            feature_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.concept_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_concepts),
        )
        self.classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        concepts = torch.sigmoid(self.concept_layer(features))
        logits = self.classifier(concepts)
        return logits, concepts

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        logits, concepts = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return {
            "class_idx": probs.argmax(dim=1),
            "class_prob": probs.max(dim=1).values,
            "concepts": concepts,
        }
