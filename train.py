"""Train the concept-bottleneck transfer learning model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.constants import CLASS_NAMES, CONCEPT_NAMES
from src.dataset import CropDiseaseDataset, default_transforms, discover_classes
from src.model import ConceptBottleneckNet


def train_one_epoch(model, loader, optimizer, device, concept_weight: float):
    model.train()
    cls_loss_fn = nn.CrossEntropyLoss()
    concept_loss_fn = nn.BCELoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, concept_targets, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        concept_targets = concept_targets.to(device)

        optimizer.zero_grad()
        logits, concepts = model(images)
        loss = cls_loss_fn(logits, labels) + concept_weight * concept_loss_fn(concepts, concept_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, concept_weight: float):
    model.eval()
    cls_loss_fn = nn.CrossEntropyLoss()
    concept_loss_fn = nn.BCELoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, concept_targets, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        concept_targets = concept_targets.to(device)
        logits, concepts = model(images)
        loss = cls_loss_fn(logits, labels) + concept_weight * concept_loss_fn(concepts, concept_targets)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train concept bottleneck crop disease model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root with class folders")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--concept-weight", type=float, default=0.5)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "mobilenet_v2"])
    parser.add_argument("--variant", type=str, default="all", choices=["all", "A", "B", "C"])
    parser.add_argument("--class-a", type=str, default="Potato___Early_blight")
    parser.add_argument("--class-b", type=str, default="Potato___Late_blight")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = discover_classes(args.data_dir)
    head_to_head = (args.class_a, args.class_b) if args.variant.upper() == "C" else None

    full_dataset = CropDiseaseDataset(
        args.data_dir,
        class_names=class_names,
        transform=default_transforms(train=True),
        variant=args.variant if args.variant.upper() in {"A", "B"} else "all",
        head_to_head=head_to_head,
    )

    if len(full_dataset) == 0:
        raise SystemExit(
            f"No images found under {args.data_dir}. "
            "Download the Kaggle dataset and point --data-dir to the folder containing class subfolders."
        )

    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=args.val_split, random_state=42, stratify=[s[1] for s in full_dataset.samples])

    train_ds = Subset(
        CropDiseaseDataset(
            args.data_dir,
            class_names=class_names,
            transform=default_transforms(train=True),
            variant=args.variant if args.variant.upper() in {"A", "B"} else "all",
            head_to_head=head_to_head,
        ),
        train_idx,
    )
    val_ds = Subset(
        CropDiseaseDataset(
            args.data_dir,
            class_names=class_names,
            transform=default_transforms(train=False),
            variant=args.variant if args.variant.upper() in {"A", "B"} else "all",
            head_to_head=head_to_head,
        ),
        val_idx,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = len(class_names)
    model = ConceptBottleneckNet(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, args.concept_weight)
        val_loss, val_acc = evaluate(model, val_loader, device, args.concept_weight)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(f"Epoch {epoch:02d} | train acc={train_acc:.3f} | val acc={val_acc:.3f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "concept_names": CONCEPT_NAMES,
                "backbone": args.backbone,
                "variant": args.variant,
            }, output_dir / "best_model.pt")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best validation accuracy: {best_acc:.3f}")
    print(f"Saved model to {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
