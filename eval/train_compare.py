"""Train Concept-Bottleneck vs plain ResNet18 baseline and write metrics.

Designed to run on CPU with a **subset** of the data. Label results clearly.

Example (subset, ~few minutes on CPU)::

    python eval/train_compare.py \
      --data-dir data/raw \
      --max-per-class 40 \
      --epochs 3 \
      --batch-size 8 \
      --out outputs/subset_cpu

Outputs
-------
``metrics.json`` with accuracy, macro-F1, per-class F1, confusion matrix paths,
plus ``confusion_cbn.png`` / ``confusion_baseline.png``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import CropDiseaseDataset, default_transforms, discover_classes  # noqa: E402
from src.model import ConceptBottleneckNet  # noqa: E402
from src.constants import CONCEPT_NAMES  # noqa: E402


class PlainResNet18(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.net = models.resnet18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def limit_per_class(dataset: CropDiseaseDataset, max_per_class: int, seed: int = 42) -> list[int]:
    rng = random.Random(seed)
    by_cls: dict[int, list[int]] = {}
    for i, (_, label, _) in enumerate(dataset.samples):
        by_cls.setdefault(label, []).append(i)
    chosen: list[int] = []
    for label, idxs in sorted(by_cls.items()):
        rng.shuffle(idxs)
        chosen.extend(idxs[:max_per_class])
    rng.shuffle(chosen)
    return chosen


def run_epoch(model, loader, optimizer, device, kind: str, concept_weight: float = 0.5):
    train = optimizer is not None
    model.train(train)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCELoss()
    total_loss = correct = total = 0
    preds, labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=kind, leave=False):
            images, y, concepts, _ = batch
            images, y, concepts = images.to(device), y.to(device), concepts.to(device)
            if train:
                optimizer.zero_grad()
            if isinstance(model, ConceptBottleneckNet):
                logits, concept_pred = model(images)
                loss = ce(logits, y) + concept_weight * bce(concept_pred, concepts)
            else:
                logits = model(images)
                loss = ce(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += images.size(0)
            preds.extend(pred.cpu().tolist())
            labels.extend(y.cpu().tolist())
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "preds": preds,
        "labels": labels,
    }


def fit(model, train_loader, val_loader, device, epochs: int, lr: float, kind: str):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    history = []
    best_state, best_acc = None, -1.0
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, opt, device, f"{kind}-train")
        va = run_epoch(model, val_loader, None, device, f"{kind}-val")
        history.append({"epoch": epoch, "train_acc": tr["acc"], "val_acc": va["acc"],
                        "train_loss": tr["loss"], "val_loss": va["loss"]})
        print(f"[{kind}] epoch {epoch}: train_acc={tr['acc']:.3f} val_acc={va['acc']:.3f}")
        if va["acc"] >= best_acc:
            best_acc = va["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    final = run_epoch(model, val_loader, None, device, f"{kind}-final")
    return history, final


def score(y_true, y_pred, class_names: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {
            class_names[i]: float(v)
            for i, v in enumerate(f1_score(y_true, y_pred, average=None, labels=list(range(len(class_names))), zero_division=0))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
        ),
    }


def plot_cm(cm, class_names, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.45), max(5, len(class_names) * 0.4)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="outputs/subset_cpu")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-per-class", type=int, default=40, help="Cap images/class (subset run)")
    ap.add_argument("--val-split", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-classes", type=int, default=0, help="0 = all discovered classes")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    class_names = discover_classes(args.data_dir)
    if args.max_classes > 0:
        class_names = class_names[: args.max_classes]
    print(f"Classes ({len(class_names)}): {class_names}")

    base_ds = CropDiseaseDataset(args.data_dir, class_names=class_names, transform=default_transforms(train=True))
    if len(base_ds) == 0:
        raise SystemExit(f"No images under {args.data_dir}")
    subset_idx = limit_per_class(base_ds, args.max_per_class, seed=args.seed)
    labels_for_strat = [base_ds.samples[i][1] for i in subset_idx]
    train_idx, val_idx = train_test_split(
        subset_idx, test_size=args.val_split, random_state=args.seed, stratify=labels_for_strat
    )

    train_full = CropDiseaseDataset(args.data_dir, class_names=class_names, transform=default_transforms(train=True))
    val_full = CropDiseaseDataset(args.data_dir, class_names=class_names, transform=default_transforms(train=False))
    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

    meta = {
        "device": str(device),
        "data_dir": str(args.data_dir),
        "n_classes": len(class_names),
        "class_names": class_names,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "max_per_class": args.max_per_class,
        "epochs": args.epochs,
        "note": "SUBSET CPU run — not a full-dataset result. Re-run without --max-per-class cap on a GPU for production numbers.",
    }
    print(json.dumps({k: meta[k] for k in ("device", "n_classes", "n_train", "n_val", "max_per_class", "epochs")}, indent=2))

    cbn = ConceptBottleneckNet(num_classes=len(class_names), backbone="resnet18", pretrained=True).to(device)
    hist_cbn, final_cbn = fit(cbn, train_loader, val_loader, device, args.epochs, args.lr, "CBN")
    metrics_cbn = score(final_cbn["labels"], final_cbn["preds"], class_names)
    plot_cm(np.array(metrics_cbn["confusion_matrix"]), class_names, out / "confusion_cbn.png",
            f"CBN ResNet18 (subset) acc={metrics_cbn['accuracy']:.3f}")

    baseline = PlainResNet18(num_classes=len(class_names), pretrained=True).to(device)
    hist_base, final_base = fit(baseline, train_loader, val_loader, device, args.epochs, args.lr, "baseline")
    metrics_base = score(final_base["labels"], final_base["preds"], class_names)
    plot_cm(np.array(metrics_base["confusion_matrix"]), class_names, out / "confusion_baseline.png",
            f"Plain ResNet18 (subset) acc={metrics_base['accuracy']:.3f}")

    torch.save({"model_state": cbn.state_dict(), "class_names": class_names,
                "concept_names": CONCEPT_NAMES, "backbone": "resnet18"}, out / "best_cbn.pt")

    result = {
        "meta": meta,
        "concept_bottleneck": {"history": hist_cbn, **{k: v for k, v in metrics_cbn.items() if k != "classification_report"},
                               "classification_report": metrics_cbn["classification_report"]},
        "resnet18_baseline": {"history": hist_base, **{k: v for k, v in metrics_base.items() if k != "classification_report"},
                              "classification_report": metrics_base["classification_report"]},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    summary = {
        "meta": meta,
        "concept_bottleneck": {"accuracy": metrics_cbn["accuracy"], "macro_f1": metrics_cbn["macro_f1"]},
        "resnet18_baseline": {"accuracy": metrics_base["accuracy"], "macro_f1": metrics_base["macro_f1"]},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
