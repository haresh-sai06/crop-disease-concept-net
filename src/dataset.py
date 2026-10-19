"""Dataset loader for folder-structured crop disease images."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import CLASS_NAMES, NUM_CONCEPTS
from .concepts import concepts_for_class


def default_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class CropDiseaseDataset(Dataset):
    """Loads images from class-named subfolders."""

    def __init__(
        self,
        root: str | Path,
        class_names: list[str] | None = None,
        transform=None,
        variant: str = "all",
        head_to_head: tuple[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.class_names = class_names or CLASS_NAMES
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.transform = transform
        self.samples: list[tuple[Path, int, torch.Tensor]] = []

        for class_name in self.class_names:
            folder = self.root / class_name
            if not folder.exists():
                # Try alternate naming (double underscores, spaces)
                alt = self.root / class_name.replace("___", "__")
                folder = alt if alt.exists() else folder

            if not folder.exists():
                continue

            concept_target = torch.tensor(
                concepts_for_class(class_name),
                dtype=torch.float32,
            )
            images = sorted(
                [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            )
            for img_path in images:
                self.samples.append((img_path, self.class_to_idx[class_name], concept_target))

        self._apply_variant(variant, head_to_head)

    def _apply_variant(
        self,
        variant: str,
        head_to_head: tuple[str, str] | None,
    ) -> None:
        if head_to_head:
            allowed = {self.class_to_idx[c] for c in head_to_head if c in self.class_to_idx}
            self.samples = [s for s in self.samples if s[1] in allowed]
            return

        if variant == "all" or not self.samples:
            return

        # Sort for reproducible first/second half split
        self.samples.sort(key=lambda s: (s[1], str(s[0])))
        midpoint = len(self.samples) // 2
        if variant.upper() == "A":
            self.samples = self.samples[:midpoint]
        elif variant.upper() == "B":
            self.samples = self.samples[midpoint:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, concept_target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, concept_target, str(path)

def discover_classes(data_dir: str | Path) -> list[str]:
    root = Path(data_dir)
    if not root.exists():
        return CLASS_NAMES
    folders = sorted([p.name for p in root.iterdir() if p.is_dir()])
    return folders or CLASS_NAMES
