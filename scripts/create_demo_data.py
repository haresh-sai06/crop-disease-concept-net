"""Generate synthetic demo images to smoke-test the pipeline without Kaggle download."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.constants import CLASS_NAMES


def make_leaf_image(seed: int, diseased: bool) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (256, 256), (34, 120, 45))
    draw = ImageDraw.Draw(img)

    # leaf outline
    draw.ellipse([40, 30, 220, 230], fill=(55, 150, 60))

    if diseased:
        for _ in range(rng.integers(5, 15)):
            x, y = rng.integers(60, 200, size=2)
            r = rng.integers(4, 14)
            color = tuple(rng.integers(80, 220, size=3).tolist())
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/demo")
    parser.add_argument("--images-per-class", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.output_dir)
    for i, class_name in enumerate(CLASS_NAMES):
        folder = root / class_name
        folder.mkdir(parents=True, exist_ok=True)
        diseased = "healthy" not in class_name.lower()
        for j in range(args.images_per_class):
            img = make_leaf_image(seed=i * 1000 + j, diseased=diseased)
            img.save(folder / f"{class_name}_{j:03d}.jpg")

    print(f"Created demo dataset at {root} ({len(CLASS_NAMES)} classes)")


if __name__ == "__main__":
    main()
