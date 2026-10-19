#!/usr/bin/env python3
"""Fetch / prepare leaf-disease images. Images are gitignored — never commit them.

Supported sources
-----------------
1. **PlantVillage already on disk** (this repo used to track ~42.7k images under
   ``Plant Village Dataset/``). Re-download from:
   https://www.kaggle.com/datasets/emmarex/plantdisease
   or any PlantVillage mirror, then::

       python scripts/download_data.py --from-plantvillage "Plant Village Dataset/Train" --out data/raw

2. **Kaggle crop-disease-detection-dataset** (14 classes matching ``src/constants.py``)::

       pip install kaggle
       # put ~/.kaggle/kaggle.json in place
       python scripts/download_data.py --kaggle snikhilrao/crop-disease-detection-dataset --out data/raw

3. **Synthetic smoke data** (no download)::

       python scripts/create_demo_data.py --output-dir data/demo
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copytree(src, dst)


def from_plantvillage(src: Path, out: Path, mode: str) -> None:
    if not src.is_dir():
        raise SystemExit(f"Not found: {src}")
    classes = sorted([p for p in src.iterdir() if p.is_dir()])
    if not classes:
        raise SystemExit(f"No class folders under {src}")
    out.mkdir(parents=True, exist_ok=True)
    for folder in classes:
        link_or_copy(folder, out / folder.name, mode)
    n_img = sum(1 for c in classes for p in c.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"Prepared {len(classes)} classes → {out} ({n_img} images, mode={mode})")


def from_kaggle(slug: str, out: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit("pip install kaggle  # and place ~/.kaggle/kaggle.json") from exc
    api = KaggleApi()
    api.authenticate()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {slug} …")
    api.dataset_download_files(slug, path=str(out), unzip=True)
    print(f"Downloaded to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-plantvillage", type=str, default=None,
                    help="Path to PlantVillage Train/ (class subfolders)")
    ap.add_argument("--kaggle", type=str, default=None, help="Kaggle dataset slug")
    ap.add_argument("--out", type=str, default="data/raw")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    args = ap.parse_args()
    out = Path(args.out)
    if args.from_plantvillage:
        from_plantvillage(Path(args.from_plantvillage), out, args.mode)
    elif args.kaggle:
        from_kaggle(args.kaggle, out)
    else:
        ap.print_help()
        print("\nTip: python scripts/create_demo_data.py --output-dir data/demo", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
