"""Smoke test against synthetic demo data (no PlantVillage required)."""
from pathlib import Path
import subprocess
import sys

from src.dataset import CropDiseaseDataset, default_transforms, discover_classes
from src.model import ConceptBottleneckNet
import torch

def test_demo_data_roundtrip(tmp_path):
    root = Path(__file__).resolve().parents[1]
    demo = tmp_path / "demo"
    subprocess.check_call([sys.executable, str(root / "scripts/create_demo_data.py"),
                           "--output-dir", str(demo), "--images-per-class", "2"])
    names = discover_classes(demo)
    assert len(names) >= 10
    ds = CropDiseaseDataset(demo, class_names=names[:3], transform=default_transforms(train=False))
    assert len(ds) == 6
    img, label, concepts, path = ds[0]
    assert img.shape[0] == 3 and concepts.numel() == 6
    model = ConceptBottleneckNet(num_classes=3, pretrained=False)
    logits, cons = model(img.unsqueeze(0))
    assert logits.shape == (1, 3) and cons.shape == (1, 6)
