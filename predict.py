"""Run inference on a single leaf image."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model import ConceptBottleneckNet


def load_model(checkpoint_path: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ConceptBottleneckNet(
        num_classes=len(ckpt["class_names"]),
        num_concepts=len(ckpt["concept_names"]),
        backbone=ckpt.get("backbone", "resnet18"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["class_names"], ckpt["concept_names"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, concept_names = load_model(args.checkpoint, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, concepts = model(image)
        probs = torch.softmax(logits, dim=1)[0]
        concept_scores = concepts[0]

    top_idx = probs.argmax().item()
    print(f"Predicted class: {class_names[top_idx]} ({probs[top_idx].item():.2%})")
    print("\nConcept activations:")
    for name, score in zip(concept_names, concept_scores.tolist()):
        print(f"  {name}: {score:.2f}")


if __name__ == "__main__":
    main()
