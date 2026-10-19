"""Gradio demo for the Concept Bottleneck crop-disease classifier.

Deploy to Hugging Face Spaces by pointing the Space at this repo and setting
the SDK to Gradio. Place a checkpoint at ``outputs/best_cbn.pt`` (or set
``CBN_CHECKPOINT``).
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from src.model import ConceptBottleneckNet

CKPT = Path(os.getenv("CBN_CHECKPOINT", "outputs/subset_cpu/best_cbn.pt"))
DEVICE = torch.device("cpu")
_model = None
_class_names: list[str] = []
_concept_names: list[str] = []

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load():
    global _model, _class_names, _concept_names
    if _model is not None:
        return
    if not CKPT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CKPT}. Train first: "
            "python eval/train_compare.py --data-dir data/raw --out outputs/subset_cpu"
        )
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    _class_names = ckpt["class_names"]
    _concept_names = ckpt["concept_names"]
    _model = ConceptBottleneckNet(
        num_classes=len(_class_names),
        num_concepts=len(_concept_names),
        backbone=ckpt.get("backbone", "resnet18"),
        pretrained=False,
    ).to(DEVICE)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()


@torch.no_grad()
def predict(image: Image.Image):
    load()
    assert _model is not None
    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits, concepts = _model(x)
    probs = torch.softmax(logits, dim=1)[0]
    topk = torch.topk(probs, k=min(5, len(_class_names)))
    class_scores = { _class_names[i]: float(probs[i]) for i in topk.indices.tolist() }
    concept_scores = { n: float(v) for n, v in zip(_concept_names, concepts[0].tolist()) }
    return class_scores, concept_scores


def build() -> gr.Blocks:
    with gr.Blocks(title="Crop Disease Concept Network") as demo:
        gr.Markdown(
            "# Crop Disease Concept Network\n"
            "ResNet18 → 6 visual concepts → disease class. "
            "Upload a leaf photo to see predicted class **and** concept activations."
        )
        with gr.Row():
            inp = gr.Image(type="pil", label="Leaf image")
            with gr.Column():
                out_cls = gr.Label(label="Disease class", num_top_classes=5)
                out_con = gr.Label(label="Concept activations", num_top_classes=6)
        inp.change(predict, inputs=inp, outputs=[out_cls, out_con])
        btn = gr.Button("Predict")
        btn.click(predict, inputs=inp, outputs=[out_cls, out_con])
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
