"""Heuristic concept targets for any leaf-disease class folder name.

Used when the dataset is PlantVillage-style (`Crop - Disease`) rather than the
14-class Kaggle naming in ``constants.CLASS_TO_CONCEPTS``.
"""
from __future__ import annotations

from .constants import CLASS_TO_CONCEPTS, CONCEPT_NAMES, NUM_CONCEPTS


def concepts_for_class(class_name: str) -> list[float]:
    if class_name in CLASS_TO_CONCEPTS:
        return list(CLASS_TO_CONCEPTS[class_name])
    low = class_name.lower().replace("_", " ").replace("-", " ")
    vec = [0.0] * NUM_CONCEPTS
    # indices: rust, brown_spots, yellowing, gray_mold, edge_burn, healthy
    if "healthy" in low:
        vec[5] = 1.0
        return vec
    if "rust" in low or "esca" in low:
        vec[0] = 1.0
    if any(k in low for k in ("blight", "spot", "scab", "scorch", "rot", "measles", "curl")):
        vec[1] = 1.0
    if any(k in low for k in ("yellow", "chlorosis", "curl")):
        vec[2] = 1.0
    if any(k in low for k in ("gray", "grey", "mold", "mildew", "powdery")):
        vec[3] = 1.0
    if any(k in low for k in ("blight", "burn", "scorch", "blast")):
        vec[4] = 1.0
    if sum(vec) == 0:
        vec[1] = 1.0  # unknown disease → brown_spots prior
    return vec


def concept_names() -> list[str]:
    return list(CONCEPT_NAMES)
