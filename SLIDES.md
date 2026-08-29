# Slide Content — Crop Disease Detection with Concept-Based Transfer Learning

Copy each **Slide** block into one Google Slides page. This covers the full Individual Deep Dive assignment.

---

## Slide 1 — Title

**Crop Disease Detection Using Concept-Based Transfer Learning**

Individual Deep Dive | Dataset: [Crop Disease Detection (Kaggle)](https://www.kaggle.com/datasets/snikhilrao/crop-disease-detection-dataset)

Architecture: Concept Bottleneck Network (CBN) with ImageNet-pretrained backbone

Student Name | Date

---

## Slide 2 — Dataset Overview

The assigned dataset is a folder-structured image classification collection of crop leaf photographs hosted on Kaggle. Each image belongs to exactly one class folder named after a crop and its condition. The dataset covers four major crops—Corn (Maize), Potato, Rice, and Wheat—with both healthy and diseased leaves. There are 14 classes in total: Corn___Common_Rust, Corn___Gray_leaf_spot, Corn___healthy, Corn___Northern_Leaf_Blight, Potato___Early_blight, Potato___healthy, Potato___Late_blight, Rice___Brown_spot, Rice___healthy, Rice___Leaf_blast, Rice___Neck_blast, Wheat___Brown_Rust, Wheat___healthy, and Wheat___Yellow_Rust. There are no CSV columns; the “fields” are the folder names (class labels) and the raw RGB pixel values inside each JPEG/PNG file. This is a medium-scale computer vision problem: thousands of leaf images, high intra-class variation (lighting, angle, background), and visually similar symptoms across different diseases.

---

## Slide 3 — ANALOGY (Concept-Based Network)

A concept-based network is like a plant clinic where a junior doctor first checks simple, understandable symptoms before naming the full disease. Instead of jumping straight from a leaf photo to “Late Blight,” the model passes through an intermediate “concept layer”—for example: Are there orange rust patches? Brown necrotic spots? Yellow chlorosis? Gray moldy areas? The pretrained backbone (ResNet18, trained on ImageNet) acts like an experienced general physician who already knows how to look at textures, edges, and colors; we only retrain the last stages for agriculture. The concept bottleneck is the symptom checklist; the final classifier is the diagnosis. This makes the network more interpretable than a black-box end-to-end classifier.

---

## Slide 4 — DETECTIVE (Why This Architecture Fits)

**Why Concept-Based Transfer Learning fits this dataset:**

1. **Data type:** RGB leaf images — convolutional backbones excel here; transfer learning avoids training millions of weights from scratch on limited agricultural data.
2. **Data size:** Crop datasets are large enough to fine-tune but not large enough to train a deep network from random initialization reliably; pretrained weights provide strong low-level feature extractors (edges, textures, color blobs).
3. **Pattern reasoning:** Diseases share visual motifs (rust, spots, blight). A concept bottleneck explicitly models these shared patterns, helping distinguish confusing pairs like Potato Early vs Late Blight.

**Compared to Architecture 1 — Plain CNN from scratch:** A vanilla CNN must learn all visual features from zero. With ~14 classes and finite images, it overfits easily and converges slowly. No interpretability when the instructor asks “why did the model predict rust?”

**Compared to Architecture 2 — Standard Transfer Learning (no concepts):** ResNet/VGG fine-tuned end-to-end works well on accuracy but is a black box. For an agricultural deployment, farmers and agronomists need explanations (“model saw orange pustules → Rust”). Concept bottlenecks add that interpretable middle layer without sacrificing much accuracy.

**Compared to Architecture 3 — RNN/LSTM:** Recurrent networks expect sequential data (time series, text). Leaf images are spatial 2D tensors, not sequences. RNNs ignore local patch structure that CNNs capture, so they are a poor fit.

**Verdict:** Concept-based transfer learning is the best trade-off for this image dataset: strong accuracy via pretraining + human-readable reasoning via concepts.

---

## Slide 5 — DESIGN (Mini Pipeline)

```
Raw Kaggle download (class folders + JPG/PNG images)
        ↓
Preprocessing: resize 224×224, normalize (ImageNet mean/std),
               random flip/crop (train), stratified train/val split
        ↓
Variant filter (assignment rule):
  A = first 50% of images | B = second 50% | C = 2-class subset
        ↓
Backbone: ImageNet-pretrained ResNet18 (feature extractor)
        ↓
Concept Bottleneck: 6 sigmoid concepts
  (rust patches, brown spots, yellowing, gray mold, edge burn, healthy green)
        ↓
Classifier head: softmax over 14 disease classes
        ↓
Output: predicted class + concept activation scores (interpretability)
```

---

## Slide 6 — Model Architecture Diagram (describe verbally)

Input tensor shape: `[batch, 3, 224, 224]`. The ResNet18 backbone outputs a 512-dimensional embedding. Two fully connected layers map this to 6 concept scores (each between 0 and 1). A final linear layer maps concepts to 14 logits. Loss = CrossEntropy(class) + λ × BCE(concept targets). Concepts use weak supervision derived from agronomic knowledge of each disease class.

---

## Slide 7 — Variant Options (pick your assigned letter)

**Variant A:** Use the first half of all images (sorted by class and filename) for training and evaluation. Tests whether early-collected samples generalize.

**Variant B:** Use the second half of images. Tests temporal/subset robustness if image ordering correlates with collection batch.

**Variant C:** Head-to-head comparison of two classes only—recommended pair: `Potato___Early_blight` vs `Potato___Late_blight` because they are visually similar and clinically important to separate.

---

## Slide 8 — Expected Results & Metrics

Report accuracy, precision, recall, and F1 on the validation split. Published work on this dataset family reports ~95%+ accuracy with MobileNetV2; our concept bottleneck ResNet18 should be comparable. Additionally report concept activation maps: for Wheat___Brown_Rust, expect high `rust_orange_patches`; for Corn___healthy, expect high `healthy_green_tissue`. This validates that the bottleneck learns meaningful symptoms.

---

## Slide 9 — GitHub Repository & Reproducibility

Code repository: `crop-disease-concept-net`

Steps:
1. Download dataset from Kaggle into `data/raw/`
2. `pip install -r requirements.txt`
3. `python scripts/create_demo_data.py` (optional smoke test)
4. `python train.py --data-dir data/raw --variant A --epochs 10`
5. `python predict.py --image path/to/leaf.jpg`

---

## Slide 10 — Key Takeaways

Concept-based transfer learning combines the best of both worlds for crop disease detection: pretrained vision models handle limited agricultural data efficiently, while concept bottlenecks make predictions explainable to non-ML stakeholders. The Kaggle dataset’s folder-per-class structure maps cleanly to a standard PyTorch `ImageFolder`-style pipeline. Among syllabus alternatives, this architecture is superior to from-scratch CNNs (data hunger), RNNs (wrong modality), and plain fine-tuning (no interpretability).

---

## One-Page Write-Up (paste into assignment document)

**ANALOGY:** A concept-based network works like a crop expert who first notes visible symptoms—rust, spots, yellowing—before giving the final disease name. The pretrained CNN backbone is prior experience recognizing shapes and colors; the concept layer is the structured checklist; the classifier is the diagnosis.

**DETECTIVE:** This dataset contains thousands of RGB leaf images across 14 classes—ideal for CNNs, not for RNNs or tabular MLPs. Transfer learning is necessary because agricultural data is smaller than ImageNet; training a deep CNN from scratch would overfit. Compared to standard fine-tuning, concept bottlenecks add interpretability critical for farmers. Compared to a plain CNN, transfer learning converges faster and generalizes better on unseen leaf photos.

**DESIGN:** Raw class folders → resize/normalize/augment → ResNet18 (pretrained) → 6 concept neurons → 14-class softmax → output label + concept scores. Variant A/B split halves the sorted sample list; Variant C filters to two potato blight classes for head-to-head analysis.
