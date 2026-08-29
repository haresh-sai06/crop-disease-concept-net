# Crop Disease Concept Network

Transfer learning with a **Concept Bottleneck Network (CBN)** for the [Crop Disease Detection Dataset](https://www.kaggle.com/datasets/snikhilrao/crop-disease-detection-dataset).

## Architecture

```
Leaf Image → Pretrained ResNet18 → Concept Layer (6 symptoms) → 14-class Disease Head
```

**Concepts:** rust patches, brown spots, yellowing, gray mold, edge burn, healthy green tissue.

## Quick Start

```bash
pip install -r requirements.txt

# Optional: synthetic demo data (no Kaggle account needed)
python scripts/create_demo_data.py

# Train (replace data path after downloading from Kaggle)
python train.py --data-dir data/demo --epochs 5 --variant all

# Predict
python predict.py --image data/demo/Potato___Early_blight/Potato___Early_blight_000.jpg
```

### Kaggle Dataset

1. Download from [Kaggle](https://www.kaggle.com/datasets/snikhilrao/crop-disease-detection-dataset)
2. Extract so class folders sit under `data/raw/`:
   ```
   data/raw/Corn___Common_Rust/
   data/raw/Potato___Early_blight/
   ...
   ```
3. Train:
   ```bash
   python train.py --data-dir data/raw --variant A --epochs 10
   ```

### Assignment Variants

| Variant | Flag | Description |
|---------|------|-------------|
| A | `--variant A` | First 50% of images |
| B | `--variant B` | Second 50% of images |
| C | `--variant C --class-a Potato___Early_blight --class-b Potato___Late_blight` | Two-class head-to-head |

## Slides

Full paragraph-wise slide content for Google Slides is in **[SLIDES.md](SLIDES.md)**.

## Project Structure

```
crop-disease-concept-net/
├── src/
│   ├── model.py          # ConceptBottleneckNet
│   ├── dataset.py        # Data loading + variant splits
│   └── constants.py      # 14 classes, 6 concepts
├── train.py
├── predict.py
├── scripts/create_demo_data.py
├── SLIDES.md             # Copy-paste slide paragraphs
└── requirements.txt
```

## License

MIT — dataset subject to Kaggle / original author terms.
