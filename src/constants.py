"""Dataset class labels for the Crop Disease Detection dataset."""

# Standard folder-based classes (14 classes across Corn, Potato, Rice, Wheat)
CLASS_NAMES = [
    "Corn___Common_Rust",
    "Corn___Gray_leaf_spot",
    "Corn___healthy",
    "Corn___Northern_Leaf_Blight",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Rice___Brown_spot",
    "Rice___healthy",
    "Rice___Leaf_blast",
    "Rice___Neck_blast",
    "Wheat___Brown_Rust",
    "Wheat___healthy",
    "Wheat___Yellow_Rust",
]

# Human-interpretable visual concepts the bottleneck layer learns
CONCEPT_NAMES = [
    "rust_orange_patches",      # fungal rust symptoms
    "brown_spots",              # necrotic / blight spots
    "yellowing_chlorosis",      # nutrient / stress discoloration
    "gray_moldy_patches",       # gray leaf spot / mildew look
    "leaf_edge_burn",           # blight along margins
    "healthy_green_tissue",     # intact chlorophyll regions
]

NUM_CLASSES = len(CLASS_NAMES)
NUM_CONCEPTS = len(CONCEPT_NAMES)

# Map each disease class to weak concept supervision (multi-hot targets)
CLASS_TO_CONCEPTS = {
    "Corn___Common_Rust":           [1, 0, 0, 0, 0, 0],
    "Corn___Gray_leaf_spot":        [0, 0, 0, 1, 0, 0],
    "Corn___healthy":               [0, 0, 0, 0, 0, 1],
    "Corn___Northern_Leaf_Blight":  [0, 1, 0, 0, 1, 0],
    "Potato___Early_blight":        [0, 1, 1, 0, 1, 0],
    "Potato___healthy":             [0, 0, 0, 0, 0, 1],
    "Potato___Late_blight":         [0, 1, 0, 1, 1, 0],
    "Rice___Brown_spot":            [0, 1, 0, 0, 0, 0],
    "Rice___healthy":               [0, 0, 0, 0, 0, 1],
    "Rice___Leaf_blast":            [0, 1, 0, 0, 1, 0],
    "Rice___Neck_blast":            [0, 1, 1, 0, 0, 0],
    "Wheat___Brown_Rust":           [1, 0, 0, 0, 0, 0],
    "Wheat___healthy":              [0, 0, 0, 0, 0, 1],
    "Wheat___Yellow_Rust":          [1, 0, 1, 0, 0, 0],
}
