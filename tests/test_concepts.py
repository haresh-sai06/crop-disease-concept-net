from src.concepts import concepts_for_class
from src.constants import CLASS_TO_CONCEPTS, NUM_CONCEPTS

def test_known_class_uses_table():
    assert concepts_for_class("Potato___Early_blight") == CLASS_TO_CONCEPTS["Potato___Early_blight"]

def test_plantvillage_healthy():
    v = concepts_for_class("Apple - Healthy")
    assert len(v) == NUM_CONCEPTS and v[5] == 1.0 and sum(v) == 1.0

def test_plantvillage_rust():
    v = concepts_for_class("Corn (Maize) - Common Rust")
    assert v[0] == 1.0

def test_plantvillage_blight():
    v = concepts_for_class("Potato - Late Blight")
    assert v[1] == 1.0 and v[4] == 1.0
