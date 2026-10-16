"""
Automated unit and integration tests for the Smart Waste Classification pipeline.
Verifies model architecture, preprocessing, inference, and database persistence.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from model.cnn_model import build_transfer_learning_model, build_custom_cnn_model
from model.predict import preprocess_image, predict_waste
from utils.recycling_guide import get_guideline, calculate_eco_savings
from utils.db_history import init_db, add_scan, get_stats, get_all_scans, clear_history
from utils.sample_images import generate_sample_images


def test_preprocessing():
    print("Testing image preprocessing pipeline...")
    test_img = Image.new("RGB", (320, 240), color=(120, 80, 40))
    batch, preview = preprocess_image(test_img)
    assert batch.shape == (1, 224, 224, 3), f"Unexpected shape {batch.shape}"
    assert preview.size == (224, 224), f"Unexpected preview size {preview.size}"
    print("  Preprocessing test PASSED.")


def test_guidelines():
    print("Testing recycling guidelines lookup...")
    for cls in config.CLASSES:
        guide = get_guideline(cls)
        assert "bin_color" in guide, f"Missing bin_color for {cls}"
        assert "prep_steps" in guide, f"Missing prep_steps for {cls}"
        assert len(guide["prep_steps"]) > 0

    savings = calculate_eco_savings("plastic", 10)
    assert savings["co2_saved_kg"] > 0
    print("  Guidelines test PASSED.")


def test_database():
    print("Testing SQLite database persistence...")
    init_db()
    clear_history()
    scan_id = add_scan("test.jpg", "plastic", 0.95, "glass", 0.03, "metal", 0.02)
    assert scan_id > 0

    stats = get_stats()
    assert stats["total_scans"] == 1
    assert stats["most_common_class"] == "plastic"

    df = get_all_scans()
    assert len(df) == 1
    print("  Database test PASSED.")


def test_model_inference():
    print("Testing model build and inference...")
    sample_images = generate_sample_images()
    test_path = sample_images["plastic"]
    
    # Run prediction
    result = predict_waste(test_path)
    assert "predicted_class" in result
    assert result["predicted_class"] in config.CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_k"]) == 3
    print(f"  Inference test PASSED. Predicted: {result['predicted_class']} ({result['confidence_pct']}%)")


if __name__ == "__main__":
    print("Running Smart Waste Assistant Pipeline Verification Tests...\n")
    test_preprocessing()
    test_guidelines()
    test_database()
    test_model_inference()
    print("\nAll pipeline verification tests passed successfully!")
