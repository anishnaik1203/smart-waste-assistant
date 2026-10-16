"""
Configuration file for Smart Waste Classification and Recycling Assistant.
Contains hyperparameters, file paths, class mappings, and UI theme constants.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
DATASET_DIR = DATA_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "waste_classifier.keras"
METRICS_PATH = DATA_DIR / "metrics.json"
HISTORY_LOG_PATH = DATA_DIR / "training_history.json"
DB_PATH = BASE_DIR / "waste_scans.db"

# Create directories if they do not exist
for path in [DATA_DIR, SAMPLES_DIR, DATASET_DIR, MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Image & Preprocessing Settings
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
CHANNELS = 3

# Dataset Classes (Standard TrashNet 6-Class Taxonomy)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)

CLASS_DISPLAY_NAMES = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "trash": "Organic / General Waste"
}

CLASS_ICONS = {
    "cardboard": "📦",
    "glass": "🍾",
    "metal": "🥫",
    "paper": "📄",
    "plastic": "🧴",
    "trash": "🍎"
}

# Standard Municipal Bin Color Codes
BIN_COLORS = {
    "cardboard": {"color": "#1E88E5", "name": "Blue Bin", "desc": "Clean Paper & Cardboard"},
    "paper": {"color": "#1565C0", "name": "Blue Bin", "desc": "Paper & Document Recycling"},
    "glass": {"color": "#2E7D32", "name": "Green Bin", "desc": "Glass Bottles & Jars"},
    "plastic": {"color": "#FBC02D", "name": "Yellow Bin", "desc": "Clean Plastic Bottles & Containers"},
    "metal": {"color": "#78909C", "name": "Silver / Gray Bin", "desc": "Aluminum & Steel Cans"},
    "trash": {"color": "#424242", "name": "Black / Brown Bin", "desc": "Organic Compost & Non-Recyclable Residuals"}
}

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0005
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Confidence Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.70
MEDIUM_CONFIDENCE_THRESHOLD = 0.45
