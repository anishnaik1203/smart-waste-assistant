"""
Waste Classification Inference Engine.
Handles image preprocessing, model caching, and multi-class probability scoring.
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
import config
from utils.recycling_guide import get_guideline

# Global model cache for CLI / backend use
_CACHED_MODEL = None


def load_waste_model(model_path: Path = config.MODEL_PATH):
    """
    Loads and caches the trained Keras model.
    Falls back gracefully if the file is not yet trained.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    model_path = Path(model_path)
    if model_path.exists():
        try:
            _CACHED_MODEL = tf.keras.models.load_model(str(model_path))
            return _CACHED_MODEL
        except Exception as e:
            print(f"Warning: Failed to load saved model from {model_path}: {e}")

    # If saved model not found on disk, build the MobileNetV2 architecture with ImageNet weights
    from model.cnn_model import build_transfer_learning_model
    _CACHED_MODEL = build_transfer_learning_model()
    return _CACHED_MODEL


def preprocess_image(image_input) -> tuple[np.ndarray, Image.Image]:
    """
    Converts various image inputs (Path, PIL Image, or Bytes) to:
    1. Preprocessed numpy batch of shape (1, 224, 224, 3) ready for model input.
    2. Resized PIL Image in RGB format for UI preview.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        # File-like object (BytesIO or Streamlit UploadedFile)
        img = Image.open(image_input)

    # Ensure 3-channel RGB (handles RGBA, Greyscale, CMYK)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to model input target
    img_resized = img.resize(config.IMAGE_SIZE, Image.Resampling.BILINEAR)

    # Convert to numpy array of float32
    img_array = np.array(img_resized, dtype=np.float32)

    # Add batch dimension: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch, img_resized


def predict_waste(image_input, model=None, top_k: int = 3) -> dict:
    """
    Runs genuine CNN model prediction on the provided image input.
    
    Returns:
        dict: Detailed classification results including top-1 prediction,
              confidence percentage, top-k probabilities, and recycling guide.
    """
    if model is None:
        model = load_waste_model()

    # Preprocess image
    img_batch, img_preview = preprocess_image(image_input)

    # Execute genuine model inference
    raw_predictions = model.predict(img_batch, verbose=0)
    probabilities = raw_predictions[0]

    # Map probabilities to classes
    class_probs = {
        config.CLASSES[i]: float(probabilities[i])
        for i in range(len(config.CLASSES))
    }

    # Sort classes by descending probability
    sorted_classes = sorted(class_probs.items(), key=lambda item: item[1], reverse=True)

    top_class, top_conf = sorted_classes[0]
    top_k_list = []
    for cls_name, conf in sorted_classes[:top_k]:
        top_k_list.append({
            "class": cls_name,
            "display_name": config.CLASS_DISPLAY_NAMES.get(cls_name, cls_name.capitalize()),
            "icon": config.CLASS_ICONS.get(cls_name, "🗑️"),
            "confidence": conf,
            "confidence_pct": round(conf * 100, 2),
            "bin": config.BIN_COLORS.get(cls_name, {}).get("name", "Standard Bin"),
            "bin_color": config.BIN_COLORS.get(cls_name, {}).get("color", "#757575")
        })

    # Confidence tier flag
    if top_conf >= config.HIGH_CONFIDENCE_THRESHOLD:
        confidence_level = "High Confidence"
        confidence_badge_color = "#2E7D32" # Green
    elif top_conf >= config.MEDIUM_CONFIDENCE_THRESHOLD:
        confidence_level = "Moderate Confidence"
        confidence_badge_color = "#F57C00" # Orange
    else:
        confidence_level = "Low Confidence (Ambiguous Object)"
        confidence_badge_color = "#D32F2F" # Red

    # Fetch domain recycling knowledge
    guideline = get_guideline(top_class)

    return {
        "predicted_class": top_class,
        "display_name": config.CLASS_DISPLAY_NAMES.get(top_class, top_class.capitalize()),
        "icon": config.CLASS_ICONS.get(top_class, "🗑️"),
        "confidence": top_conf,
        "confidence_pct": round(top_conf * 100, 2),
        "confidence_level": confidence_level,
        "confidence_badge_color": confidence_badge_color,
        "top_k": top_k_list,
        "all_probabilities": class_probs,
        "guideline": guideline,
        "preview_image": img_preview
    }
