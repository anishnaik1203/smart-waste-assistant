"""
Dataset loader and preparation pipeline for TrashNet waste taxonomy.
Supports automatic train/val/test split generation, class imbalance weighting,
and on-demand starter dataset generation for immediate end-to-end training.
"""

import os
import shutil
import random
import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from PIL import Image, ImageDraw, ImageFilter
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import config

DATASET_DIR = config.DATASET_DIR


def get_dataset_splits(data_dir: Path = DATASET_DIR, batch_size: int = config.BATCH_SIZE):
    """
    Loads images from directory into Train, Validation, and Test tf.data.Dataset objects.
    Applies stratified batching, one-hot label encoding, and GPU/CPU prefetching.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print(f"Dataset directory {data_dir} is empty or missing. Creating starter dataset...")
        create_starter_dataset(num_samples_per_class=35)

    # 1. Load full training + validation split (85%)
    train_val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=config.IMAGE_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED,
        validation_split=config.VALIDATION_SPLIT + config.TEST_SPLIT,
        subset="training"
    )

    # 2. Load validation + test split (30% total)
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=config.IMAGE_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED,
        validation_split=config.VALIDATION_SPLIT + config.TEST_SPLIT,
        subset="validation"
    )

    # Split the validation + test set equally into val_ds (15%) and test_ds (15%)
    val_test_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_batches = max(1, val_test_batches // 2)

    val_ds = val_test_ds.take(val_batches)
    test_ds = val_test_ds.skip(val_batches)

    # Optimize pipeline with AUTOTUNE prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_val_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


def calculate_class_weights(data_dir: Path = DATASET_DIR) -> dict:
    """
    Computes balanced class weights to compensate for class imbalance
    (e.g., TrashNet has significantly fewer 'trash' images than 'paper').
    """
    data_dir = Path(data_dir)
    class_counts = {}
    for idx, class_name in enumerate(config.CLASSES):
        class_folder = data_dir / class_name
        if class_folder.exists():
            count = len(list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")))
            class_counts[idx] = max(1, count)
        else:
            class_counts[idx] = 1

    total_samples = sum(class_counts.values())
    n_classes = len(config.CLASSES)
    weights = {}
    for idx, count in class_counts.items():
        weights[idx] = float(total_samples / (n_classes * count))

    return weights


def create_starter_dataset(target_dir: Path = DATASET_DIR, num_samples_per_class: int = 35):
    """
    Generates a starter synthetic waste dataset with distinct textures, shapes, and color
    profiles for cardboard, glass, metal, paper, plastic, and organic waste.
    This enables immediate training and verification on any local machine.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.RANDOM_SEED)

    category_palettes = {
        "cardboard": [(180, 140, 100), (160, 120, 80), (139, 90, 43), (205, 170, 125)],
        "glass": [(140, 200, 190), (100, 180, 170), (40, 140, 130), (220, 245, 240)],
        "metal": [(180, 190, 200), (140, 150, 160), (210, 215, 220), (90, 100, 110)],
        "paper": [(245, 245, 245), (235, 235, 240), (210, 210, 215), (255, 255, 250)],
        "plastic": [(255, 220, 100), (255, 170, 50), (240, 100, 80), (100, 180, 255)],
        "trash": [(120, 100, 80), (90, 70, 50), (70, 120, 60), (150, 100, 60)]
    }

    print(f"Generating starter dataset ({num_samples_per_class} images per category) in {target_dir}...")

    for category in config.CLASSES:
        cat_dir = target_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        colors = category_palettes[category]

        for i in range(num_samples_per_class):
            img_path = cat_dir / f"{category}_{i+1:04d}.jpg"
            if img_path.exists():
                continue

            # Random background shade from palette
            bg_color = random.choice(colors)
            img = Image.new("RGB", (256, 256), color=bg_color)
            draw = ImageDraw.Draw(img)

            # Draw random geometric features representing waste objects
            num_shapes = random.randint(3, 8)
            for _ in range(num_shapes):
                obj_color = random.choice(colors)
                x1 = random.randint(20, 180)
                y1 = random.randint(20, 180)
                x2 = x1 + random.randint(40, 120)
                y2 = y1 + random.randint(40, 120)

                shape_type = random.choice(["rectangle", "ellipse", "polygon", "line"])
                if shape_type == "rectangle":
                    draw.rectangle([x1, y1, x2, y2], fill=obj_color, outline=(30, 30, 30))
                elif shape_type == "ellipse":
                    draw.ellipse([x1, y1, x2, y2], fill=obj_color, outline=(30, 30, 30))
                elif shape_type == "polygon":
                    points = [(x1, y1), (x2, y1 + 20), (x2 - 10, y2), (x1 - 10, y2 - 20)]
                    draw.polygon(points, fill=obj_color, outline=(30, 30, 30))
                else:
                    draw.line([(x1, y1), (x2, y2)], fill=obj_color, width=random.randint(2, 6))

            # Apply slight blur to simulate camera depth
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
            img.save(img_path, quality=90)

    print("Starter dataset generation complete.")


def get_dataset_statistics(data_dir: Path = DATASET_DIR) -> dict:
    """Returns image counts and percentage distribution per class."""
    data_dir = Path(data_dir)
    stats = {}
    total = 0
    for cls in config.CLASSES:
        folder = data_dir / cls
        count = len(list(folder.glob("*.jpg")) + list(folder.glob("*.png"))) if folder.exists() else 0
        stats[cls] = count
        total += count

    percentages = {
        cls: round((count / total) * 100, 1) if total > 0 else 0.0
        for cls, count in stats.items()
    }

    return {
        "total_images": total,
        "counts": stats,
        "percentages": percentages
    }
