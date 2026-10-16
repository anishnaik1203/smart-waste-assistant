"""
Sample image utility module.
Provides generated representative images for each waste category to allow
instant 1-click testing and demonstration of the machine learning classifier.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import config

SAMPLES_DIR = config.SAMPLES_DIR


def generate_sample_images():
    """
    Generates representative sample images for each waste category
    if they do not already exist on disk.
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    sample_specs = {
        "cardboard": {
            "bg": "#C19A6B",
            "accent": "#8B5A2B",
            "text": "Cardboard Box\nCorrugated Fiber",
            "shapes": "box"
        },
        "glass": {
            "bg": "#80CBC4",
            "accent": "#004D40",
            "text": "Glass Bottle\nClean Transparent Jar",
            "shapes": "bottle"
        },
        "metal": {
            "bg": "#B0BEC5",
            "accent": "#37474F",
            "text": "Aluminum Beverage Can\nRecyclable Tin",
            "shapes": "can"
        },
        "paper": {
            "bg": "#F5F5F5",
            "accent": "#424242",
            "text": "Printed Document\nClean White Paper",
            "shapes": "paper"
        },
        "plastic": {
            "bg": "#FFF59D",
            "accent": "#E65100",
            "text": "Plastic PET Bottle\nMineral Water Container",
            "shapes": "plastic_bottle"
        },
        "trash": {
            "bg": "#A1887F",
            "accent": "#3E2723",
            "text": "Organic Apple Core\nFood Waste Scrap",
            "shapes": "organic"
        }
    }

    created = {}
    for category, spec in sample_specs.items():
        img_path = SAMPLES_DIR / f"{category}_sample.jpg"
        if not img_path.exists():
            img = Image.new("RGB", (300, 300), color=spec["bg"])
            draw = ImageDraw.Draw(img)

            # Draw visual visual cues based on category
            if spec["shapes"] == "box":
                # Draw 3D-like carton flaps
                draw.rectangle([50, 70, 250, 230], outline=spec["accent"], width=6, fill="#D2B48C")
                draw.line([(50, 70), (150, 140), (250, 70)], fill=spec["accent"], width=4)
                draw.line([(150, 140), (150, 230)], fill=spec["accent"], width=4)
            elif spec["shapes"] == "bottle":
                # Draw glass bottle silhouette
                draw.rectangle([125, 40, 175, 90], outline=spec["accent"], width=5, fill="#B2DFDB")
                draw.polygon([(125, 90), (80, 140), (80, 250), (220, 250), (220, 140), (175, 90)], outline=spec["accent"], fill="#E0F2F1")
            elif spec["shapes"] == "can":
                # Draw aluminum beverage can
                draw.ellipse([90, 45, 210, 85], outline=spec["accent"], width=5, fill="#CFD8DC")
                draw.rectangle([90, 65, 210, 225], outline=spec["accent"], width=5, fill="#ECEFF1")
                draw.ellipse([90, 205, 210, 245], outline=spec["accent"], width=5, fill="#CFD8DC")
            elif spec["shapes"] == "paper":
                # Draw stacked paper sheet
                draw.rectangle([60, 45, 240, 245], outline=spec["accent"], width=4, fill="#FFFFFF")
                for y in range(80, 220, 22):
                    draw.line([(80, y), (220, y)], fill="#9E9E9E", width=2)
            elif spec["shapes"] == "plastic_bottle":
                # Draw PET plastic bottle
                draw.rectangle([130, 40, 170, 75], outline=spec["accent"], width=5, fill="#FFE082")
                draw.polygon([(130, 75), (95, 115), (95, 245), (205, 245), (205, 115), (170, 75)], outline=spec["accent"], fill="#FFF8E1")
                # Ribs
                draw.line([(95, 150), (205, 150)], fill=spec["accent"], width=3)
                draw.line([(95, 180), (205, 180)], fill=spec["accent"], width=3)
            elif spec["shapes"] == "organic":
                # Draw apple core / organic compost scrap
                draw.ellipse([100, 70, 200, 210], outline=spec["accent"], width=5, fill="#8D6E63")
                draw.line([(150, 45), (150, 75)], fill="#4E342E", width=6)
                draw.ellipse([155, 45, 180, 65], fill="#66BB6A")

            # Draw caption overlay at bottom
            draw.rectangle([0, 255, 300, 300], fill=(0, 0, 0, 180))
            draw.text((15, 265), f"Sample: {category.upper()}", fill="#FFFFFF")

            img.save(img_path, quality=95)
        created[category] = img_path

    return created


def get_sample_images() -> dict:
    """Returns a dictionary mapping class names to sample image paths."""
    generate_sample_images()
    return {
        cat: SAMPLES_DIR / f"{cat}_sample.jpg"
        for cat in config.CLASSES
        if (SAMPLES_DIR / f"{cat}_sample.jpg").exists()
    }
