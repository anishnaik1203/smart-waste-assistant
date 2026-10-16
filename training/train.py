"""
Training script for Waste Classification Convolutional Neural Network.
Supports both MobileNetV2 Transfer Learning and Custom Deep CNN architectures.
Includes callbacks for early stopping, learning rate reduction, and model checkpointing.
"""

import argparse
import json
import time
import sys
from pathlib import Path

# Configure UTF-8 for console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
import config
from training.dataset import get_dataset_splits, calculate_class_weights, create_starter_dataset
from model.cnn_model import build_transfer_learning_model, build_custom_cnn_model
from training.evaluate import evaluate_model_on_dataset


class TrainingHistoryLogger(tf.keras.callbacks.Callback):
    """Custom callback to save epoch-by-epoch training and validation metrics to JSON."""
    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = Path(log_path)
        self.history_dict = {
            "epoch": [],
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "lr": []
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history_dict["epoch"].append(epoch + 1)
        self.history_dict["loss"].append(round(float(logs.get("loss", 0)), 4))
        self.history_dict["accuracy"].append(round(float(logs.get("accuracy", 0)), 4))
        self.history_dict["val_loss"].append(round(float(logs.get("val_loss", 0)), 4))
        self.history_dict["val_accuracy"].append(round(float(logs.get("val_accuracy", 0)), 4))
        
        # Capture current learning rate safely
        lr = float(self.model.optimizer.learning_rate.numpy()) if hasattr(self.model.optimizer.learning_rate, "numpy") else config.LEARNING_RATE
        self.history_dict["lr"].append(lr)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.history_dict, f, indent=2)


def train(
    model_type: str = "mobilenet",
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    data_dir: Path = config.DATASET_DIR,
    model_save_path: Path = config.MODEL_PATH,
    history_save_path: Path = config.HISTORY_LOG_PATH
):
    """Executes end-to-end model training, checkpointing, and evaluation."""
    start_time = time.time()
    data_dir = Path(data_dir)
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"🚀 Starting Waste Classification Training Pipeline")
    print(f"   Model Architecture: {model_type.upper()}")
    print(f"   Target Epochs:      {epochs}")
    print(f"   Batch Size:         {batch_size}")
    print(f"   Initial LR:         {learning_rate}")
    print(f"   Data Directory:     {data_dir}")
    print("=" * 65)

    # 1. Load dataset splits
    train_ds, val_ds, test_ds = get_dataset_splits(data_dir=data_dir, batch_size=batch_size)
    class_weights = calculate_class_weights(data_dir=data_dir)
    print(f"Class imbalance weights: {class_weights}")

    # 2. Build model architecture
    if model_type.lower() == "custom":
        model = build_custom_cnn_model(num_classes=config.NUM_CLASSES, learning_rate=learning_rate)
    else:
        model = build_transfer_learning_model(
            num_classes=config.NUM_CLASSES,
            fine_tune_at=100,
            learning_rate=learning_rate
        )

    model.summary()

    # 3. Define Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        TrainingHistoryLogger(history_save_path)
    ]

    # 4. Train Model
    print("\nStarting model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    elapsed = round(time.time() - start_time, 2)
    print(f"\n✅ Training completed in {elapsed} seconds.")

    # 5. Save final model if checkpoint didn't save (e.g. if single epoch)
    if not model_save_path.exists():
        model.save(str(model_save_path))
        print(f"Model saved to {model_save_path}")

    # 6. Evaluate on Test Split
    print("\nRunning final test evaluation...")
    metrics = evaluate_model_on_dataset(model, test_ds, save_path=config.METRICS_PATH)
    print(f"Test Accuracy: {metrics.get('accuracy_pct', 0)}%")

    return model, history, metrics


def main():
    parser = argparse.ArgumentParser(description="Train CNN Waste Classifier")
    parser.add_argument("--model", type=str, choices=["mobilenet", "custom"], default="mobilenet", help="Model type")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--create_starter", action="store_true", help="Generate starter dataset before training")
    args = parser.parse_args()

    if args.create_starter:
        create_starter_dataset()

    train(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
