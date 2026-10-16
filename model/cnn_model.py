"""
Convolutional Neural Network (CNN) Model Architecture Definitions.
Provides both a Transfer Learning MobileNetV2 architecture (recommended for high accuracy
and edge efficiency) and a Custom Deep CNN architecture (for academic comparative study).
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import config


def get_data_augmentation():
    """Builds a Keras Sequential preprocessing layer for real-time data augmentation."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=config.RANDOM_SEED),
        layers.RandomRotation(0.15, seed=config.RANDOM_SEED),
        layers.RandomZoom(0.15, seed=config.RANDOM_SEED),
        layers.RandomContrast(0.1, seed=config.RANDOM_SEED),
    ], name="data_augmentation")


def build_transfer_learning_model(
    num_classes: int = config.NUM_CLASSES,
    input_shape: tuple = config.INPUT_SHAPE,
    fine_tune_at: int = 100,
    learning_rate: float = config.LEARNING_RATE
) -> tf.keras.Model:
    """
    Constructs a Transfer Learning model using MobileNetV2 as the feature extractor.
    
    MobileNetV2 uses Depthwise Separable Convolutions and Inverted Residual blocks,
    making it computationally efficient with only ~2.2M - 3.5M parameters, ideal for
    real-time waste classification on local systems and edge devices.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Real-time data augmentation pipeline
    x = get_data_augmentation()(inputs)

    # Preprocessing: scales pixel values from [0, 255] to [-1, 1] for MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Base feature extractor pretrained on ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Fine-tuning configuration: Freeze early general features, fine-tune higher layers
    base_model.trainable = True
    if fine_tune_at > 0:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    x = base_model(x, training=False)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_256"
    )(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)

    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_WasteClassifier")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model


def build_custom_cnn_model(
    num_classes: int = config.NUM_CLASSES,
    input_shape: tuple = config.INPUT_SHAPE,
    learning_rate: float = config.LEARNING_RATE
) -> tf.keras.Model:
    """
    Constructs a 4-block custom Convolutional Neural Network from scratch.
    
    Architecture:
    [Conv2D (32) -> BN -> ReLU -> MaxPool]
    [Conv2D (64) -> BN -> ReLU -> MaxPool]
    [Conv2D (128) -> BN -> ReLU -> MaxPool]
    [Conv2D (256) -> BN -> ReLU -> MaxPool]
    [GlobalAveragePooling2D -> Dense(256) -> Dropout(0.5) -> Dense(6, Softmax)]
    """
    inputs = layers.Input(shape=input_shape, name="custom_input_image")

    # Rescaling [0, 255] -> [0, 1]
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = get_data_augmentation()(x)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

    # Dense Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="custom_predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Custom_CNN_WasteClassifier")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model
