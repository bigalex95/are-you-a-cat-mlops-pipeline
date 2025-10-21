"""
Model training module for Cats vs Dogs classification.

This module provides functions to build, train, and save CNN models for
binary image classification (cat vs dog).
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (150, 150, 3),
    num_classes: int = 1,
) -> keras.Model:
    """
    Build a simple Convolutional Neural Network (CNN) for image classification.

    Architecture:
        - Conv2D (32 filters, 3x3 kernel) + ReLU + MaxPooling (2x2)
        - Conv2D (64 filters, 3x3 kernel) + ReLU + MaxPooling (2x2)
        - Conv2D (128 filters, 3x3 kernel) + ReLU + MaxPooling (2x2)
        - Flatten layer
        - Dense layer (128 units, ReLU)
        - Output layer (1 unit, sigmoid for binary classification)

    Args:
        input_shape (Tuple[int, int, int]): Shape of input images (height, width, channels).
            Default: (150, 150, 3)
        num_classes (int): Number of output classes. Use 1 for binary classification
            with sigmoid activation (predicts probability of class 1).
            Default: 1

    Returns:
        keras.Model: Compiled Keras model ready for training

    Example:
        >>> model = build_cnn_model(input_shape=(150, 150, 3))
        >>> model.summary()

    Learning Goals:
        - Understand CNN architecture components:
          * Convolutional layers: Extract features from images
          * MaxPooling: Reduce spatial dimensions and computation
          * Flatten: Convert 2D feature maps to 1D vector
          * Dense layers: Learn complex patterns for classification
        - Learn about filter sizes and their impact on feature extraction
        - Understand activation functions (ReLU for hidden layers, sigmoid for binary output)
        - Practice building sequential models with Keras
    """
    logger.info("Building CNN model...")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Output classes: {num_classes}")

    # Create sequential model
    model = models.Sequential(name="simple_cnn")

    # First Convolutional Block
    # Conv2D with 32 filters extracts 32 different features from the image
    # 3x3 kernel size is standard for capturing local patterns
    # padding='same' keeps the spatial dimensions unchanged
    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=input_shape,
            name="conv1",
        )
    )
    # MaxPooling reduces the spatial dimensions by half (150x150 -> 75x75)
    # This reduces computation and helps the model learn more abstract features
    model.add(layers.MaxPooling2D((2, 2), name="pool1"))

    # Second Convolutional Block
    # 64 filters learn more complex features from the features extracted by the first layer
    model.add(
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2")
    )
    # Further reduce spatial dimensions (75x75 -> 37x37)
    model.add(layers.MaxPooling2D((2, 2), name="pool2"))

    # Third Convolutional Block
    # 128 filters learn even more complex, high-level features
    model.add(
        layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3")
    )
    # Final pooling (37x37 -> 18x18)
    model.add(layers.MaxPooling2D((2, 2), name="pool3"))

    # Flatten the 3D feature maps to 1D feature vector
    # This prepares the data for the fully connected layers
    model.add(layers.Flatten(name="flatten"))

    # Dense (fully connected) layer with 128 units
    # This layer learns complex combinations of the extracted features
    # ReLU activation introduces non-linearity
    model.add(layers.Dense(128, activation="relu", name="fc1"))

    # Output layer
    # For binary classification, we use 1 unit with sigmoid activation
    # Sigmoid outputs a probability between 0 and 1
    # Output close to 0 = cat, close to 1 = dog
    model.add(layers.Dense(num_classes, activation="sigmoid", name="output"))

    logger.info("Model architecture created successfully!")

    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
    loss: str = "binary_crossentropy",
    metrics: list = None,
) -> keras.Model:
    """
    Compile the model with optimizer, loss function, and metrics.

    Args:
        model (keras.Model): The model to compile
        learning_rate (float): Learning rate for the optimizer.
            Default: 0.001
        optimizer (str): Optimizer to use ('adam', 'sgd', 'rmsprop').
            Default: 'adam'
        loss (str): Loss function to use. For binary classification, use 'binary_crossentropy'.
            Default: 'binary_crossentropy'
        metrics (list): List of metrics to track during training.
            Default: None (uses ['accuracy'])

    Returns:
        keras.Model: Compiled model

    Example:
        >>> model = build_cnn_model()
        >>> model = compile_model(model, learning_rate=0.001)

    Learning Goals:
        - Understand the role of optimizers in training:
          * Adam: Adaptive learning rate, good default choice
          * Combines benefits of RMSprop and momentum
        - Learn about loss functions:
          * Binary crossentropy: Standard loss for binary classification
          * Measures the difference between predicted and true probabilities
        - Understand evaluation metrics:
          * Accuracy: Percentage of correct predictions
        - Practice model compilation in Keras
    """
    if metrics is None:
        metrics = ["accuracy"]

    logger.info("Compiling model...")
    logger.info(f"  Optimizer: {optimizer}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Loss: {loss}")
    logger.info(f"  Metrics: {metrics}")

    # Create optimizer instance
    if optimizer.lower() == "adam":
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == "sgd":
        opt = optimizers.SGD(learning_rate=learning_rate)
    elif optimizer.lower() == "rmsprop":
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Compile model
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    logger.info("Model compiled successfully!")

    return model


def train_model(
    model: keras.Model,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    epochs: int = 20,
    batch_size: int = 32,
    callbacks: list = None,
    verbose: int = 1,
) -> keras.callbacks.History:
    """
    Train the CNN model on the provided data.

    Args:
        model (keras.Model): Compiled model to train
        train_data (Tuple[np.ndarray, np.ndarray]): Training data (images, labels)
        val_data (Tuple[np.ndarray, np.ndarray]): Validation data (images, labels)
        epochs (int): Number of training epochs.
            Default: 20
        batch_size (int): Batch size for training.
            Default: 32
        callbacks (list): List of Keras callbacks to use during training.
            Default: None
        verbose (int): Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch).
            Default: 1

    Returns:
        keras.callbacks.History: Training history containing loss and metrics

    Example:
        >>> model = build_cnn_model()
        >>> model = compile_model(model)
        >>> history = train_model(
        ...     model,
        ...     (X_train, y_train),
        ...     (X_val, y_val),
        ...     epochs=20,
        ...     batch_size=32
        ... )

    Learning Goals:
        - Understand the training loop:
          * Forward pass: Make predictions
          * Calculate loss: Compare predictions with true labels
          * Backward pass: Calculate gradients
          * Update weights: Adjust model parameters
        - Learn about epochs and batches:
          * Epoch: One complete pass through the training data
          * Batch: Subset of data processed at once
        - Understand validation during training:
          * Monitor performance on unseen data
          * Detect overfitting early
    """
    X_train, y_train = train_data
    X_val, y_val = val_data

    logger.info("Starting model training...")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Validation samples: {len(X_val)}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    logger.info("Training complete!")

    # Log final metrics
    final_train_loss = history.history["loss"][-1]
    final_train_acc = history.history["accuracy"][-1]
    final_val_loss = history.history["val_loss"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    logger.info(f"Final training loss: {final_train_loss:.4f}")
    logger.info(f"Final training accuracy: {final_train_acc:.4f}")
    logger.info(f"Final validation loss: {final_val_loss:.4f}")
    logger.info(f"Final validation accuracy: {final_val_acc:.4f}")

    return history


def create_callbacks(
    model_save_path: str = "models/best_model.keras",
    monitor: str = "val_loss",
    patience: int = 5,
) -> list:
    """
    Create a list of useful training callbacks.

    Callbacks:
        - EarlyStopping: Stop training when validation loss stops improving
        - ModelCheckpoint: Save the best model during training
        - ReduceLROnPlateau: Reduce learning rate when validation loss plateaus

    Args:
        model_save_path (str): Path to save the best model.
            Default: 'models/best_model.keras'
        monitor (str): Metric to monitor ('val_loss', 'val_accuracy', etc.).
            Default: 'val_loss'
        patience (int): Number of epochs with no improvement before taking action.
            Default: 5

    Returns:
        list: List of Keras callbacks

    Example:
        >>> callbacks = create_callbacks(
        ...     model_save_path='models/best_model.keras',
        ...     patience=5
        ... )
        >>> history = train_model(model, train_data, val_data, callbacks=callbacks)

    Learning Goals:
        - Understand training callbacks:
          * EarlyStopping: Prevent overfitting by stopping when validation performance degrades
          * ModelCheckpoint: Save the best version of the model
          * ReduceLROnPlateau: Adjust learning rate to escape local minima
        - Learn about patience parameter: Balance between thorough training and efficiency
    """
    # Create directory for saving models
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    logger.info("Creating training callbacks...")
    logger.info(f"  Monitor metric: {monitor}")
    logger.info(f"  Patience: {patience}")
    logger.info(f"  Model save path: {model_save_path}")

    callbacks_list = [
        # Stop training when validation loss stops improving
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Save the best model
        ModelCheckpoint(
            model_save_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    return callbacks_list


def save_model(model: keras.Model, save_path: str) -> None:
    """
    Save the trained model to disk.

    Args:
        model (keras.Model): Trained model to save
        save_path (str): Path to save the model

    Example:
        >>> save_model(model, 'models/final_model.keras')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    logger.info(f"Saving model to {save_path}")
    model.save(save_path)
    logger.info("Model saved successfully!")


def load_model(model_path: str) -> keras.Model:
    """
    Load a saved model from disk.

    Args:
        model_path (str): Path to the saved model

    Returns:
        keras.Model: Loaded model

    Example:
        >>> model = load_model('models/best_model.keras')
    """
    logger.info(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    logger.info("Model loaded successfully!")

    return model


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 80)
    print("Testing model_train.py")
    print("=" * 80)

    # 1. Build the model
    print("\n1. Building CNN model...")
    model = build_cnn_model(input_shape=(150, 150, 3), num_classes=1)

    # Display model architecture
    print("\nModel Architecture:")
    model.summary()

    # 2. Compile the model
    print("\n2. Compiling model...")
    model = compile_model(
        model, learning_rate=0.001, optimizer="adam", loss="binary_crossentropy"
    )

    # 3. Create dummy data for testing
    print("\n3. Creating dummy training data...")
    X_train = np.random.rand(100, 150, 150, 3).astype(np.float32)
    y_train = np.random.randint(0, 2, 100).astype(np.float32)
    X_val = np.random.rand(20, 150, 150, 3).astype(np.float32)
    y_val = np.random.randint(0, 2, 20).astype(np.float32)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")

    # 4. Create callbacks
    print("\n4. Creating training callbacks...")
    callbacks = create_callbacks(model_save_path="models/test_model.keras", patience=3)

    # 5. Train the model (just 2 epochs for testing)
    print("\n5. Training model (2 epochs for testing)...")
    history = train_model(
        model,
        (X_train, y_train),
        (X_val, y_val),
        epochs=2,
        batch_size=16,
        callbacks=callbacks,
    )

    # 6. Test model saving
    print("\n6. Testing model save/load...")
    save_model(model, "models/test_model_final.keras")
    loaded_model = load_model("models/test_model_final.keras")
    print("Model saved and loaded successfully!")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
