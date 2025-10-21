"""
Image preprocessing and data augmentation module.

This module provides functions for preprocessing images (resize, normalize)
and applying data augmentation techniques (rotation, flip, zoom) for the
Cats vs Dogs classification task.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple, Optional, List, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_images(
    images: Union[np.ndarray, List[np.ndarray]],
    target_size: Tuple[int, int] = (150, 150),
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess images by resizing and normalizing.

    This function takes a batch of images and applies standard preprocessing:
    - Resizes images to a target size
    - Normalizes pixel values to [0, 1] range

    Args:
        images (Union[np.ndarray, List[np.ndarray]]): Input images as either:
            - numpy array of shape (n_samples, height, width, channels)
            - list of numpy arrays with varying shapes (from data_loader)
        target_size (Tuple[int, int]): Target image size (height, width).
            Default: (150, 150)
        normalize (bool): Whether to normalize pixel values to [0, 1] range.
            Default: True

    Returns:
        np.ndarray: Preprocessed images of shape (n_samples, target_height, target_width, channels)

    Example:
        >>> # With numpy array
        >>> images = np.random.randint(0, 255, (10, 200, 200, 3), dtype=np.uint8)
        >>> processed = preprocess_images(images, target_size=(150, 150))
        >>>
        >>> # With list of images (from data_loader)
        >>> images_list, labels = load_dataset(split='train[:100]')
        >>> processed = preprocess_images(images_list, target_size=(150, 150))
        >>> print(f"Processed shape: {processed.shape}")

    Learning Goals:
        - Understand why image resizing is important (consistent input size for neural networks)
        - Learn about pixel normalization (helps with gradient descent convergence)
        - Practice working with numpy arrays and image data
    """
    logger.info(f"Preprocessing {len(images)} images")

    # Handle both list and numpy array inputs
    if isinstance(images, list):
        logger.info(f"Input: List of {len(images)} images with varying dimensions")
        if len(images) > 0:
            sample_shapes = [img.shape for img in images[:3]]
            logger.info(f"Sample input shapes: {sample_shapes}")
    else:
        logger.info(f"Input shape: {images.shape}")

    logger.info(f"Target size: {target_size}")

    preprocessed_images = []

    for img in images:
        # Resize image
        img_resized = tf.image.resize(img, target_size).numpy()

        # Normalize pixel values to [0, 1] if requested
        if normalize:
            if img_resized.max() > 1.0:
                img_resized = img_resized / 255.0

        preprocessed_images.append(img_resized)

    preprocessed_images = np.array(preprocessed_images, dtype=np.float32)

    logger.info(f"Output shape: {preprocessed_images.shape}")
    logger.info(
        f"Value range: [{preprocessed_images.min():.3f}, {preprocessed_images.max():.3f}]"
    )

    return preprocessed_images


def create_augmentation_generator(
    rotation_range: int = 20,
    width_shift_range: float = 0.2,
    height_shift_range: float = 0.2,
    shear_range: float = 0.2,
    zoom_range: float = 0.2,
    horizontal_flip: bool = True,
    vertical_flip: bool = False,
    fill_mode: str = "nearest",
) -> ImageDataGenerator:
    """
    Create an image data augmentation generator.

    Data augmentation helps prevent overfitting by creating variations of training images.
    This function creates a Keras ImageDataGenerator with common augmentation techniques.

    Args:
        rotation_range (int): Degree range for random rotations (0-180).
            Default: 20
        width_shift_range (float): Fraction of total width for horizontal shifts.
            Default: 0.2
        height_shift_range (float): Fraction of total height for vertical shifts.
            Default: 0.2
        shear_range (float): Shear intensity (shear angle in radians).
            Default: 0.2
        zoom_range (float): Range for random zoom [1-zoom_range, 1+zoom_range].
            Default: 0.2
        horizontal_flip (bool): Whether to randomly flip images horizontally.
            Default: True
        vertical_flip (bool): Whether to randomly flip images vertically.
            Default: False (typically not used for cats/dogs)
        fill_mode (str): Strategy for filling in newly created pixels.
            Options: 'constant', 'nearest', 'reflect', 'wrap'
            Default: 'nearest'

    Returns:
        ImageDataGenerator: Configured data augmentation generator

    Example:
        >>> # Create augmentation generator
        >>> aug_gen = create_augmentation_generator(rotation_range=30, zoom_range=0.3)
        >>>
        >>> # Use with training data
        >>> images = np.random.rand(100, 150, 150, 3)
        >>> labels = np.random.randint(0, 2, 100)
        >>>
        >>> # Generate augmented batches
        >>> for batch_images, batch_labels in aug_gen.flow(images, labels, batch_size=32):
        >>>     # Train your model with augmented data
        >>>     break

    Learning Goals:
        - Understand data augmentation and why it's important
        - Learn different augmentation techniques (rotation, flip, zoom, shift)
        - Practice using Keras ImageDataGenerator
        - Understand how augmentation helps prevent overfitting
    """
    logger.info("Creating data augmentation generator")
    logger.info(f"  - Rotation range: {rotation_range} degrees")
    logger.info(f"  - Width shift range: {width_shift_range}")
    logger.info(f"  - Height shift range: {height_shift_range}")
    logger.info(f"  - Shear range: {shear_range}")
    logger.info(f"  - Zoom range: {zoom_range}")
    logger.info(f"  - Horizontal flip: {horizontal_flip}")
    logger.info(f"  - Vertical flip: {vertical_flip}")

    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        fill_mode=fill_mode,
    )

    return datagen


def augment_images(
    images: np.ndarray,
    labels: np.ndarray,
    augmentation_factor: int = 2,
    **augmentation_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to create additional training samples.

    This function creates augmented versions of the input images to increase
    the effective size of the training dataset.

    Args:
        images (np.ndarray): Input images of shape (n_samples, height, width, channels)
        labels (np.ndarray): Corresponding labels of shape (n_samples,)
        augmentation_factor (int): Number of augmented versions per original image.
            Final dataset size will be n_samples * augmentation_factor.
            Default: 2
        **augmentation_kwargs: Additional keyword arguments passed to create_augmentation_generator()

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - augmented_images: Augmented images including originals
            - augmented_labels: Corresponding labels

    Example:
        >>> images = np.random.rand(100, 150, 150, 3)
        >>> labels = np.random.randint(0, 2, 100)
        >>> aug_images, aug_labels = augment_images(images, labels, augmentation_factor=3)
        >>> print(f"Original size: {len(images)}")
        >>> print(f"Augmented size: {len(aug_images)}")
    """
    logger.info(f"Augmenting {len(images)} images with factor {augmentation_factor}")

    # Create augmentation generator
    datagen = create_augmentation_generator(**augmentation_kwargs)

    augmented_images = []
    augmented_labels = []

    # Add original images
    augmented_images.extend(images)
    augmented_labels.extend(labels)

    # Generate augmented versions
    for i in range(len(images)):
        img = images[i : i + 1]  # Keep batch dimension
        label = labels[i]

        # Generate augmented versions
        aug_iter = datagen.flow(img, batch_size=1, shuffle=False)

        for _ in range(augmentation_factor - 1):
            aug_img = next(aug_iter)[0]
            augmented_images.append(aug_img)
            augmented_labels.append(label)

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    logger.info(f"Augmentation complete: {len(augmented_images)} images")

    return augmented_images, augmented_labels


def split_data(
    images: np.ndarray,
    labels: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    shuffle: bool = True,
    random_seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Split data into training, validation, and test sets.

    This function divides the dataset into three sets:
    - Training set: Used to train the model
    - Validation set: Used to tune hyperparameters and monitor training
    - Test set: Used for final model evaluation

    Args:
        images (np.ndarray): Input images of shape (n_samples, height, width, channels)
        labels (np.ndarray): Corresponding labels of shape (n_samples,)
        train_size (float): Proportion of data for training (0.0 to 1.0).
            Default: 0.7 (70%)
        val_size (float): Proportion of data for validation (0.0 to 1.0).
            Default: 0.15 (15%)
        test_size (float): Proportion of data for testing (0.0 to 1.0).
            Default: 0.15 (15%)
        shuffle (bool): Whether to shuffle data before splitting.
            Default: True
        random_seed (int): Random seed for reproducibility.
            Default: 42

    Returns:
        Tuple containing three tuples:
            - (train_images, train_labels): Training set
            - (val_images, val_labels): Validation set
            - (test_images, test_labels): Test set

    Raises:
        ValueError: If train_size + val_size + test_size != 1.0

    Example:
        >>> images = np.random.rand(1000, 150, 150, 3)
        >>> labels = np.random.randint(0, 2, 1000)
        >>>
        >>> (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        ...     images, labels, train_size=0.7, val_size=0.15, test_size=0.15
        ... )
        >>>
        >>> print(f"Training set: {len(X_train)} samples")
        >>> print(f"Validation set: {len(X_val)} samples")
        >>> print(f"Test set: {len(X_test)} samples")

    Learning Goals:
        - Understand the importance of train/val/test splits
        - Learn why validation sets are necessary for hyperparameter tuning
        - Practice maintaining separate test sets for unbiased evaluation
        - Understand the impact of data shuffling
    """
    # Validate split sizes
    total_size = train_size + val_size + test_size
    if not np.isclose(total_size, 1.0):
        raise ValueError(
            f"Split sizes must sum to 1.0, got {total_size} "
            f"(train={train_size}, val={val_size}, test={test_size})"
        )

    logger.info(f"Splitting {len(images)} samples into train/val/test sets")
    logger.info(
        f"Split ratios - Train: {train_size}, Val: {val_size}, Test: {test_size}"
    )

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Shuffle data if requested
    if shuffle:
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]

    # Calculate split indices
    n_samples = len(images)
    train_end = int(n_samples * train_size)
    val_end = train_end + int(n_samples * val_size)

    # Split data
    train_images = images[:train_end]
    train_labels = labels[:train_end]

    val_images = images[train_end:val_end]
    val_labels = labels[train_end:val_end]

    test_images = images[val_end:]
    test_labels = labels[val_end:]

    # Log split information
    logger.info(f"Training set: {len(train_images)} samples")
    logger.info(
        f"  - Cats: {np.sum(train_labels == 0)}, Dogs: {np.sum(train_labels == 1)}"
    )
    logger.info(f"Validation set: {len(val_images)} samples")
    logger.info(f"  - Cats: {np.sum(val_labels == 0)}, Dogs: {np.sum(val_labels == 1)}")
    logger.info(f"Test set: {len(test_images)} samples")
    logger.info(
        f"  - Cats: {np.sum(test_labels == 0)}, Dogs: {np.sum(test_labels == 1)}"
    )

    return (
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, test_labels),
    )


def save_processed_data(
    data: Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ],
    save_dir: str = None,
) -> None:
    """
    Save processed data splits to disk.

    This function saves the train/val/test splits as numpy arrays for later use.

    Args:
        data: Tuple containing (train, val, test) data, where each is (images, labels)
        save_dir (str): Directory to save processed data.
            Default: None (uses ../data/processed/)

    Example:
        >>> # After splitting data
        >>> train_data, val_data, test_data = split_data(images, labels)
        >>> save_processed_data((train_data, val_data, test_data))
    """
    import os

    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "processed"
        )

    os.makedirs(save_dir, exist_ok=True)

    (
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, test_labels),
    ) = data

    logger.info(f"Saving processed data to {save_dir}")

    # Save training data
    np.save(os.path.join(save_dir, "train_images.npy"), train_images)
    np.save(os.path.join(save_dir, "train_labels.npy"), train_labels)

    # Save validation data
    np.save(os.path.join(save_dir, "val_images.npy"), val_images)
    np.save(os.path.join(save_dir, "val_labels.npy"), val_labels)

    # Save test data
    np.save(os.path.join(save_dir, "test_images.npy"), test_images)
    np.save(os.path.join(save_dir, "test_labels.npy"), test_labels)

    logger.info("Data saved successfully!")


def load_processed_data(
    load_dir: str = None,
    mmap_mode: str = "r",
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Load previously saved processed data.

    Args:
        load_dir (str): Directory containing processed data.
            Default: None (uses ../data/processed/)
        mmap_mode (str): Memory-map mode for numpy arrays. Options:
            - 'r': Read-only (memory-mapped, efficient for large files)
            - None: Load entire array into memory
            Default: 'r' (memory-mapped)

    Returns:
        Tuple containing (train, val, test) data, where each is (images, labels)

    Example:
        >>> # Memory-mapped loading (efficient, doesn't load all into RAM)
        >>> train_data, val_data, test_data = load_processed_data(mmap_mode='r')
        >>> X_train, y_train = train_data
        >>> print(f"Loaded {len(X_train)} training samples (memory-mapped)")

        >>> # Load all into memory (use only if you have enough RAM/GPU memory)
        >>> train_data, val_data, test_data = load_processed_data(mmap_mode=None)
    """
    import os

    if load_dir is None:
        load_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "processed"
        )

    logger.info(f"Loading processed data from {load_dir}")
    if mmap_mode:
        logger.info(
            f"Using memory-mapped mode: {mmap_mode} (efficient for large files)"
        )
    else:
        logger.info(
            "Loading all data into memory (may require significant RAM/GPU memory)"
        )

    # Load training data
    train_images = np.load(
        os.path.join(load_dir, "train_images.npy"), mmap_mode=mmap_mode
    )
    train_labels = np.load(
        os.path.join(load_dir, "train_labels.npy"), mmap_mode=mmap_mode
    )

    # Load validation data
    val_images = np.load(os.path.join(load_dir, "val_images.npy"), mmap_mode=mmap_mode)
    val_labels = np.load(os.path.join(load_dir, "val_labels.npy"), mmap_mode=mmap_mode)

    # Load test data
    test_images = np.load(
        os.path.join(load_dir, "test_images.npy"), mmap_mode=mmap_mode
    )
    test_labels = np.load(
        os.path.join(load_dir, "test_labels.npy"), mmap_mode=mmap_mode
    )

    logger.info(f"Loaded {len(train_images)} training samples")
    logger.info(f"Loaded {len(val_images)} validation samples")
    logger.info(f"Loaded {len(test_images)} test samples")

    return (
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, test_labels),
    )


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 80)
    print("Testing preprocess.py")
    print("=" * 80)

    # Create dummy data for testing
    print("\n1. Creating dummy test data...")
    dummy_images = np.random.randint(0, 255, (100, 200, 200, 3), dtype=np.uint8)
    dummy_labels = np.random.randint(0, 2, 100)
    print(f"Created {len(dummy_images)} dummy images")

    # Test preprocessing
    print("\n2. Testing image preprocessing...")
    processed = preprocess_images(dummy_images, target_size=(150, 150))
    print(f"Processed shape: {processed.shape}")
    print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")

    # Test data splitting
    print("\n3. Testing data splitting...")
    train_data, val_data, test_data = split_data(
        processed, dummy_labels, train_size=0.7, val_size=0.15, test_size=0.15
    )
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    print(f"\nSplit results:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Test augmentation generator
    print("\n4. Testing augmentation generator...")
    aug_gen = create_augmentation_generator(rotation_range=20, zoom_range=0.2)
    print("Augmentation generator created successfully!")

    # Generate one batch to test
    batch_gen = aug_gen.flow(X_train[:10], y_train[:10], batch_size=5)
    batch_images, batch_labels = next(batch_gen)
    print(f"Generated augmented batch: {batch_images.shape}")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
