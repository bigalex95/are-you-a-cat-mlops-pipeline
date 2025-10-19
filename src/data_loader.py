"""
Data loader module for Cats vs Dogs dataset.

This module provides functionality to download and load the Cats vs Dogs dataset
using TensorFlow Datasets with automatic caching support.
"""

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Tuple, Optional, List, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(
    split: str = "train",
    data_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    as_supervised: bool = True,
) -> Union[Tuple[List[np.ndarray], np.ndarray], object]:
    """
    Load the Cats vs Dogs dataset from TensorFlow Datasets.

    This function downloads (if not cached) and loads the cats_vs_dogs dataset,
    which contains images of cats and dogs for binary classification.

    Args:
        split (str): Dataset split to load. Options:
            - 'train[:80%]': First 80% of training data
            - 'train[80%:]': Last 20% of training data (can be used as validation)
            - 'train': Full training dataset
            Default: 'train'
        data_dir (Optional[str]): Directory to cache the dataset.
            If None, uses default TFDS cache directory.
            Default: None
        batch_size (Optional[int]): If provided, returns batched dataset.
            If None, loads entire dataset into memory as numpy arrays.
            Default: None
        shuffle (bool): Whether to shuffle the dataset.
            Default: True
        as_supervised (bool): If True, returns (image, label) tuples.
            Default: True

    Returns:
        Union[Tuple[List[np.ndarray], np.ndarray], tf.data.Dataset]:
            If batch_size is None:
                - images: list of numpy arrays with varying shapes (height, width, channels)
                - labels: numpy array of shape (n_samples,) with values 0 (cat) or 1 (dog)
            If batch_size is provided:
                - returns a batched tf.data.Dataset

    Example:
        >>> # Load training data
        >>> images, labels = load_dataset(split='train[:80%]')
        >>> print(f"Images shape: {images.shape}")
        >>> print(f"Labels shape: {labels.shape}")

        >>> # Load validation data
        >>> val_images, val_labels = load_dataset(split='train[80%:]')

    Learning Goals:
        - Understand how to use TensorFlow Datasets (TFDS)
        - Learn about dataset caching for efficient reuse
        - Practice handling different data splits
        - Understand the importance of data loading in ML pipelines
    """
    # Set data directory
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "raw"
        )
        os.makedirs(data_dir, exist_ok=True)

    logger.info(f"Loading cats_vs_dogs dataset, split: {split}")
    logger.info(f"Data directory: {data_dir}")

    try:
        # Load dataset from TensorFlow Datasets
        # as_supervised=True returns (image, label) tuples
        dataset, info = tfds.load(  # type: ignore[misc]
            "cats_vs_dogs",
            split=split,
            data_dir=data_dir,
            as_supervised=as_supervised,
            with_info=True,
            shuffle_files=shuffle,
        )

        logger.info(f"Dataset info: {info}")
        logger.info(
            f"Number of examples: {info.splits[split.split('[')[0]].num_examples}"
        )
        logger.info(f"Features: {info.features}")

        # If batch_size is specified, return the batched dataset
        if batch_size is not None:
            if shuffle:
                dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(batch_size)
            return dataset

        # Otherwise, load all data into memory
        # Note: Images have different sizes, so we keep them as a list
        # They will be resized during preprocessing
        images = []
        labels = []

        logger.info("Converting dataset to numpy arrays...")
        for image, label in tfds.as_numpy(dataset):
            images.append(image)
            labels.append(label)

        # Convert labels to numpy array (they have uniform size)
        # Keep images as list since they have varying dimensions
        labels = np.array(labels)

        logger.info(f"Loaded {len(images)} images")
        logger.info(f"Images: List of {len(images)} images with varying dimensions")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(
            f"Label distribution - Cats (0): {np.sum(labels == 0)}, Dogs (1): {np.sum(labels == 1)}"
        )

        # Sample image dimensions
        if len(images) > 0:
            sample_shapes = [img.shape for img in images[:5]]
            logger.info(f"Sample image shapes: {sample_shapes}")

        return images, labels

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def get_dataset_info(data_dir: Optional[str] = None) -> dict:
    """
    Get information about the cats_vs_dogs dataset.

    Args:
        data_dir (Optional[str]): Directory where dataset is cached.
            Default: None

    Returns:
        dict: Dictionary containing dataset information including:
            - description: Dataset description
            - num_examples: Number of examples in each split
            - features: Feature specifications
            - splits: Available data splits

    Example:
        >>> info = get_dataset_info()
        >>> print(info['description'])
        >>> print(f"Total examples: {info['num_examples']}")
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "raw"
        )

    try:
        builder = tfds.builder("cats_vs_dogs", data_dir=data_dir)
        info = builder.info

        return {
            "description": info.description,
            "num_examples": info.splits["train"].num_examples,
            "features": str(info.features),
            "splits": list(info.splits.keys()),
            "homepage": info.homepage,
            "citation": info.citation,
        }
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        return {}


def download_dataset(data_dir: Optional[str] = None) -> None:
    """
    Download and cache the cats_vs_dogs dataset.

    This function only downloads the dataset without loading it into memory.
    Useful for pre-downloading datasets before running training pipelines.

    Args:
        data_dir (Optional[str]): Directory to cache the dataset.
            Default: None (uses default cache directory)

    Example:
        >>> # Pre-download dataset
        >>> download_dataset()
        >>> print("Dataset downloaded and cached successfully!")
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "raw"
        )
        os.makedirs(data_dir, exist_ok=True)

    logger.info(f"Downloading cats_vs_dogs dataset to {data_dir}")

    try:
        # Download and prepare the dataset
        builder = tfds.builder("cats_vs_dogs", data_dir=data_dir)
        builder.download_and_prepare()

        logger.info("Dataset downloaded and prepared successfully!")

    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 80)
    print("Testing data_loader.py")
    print("=" * 80)

    # Get dataset info
    print("\n1. Getting dataset information...")
    info = get_dataset_info()
    if info:
        print(f"\nDataset: Cats vs Dogs")
        print(f"Total examples: {info['num_examples']}")
        print(f"Available splits: {info['splits']}")

    # Download dataset (if not already cached)
    print("\n2. Downloading dataset (if not cached)...")
    download_dataset()

    # Load a small sample for testing
    print("\n3. Loading a small sample (first 100 examples)...")
    images, labels = load_dataset(split="train[:100]", shuffle=False)
    print(f"\nSample loaded successfully!")
    print(f"Number of images: {len(images)}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample image shapes: {[img.shape for img in images[:5]]}")
    print(f"Label dtype: {labels.dtype}")
    print(
        f"Image value ranges: min={min(img.min() for img in images)}, max={max(img.max() for img in images)}"
    )
    print(f"Unique labels: {np.unique(labels)}")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
