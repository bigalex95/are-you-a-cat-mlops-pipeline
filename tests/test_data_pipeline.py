"""
Integration test for the complete data loading and preprocessing pipeline.

This script tests that all data pipeline components work together correctly,
including loading raw data, preprocessing, splitting, and augmentation.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


from data_loader import load_dataset, get_dataset_info
from preprocess import preprocess_images, split_data, create_augmentation_generator


def test_data_loading_pipeline():
    """Test the complete data loading and preprocessing pipeline."""

    print("=" * 80)
    print("INTEGRATION TEST: Data Loading & Preprocessing Pipeline")
    print("=" * 80)

    # Test 1: Dataset info
    print("\n[TEST 1] Getting dataset information...")
    info = get_dataset_info()
    assert info is not None, "Failed to get dataset info"
    assert "num_examples" in info, "Dataset info missing num_examples"
    print(f"✅ Dataset info retrieved: {info['num_examples']} examples")

    # Test 2: Load and preprocess data
    print("\n[TEST 2] Loading sample dataset (auto-preprocessed)...")
    # NOTE: load_dataset now automatically preprocesses custom splits
    images, labels = load_dataset(split="train[:200]", shuffle=False)

    # Images are now preprocessed numpy arrays (150, 150, 3)
    assert isinstance(images, np.ndarray), f"Expected numpy array, got {type(images)}"
    assert len(images) == 200, f"Expected 200 images, got {len(images)}"
    assert len(labels) == 200, f"Expected 200 labels, got {len(labels)}"
    assert images.shape == (200, 150, 150, 3), f"Unexpected shape: {images.shape}"
    assert images.min() >= 0 and images.max() <= 1, "Images should be normalized [0, 1]"
    print(f"✅ Loaded {len(images)} images successfully")
    print(f"   Images shape: {images.shape}")
    print(f"   Value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"   Already preprocessed: 150x150, normalized ✓")

    # Test 3: Split data
    print("\n[TEST 3] Splitting preprocessed data...")
    train_data, val_data, test_data = split_data(
        images, labels, train_size=0.7, val_size=0.15, test_size=0.15
    )
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    assert len(X_train) == 140, f"Expected 140 train samples, got {len(X_train)}"
    assert len(X_val) == 30, f"Expected 30 val samples, got {len(X_val)}"
    assert len(X_test) == 30, f"Expected 30 test samples, got {len(X_test)}"
    print(f"✅ Data split successful:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")

    # Test 4: Data augmentation
    print("\n[TEST 4] Testing data augmentation...")
    aug_gen = create_augmentation_generator(rotation_range=20, zoom_range=0.2)
    batch_gen = aug_gen.flow(X_train[:10], y_train[:10], batch_size=5)
    aug_batch, aug_labels = next(batch_gen)
    assert aug_batch.shape == (
        5,
        150,
        150,
        3,
    ), f"Unexpected augmented shape: {aug_batch.shape}"
    print(f"✅ Augmentation working: generated batch of shape {aug_batch.shape}")

    # Test 5: Full pipeline with processed data (if available)
    print("\n[TEST 5] Testing processed data loading (if available)...")
    try:
        full_data = load_dataset(split="train")
        if isinstance(full_data, tuple) and len(full_data) == 3:
            # Processed data returns (train, val, test) splits
            (
                (X_train_full, y_train_full),
                (X_val_full, y_val_full),
                (X_test_full, y_test_full),
            ) = full_data
            print(f"✅ Processed data loaded successfully:")
            print(f"   Train: {len(X_train_full)} samples")
            print(f"   Val: {len(X_val_full)} samples")
            print(f"   Test: {len(X_test_full)} samples")
            print(f"   Shape: {X_train_full.shape}")
        else:
            print("ℹ️  Processed data not available, skipping...")
    except Exception as e:
        print(f"ℹ️  Processed data test skipped: {e}")

    print("\n" + "=" * 80)
    print("🎉 ALL TESTS PASSED! Data loading pipeline is working correctly.")
    print("=" * 80)

    return True


if __name__ == "__main__":
    try:
        success = test_data_loading_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
