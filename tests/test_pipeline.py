"""
Integration test for data pipeline.

This script tests the complete data loading and preprocessing pipeline
to ensure all components work together correctly.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


from data_loader import load_dataset, get_dataset_info
from preprocess import preprocess_images, split_data, create_augmentation_generator


def test_data_pipeline():
    """Test the complete data pipeline."""

    print("=" * 80)
    print("INTEGRATION TEST: Data Pipeline")
    print("=" * 80)

    # Test 1: Dataset info
    print("\n[TEST 1] Getting dataset information...")
    info = get_dataset_info()
    assert info is not None, "Failed to get dataset info"
    assert "num_examples" in info, "Dataset info missing num_examples"
    print(f"✅ Dataset info retrieved: {info['num_examples']} examples")

    # Test 2: Load data
    print("\n[TEST 2] Loading sample dataset...")
    images, labels = load_dataset(split="train[:200]", shuffle=False)
    assert len(images) == 200, f"Expected 200 images, got {len(images)}"
    assert len(labels) == 200, f"Expected 200 labels, got {len(labels)}"
    assert isinstance(images, list), "Images should be a list"
    print(f"✅ Loaded {len(images)} images successfully")
    print(f"   Sample image shapes: {[img.shape for img in images[:3]]}")

    # Test 3: Preprocess images
    print("\n[TEST 3] Preprocessing images...")
    processed_images = preprocess_images(images, target_size=(150, 150), normalize=True)
    assert processed_images.shape == (
        200,
        150,
        150,
        3,
    ), f"Unexpected shape: {processed_images.shape}"
    assert (
        processed_images.min() >= 0 and processed_images.max() <= 1
    ), "Images not normalized"
    print(f"✅ Preprocessed to shape: {processed_images.shape}")
    print(
        f"   Value range: [{processed_images.min():.3f}, {processed_images.max():.3f}]"
    )

    # Test 4: Split data
    print("\n[TEST 4] Splitting data...")
    train_data, val_data, test_data = split_data(
        processed_images, labels, train_size=0.7, val_size=0.15, test_size=0.15
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

    # Test 5: Augmentation
    print("\n[TEST 5] Testing data augmentation...")
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

    print("\n" + "=" * 80)
    print("🎉 ALL TESTS PASSED! Data pipeline is working correctly.")
    print("=" * 80)

    return True


if __name__ == "__main__":
    try:
        success = test_data_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
