"""
Test script to demonstrate:
1. Training model with real cats vs dogs data
2. Testing with human photos to show it only detects cats/dogs
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add src directory to path (now we're in tests/ folder, so go up one level)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from data_loader import load_dataset
from preprocess import preprocess_images, split_data
from model_train import build_cnn_model, compile_model, train_model, create_callbacks


def test_with_real_data():
    """Test 1: Train model with real cats vs dogs dataset"""
    print("=" * 80)
    print("TEST 1: Training with Real Cats vs Dogs Data")
    print("=" * 80)

    # Load a subset of the real dataset (first 1000 samples for quick training)
    print("\n1. Loading real cats vs dogs dataset...")
    images, labels = load_dataset(split="train[:1000]", shuffle=True)
    print(f"Loaded {len(images)} images")

    # Preprocess images
    print("\n2. Preprocessing images to 150x150...")
    processed_images = preprocess_images(images, target_size=(150, 150), normalize=True)

    # Split data
    print("\n3. Splitting data into train/val/test sets...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        processed_images,
        labels,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        shuffle=True,
    )

    # Build and compile model
    print("\n4. Building and compiling CNN model...")
    model = build_cnn_model(input_shape=(150, 150, 3), num_classes=1)
    model = compile_model(model, learning_rate=0.001)

    # Create callbacks
    callbacks = create_callbacks(
        model_save_path="models/cats_vs_dogs_model.keras", patience=5
    )

    # Train model (just 5 epochs for demo)
    print("\n5. Training model (5 epochs for demonstration)...")
    history = train_model(
        model,
        (X_train, y_train),
        (X_val, y_val),
        epochs=5,
        batch_size=32,
        callbacks=callbacks,
    )

    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Make predictions on a few test samples
    print("\n7. Sample predictions:")
    sample_predictions = model.predict(X_test[:5], verbose=0)
    for i, (pred, true_label) in enumerate(zip(sample_predictions, y_test[:5])):
        pred_class = "Dog" if pred[0] > 0.5 else "Cat"
        true_class = "Dog" if true_label == 1 else "Cat"
        confidence = pred[0] if pred[0] > 0.5 else (1 - pred[0])
        print(
            f"  Sample {i+1}: Predicted={pred_class} (confidence={confidence:.2%}), True={true_class}"
        )

    print("\n✅ TEST 1 PASSED: Model trains successfully with real data!")
    return model


def test_human_photo(model):
    """Test 2: What happens when you give it a human photo?"""
    print("\n" + "=" * 80)
    print("TEST 2: What About Human Photos?")
    print("=" * 80)

    print("\n⚠️  IMPORTANT CONCEPT: Model Limitations")
    print("-" * 80)
    print("This model is trained ONLY on cats and dogs.")
    print("It doesn't know about humans, cars, trees, or anything else!")
    print("\nWhen you give it a human photo:")
    print("  • It will still output a probability between 0 and 1")
    print("  • 0-0.5 means 'more cat-like features detected'")
    print("  • 0.5-1.0 means 'more dog-like features detected'")
    print("  • It CAN'T say 'this is not a cat or dog'")
    print("\nWhy? Because it's a BINARY classifier - it only learned to distinguish")
    print("between TWO classes (cat vs dog), not to detect 'unknown' objects.")
    print("-" * 80)

    # Create a synthetic "human-like" image (random noise as example)
    print("\n1. Simulating a human photo test...")
    print("   (Using random noise as a stand-in for a human photo)")

    # Random image to simulate an "out-of-distribution" input
    random_image = np.random.rand(1, 150, 150, 3).astype(np.float32)

    prediction = model.predict(random_image, verbose=0)
    pred_value = prediction[0][0]

    print(f"\n2. Model output: {pred_value:.4f}")

    if pred_value < 0.3:
        print(f"   ➡️  Model says: 'Cat-like' (confidence: {(1-pred_value):.2%})")
    elif pred_value > 0.7:
        print(f"   ➡️  Model says: 'Dog-like' (confidence: {pred_value:.2%})")
    else:
        print(f"   ➡️  Model says: 'Uncertain' (between cat and dog)")

    print("\n3. Reality check:")
    print("   • The model is forced to choose between cat-like or dog-like")
    print("   • It has NO concept of 'not a cat' or 'not a dog'")
    print("   • It will find the closest match to what it learned")

    print("\n" + "=" * 80)
    print("SOLUTION: How to make it say 'You are not a cat'?")
    print("=" * 80)
    print("\n📝 You need to train with THREE classes:")
    print("   1. Cat")
    print("   2. Dog")
    print("   3. Other/Human/Unknown")
    print("\nOr use a more sophisticated approach:")
    print("   • Anomaly detection")
    print("   • Confidence thresholding (reject low-confidence predictions)")
    print("   • Multi-stage classification (first detect if it's an animal)")
    print("=" * 80)


def demonstrate_binary_classification_concept():
    """Explain the concept of binary classification"""
    print("\n" + "=" * 80)
    print("UNDERSTANDING YOUR MODEL: Binary Classification")
    print("=" * 80)

    print("\n🎯 What your current model does:")
    print("   Input: An image (150x150x3)")
    print("   Output: A single number between 0 and 1")
    print("   Interpretation:")
    print("      • 0.0 = 100% Cat")
    print("      • 0.5 = Uncertain (50-50)")
    print("      • 1.0 = 100% Dog")

    print("\n❌ What your model CANNOT do:")
    print("   • It cannot say 'This is not a cat or dog'")
    print("   • It cannot detect humans, cars, trees, etc.")
    print("   • It was only trained on cat and dog images")

    print("\n✅ To answer 'Are you a cat?' correctly:")
    print("   Option 1: Train a 3-class model (Cat, Dog, Human)")
    print("   Option 2: Use confidence thresholding")
    print("      - If confidence < 80%, reject and say 'Not a cat or dog'")
    print("   Option 3: Add anomaly detection")
    print("      - Detect if input is too different from training data")

    print("\n📊 Example predictions:")
    print("   Cat image → 0.05 → 'Cat' ✅")
    print("   Dog image → 0.95 → 'Dog' ✅")
    print("   Human photo → 0.63 → 'Dog' ❌ (Wrong! But model doesn't know)")
    print("   Car photo → 0.42 → 'Cat' ❌ (Wrong! But model doesn't know)")

    print("\n💡 The model is like someone who only knows two words:")
    print("   'Cat' and 'Dog' - they'll call everything one of these!")
    print("=" * 80)


if __name__ == "__main__":
    # First, explain the concept
    demonstrate_binary_classification_concept()

    # Ask user if they want to run the full test
    print("\n" + "=" * 80)
    print("Do you want to run the full test with real data?")
    print("This will:")
    print("  • Load 1000 cat and dog images")
    print("  • Train the model for 5 epochs (~2-3 minutes)")
    print("  • Test with sample predictions")
    print("=" * 80)

    response = input("\nRun full test? (y/n): ").lower().strip()

    if response == "y":
        # Test 1: Train with real data
        model = test_with_real_data()

        # Test 2: Show what happens with human photos
        test_human_photo(model)

        print("\n" + "=" * 80)
        print("TESTS COMPLETE!")
        print("=" * 80)
    else:
        print("\nSkipping full test. The explanation above shows how your model works!")
