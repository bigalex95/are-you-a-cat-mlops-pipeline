# GPU Memory Management Guide

## The Problem: GPU Out of Memory (OOM) Errors

### What Happened?

When loading processed `.npy` files, you encountered this error:

```
Allocator (GPU_0_bfc) ran out of memory trying to allocate 4.09GiB
```

### Why Does This Happen?

**Raw Data Loading (Before - WORKED ✅):**

```python
# Loading from TFRecords - streams data incrementally
images, labels = load_dataset(split="train[:1000]")
```

- Loads only **1,000 images** (~270 MB)
- TensorFlow streams data from TFRecord files
- Memory efficient - only loads what you request
- GPU memory: ~500 MB during training

**Processed Data Loading (After - FAILED ❌):**

```python
# Loading from .npy files - loads everything at once
(train, val, test) = load_dataset(split="train")
```

- Loads **ALL 23,262 images** (~6 GB total):
  - Training: 16,283 images × 150×150×3 × 4 bytes = **4.1 GB**
  - Validation: 3,489 images = **0.9 GB**
  - Test: 3,490 images = **0.9 GB**
- Entire dataset loaded into RAM at once
- TensorFlow tries to copy **all training data** to GPU
- **Your GPU**: RTX 3060 Laptop (2.7 GB available)
- **Result**: 4.1 GB doesn't fit in 2.7 GB → **OOM Error**

## Solutions

### Solution 1: Use Smaller Dataset (Current Fix)

**Best for**: Testing, development, prototyping

```python
# In your test script
data = load_dataset(split="train")
(X_train, y_train), (X_val, y_val), (X_test, y_test) = data

# Reduce to fit in GPU memory
X_train = X_train[:1000]  # Use first 1000 samples
y_train = y_train[:1000]
X_val = X_val[:200]
y_val = y_val[:200]
```

**Pros**: ✅ Works immediately, ✅ Fast for testing
**Cons**: ❌ Not training on full dataset

### Solution 2: Train on CPU

**Best for**: Full dataset training when GPU is too small

```python
# Add at the very top of your script (before importing TensorFlow)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

import tensorflow as tf
# ... rest of your code
```

**Pros**: ✅ Works with full dataset, ✅ No code changes needed
**Cons**: ❌ 10-20x slower than GPU

### Solution 3: Use Raw Data with Batching

**Best for**: Production training pipelines

```python
# Don't use processed .npy files - use raw TFRecords with batching
dataset = load_dataset(split="train", batch_size=32)

# Or manually create batched dataset
from data_loader import load_dataset
import tensorflow as tf

# This loads data incrementally, not all at once
images, labels = load_dataset(split="train")  # Returns iterator
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)

# Train with dataset directly
model.fit(dataset, epochs=10, validation_data=val_dataset)
```

**Pros**: ✅ Memory efficient, ✅ Full dataset, ✅ GPU accelerated
**Cons**: ❌ Requires code refactoring

### Solution 4: Use Mixed Precision Training

**Best for**: Reducing memory usage while maintaining accuracy

```python
import tensorflow as tf

# Enable mixed precision (uses float16 instead of float32)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Your model now uses ~50% less GPU memory
model = build_cnn_model(...)
```

**Pros**: ✅ ~50% memory reduction, ✅ Often faster training
**Cons**: ❌ Slight accuracy loss (usually negligible)

### Solution 5: Gradient Accumulation (Advanced)

**Best for**: Simulating large batch sizes with limited memory

```python
# Instead of batch_size=128 (requires 4x memory)
# Use batch_size=32 with 4 gradient accumulation steps

# This requires custom training loop
# See: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
```

### Solution 6: Use Memory-Mapped Arrays (Partial Solution)

**Best for**: Reducing RAM usage, but doesn't solve GPU OOM

The processed data loader now supports memory-mapping:

```python
from preprocess import load_processed_data

# Memory-mapped (doesn't load all into RAM)
train, val, test = load_processed_data(mmap_mode='r')

# Still causes GPU OOM because TensorFlow copies to GPU during training
```

**Pros**: ✅ Reduces RAM usage
**Cons**: ❌ Still causes GPU OOM during training

## Recommended Approaches

### For Development/Testing:

```python
# Use Solution 1: Reduce dataset size
data = load_dataset(split="train")
(X_train, y_train), _, _ = data
X_train, y_train = X_train[:1000], y_train[:1000]
```

### For Full Training on Your Laptop:

```python
# Use Solution 2: Train on CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### For Production/Cloud Training:

```python
# Use Solution 3 + 4: Raw data with batching + Mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
dataset = load_dataset(split="train", batch_size=32)
model.fit(dataset, ...)
```

## Why Raw Data Worked Before

When you used raw TFRecords:

```python
# This loaded only what you requested
images, labels = load_dataset(split="train[:1000]")
```

The function would:

1. Open TFRecord file
2. Read **only 1,000 examples**
3. Process them incrementally
4. ✅ Memory efficient!

When you added processed .npy files:

```python
# This loads EVERYTHING
data = load_dataset(split="train")
```

The function would:

1. Check if processed files exist ✅ (they do)
2. Load **ALL** training data: `np.load("train_images.npy")` → 4.1 GB
3. Load **ALL** validation data: `np.load("val_images.npy")` → 0.9 GB
4. Load **ALL** test data: `np.load("test_images.npy")` → 0.9 GB
5. ❌ GPU runs out of memory!

## Summary

| Approach        | Memory Usage | Speed   | Full Dataset | Complexity |
| --------------- | ------------ | ------- | ------------ | ---------- |
| Reduce dataset  | ✅ Low       | ✅ Fast | ❌ No        | ✅ Easy    |
| CPU training    | ✅ Low       | ❌ Slow | ✅ Yes       | ✅ Easy    |
| Raw + batching  | ✅ Low       | ✅ Fast | ✅ Yes       | ⚠️ Medium  |
| Mixed precision | ⚠️ Medium    | ✅ Fast | ✅ Yes       | ⚠️ Medium  |

**Recommendation**: For your laptop (RTX 3060 with 2.7 GB), use **Solution 1** (reduced dataset) for testing and **Solution 2** (CPU) when you need to train on the full dataset.
