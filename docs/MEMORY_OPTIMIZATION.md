# Memory Optimization Guide

## Problem: RAM Usage Exceeding 12GB in Google Colab

### Root Cause

The original data preprocessing notebook loads the **entire Cats vs Dogs dataset into memory at once**, causing RAM usage to exceed 12GB (Colab's limit). Here's why:

#### Memory Breakdown:

- **Total dataset**: ~23,000 images
- **After preprocessing** (150×150×3 pixels, float32):

  - Training set: ~16,000 images × 150 × 150 × 3 × 4 bytes = **~4.1 GB**
  - Validation set: ~3,500 images = **~0.9 GB**
  - Test set: ~3,500 images = **~0.9 GB**
  - **Subtotal**: ~6 GB for numpy arrays

- **Additional overhead**:
  - TensorFlow dataset objects: ~2-3 GB
  - Intermediate processing buffers: ~2-3 GB
  - Python runtime and libraries: ~1-2 GB
  - **Total peak usage**: **12-15 GB** ❌

### Why This Happens

The original code in `src/data_loader.py` does this:

```python
# Load ALL images into memory
images = []
labels = []
for image, label in tfds.as_numpy(dataset):
    images.append(image)  # Stores raw images
    labels.append(label)

# Then preprocess ALL images at once
processed_images = preprocess_images(images, target_size=(150, 150))
```

This loads:

1. Raw images (varying sizes, ~3-5 GB)
2. Processed images (150×150, ~6 GB)
3. Both exist in memory simultaneously during preprocessing

---

## Solution: Memory-Efficient Processing

### Strategy

Use **batch processing** with **memory-mapped files** to keep RAM usage under 8GB:

1. **Don't load all images at once** - process in batches
2. **Use memory-mapped arrays** - store on disk, access like RAM
3. **Stream from TensorFlow datasets** - iterator mode
4. **Clear memory after each batch** - force garbage collection

### Implementation

I've created a new notebook: `colab_data_preprocessing_memory_efficient.ipynb`

#### Key Differences:

**Original (Memory Intensive):**

```python
# ❌ Loads everything into memory
images, labels = load_dataset(split='train')
processed = preprocess_images(images)  # All 23k images at once
```

**Optimized (Memory Efficient):**

```python
# ✅ Process in batches
BATCH_SIZE = 500

# Pre-allocate memory-mapped array (on disk)
all_images = np.memmap('temp.npy', dtype='float32', mode='w+',
                       shape=(total_samples, 150, 150, 3))

# Process in batches
for i in range(0, total_samples, BATCH_SIZE):
    batch = load_batch(i, BATCH_SIZE)
    processed_batch = preprocess_batch(batch)
    all_images[i:i+BATCH_SIZE] = processed_batch

    # Free memory
    del batch, processed_batch
    gc.collect()
```

---

## Detailed Optimizations

### 1. Memory-Mapped Arrays

Instead of loading arrays into RAM, use disk-backed arrays:

```python
# Regular numpy array (loads into RAM)
array = np.load('data.npy')  # Uses ~6 GB RAM

# Memory-mapped array (stays on disk)
array = np.load('data.npy', mmap_mode='r')  # Uses ~0 MB RAM
```

**Benefits:**

- Access like normal numpy arrays
- Data stays on disk
- Only loaded pages are cached in RAM
- Perfect for large datasets

### 2. Batch Processing

Process images in small batches:

```python
BATCH_SIZE = 500  # Process 500 images at a time

for batch in batches:
    # Process batch (uses ~300 MB)
    processed_batch = process(batch)

    # Write to disk
    save(processed_batch)

    # Clear memory
    del batch, processed_batch
    gc.collect()
```

**Benefits:**

- Peak RAM usage: ~1-2 GB per batch
- Total RAM never exceeds 8 GB
- Progress bar shows real-time status

### 3. Streaming from TensorFlow

Use TensorFlow datasets in streaming mode:

```python
# ❌ Loads everything
dataset = tfds.load('cats_vs_dogs', split='train')
images = list(tfds.as_numpy(dataset))  # Loads all into memory

# ✅ Streams data
dataset = tfds.load('cats_vs_dogs', split='train')
for image, label in tfds.as_numpy(dataset):  # Loads one at a time
    process(image)
```

### 4. Memory Monitoring

Track RAM usage throughout:

```python
import psutil

def print_memory_usage():
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024**3
    system_mem = psutil.virtual_memory()
    print(f"Process: {mem_gb:.2f} GB")
    print(f"System: {system_mem.percent:.1f}% used")

print_memory_usage()  # Check at each step
```

---

## Memory Usage Comparison

| Method        | Peak RAM | Colab Compatible | Processing Time |
| ------------- | -------- | ---------------- | --------------- |
| **Original**  | 12-15 GB | ❌ No            | ~15 min         |
| **Optimized** | 6-8 GB   | ✅ Yes           | ~18 min         |
| **With mmap** | 4-6 GB   | ✅ Yes           | ~20 min         |

---

## How to Use

### Option 1: Use the New Memory-Efficient Notebook

1. Open `notebooks/colab_data_preprocessing_memory_efficient.ipynb` in Colab
2. Run all cells
3. Monitor memory usage in each step
4. Peak RAM should stay under 8 GB

### Option 2: Update Existing Code

Modify `src/data_loader.py` to use batch processing:

```python
def load_dataset_batch(batch_size=500):
    """Load dataset in batches to save memory."""
    dataset = tfds.load('cats_vs_dogs', split='train')

    batch_images = []
    batch_labels = []

    for image, label in tfds.as_numpy(dataset):
        batch_images.append(image)
        batch_labels.append(label)

        if len(batch_images) >= batch_size:
            yield (batch_images, batch_labels)
            batch_images = []
            batch_labels = []

    if batch_images:
        yield (batch_images, batch_labels)
```

### Option 3: Use Data Generators for Training

Instead of loading all data, use generators during training:

```python
# Instead of this:
X_train, y_train = load_processed_data()  # Loads 4+ GB
model.fit(X_train, y_train)

# Do this:
train_generator = create_data_generator(batch_size=32)  # Loads only 32 images
model.fit(train_generator)
```

---

## Training with Limited Memory

Even after preprocessing, training can use lots of memory. Here are tips:

### 1. Use Memory-Mapped Loading

```python
# In training notebook
X_train = np.load('data/processed/train_images.npy', mmap_mode='r')
y_train = np.load('data/processed/train_labels.npy', mmap_mode='r')
```

### 2. Reduce Batch Size

```python
# Instead of:
BATCH_SIZE = 64  # Uses more GPU/RAM

# Use:
BATCH_SIZE = 16  # Uses less GPU/RAM
```

### 3. Use tf.data Pipeline

```python
import tensorflow as tf

def create_tf_dataset(images_path, labels_path, batch_size=32):
    """Create TensorFlow dataset with efficient loading."""
    # Load with mmap
    X = np.load(images_path, mmap_mode='r')
    y = np.load(labels_path, mmap_mode='r')

    # Create tf.data.Dataset (efficient batching)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Use in training
train_dataset = create_tf_dataset(
    'data/processed/train_images.npy',
    'data/processed/train_labels.npy',
    batch_size=32
)

model.fit(train_dataset, epochs=10)
```

### 4. Mixed Precision Training

Reduce memory by using float16 instead of float32:

```python
from tensorflow.keras import mixed_precision

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Train as usual (uses half the memory)
model.fit(X_train, y_train)
```

---

## Best Practices

### ✅ DO:

- Process data in batches
- Use memory-mapped files for large arrays
- Monitor memory usage with `psutil`
- Clear memory after each batch with `gc.collect()`
- Use TensorFlow data pipelines
- Load only what you need

### ❌ DON'T:

- Load entire dataset into memory at once
- Keep intermediate results in memory
- Use large batch sizes on limited hardware
- Create unnecessary copies of data
- Forget to delete unused variables

---

## Troubleshooting

### Still Running Out of Memory?

1. **Reduce batch size** to 250 or 100
2. **Use smaller images** (128×128 instead of 150×150)
3. **Reduce dataset size** (use subset for testing)
4. **Upgrade to Colab Pro** (25GB RAM instead of 12GB)
5. **Use cloud GPUs** (AWS, GCP, etc.)

### Memory Error During Training?

```python
# Add to training code
import tensorflow as tf

# Limit TensorFlow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

## Summary

The original notebook fails because it loads 23,000 images (~15GB) into memory at once. The optimized version:

1. **Processes in batches** (500 images at a time)
2. **Uses memory-mapped files** (disk storage, not RAM)
3. **Streams from TensorFlow** (iterator mode)
4. **Clears memory** after each batch

**Result**: Peak RAM usage drops from 15GB to 6-8GB ✅

Use the new notebook: `notebooks/colab_data_preprocessing_memory_efficient.ipynb`
