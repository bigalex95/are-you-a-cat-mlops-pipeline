# Data Pipeline Test Report

**Date:** October 19, 2025  
**Status:** ✅ ALL TESTS PASSED

## Problem Identified

The original `data_loader.py` had an issue when trying to convert images to a numpy array:

```
ValueError: setting an array element with a sequence. The requested array has an
inhomogeneous shape after 1 dimensions. The detected shape was (100,) +
inhomogeneous part.
```

### Root Cause

The Cats vs Dogs dataset contains images with **varying dimensions** (different heights and widths). When we tried to convert a list of these variable-sized images into a single numpy array, it failed because numpy arrays require uniform shapes.

## Solution Applied

### Changes to `src/data_loader.py`:

1. **Modified return type**: Changed to return images as a `List[np.ndarray]` instead of a single `np.ndarray`
2. **Updated type hints**: Added `Union[Tuple[List[np.ndarray], np.ndarray], object]` to the return type
3. **Improved logging**: Added logging to show sample image shapes instead of trying to show shape of list
4. **Updated docstring**: Clarified that images are returned as a list with varying dimensions

### Changes to `src/preprocess.py`:

1. **Updated input type**: Modified `preprocess_images()` to accept both `np.ndarray` and `List[np.ndarray]`
2. **Added input detection**: Function now detects whether input is a list or array and handles accordingly
3. **Improved logging**: Shows sample shapes when input is a list

## Test Results

### Integration Test (`test_pipeline.py`)

All 5 tests passed successfully:

✅ **Test 1: Dataset Info**

- Successfully retrieved dataset information
- Dataset contains 23,262 examples

✅ **Test 2: Data Loading**

- Loaded 200 sample images successfully
- Images returned as list with varying dimensions
- Sample shapes: `[(262, 350, 3), (409, 336, 3), (493, 500, 3)]`
- Labels: 103 cats, 97 dogs

✅ **Test 3: Image Preprocessing**

- Successfully preprocessed 200 images
- Resized to uniform (150, 150, 3) shape
- Normalized to [0.0, 1.0] range
- Output shape: (200, 150, 150, 3)

✅ **Test 4: Data Splitting**

- Successfully split into train/val/test sets
- Train: 140 samples (70%)
- Validation: 30 samples (15%)
- Test: 30 samples (15%)
- Class distribution maintained across splits

✅ **Test 5: Data Augmentation**

- Successfully created augmentation generator
- Generated augmented batches of shape (5, 150, 150, 3)
- Augmentation techniques working (rotation, zoom, flip, shift)

## Files Modified

1. **`src/data_loader.py`**

   - Returns images as list instead of numpy array
   - Updated type hints and documentation
   - Improved error handling and logging

2. **`src/preprocess.py`**

   - Accepts both list and numpy array inputs
   - Automatic input type detection
   - Maintains backward compatibility

3. **`requirements.txt`** (already updated)
   - Added `tensorflow-datasets`
   - Added `matplotlib` and `seaborn` for visualizations

## Files Created

1. **`test_pipeline.py`**
   - Comprehensive integration test
   - Tests all pipeline components
   - Validates data shapes and values

## How to Run Tests

```bash
# Test individual modules
python3 src/data_loader.py
python3 src/preprocess.py

# Run integration test
python3 test_pipeline.py

# Use the Jupyter notebook for interactive exploration
jupyter notebook notebooks/data_pipeline_test.ipynb
```

## Next Steps

The data pipeline is now fully functional and ready for:

1. **Phase 3: Model Training**

   - Build CNN architecture
   - Train model using the preprocessed data
   - Implement MLflow experiment tracking

2. **Creating Actual Unit Tests**
   - Create proper pytest tests in `tests/` directory
   - Test edge cases and error handling
   - Add continuous integration

## Key Learnings

1. **Variable-sized images**: Real-world datasets often have images with different dimensions
2. **Preprocessing necessity**: Resizing is crucial to create uniform inputs for neural networks
3. **Type flexibility**: Functions should handle multiple input types for better usability
4. **Testing importance**: Integration tests catch issues that unit tests might miss

## Summary

✅ **Data pipeline is working correctly**  
✅ **All components tested and validated**  
✅ **Ready for model training phase**

The pipeline successfully:

- Downloads and caches the Cats vs Dogs dataset
- Loads images with varying dimensions
- Preprocesses images to uniform size and normalized values
- Splits data into train/validation/test sets
- Applies data augmentation techniques
