# PaddleOCR Setup Guide for Soccer Project

## Current Status

The OCR adapter has been successfully implemented and tested with:
- ✅ **EasyOCR**: Working correctly
- ❌ **PaddleOCR v3.3.2**: Library compatibility issue

## Issue with PaddleOCR v3.3.2

### Error
```
ImportError: /workspace/miniconda3/envs/soccer/lib/python3.10/site-packages/pydisort/pydisort.cpython-310-x86_64-linux-gnu.so:
undefined symbol: _ZN5torch9TypeErrorC1EPKcz
```

### Root Cause
Torch and pydisort version mismatch. The compiled pydisort binary is incompatible with the installed torch version.

## Solution Options

### Option 1: Downgrade PaddleOCR (Recommended for stability)

PaddleOCR v2.7.0 uses simpler dependency management and doesn't have the pydisort issue:

```bash
conda activate soccer
pip uninstall -y paddleocr paddlepaddle paddlex pydisort

# Install PaddleOCR v2.7.0 (stable version)
pip install paddleocr==2.7.0

# Test
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_gpu=False); print('✓ PaddleOCR v2.7 ready')"
```

**Adapter changes needed for v2.7:**
```python
# In ocr_adapter.py, PaddleOCRAdapter.__init__():
self.ocr = PaddleOCR(
    lang=lang,
    use_gpu=use_gpu,      # v2.7 uses this parameter
    gpu_id=0 if use_gpu else None,
    use_angle_cls=use_angle_cls  # v2.7 has this parameter
)
```

### Option 2: Fix Torch/Pydisort Version Mismatch (Advanced)

Reinstall torch to match pydisort requirements:

```bash
conda activate soccer

# Remove conflicting packages
pip uninstall -y torch pydisort

# Reinstall with compatible versions
pip install torch==2.0.0  # Match pydisort compiled version
pip install pydisort

# Reinstall PaddleOCR
pip install paddleocr==3.3.2
```

### Option 3: Use CPU-Only PaddleOCR v3.3.2

Force PaddleOCR to skip GPU initialization:

```bash
conda activate soccer

# Set environment variable to skip GPU check
export DISABLE_MODEL_SOURCE_CHECK=1
export HIP_VISIBLE_DEVICES=""  # Disable AMD GPU if any

python script/detect_jersey.py --color_csv ... --no_gpu
```

This may work even with the torch mismatch since torch won't be fully loaded.

## Recommended Approach

**For the soccer project, Option 1 (downgrade to v2.7) is recommended because:**

1. ✅ PaddleOCR v2.7 is stable and widely used
2. ✅ No pydisort/torch compatibility issues
3. ✅ Works well on CPU
4. ✅ Minimal adapter code changes
5. ❌ Performance is slightly slower than v3.3.2
6. ✅ Much easier to troubleshoot

## Implementation Steps

### Step 1: Update Adapter for PaddleOCR v2.7

Edit `/workspace/Soccer/script/utils/ocr_adapter.py`:

```python
# Around line 93-107
# PaddleOCR v2.7 uses 'use_gpu' parameter instead of 'device'
self.ocr = PaddleOCR(
    lang=lang,
    use_gpu=use_gpu,        # v2.7 parameter
    use_angle_cls=use_angle_cls,  # v2.7 supports this
    show_log=False          # v2.7 doesn't support this, so it's omitted
)
```

### Step 2: Reinstall Dependencies

```bash
# Activate conda environment
conda activate soccer
cd /workspace/Soccer

# Remove current installation
pip uninstall -y paddleocr paddlepaddle

# Install v2.7
pip install paddleocr==2.7.0

# Test
python -c "from paddleocr import PaddleOCR; print('PaddleOCR v2.7 ready')"
```

### Step 3: Test

```bash
# Test with CPU first
python script/detect_jersey.py \
    --color_csv results/player/yolo11x/test_color.csv \
    --frames_dir original_frames \
    --output_dir results/player/yolo11x \
    --ocr_engine paddleocr \
    --no_gpu

# If CPU works, test with GPU
python script/detect_jersey.py \
    --color_csv results/player/yolo11x/test_color.csv \
    --frames_dir original_frames \
    --output_dir results/player/yolo11x \
    --ocr_engine paddleocr \
    --use_gpu
```

## Current Adapter Support

The current adapter already supports both v2.7 and v3.3.2 (theoretically), but needs version-specific tweaks:

### Version Detection (Future Enhancement)

```python
import paddleocr
version = tuple(map(int, paddleocr.__version__.split('.')[:2]))

if version >= (3, 0):
    # v3.0+ API
    device = "gpu:0" if use_gpu else "cpu"
    self.ocr = PaddleOCR(lang=lang, device=device, ...)
else:
    # v2.7 API
    self.ocr = PaddleOCR(lang=lang, use_gpu=use_gpu, ...)
```

## Testing the Full Pipeline

Once PaddleOCR is fixed, test with:

```bash
# Single file
python script/detect_jersey.py \
    --color_csv results/player/yolo11x/test_color.csv \
    --frames_dir original_frames \
    --output_dir results/player/yolo11x \
    --ocr_engine paddleocr

# Batch mode
python script/detect_jersey.py \
    --color_csv_dir results/player \
    --frames_dir original_frames \
    --output_dir results/player \
    --ocr_engine paddleocr

# With options
python script/detect_jersey.py \
    --color_csv results/player/yolo11x/test_color.csv \
    --frames_dir original_frames \
    --output_dir results/player/yolo11x \
    --ocr_engine paddleocr \
    --no_angle_cls \
    --continue_on_error
```

## Fallback Strategy

If PaddleOCR continues to have issues:
```bash
# Use EasyOCR as fallback
python script/detect_jersey.py \
    --color_csv results/player/yolo11x/test_color.csv \
    --frames_dir original_frames \
    --output_dir results/player/yolo11x \
    --ocr_engine easyocr
```

The adapter pattern ensures you can easily switch between engines without code changes.

## References

- [PaddleOCR v2.7 Documentation](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.7)
- [PaddleOCR v3.3 Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR Issue #10429 - GPU Parameter](https://github.com/PaddlePaddle/PaddleOCR/issues/10429)