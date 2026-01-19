# Quick Start: OCR Engine Selection

## Current Status
- ✅ **EasyOCR**: Ready to use
- ⚠️ **PaddleOCR**: Needs environment fix

## Use EasyOCR (Working Now)

```bash
cd /workspace/Soccer
conda activate soccer

# Single file
python script/detect_jersey.py \
    --color_csv results/player/yolo11x/test_color.csv \
    --frames_dir original_frames \
    --output_dir results/player/yolo11x \
    --ocr_engine easyocr

# Batch mode
python script/detect_jersey.py \
    --color_csv_dir results/player \
    --frames_dir original_frames \
    --output_dir results/player \
    --ocr_engine easyocr
```

## Fix PaddleOCR Environment

### Option 1: Recommended - Use v2.7 (Easiest)

```bash
conda activate soccer

# Clean install
pip uninstall -y paddleocr paddlepaddle paddlex pydisort
pip install paddleocr==2.7.0

# Test
python -c "from paddleocr import PaddleOCR; print('✓ PaddleOCR v2.7 ready')"
```

### Option 2: Keep v3.3.2 - Fix Dependencies

```bash
conda activate soccer
pip install --upgrade torch pydisort
```

## Use PaddleOCR (After Environment Fix)

```bash
cd /workspace/Soccer
conda activate soccer

# With GPU
python script/detect_jersey.py \
    --color_csv results/player/yolo11x/test_color.csv \
    --frames_dir original_frames \
    --output_dir results/player/yolo11x \
    --ocr_engine paddleocr

# With CPU only
python script/detect_jersey.py \
    --color_csv results/player/yolo11x/test_color.csv \
    --frames_dir original_frames \
    --output_dir results/player/yolo11x \
    --ocr_engine paddleocr \
    --no_gpu

# With options
python script/detect_jersey.py \
    --color_csv results/player/yolo11x/test_color.csv \
    --frames_dir original_frames \
    --output_dir results/player/yolo11x \
    --ocr_engine paddleocr \
    --no_angle_cls
```

## CLI Options

### Engine Selection
```bash
--ocr_engine paddleocr   # Default
--ocr_engine easyocr     # Alternative
```

### PaddleOCR Options
```bash
--no_angle_cls    # Disable angle classification (faster)
--ocr_show_log    # Show logs (v2.7 only)
--no_gpu          # Force CPU mode
```

### Batch Processing
```bash
--color_csv_dir DIRECTORY          # Process all CSVs in directory
--color_csv_pattern "*.csv"        # Filter with pattern
--frames_dir_pattern "*/frames"    # Match frames directories
--skip_existing                    # Skip already processed
--continue_on_error                # Continue on failures
```

## Check Current Setup

```bash
conda activate soccer

# Check installed version
python -c "import paddleocr; print('PaddleOCR:', paddleocr.__version__)"
python -c "import easyocr; print('EasyOCR: OK')"

# Test adapter
python -c "
from script.utils.ocr_adapter import create_ocr_engine
ocr = create_ocr_engine(engine_type='easyocr', use_gpu=False)
print(f'✓ {ocr.engine_name} adapter working')
"
```

## Troubleshooting

### "Unknown argument: gpu" error
**Cause**: PaddleOCR v3.3.2 installed with torch mismatch
**Fix**: See "Fix PaddleOCR Environment" above

### "ImportError: pydisort" error
**Cause**: Torch and pydisort versions incompatible
**Fix**: Option 1 (downgrade to v2.7) or Option 2 (upgrade torch)

### EasyOCR slow on first run
**Cause**: Downloading language models
**Solution**: First run will be slow, subsequent runs use cached models

### GPU not detected
**Solution**: Use `--no_gpu` flag or check CUDA installation

## Performance Tips

### Fast Mode (EasyOCR)
```bash
--ocr_engine easyocr --no_preprocessing
```

### Accurate Mode (PaddleOCR)
```bash
--ocr_engine paddleocr --use_gpu
```

### Balanced Mode
```bash
--ocr_engine easyocr  # Default, good balance
```

## See Also
- `PADDLE_OCR_SETUP_GUIDE.md` - Detailed PaddleOCR setup
- `IMPLEMENTATION_SUMMARY.md` - Technical overview
- `script/detect_jersey.py --help` - Full options
