# SnapGoal_Test

A YOLO-based object detection project for soccer analysis.

## Project Structure

- `detection.py` - Main detection script
- `yolo11x.pt` - YOLO11x model weights
- `results/` - Detection results directory

## Setup

1. Install dependencies:
```bash
pip install ultralytics opencv-python
```

2. Run detection:
```bash
python detection.py
```

---

## Soccer Tracking Pipeline

### 1. run_full_pipeline.py

4-Stage Soccer Tracking Pipeline with Re-ID:

- **기능:**
    1. SORT Tracking (short tracklets)
    2. Tracklet Interpolation (ID별 결손 프레임 보간)
    3. Re-ID 기반 트랙 병합 (ID 통합)
    4. 병합 후 추가 보간
    - 각 단계별 중간 CSV 결과 저장

- **입력:**
    - `--detections` : 탐지 결과 CSV 파일 경로 (필수)
    - `--frames-dir` : 프레임 이미지 폴더 경로 (필수)
    - `--output-dir` : 결과 저장 폴더 (기본값: tracking_results)
    - 기타 파라미터: 프레임 범위, SORT/보간/ReID 관련 옵션 등

- **출력:**
    - 단계별 트래킹 결과 CSV (output-dir 하위)
        - step1_sort_raw.csv
        - step2_interpolated.csv
        - step3_reid_merged.csv
        - step4_post_interpolated.csv
    - Re-ID feature pickle 파일 (reid_features.pkl)

- **실행 예시:**
```bash
python run_full_pipeline.py \
  --detections detection_results/yolo11x/detections.csv \
  --frames-dir detection_results/yolo11x/detected_frames \
  --output-dir tracking_results
```


### 2. visualize_tracking.py

- **기능:**
    - 트래킹 결과 CSV를 읽어 프레임별로 bounding box와 track ID를 시각화하여 이미지로 저장

- **입력:**
    - tracking_csv : 트래킹 결과 CSV 파일 (예: step4_post_interpolated.csv)
    - --frames-dir : 원본 프레임 이미지 폴더
    - --output-dir : 시각화 결과 저장 폴더 (기본값: tracking_results/tracking_visualization)

- **출력:**
    - bounding box와 ID가 그려진 프레임 이미지들 (output-dir 하위)

- **실행 예시:**
```bash
python visualize_tracking.py tracking_results/step4_post_interpolated.csv \
  --frames-dir detection_results/yolo11x/detected_frames \
  --output-dir tracking_results/tracking_visualization
```

---
