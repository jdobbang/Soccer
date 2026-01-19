# Soccer Analysis Pipeline

YOLO 기반의 축구 경기 분석 시스템입니다. 선수 탐지, 추적, 저지 번호 인식, 공 감지 등을 지원합니다.

## 분석 결과 시각화

**선수 탐지 및 추적:**

![Player Detection Example](asset/frame_024963_annotated.jpg)

**저지 번호(NO.10) 추적 분포:**

![Jersey Number Tracking Distribution](asset/유니폼_등번호_추적_프레임_분포(NO.10).png)

## 프로젝트 구조

```
Soccer/
├── script/                    # 분석 스크립트
│   ├── tools/                 # 탐지/추적/필터링 도구
│   ├── visualizer/            # 시각화 모듈
│   ├── preprocessor/          # 전처리 모듈
│   └── utils/                 # 유틸리티
├── asset/                     # 리소스 파일 (Git 추적 ✓)
├── ball_dataset/              # 공 감지 데이터셋
├── results/                   # 분석 결과 (Git 제외)
├── tracking_results/          # 추적 결과 (Git 제외)
├── vis_result/                # 시각화 결과 (Git 제외)
├── original_frames/           # 원본 프레임 (Git 제외)
└── doc/                       # 문서
```

## 설치

```bash
pip install ultralytics opencv-python numpy pandas scikit-learn easyocr
```

## 빠른 시작

### 선수 분석 파이프라인

```bash
# 1단계: 선수 탐지
python script/tools/detection.py --input video.mp4 --output results/

# 2단계: 선수 추적 (SORT + Re-ID)
python script/tools/track.py \
  --detections results/detections.csv \
  --frames-dir results/frames \
  --output-dir tracking_results/

# 3단계: 팀 색상 분류
python script/tools/classify_uniform.py \
  --detection_csv results/detections.csv \
  --image_folder results/frames \
  --output_csv results/uniform_color.csv

# 4단계: 저지 번호 인식
python script/tools/detect_jersey.py \
  --color_csv results/uniform_color.csv \
  --frames_dir results/frames \
  --output_dir results/

# 5단계: 특정 선수 필터링 (예: 번호 10)
python script/tools/filter_tracklets.py \
  --tracking tracking_results/step4_post_interpolated.csv \
  --jersey results/jersey_numbers_detailed.csv \
  --number 10 \
  --output tracking_results/player_10_filtered.csv

# 6단계: 시각화
python script/visualizer/visualize_tracking.py \
  tracking_results/player_10_filtered.csv \
  --frames-dir results/frames \
  --output-dir vis_result/player_10
```

### 공 감지 및 하이라이트 추출 파이프라인

```bash
# 1단계: 공 탐지
python script/tools/ball_detection.py \
  --input video.mp4 \
  --model ball_yolo11x \
  --output results/ball_detection.csv

# 2단계: 공-선수 근접 구간 추출
python script/tools/extract_ball_player_proximity.py \
  --player-csv tracking_results/player_10_filtered.csv \
  --ball-csv results/ball_detection.csv \
  --distance-threshold 100 \
  --output tracking_results/player_10_ball_proximity.csv

# 3단계: 하이라이트 시각화
python script/visualizer/visualize_proximity.py \
  tracking_results/player_10_ball_proximity.csv \
  --frames-dir results/frames \
  --output-dir vis_result/highlights_player_10
```

## 주요 기능

### 선수 분석
| 기능 | 설명 | 입력 | 출력 |
|------|------|------|------|
| **Player Detection** | YOLO 기반 선수 탐지 | 비디오 | detections.csv |
| **Player Tracking** | 4단계 추적 파이프라인 | detections.csv | step4_post_interpolated.csv |
| **Uniform Classification** | HSV 기반 팀 색상 분류 | 프레임 + 탐지 | uniform_color.csv |
| **Jersey Recognition** | OCR 기반 저지 번호 인식 | 프레임 + 색상 | jersey_numbers_*.csv |
| **Track Filtering** | 저지 번호 기반 필터링 | 추적 + 저지 | player_X_filtered.csv |

### 공 감지 및 하이라이트
| 기능 | 설명 | 입력 | 출력 |
|------|------|------|------|
| **Ball Detection** | YOLO 기반 공 감지 | 비디오 | ball_detection.csv |
| **Proximity Extraction** | 공-선수 근접 구간 추출 | 추적 + 공 | proximity_segments.csv |
| **Highlight Visualization** | 하이라이트 구간 시각화 | CSV + 프레임 | 주석된 이미지 |

### 시각화
| 기능 | 설명 | 입력 | 출력 |
|------|------|------|------|
| **Visualization** | 추적 결과 시각화 | CSV + 프레임 | 주석된 이미지 |
| **Highlight Rendering** | 하이라이트 렌더링 | 근접 CSV | 하이라이트 영상 |

## 주요 파일

**선수 분석:**
- `script/tools/detection.py` - 선수 탐지 (YOLO11x)
- `script/tools/track.py` - 추적 파이프라인 (SORT + Re-ID)
- `script/tools/classify_uniform.py` - 팀 색상 분류
- `script/tools/detect_jersey.py` - 저지 번호 인식 (OCR)
- `script/tools/filter_tracklets.py` - Track 필터링

**공 감지 및 하이라이트:**
- `script/tools/ball_detection.py` - 공 탐지 (YOLO)
- `script/tools/extract_ball_player_proximity.py` - 공-선수 근접 추출
- `script/visualizer/visualize_proximity.py` - 하이라이트 시각화

**시각화:**
- `script/visualizer/visualize_tracking.py` - 추적 결과 시각화

## 출력 형식

**detections.csv:**
```csv
frame_id,x1,y1,x2,y2,confidence
0,100,150,200,350,0.95
```

**step4_post_interpolated.csv:**
```csv
frame_id,track_id,x1,y1,x2,y2,confidence
0,1,100,150,200,350,0.95
```

**jersey_numbers_consolidated.csv:**
```csv
track_id,jersey_number,consolidated_confidence,detection_count
1,10,85.234,12
```

## Git 정책

**저장됨:**
- `script/` - 모든 분석 스크립트
- `asset/` - 이미지, 리소스 파일 ✓
- `ball_dataset/` - 데이터셋 구조

**제외됨:**
- `*.pt` - 모델 가중치
- `*.mp4` - 비디오 파일
- `results/`, `tracking_results/`, `vis_result/` - 생성된 결과
- `original_frames/` - 원본 프레임
- `__pycache__/` - Python 캐시

## 라이센스

[라이센스 정보]
