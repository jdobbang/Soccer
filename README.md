# Soccer Analysis Pipeline

YOLO 기반의 축구 경기 분석 시스템입니다. 선수 탐지, 추적, 유니폼 인식 및 활동 구간 추출을 통해 종합적인 경기 분석을 제공합니다.

## 프로젝트 구조

```
Soccer/
├── detection.py                 # YOLO 기반 선수 탐지
├── yolo11x.pt                   # YOLO11x 모델 가중치
├── run_full_pipeline.py         # 4단계 추적 파이프라인 (SORT + Re-ID)
├── visualize_tracking.py        # 추적 결과 시각화
├── results/                     # 탐지 결과 디렉토리
└── README.md
```

## 설치 및 사용

### 1. 의존성 설치

```bash
pip install ultralytics opencv-python numpy pandas scikit-learn
```

### 2. 기본 실행 흐름

```bash
# 1단계: 선수 탐지
python detection.py --input video.mp4 --output results/

# 2단계: 선수 추적 및 Re-ID
python run_full_pipeline.py \
  --detections results/detections.csv \
  --frames-dir results/frames \
  --output-dir tracking_results/

# 3단계: 결과 시각화
python visualize_tracking.py tracking_results/step4_post_interpolated.csv \
  --frames-dir results/frames \
  --output-dir tracking_results/visualization
```

---

## 핵심 기능

### 1️⃣ 선수 탐지 (Player Detection)

**detection.py** - YOLO11x 기반 실시간 선수 탐지

- 동영상 프레임에서 선수 위치 탐지
- 각 선수에 대해 bounding box 및 신뢰도 점수 제공
- CSV 형식으로 탐지 결과 저장

---

### 2️⃣ 선수 추적 (Player Tracking)

**run_full_pipeline.py** - 4단계 멀티스케일 추적 파이프라인

#### 파이프라인 구성

1. **SORT Tracking** - 단거리 tracklet 생성
2. **Tracklet Interpolation** - ID별 결손 프레임 보간
3. **Re-ID 기반 병합** - 같은 선수의 끊긴 track 통합
4. **Post-Interpolation** - 병합 후 최종 보간

#### 입력/출력

**입력:**
- `--detections` : 탐지 결과 CSV (필수)
- `--frames-dir` : 원본 프레임 이미지 폴더 (필수)
- `--output-dir` : 결과 저장 경로 (기본값: tracking_results)
- `--start-frame`, `--end-frame` : 처리 프레임 범위
- `--max-age`, `--min-hits` : SORT 파라미터

**출력:**
```
tracking_results/
├── step1_sort_raw.csv              # SORT 초기 추적
├── step2_interpolated.csv          # 보간 후 추적
├── step3_reid_merged.csv           # Re-ID 병합 후
├── step4_post_interpolated.csv     # 최종 추적 결과 ⭐
└── reid_features.pkl               # Re-ID 특징 벡터
```

#### 실행 예시

```bash
python run_full_pipeline.py \
  --detections detection_results/detections.csv \
  --frames-dir detection_results/frames \
  --output-dir tracking_results \
  --start-frame 0 \
  --end-frame 1000
```

---

### 3️⃣ 유니폼 분류 (Uniform Classification)

각 추적된 선수의 유니폼 색상/팀 분류 (구현 필요)

**목표:**
- 선수 bounding box에서 유니폼 색상 추출
- 팀 소속 분류 (Team A / Team B)
- 추적 결과에 팀 정보 통합

**제안 방식:**
- K-means 클러스터링으로 지배적 색상 추출
- 색상 거리 기반 팀 분류 또는 CNN 분류기 사용

---

### 4️⃣ 저지 탐지 (Jersey Detection)

선수 유니폼 번호 인식 (구현 필요)

**목표:**
- 선수 저지 이미지에서 번호 추출
- OCR 또는 숫자 분류 모델 활용
- 선수 고유 식별

**제안 방식:**
- EasyOCR 또는 PaddleOCR 사용
- 또는 숫자 분류 YOLO 모델 학습

---

### 5️⃣ 출현 구간 추출 (Appearance Duration Extraction)

각 선수의 활동 시간대 추출

**목표:**
- 각 track ID별로 첫 등장 프레임과 마지막 등장 프레임 기록
- 활동 시간대 요약 (시작 시간, 종료 시간, 총 플레이 시간)
- 교체 또는 부상으로 인한 퇴장 분석

**출력 예시:**
```
player_id | team | jersey_number | first_frame | last_frame | duration | appearance_count
    1     |  A   |      10       |     0       |    2500    |   2500   |      2345
    2     |  B   |       7       |    150      |    2400    |   2250   |      2100
```

---

### ⚽ 결과 시각화 (Visualization)

**visualize_tracking.py** - 추적 결과를 프레임 이미지에 렌더링

**기능:**
- 선수별 bounding box 그리기
- Track ID 표시
- 선택적으로 팀 색상 / 저지 번호 표시

**입력:**
- 추적 결과 CSV (예: step4_post_interpolated.csv)
- 원본 프레임 이미지 폴더

**출력:**
- 주석이 추가된 프레임 이미지들

**실행 예시:**

```bash
python visualize_tracking.py \
  tracking_results/step4_post_interpolated.csv \
  --frames-dir detection_results/frames \
  --output-dir tracking_results/visualization
```

#### 예시 결과

**선수 탐지 및 추적 결과:**

![Player Detection Example](asset/frame_024963_annotated.jpg)

**유니폼 번호(NO.10) 추적 프레임 분포:**

![Jersey Number Tracking Distribution](asset/유니폼_등번호_추적_프레임_분포(NO.10).png)

---

## 확장 기능 (Future Work)

- [ ] 유니폼 색상 기반 팀 분류
- [ ] OCR을 이용한 저지 번호 인식
- [ ] 선수 활동 구간 통계 분석
- [ ] 선수 포지션 추정
- [ ] 공 추적 (Ball Detection & Tracking)
- [ ] 골장 인식 (Pitch Detection)
- [ ] 3D 궤적 재구성

---

## 데이터 형식

### 탐지 결과 CSV (detections.csv)

```
frame_id,x1,y1,x2,y2,confidence,class
0,100,150,200,350,0.95,person
0,400,100,550,400,0.92,person
1,105,155,205,355,0.94,person
...
```

### 추적 결과 CSV (step4_post_interpolated.csv)

```
frame_id,track_id,x1,y1,x2,y2,confidence
0,1,100,150,200,350,0.95
0,2,400,100,550,400,0.92
1,1,105,155,205,355,0.94
1,2,405,105,555,405,0.91
...
```

---

## 성능 최적화

- **멀티프로세싱** : 대규모 영상 처리 시 병렬 처리 구현
- **배치 처리** : 프레임 배치로 추론 성능 향상
- **특징 캐싱** : Re-ID 특징 사전 계산 및 저장

---
