#!/usr/bin/env python3
"""
Shoe Class Pseudo Label Generation
===================================

Person detection + Pose estimation을 활용하여
발 영역에 대한 shoe class pseudo label을 자동 생성합니다.

Ball detection 모델의 false positive (발 → ball 오인식)을 줄이기 위해
ball class에 shoe class를 추가하는 2-class detection 데이터셋을 생성합니다.

사용 모델:
- YOLO11x: Person detection
- YOLO11x-pose: Ankle keypoint detection
"""

import argparse
import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
import random


# COCO Keypoint 인덱스 (YOLO11-pose)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16


def normalize_bbox(x1: float, y1: float, x2: float, y2: float,
                   img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    픽셀 좌표를 YOLO 정규화 형식으로 변환

    Args:
        x1, y1, x2, y2: 픽셀 좌표 (top-left, bottom-right)
        img_width, img_height: 이미지 크기

    Returns:
        (cx, cy, w, h): 정규화된 center x, y, width, height (0-1 범위)
    """
    cx = ((x1 + x2) / 2) / img_width
    cy = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    # 범위 제한
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    w = max(0, min(1, w))
    h = max(0, min(1, h))

    return cx, cy, w, h


def generate_shoe_bbox(ankle_x: float, ankle_y: float,
                       img_height: int, img_width: int,
                       person_bbox: Optional[Dict] = None,
                       bbox_size_ratio: float = 0.1) -> Optional[Tuple[float, float, float, float]]:
    """
    Ankle keypoint를 기반으로 shoe bounding box 생성
    Person bbox 크기에 비례하여 동적으로 조정

    Args:
        ankle_x, ankle_y: ankle keypoint 좌표 (픽셀)
        img_height, img_width: 이미지 크기
        person_bbox: person detection bbox (x1, y1, x2, y2) - 동적 크기 계산용
        bbox_size_ratio: person bbox 크기 대비 shoe bbox 크기 비율 (기본값 0.1 = 10%)

    Returns:
        (cx, cy, w, h) YOLO 정규화 형식, 또는 None
    """
    # Person bbox 기반으로 동적 크기 계산
    if person_bbox is not None:
        person_w = person_bbox['x2'] - person_bbox['x1']
        person_h = person_bbox['y2'] - person_bbox['y1']
        # Person bbox의 10%를 기준으로 하되, 더 큰 값 선택 (발이 person bbox보다 작을 수 있음)
        bbox_size = max(person_w * bbox_size_ratio, person_h * bbox_size_ratio) * 1.2
    else:
        bbox_size = 70  # fallback

    half_size = bbox_size / 2

    # Ankle 기준으로 bbox 생성 (위쪽 30%, 아래쪽 70%)
    x1 = max(0, ankle_x - half_size)
    y1 = max(0, ankle_y - bbox_size * 0.3)
    x2 = min(img_width, ankle_x + half_size)
    y2 = min(img_height, ankle_y + bbox_size * 0.7)

    # 유효성 검사 (최소 크기)
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return None

    return normalize_bbox(x1, y1, x2, y2, img_width, img_height)


def extract_person_bboxes(image: np.ndarray, person_model: YOLO,
                          conf_threshold: float = 0.1) -> List[Dict]:
    """
    이미지에서 사람 감지

    Returns:
        List of dicts: {x1, y1, x2, y2, conf, class_id}
    """
    results = person_model.predict(image, conf=conf_threshold, verbose=False)

    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            # YOLO coco는 person class = 0
            if cls_id == 0:
                detections.append({
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'conf': conf
                })

    return detections


def extract_ankle_keypoints(person_crop: np.ndarray,
                            pose_model: YOLO,
                            person_bbox: Dict,
                            crop_info: Dict,
                            keypoint_conf_threshold: float = 0.1) -> List[Tuple[float, float, float]]:
    """
    Person crop에서 ankle keypoint 추출

    Returns:
        List of (x, y, conf) in original image coordinates
    """
    if person_crop.size == 0:
        return []

    results = pose_model.predict(person_crop, conf=0.1, verbose=False)

    ankles = []
    if len(results) > 0 and results[0].keypoints is not None:
        keypoints_data = results[0].keypoints

        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return ankles

        # 가장 큰 detection (가장 가능성 높은 사람) 선택
        best_idx = 0
        best_area = 0

        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            for idx in range(len(boxes)):
                box = boxes[idx].xyxy[0].cpu().numpy()
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > best_area:
                    best_area = area
                    best_idx = idx

        kpts_xy = keypoints_data.xy[best_idx].cpu().numpy()  # (17, 2)
        kpts_conf = keypoints_data.conf[best_idx].cpu().numpy() if keypoints_data.conf is not None else np.ones(17)

        # Left ankle
        left_ankle_x = float(kpts_xy[LEFT_ANKLE_IDX][0])
        left_ankle_y = float(kpts_xy[LEFT_ANKLE_IDX][1])
        left_conf = float(kpts_conf[LEFT_ANKLE_IDX])

        # Right ankle
        right_ankle_x = float(kpts_xy[RIGHT_ANKLE_IDX][0])
        right_ankle_y = float(kpts_xy[RIGHT_ANKLE_IDX][1])
        right_conf = float(kpts_conf[RIGHT_ANKLE_IDX])

        # 원본 이미지 좌표로 변환
        for ankle_x, ankle_y, conf in [
            (left_ankle_x, left_ankle_y, left_conf),
            (right_ankle_x, right_ankle_y, right_conf)
        ]:
            if conf > keypoint_conf_threshold:
                orig_x = ankle_x + crop_info['offset_x']
                orig_y = ankle_y + crop_info['offset_y']
                ankles.append((orig_x, orig_y, conf))

    return ankles


def crop_person_bbox(image: np.ndarray, bbox: Dict, padding: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """
    Person bbox 주변을 crop

    Returns:
        (cropped_image, crop_info)
    """
    h, w = image.shape[:2]

    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

    # Padding 적용
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * padding
    pad_y = box_h * padding

    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(w, int(x2 + pad_x))
    crop_y2 = min(h, int(y2 + pad_y))

    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    crop_info = {
        'offset_x': crop_x1,
        'offset_y': crop_y1,
        'crop_w': crop_x2 - crop_x1,
        'crop_h': crop_y2 - crop_y1
    }

    return cropped, crop_info


def add_padding_to_bbox(cx: float, cy: float, w: float, h: float,
                       padding_ratio: float = 0.1) -> Tuple[float, float, float, float]:
    """
    YOLO 정규화 좌표 bbox에 padding 추가

    Args:
        cx, cy, w, h: 정규화된 center x, y, width, height (0-1 범위)
        padding_ratio: padding 비율 (기본값 0.1 = 10%)

    Returns:
        (cx, cy, w, h) padding이 적용된 정규화 좌표
    """
    # Width와 height에 padding 추가
    new_w = w * (1 + padding_ratio)
    new_h = h * (1 + padding_ratio)

    # 범위 제한 (0-1)
    new_w = min(1.0, new_w)
    new_h = min(1.0, new_h)

    return cx, cy, new_w, new_h


def process_image(image_path: str, label_path: str,
                  person_model: YOLO, pose_model: YOLO,
                  person_conf: float = 0.1,
                  keypoint_conf: float = 0.5,
                  bbox_size_ratio: float = 0.1,
                  bottom_ratio: float = 0.3,
                  padding_ratio: float = 0.1) -> Tuple[List[str], List[Dict]]:
    """
    이미지 처리: ball label + shoe pseudo label 생성
    모든 bbox에 10% padding 적용

    Returns:
        Tuple of (label_lines, person_bboxes)
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        return [], []

    img_h, img_w = image.shape[:2]

    # 기존 ball label 로드 및 padding 적용
    ball_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    ball_labels.append(line)
                    continue

                class_id = parts[0]
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                # Padding 적용
                cx, cy, w, h = add_padding_to_bbox(cx, cy, w, h, padding_ratio)

                # Padding 적용된 label 저장
                ball_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # Person detection
    person_bboxes = extract_person_bboxes(image, person_model, conf_threshold=person_conf)

    shoe_labels = []

    # 각 사람에 대해 pose estimation 수행
    for person_bbox in person_bboxes:
        # Person crop
        person_crop, crop_info = crop_person_bbox(image, person_bbox, padding=0.1)

        # Ankle keypoint 추출
        ankles = extract_ankle_keypoints(person_crop, pose_model, person_bbox,
                                         crop_info, keypoint_conf_threshold=keypoint_conf)

        # 이미지 하단 30% 영역에 있는 ankle만 사용
        for ankle_x, ankle_y, ankle_conf in ankles:
            # Y 좌표 기반 필터링 (하단 bottom_ratio 영역만)
            if ankle_y > img_h * (1 - bottom_ratio):
                shoe_bbox = generate_shoe_bbox(ankle_x, ankle_y, img_h, img_w,
                                              person_bbox=person_bbox,
                                              bbox_size_ratio=bbox_size_ratio)
                if shoe_bbox is not None:
                    cx, cy, w, h = shoe_bbox

                    # Padding 적용
                    cx, cy, w, h = add_padding_to_bbox(cx, cy, w, h, padding_ratio)

                    # Class 1 = shoe
                    shoe_labels.append(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # Ball label + shoe pseudo label 병합
    all_labels = ball_labels + shoe_labels

    return all_labels, person_bboxes


def process_dataset(input_dir: str, output_dir: str,
                    person_model: YOLO, pose_model: YOLO,
                    person_conf: float = 0.1,
                    keypoint_conf: float = 0.5,
                    bbox_size_ratio: float = 0.1,
                    bottom_ratio: float = 0.3,
                    padding_ratio: float = 0.1,
                    visualize: bool = False,
                    visualization_count: int = 50):
    """
    전체 dataset 처리: train, valid, test
    """
    splits = ['train', 'valid', 'test']

    # 출력 디렉토리 준비
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        input_images_dir = os.path.join(input_dir, split, 'images')
        input_labels_dir = os.path.join(input_dir, split, 'labels')

        output_images_dir = os.path.join(output_dir, split, 'images')
        output_labels_dir = os.path.join(output_dir, split, 'labels')

        # 디렉토리 생성
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        if not os.path.exists(input_images_dir):
            print(f"Warning: {input_images_dir} does not exist, skipping...")
            continue

        print(f"\nProcessing {split} split...")

        # 이미지 파일 목록
        image_files = sorted([f for f in os.listdir(input_images_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

        # 시각화용 샘플 선택
        viz_samples = set()
        if visualize and split == 'train':
            viz_samples = set(random.sample(image_files, min(visualization_count, len(image_files))))

        for image_file in tqdm(image_files, desc=f"Processing {split} images"):
            # 원본 이미지 경로
            image_path = os.path.join(input_images_dir, image_file)
            label_name = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(input_labels_dir, label_name)

            # 이미지 처리
            try:
                label_lines, person_bboxes = process_image(
                    image_path, label_path,
                    person_model, pose_model,
                    person_conf=person_conf,
                    keypoint_conf=keypoint_conf,
                    bbox_size_ratio=bbox_size_ratio,
                    bottom_ratio=bottom_ratio,
                    padding_ratio=padding_ratio
                )

                # 이미지 복사
                output_image_path = os.path.join(output_images_dir, image_file)
                shutil.copy2(image_path, output_image_path)

                # Label 저장
                output_label_path = os.path.join(output_labels_dir, label_name)
                with open(output_label_path, 'w') as f:
                    for line in label_lines:
                        f.write(line + '\n')

                # 시각화
                if image_file in viz_samples:
                    visualize_labels(image_path, label_lines, output_dir, image_file, person_bboxes=person_bboxes)

            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue

    print(f"\nDataset processing completed!")
    print(f"Output directory: {output_dir}")


def visualize_labels(image_path: str, label_lines: List[str],
                     output_dir: str, image_file: str,
                     person_bboxes: Optional[List[Dict]] = None):
    """
    생성된 label을 이미지에 시각화
    Person bbox와 Detection label (ball, shoe)를 함께 표시
    """
    image = cv2.imread(image_path)
    if image is None:
        return

    img_h, img_w = image.shape[:2]

    # Person bbox 시각화 (초록색)
    if person_bboxes:
        for person_bbox in person_bboxes:
            x1 = int(person_bbox['x1'])
            y1 = int(person_bbox['y1'])
            x2 = int(person_bbox['x2'])
            y2 = int(person_bbox['y2'])
            conf = person_bbox.get('conf', 0)

            # Person bbox: 초록색
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'person ({conf:.2f})',
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Ball (class 0): 빨강, Shoe (class 1): 파랑
    colors = {0: (0, 0, 255), 1: (255, 0, 0)}
    class_names = {0: 'ball', 1: 'shoe'}

    for line in label_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # 정규화 좌표 -> 픽셀 좌표
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)

        color = colors.get(class_id, (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, class_names.get(class_id, 'unknown'),
                   (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 시각화 디렉토리 생성 및 저장
    viz_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(viz_dir, exist_ok=True)

    output_path = os.path.join(viz_dir, image_file)
    cv2.imwrite(output_path, image)


def create_data_yaml(output_dir: str):
    """
    2-class dataset을 위한 data.yaml 생성
    """
    yaml_content = """names:
- ball
- shoe
nc: 2
train: train/images
val: valid/images
test: test/images
"""

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate shoe pseudo labels for ball detection dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 설정으로 실행
  python generate_shoe_pseudo_labels.py

  # 커스텀 경로 및 파라미터
  python generate_shoe_pseudo_labels.py \\
    --input_dir /path/to/ball/merged \\
    --output_dir /path/to/ball_shoe/merged \\
    --bbox_size 80 \\
    --visualize

  # 시각화 샘플 개수 지정
  python generate_shoe_pseudo_labels.py --visualize --viz_count 100
        """
    )

    parser.add_argument('--input_dir', type=str,
                       default='/workspace/Soccer/dataset/ball/merged',
                       help='Input dataset directory (default: ball dataset)')
    parser.add_argument('--output_dir', type=str,
                       default='/workspace/Soccer/dataset/ball_shoe/merged',
                       help='Output dataset directory (default: ball_shoe dataset)')
    parser.add_argument('--person_model', type=str, default='yolo11x.pt',
                       help='YOLO person detection model (default: yolo11x.pt)')
    parser.add_argument('--pose_model', type=str, default='yolo11x-pose.pt',
                       help='YOLO pose estimation model (default: yolo11x-pose.pt)')
    parser.add_argument('--person_conf', type=float, default=0.7,
                       help='Person detection confidence threshold (default: 0.7)')
    parser.add_argument('--keypoint_conf', type=float, default=0.1,
                       help='Keypoint confidence threshold (default: 0.1)')
    parser.add_argument('--bbox_size_ratio', type=float, default=0.1,
                       help='Shoe bbox size ratio to person bbox (default: 0.1 = 10% of person bbox)')
    parser.add_argument('--bottom_ratio', type=float, default=1,
                       help='Image bottom ratio for ankle filtering (default: 0.0)')
    parser.add_argument('--padding_ratio', type=float, default=0.1,
                       help='Padding ratio for bbox (default: 0.1 = 10% padding)')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization of generated labels')
    parser.add_argument('--viz_count', type=int, default=10000,
                       help='Number of samples to visualize (default: 10000)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu, default: cuda)')

    args = parser.parse_args()

    print("=" * 70)
    print("Shoe Class Pseudo Label Generation")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Person model: {args.person_model}")
    print(f"Pose model: {args.pose_model}")
    print(f"Bbox size ratio: {args.bbox_size_ratio} (person bbox 대비)")
    print(f"Padding ratio: {args.padding_ratio} (10% padding added to bbox)")
    print(f"Bottom ratio: {args.bottom_ratio}")
    print(f"Visualization: {args.visualize}")
    print("=" * 70)

    # 모델 로드
    print("\nLoading models...")
    person_model = YOLO(args.person_model).to(args.device)
    pose_model = YOLO(args.pose_model).to(args.device)
    print("Models loaded successfully!")

    # Dataset 처리
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        person_model=person_model,
        pose_model=pose_model,
        person_conf=args.person_conf,
        keypoint_conf=args.keypoint_conf,
        bbox_size_ratio=args.bbox_size_ratio,
        bottom_ratio=args.bottom_ratio,
        padding_ratio=args.padding_ratio,
        visualize=args.visualize,
        visualization_count=args.viz_count
    )

    # data.yaml 생성
    create_data_yaml(args.output_dir)

    print("\n" + "=" * 70)
    print("Successfully completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
