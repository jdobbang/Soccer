#!/usr/bin/env python3
"""
유니폼 색상 분류
================

detection CSV의 bbox 상체 영역에서 유니폼 색상을 분류
현재 지원: 주황색(orange), 검은색(black)
"""

import argparse
import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict


# 팀 색상 정의 (HSV 범위)
# H: 0-179, S: 0-255, V: 0-255
TEAM_COLORS = {
    'orange': {
        'hsv_lower': np.array([5, 100, 100]),
        'hsv_upper': np.array([25, 255, 255]),
        'display_bgr': (0, 165, 255),  # 시각화용 BGR
    },
    'black': {
        'hsv_lower': np.array([0, 0, 0]),
        'hsv_upper': np.array([179, 255, 80]),  # 낮은 V값 = 검은색
        'display_bgr': (0, 0, 0),
    },
}

# 제외할 색상 (잔디, 피부)
EXCLUDE_COLORS = {
    'grass': {
        'hsv_lower': np.array([35, 40, 40]),
        'hsv_upper': np.array([85, 255, 255]),
    },
    'skin': {
        'hsv_lower': np.array([0, 20, 70]),
        'hsv_upper': np.array([20, 150, 255]),
    },
}


def load_detection_csv(csv_path: str, image_pattern: str = None) -> dict:
    """detection CSV 로드"""
    frame_data = defaultdict(list)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # 형식 감지
        has_object_id = 'object_id' in fieldnames
        has_image_name = 'image_name' in fieldnames

        for row in reader:
            frame = int(row['frame'])

            if has_object_id and not has_image_name:
                # detection 형식
                if image_pattern:
                    image_name = image_pattern.format(frame)
                else:
                    image_name = f"frame_{frame:06d}.jpg"
                track_id = int(row['object_id'])
            else:
                # tracking 형식
                image_name = row['image_name']
                track_id = int(row['track_id'])

            frame_data[frame].append({
                'image_name': image_name,
                'track_id': track_id,
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2']),
                'confidence': float(row['confidence'])
            })

    return frame_data


def get_upper_body_region(image: np.ndarray, bbox: dict, upper_ratio: float = 0.5) -> np.ndarray:
    """
    bbox에서 상체 영역만 추출

    Args:
        image: 원본 이미지
        bbox: bounding box (x1, y1, x2, y2)
        upper_ratio: 상단에서 사용할 비율 (0.5 = 상위 50%)

    Returns:
        상체 영역 이미지
    """
    h, w = image.shape[:2]

    x1 = max(0, int(bbox['x1']))
    y1 = max(0, int(bbox['y1']))
    x2 = min(w, int(bbox['x2']))
    y2 = min(h, int(bbox['y2']))

    box_h = y2 - y1
    upper_y2 = y1 + int(box_h * upper_ratio)

    return image[y1:upper_y2, x1:x2]


def create_exclude_mask(hsv_image: np.ndarray) -> np.ndarray:
    """잔디/피부색 제외 마스크 생성"""
    mask = np.ones(hsv_image.shape[:2], dtype=np.uint8) * 255

    for color_name, color_range in EXCLUDE_COLORS.items():
        exclude_mask = cv2.inRange(hsv_image, color_range['hsv_lower'], color_range['hsv_upper'])
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_mask))

    return mask


def classify_color(crop: np.ndarray) -> tuple:
    """
    상체 영역에서 유니폼 색상 분류

    Args:
        crop: 상체 영역 이미지 (BGR)

    Returns:
        (color_name, confidence)
    """
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return 'unknown', 0.0

    # HSV 변환
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 잔디/피부색 제외
    valid_mask = create_exclude_mask(hsv)
    valid_pixels = cv2.countNonZero(valid_mask)

    if valid_pixels < 50:  # 유효 픽셀이 너무 적으면
        return 'unknown', 0.0

    # 각 팀 색상에 대해 매칭 비율 계산
    scores = {}
    for team_name, team_color in TEAM_COLORS.items():
        color_mask = cv2.inRange(hsv, team_color['hsv_lower'], team_color['hsv_upper'])
        # 유효 영역 내에서만 계산
        color_mask = cv2.bitwise_and(color_mask, valid_mask)
        match_pixels = cv2.countNonZero(color_mask)
        scores[team_name] = match_pixels / valid_pixels

    # 가장 높은 점수의 색상 선택
    best_color = max(scores, key=scores.get)
    best_score = scores[best_color]

    # threshold 적용
    if best_score < 0.15:  # 15% 미만이면 unknown
        return 'unknown', best_score

    return best_color, best_score


def process_sequence(
    detection_csv: str,
    image_folder: str,
    output_csv: str,
    image_pattern: str = None,
    upper_ratio: float = 0.5
):
    """
    시퀀스 전체에 대해 유니폼 색상 분류

    Args:
        detection_csv: detection CSV 경로
        image_folder: 이미지 폴더 경로
        output_csv: 출력 CSV 경로
        image_pattern: 이미지 이름 패턴
        upper_ratio: 상체 영역 비율
    """
    # 데이터 로드
    print(f"Loading detection data from: {detection_csv}")
    frame_data = load_detection_csv(detection_csv, image_pattern)

    if not frame_data:
        print("No detection data found!")
        return

    print(f"Loaded {len(frame_data)} frames")

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)

    # 결과 저장
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'image_name', 'track_id', 'x1', 'y1', 'x2', 'y2',
                         'confidence', 'uniform_color', 'color_confidence'])

        sorted_frames = sorted(frame_data.keys())

        # 통계
        color_counts = defaultdict(int)

        for frame_num in tqdm(sorted_frames, desc="Classifying"):
            detections = frame_data[frame_num]
            image_name = detections[0]['image_name']
            image_path = os.path.join(image_folder, image_name)

            if not os.path.exists(image_path):
                continue

            image = cv2.imread(image_path)
            if image is None:
                continue

            for det in detections:
                # 상체 영역 추출
                upper_body = get_upper_body_region(image, det, upper_ratio)

                # 색상 분류
                color, color_conf = classify_color(upper_body)
                color_counts[color] += 1

                # 결과 저장
                writer.writerow([
                    frame_num,
                    image_name,
                    det['track_id'],
                    det['x1'], det['y1'], det['x2'], det['y2'],
                    det['confidence'],
                    color,
                    f"{color_conf:.3f}"
                ])

    print(f"\nResults saved to: {output_csv}")
    print(f"\nColor distribution:")
    for color, count in sorted(color_counts.items()):
        print(f"  {color}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Classify uniform colors from detection results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python classify_uniform_color.py --detection_csv detection_results/yolo11x/test.csv \\
                                   --image_folder images/ \\
                                   --output_csv detection_results/yolo11x/uniform_color.csv

  # 이미지 패턴 지정
  python classify_uniform_color.py --detection_csv detection_results/test.csv \\
                                   --image_folder images/ \\
                                   --output_csv uniform_color.csv \\
                                   --image_pattern "frame_{:06d}.jpg"
        """
    )

    parser.add_argument('--detection_csv', type=str, required=True,
                        help='Detection CSV 경로')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='이미지 폴더 경로')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='출력 CSV 경로')
    parser.add_argument('--image_pattern', type=str, default=None,
                        help='이미지 이름 패턴 (예: "frame_{:06d}.jpg")')
    parser.add_argument('--upper_ratio', type=float, default=0.5,
                        help='상체 영역 비율 (default: 0.5)')

    args = parser.parse_args()

    process_sequence(
        detection_csv=args.detection_csv,
        image_folder=args.image_folder,
        output_csv=args.output_csv,
        image_pattern=args.image_pattern,
        upper_ratio=args.upper_ratio
    )


if __name__ == '__main__':
    main()
