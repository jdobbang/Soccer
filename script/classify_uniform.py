#!/usr/bin/env python3
"""
유니폼 색상 분류
==============

detection CSV의 bbox 상체 영역에서 유니폼 색상을 분류
현재 지원: 주황색(orange), 검은색(black)
배치 처리로 최적화됨
"""

import argparse
import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.csv_handler import CSVHandler
from utils.color_analyzer import ColorAnalyzer


def load_images_batch(frame_data, image_folder, sorted_frames, batch_size=32):
    """
    배치 단위로 이미지 로드 (멀티스레딩)

    Args:
        frame_data: 프레임 데이터
        image_folder: 이미지 폴더 경로
        sorted_frames: 정렬된 프레임 번호 리스트
        batch_size: 배치 크기

    Yields:
        (batch_frames, batch_images) 튜플
    """
    for i in range(0, len(sorted_frames), batch_size):
        batch_frames = sorted_frames[i:i+batch_size]
        batch_images = {}

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_frame = {}

            for frame_num in batch_frames:
                detections = frame_data[frame_num]
                image_name = detections[0]['image_name']
                image_path = os.path.join(image_folder, image_name)

                if os.path.exists(image_path):
                    future = executor.submit(cv2.imread, image_path)
                    future_to_frame[future] = (frame_num, image_name)

            for future in as_completed(future_to_frame):
                frame_num, image_name = future_to_frame[future]
                try:
                    image = future.result()
                    if image is not None:
                        batch_images[frame_num] = image
                except Exception:
                    pass

        yield batch_frames, batch_images


def classify_colors_batch(
    detections,
    image,
    color_analyzer,
    upper_ratio=0.5
):
    """
    한 프레임의 모든 detection에 대해 색상 분류

    Args:
        detections: detection 리스트
        image: 프레임 이미지
        color_analyzer: 색상 분석기
        upper_ratio: 상체 영역 비율

    Returns:
        [(color, color_conf), ...] 리스트
    """
    results = []
    for det in detections:
        color, color_conf = color_analyzer.analyze_region(image, det, upper_ratio)
        results.append((color, color_conf))
    return results


def classify_colors(
    detection_csv: str,
    image_folder: str,
    output_csv: str,
    image_pattern: str = None,
    upper_ratio: float = 0.5,
    batch_size: int = 32
):
    """
    시퀀스 전체에 대해 유니폼 색상 분류 (배치 처리)

    Args:
        detection_csv: detection CSV 경로
        image_folder: 이미지 폴더 경로
        output_csv: 출력 CSV 경로
        image_pattern: 이미지 이름 패턴 (e.g., "frame_{:06d}.jpg")
        upper_ratio: 상체 영역 비율 (기본값: 0.5)
        batch_size: 배치 크기 (기본값: 32)
    """
    # 데이터 로드
    print(f"Loading detection data from: {detection_csv}")
    frame_data = CSVHandler.load_detection_csv(detection_csv, image_pattern)

    if not frame_data:
        print("No detection data found!")
        return

    print(f"Loaded {len(frame_data)} frames")

    # 색상 분석기 초기화
    color_analyzer = ColorAnalyzer()

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)

    # 결과 저장
    sorted_frames = sorted(frame_data.keys())
    color_counts = defaultdict(int)
    detection_index = 0

    results_buffer = []
    buffer_size = 1000  # 버퍼에 모았다가 한번에 쓰기

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame', 'image_name', 'track_id', 'x1', 'y1', 'x2', 'y2',
            'confidence', 'uniform_color', 'color_confidence'
        ])

        # 배치 단위로 이미지 로드 및 처리
        for batch_frames, batch_images in tqdm(
            load_images_batch(frame_data, image_folder, sorted_frames, batch_size),
            total=(len(sorted_frames) + batch_size - 1) // batch_size,
            desc="Classifying (batch)"
        ):
            for frame_num in batch_frames:
                if frame_num not in batch_images:
                    continue

                image = batch_images[frame_num]
                detections = frame_data[frame_num]
                image_name = detections[0]['image_name']

                # 배치 분류
                color_results = classify_colors_batch(
                    detections, image, color_analyzer, upper_ratio
                )

                for det, (color, color_conf) in zip(detections, color_results):
                    color_counts[color] += 1

                    # track_id가 -1이면 detection index 사용
                    track_id = det['track_id'] if det['track_id'] != -1 else detection_index
                    detection_index += 1

                    # 버퍼에 추가
                    results_buffer.append([
                        frame_num,
                        image_name,
                        track_id,
                        int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2']),
                        f"{det['confidence']:.4f}",
                        color,
                        f"{color_conf:.3f}"
                    ])

                    # 버퍼 플러시
                    if len(results_buffer) >= buffer_size:
                        writer.writerows(results_buffer)
                        f.flush()
                        results_buffer = []

        # 남은 버퍼 플러시
        if results_buffer:
            writer.writerows(results_buffer)
            f.flush()

    print(f"\nResults saved to: {output_csv}")
    print(f"Color distribution:")
    for color, count in sorted(color_counts.items()):
        print(f"  {color}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Classify uniform colors from detection results (Batch Processing)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/classify_uniform.py --detection_csv results/player/yolo11x/test.csv --image_folder original_frames/ --output_csv results/player/yolo11x/test_color.csv
  python script/classify_uniform.py --detection_csv results/player/yolo11x/test.csv --image_folder original_frames/ --output_csv results/player/yolo11x/test_color.csv --batch_size 64
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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기 (default: 32)')

    args = parser.parse_args()

    classify_colors(
        detection_csv=args.detection_csv,
        image_folder=args.image_folder,
        output_csv=args.output_csv,
        image_pattern=args.image_pattern,
        upper_ratio=args.upper_ratio,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
