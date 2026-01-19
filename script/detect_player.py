#!/usr/bin/env python3
"""
선수 탐지 (YOLO 배치 추론)
========================

YOLO11x 모델을 사용하여 비디오에서 선수 탐지
배치 처리로 GPU 효율 최적화
"""

import argparse
import os
import csv
from utils.yolo_inference import YOLOBatchInference


def write_detection_result(writer, frame_idx: int, result):
    """탐지 결과를 CSV에 기록하는 콜백 함수"""
    boxes = result.boxes

    for box_idx, box in enumerate(boxes):
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1

        writer.writerow([
            frame_idx, box_idx, x1, y1, x2, y2, f"{conf:.4f}", width, height
        ])


def detect_players(
    video_path: str,
    model_path: str,
    output_folder: str = "results/player",
    detection_interval: int = 1,
    batch_size: int = 32
):
    """
    비디오에서 선수 탐지

    Args:
        video_path: 입력 비디오 경로
        model_path: YOLO 모델 경로
        output_folder: 출력 폴더
        detection_interval: 탐지 간격 (기본값: 1)
        batch_size: 배치 크기 (기본값: 32)
    """
    # 경로 설정
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_path = os.path.join(output_folder, model_name)
    csv_path = os.path.join(output_path, f"{video_name}.csv")

    # YOLO 추론 초기화
    inference = YOLOBatchInference(
        model_path=model_path,
        batch_size=batch_size,
        conf_threshold=0.1,
        classes=[0]  # 0 = person
    )

    # 비디오 처리
    inference.process_video(
        video_path=video_path,
        csv_path=csv_path,
        detection_interval=detection_interval,
        result_writer=write_detection_result,
        header=['frame', 'object_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'width', 'height']
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detect players in video using YOLO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_player.py yolo11x.pt test.mp4
        """
    )
    parser.add_argument("model_name", nargs="?", default='asset/yolo11x.pt', help="YOLO model path (e.g., yolo11x.pt)")
    parser.add_argument("video_file", nargs="?", default="test.mp4", help="Input video path")
    parser.add_argument("--interval", type=int, default=1, help="Detection interval (frames)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--output", type=str, default="results/player", help="Output folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect_players(
        video_path=args.video_file,
        model_path=args.model_name,
        output_folder=args.output,
        detection_interval=args.interval,
        batch_size=args.batch
    )
