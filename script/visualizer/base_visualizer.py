"""
기본 시각화 모듈 (refactored)
===========================

모든 시각화 도구의 기본 클래스
utils.visualization.Visualizer를 기반으로 함
"""

import cv2
import os
import csv
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.visualization import Visualizer
from utils.csv_handler import CSVHandler


class BaseVisualizer:
    """모든 시각화 도구의 기본 클래스"""

    def __init__(
        self,
        output_folder: str = "visualized_frames",
        font_size: float = 0.5,
        line_width: int = 2
    ):
        """
        Args:
            output_folder: 출력 폴더
            font_size: 폰트 크기
            line_width: 선 두께
        """
        self.output_folder = output_folder
        self.visualizer = Visualizer(font_size=font_size, line_width=line_width)
        self.frame_count = 0

        os.makedirs(output_folder, exist_ok=True)

    def _get_frame_path(self, frames_dir: str, frame_num: int) -> Optional[str]:
        """프레임 경로 찾기"""
        from utils.tracking import FramePathHandler
        return FramePathHandler.get_frame_path(frames_dir, frame_num)

    def _load_frame(self, frames_dir: str, frame_num: int) -> Optional[np.ndarray]:
        """프레임 로드"""
        frame_path = self._get_frame_path(frames_dir, frame_num)
        if frame_path is None:
            return None

        from utils.tracking import ImageProcessor
        return ImageProcessor.safe_load_image(frame_path)

    def _save_frame(self, image: np.ndarray, output_name: str):
        """프레임 저장"""
        output_path = os.path.join(self.output_folder, output_name)
        cv2.imwrite(output_path, image)

    def _add_frame_info(
        self,
        image: np.ndarray,
        frame_num: int,
        extra_info: str = ""
    ) -> np.ndarray:
        """프레임 번호 및 추가 정보 표시"""
        text = f"Frame: {frame_num}"
        if extra_info:
            text += f" | {extra_info}"

        return self.visualizer.draw_text(
            image,
            text,
            x=10, y=30,
            color=(255, 255, 255),
            font_size=0.6,
            thickness=1,
            bg_color=(0, 0, 0)
        )


class DetectionVisualizer(BaseVisualizer):
    """탐지 결과 시각화"""

    def visualize_detections(
        self,
        csv_path: str,
        frames_dir: str,
        output_prefix: str = "detection"
    ):
        """
        탐지 결과 시각화

        Args:
            csv_path: 탐지 CSV 경로
            frames_dir: 프레임 폴더 경로
            output_prefix: 출력 파일명 접두사
        """
        print(f"Visualizing detections from {csv_path}")

        frame_data = CSVHandler.load_detection_csv(csv_path)

        for frame_num in sorted(frame_data.keys()):
            frame = self._load_frame(frames_dir, frame_num)
            if frame is None:
                continue

            detections = frame_data[frame_num]

            # bbox 그리기
            for det in detections:
                x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                confidence = det['confidence']
                label = f"conf:{confidence:.2f}"

                frame = self.visualizer.draw_bbox(
                    frame, x1, y1, x2, y2,
                    color=(0, 255, 0),
                    thickness=2,
                    label=label
                )

            # 프레임 정보 추가
            frame = self._add_frame_info(frame, frame_num, f"Detections: {len(detections)}")

            # 저장
            output_name = f"{output_prefix}_{frame_num:06d}.jpg"
            self._save_frame(frame, output_name)

            self.frame_count += 1

        print(f"Visualized {self.frame_count} frames -> {self.output_folder}")


class TrackingVisualizer(BaseVisualizer):
    """추적 결과 시각화"""

    def visualize_tracking(
        self,
        csv_path: str,
        frames_dir: str,
        output_prefix: str = "tracking"
    ):
        """
        추적 결과 시각화

        Args:
            csv_path: 추적 CSV 경로
            frames_dir: 프레임 폴더 경로
            output_prefix: 출력 파일명 접두사
        """
        print(f"Visualizing tracking from {csv_path}")

        frame_data = CSVHandler.load_detection_csv(csv_path)

        for frame_num in sorted(frame_data.keys()):
            frame = self._load_frame(frames_dir, frame_num)
            if frame is None:
                continue

            detections = frame_data[frame_num]

            # track_id별로 그리기
            for det in detections:
                x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                track_id = int(det['track_id'])

                frame = self.visualizer.draw_track(
                    frame, track_id, x1, y1, x2, y2,
                    thickness=2
                )

            # 프레임 정보 추가
            frame = self._add_frame_info(frame, frame_num, f"Tracks: {len(detections)}")

            # 저장
            output_name = f"{output_prefix}_{frame_num:06d}.jpg"
            self._save_frame(frame, output_name)

            self.frame_count += 1

        print(f"Visualized {self.frame_count} frames -> {self.output_folder}")


class ColorClassificationVisualizer(BaseVisualizer):
    """색상 분류 결과 시각화"""

    def visualize_colors(
        self,
        csv_path: str,
        frames_dir: str,
        output_prefix: str = "color"
    ):
        """
        색상 분류 결과 시각화

        Args:
            csv_path: 색상 분류 CSV 경로
            frames_dir: 프레임 폴더 경로
            output_prefix: 출력 파일명 접두사
        """
        print(f"Visualizing color classification from {csv_path}")

        frame_data = defaultdict(list)

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_num = int(row['frame'])
                frame_data[frame_num].append(row)

        # 색상 매핑
        color_map = {
            'orange': (0, 165, 255),  # BGR
            'black': (0, 0, 0),
            'unknown': (128, 128, 128)
        }

        for frame_num in sorted(frame_data.keys()):
            frame = self._load_frame(frames_dir, frame_num)
            if frame is None:
                continue

            detections = frame_data[frame_num]

            # 색상별로 그리기
            for det in detections:
                x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                color_name = det['uniform_color']
                color_conf = float(det['color_confidence'])

                color_bgr = color_map.get(color_name, (128, 128, 128))
                label = f"{color_name}:{color_conf:.2f}"

                frame = self.visualizer.draw_bbox(
                    frame, x1, y1, x2, y2,
                    color=color_bgr,
                    thickness=2,
                    label=label
                )

            # 프레임 정보 추가
            frame = self._add_frame_info(frame, frame_num)

            # 저장
            output_name = f"{output_prefix}_{frame_num:06d}.jpg"
            self._save_frame(frame, output_name)

            self.frame_count += 1

        print(f"Visualized {self.frame_count} frames -> {self.output_folder}")


if __name__ == "__main__":
    # 예제
    import argparse

    parser = argparse.ArgumentParser(description='Visualize soccer analysis results')
    parser.add_argument('--type', choices=['detection', 'tracking', 'color'],
                        required=True, help='Visualization type')
    parser.add_argument('--csv', required=True, help='Input CSV file')
    parser.add_argument('--frames', required=True, help='Frames directory')
    parser.add_argument('--output', default='visualized_frames', help='Output directory')

    args = parser.parse_args()

    if args.type == 'detection':
        viz = DetectionVisualizer(output_folder=args.output)
        viz.visualize_detections(args.csv, args.frames)
    elif args.type == 'tracking':
        viz = TrackingVisualizer(output_folder=args.output)
        viz.visualize_tracking(args.csv, args.frames)
    elif args.type == 'color':
        viz = ColorClassificationVisualizer(output_folder=args.output)
        viz.visualize_colors(args.csv, args.frames)
