#!/usr/bin/env python3
"""
YOLO 볼 탐지 결과를 프레임에 시각화하고 영상으로 저장
detection_ball.py가 생성한 CSV 결과를 입력 영상의 프레임에 표시하고 저장합니다.
"""

import argparse
import os
import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

class BallDetectionVisualizer:
    def __init__(self, video_path, csv_path, output_dir="frames"):
        """
        Args:
            video_path: 입력 비디오 파일 경로
            csv_path: 탐지 결과 CSV 파일 경로
            output_dir: 결과 저장 디렉토리
        """
        self.video_path = video_path
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # CSV에서 탐지 결과 로드
        self.detections = self._load_detections()

        # 비디오 정보 로드
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📹 비디오 정보:")
        print(f"   - 해상도: {self.width}x{self.height}")
        print(f"   - FPS: {self.fps}")
        print(f"   - 총 프레임: {self.total_frames}")

    def _load_detections(self):
        """CSV 파일에서 탐지 결과 로드"""
        detections = {}

        if not os.path.exists(self.csv_path):
            print(f"Warning: CSV file not found: {self.csv_path}")
            return detections

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_idx = int(row['frame'])

                # class_id가 1인 경우만 로드 (class_id = 1은 ball, 0은 shoe)
                # CSV에 class_id 또는 object_id 컬럼이 있으면 필터링
                if 'class_id' in row:
                    class_id = int(row['class_id'])
                    if class_id != 1:  # 1(ball)만 필터링
                        continue
                elif 'object_id' in row:
                    object_id = int(row['object_id'])
                    if object_id != 1:  # 1(ball)만 필터링
                        continue

                if frame_idx not in detections:
                    detections[frame_idx] = []

                detection = {
                    'x1': int(row['x1']),
                    'y1': int(row['y1']),
                    'x2': int(row['x2']),
                    'y2': int(row['y2']),
                    'confidence': float(row['confidence']),
                    'width': int(row['width']),
                    'height': int(row['height']),
                    'class_id': int(row.get('class_id', row.get('object_id', 0)))
                }
                detections[frame_idx].append(detection)

        print(f"✓ 탐지 결과 로드: {len(detections)} 프레임에 총 {sum(len(d) for d in detections.values())} 탐지 (Ball만 표시)")
        return detections

    def draw_detections(self, frame, frame_idx, draw_info=True):
        """
        프레임에 탐지 박스 그리기

        Args:
            frame: 입력 프레임
            frame_idx: 프레임 번호
            draw_info: 좌측 상단에 정보 표시 여부
        """
        if frame_idx in self.detections:
            detections = self.detections[frame_idx]

            for det in detections:
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                conf = det['confidence']
                class_id = det.get('class_id', 0)

                # 박스 색상 (confidence에 따라 변함)
                color = self._get_color_by_confidence(conf)
                thickness = 2

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # 신뢰도 텍스트 (class_id 포함)
                class_name = "Ball" if class_id == 1 else "Shoe"
                label = f"{class_name} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 1

                # 텍스트 배경 (가독성 향상)
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_x, text_y = x1, y1 - 10

                # 배경 사각형
                cv2.rectangle(frame,
                            (text_x - 2, text_y - text_size[1] - 4),
                            (text_x + text_size[0] + 2, text_y + 2),
                            color, -1)

                # 텍스트
                cv2.putText(frame, label, (text_x, text_y),
                          font, font_scale, (255, 255, 255), font_thickness)

        # 프레임 정보 표시
        if draw_info:
            self._draw_frame_info(frame, frame_idx)

        return frame

    def _get_color_by_confidence(self, confidence):
        """신뢰도에 따른 색상 반환"""
        if confidence >= 0.9:
            return (0, 255, 0)  # Green: 매우 높음
        elif confidence >= 0.7:
            return (0, 255, 255)  # Yellow: 높음
        elif confidence >= 0.5:
            return (0, 165, 255)  # Orange: 중간
        else:
            return (0, 0, 255)  # Red: 낮음

    def _draw_frame_info(self, frame, frame_idx):
        """좌측 상단에 프레임 정보 표시"""
        info_text = [
            f"Frame: {frame_idx}/{self.total_frames}",
            f"Detections: {len(self.detections.get(frame_idx, []))}",
            f"Time: {frame_idx/self.fps:.2f}s"
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        y_offset = 25

        # 배경 (가독성)
        cv2.rectangle(frame, (5, 5), (250, 5 + y_offset * len(info_text)), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (250, 5 + y_offset * len(info_text)), (255, 255, 255), 1)

        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 25 + i * y_offset),
                      font, font_scale, (255, 255, 255), font_thickness)

    def visualize_frames(self, output_format="png", confidence_threshold=0.0):
        """
        탐지 결과를 시각화하여 프레임으로 저장

        Args:
            output_format: 저장 포맷 (png, jpg)
            confidence_threshold: 표시할 최소 신뢰도
        """
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        frame_idx = 0
        saved_count = 0

        print(f"\n📸 프레임 시각화 중...")
        pbar = tqdm(total=self.total_frames, desc="Visualizing frames", unit="frame")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # 신뢰도 필터링
            if frame_idx in self.detections:
                filtered_dets = [d for d in self.detections[frame_idx]
                               if d['confidence'] >= confidence_threshold]
                self.detections[frame_idx] = filtered_dets

            # 탐지 결과 그리기
            frame = self.draw_detections(frame, frame_idx, draw_info=True)

            # 프레임 저장
            if frame_idx in self.detections and len(self.detections[frame_idx]) > 0:
                output_path = frames_dir / f"frame_{frame_idx:06d}.{output_format}"
                cv2.imwrite(str(output_path), frame)
                saved_count += 1

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        self.cap.release()

        print(f"✓ 프레임 저장 완료: {saved_count}/{frame_idx} 프레임")
        print(f"📁 저장 위치: {frames_dir}")

        return frames_dir

    def create_output_video(self, output_video_path=None, confidence_threshold=0.0):
        """
        시각화된 프레임으로 비디오 생성

        Args:
            output_video_path: 출력 비디오 경로 (기본값: results.mp4)
            confidence_threshold: 표시할 최소 신뢰도
        """
        if output_video_path is None:
            video_name = Path(self.video_path).stem
            output_video_path = str(self.output_dir / f"{video_name}_detected.mp4")

        print(f"\n🎬 비디오 생성 중...")

        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width, self.height))

        # 재설정
        self.cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0

        pbar = tqdm(total=self.total_frames, desc="Creating video", unit="frame")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # 신뢰도 필터링
            if frame_idx in self.detections:
                filtered_dets = [d for d in self.detections[frame_idx]
                               if d['confidence'] >= confidence_threshold]
                self.detections[frame_idx] = filtered_dets

            # 탐지 결과 그리기
            frame = self.draw_detections(frame, frame_idx, draw_info=True)

            # 비디오에 프레임 쓰기
            out.write(frame)

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        self.cap.release()
        out.release()

        print(f"✓ 비디오 생성 완료")
        print(f"📁 저장 위치: {output_video_path}")

        return output_video_path

    def create_summary_stats(self):
        """탐지 통계 생성"""
        total_detections = sum(len(dets) for dets in self.detections.values())
        frames_with_detection = len(self.detections)

        all_confidences = []
        for dets in self.detections.values():
            all_confidences.extend([d['confidence'] for d in dets])

        if all_confidences:
            avg_confidence = np.mean(all_confidences)
            max_confidence = np.max(all_confidences)
            min_confidence = np.min(all_confidences)
        else:
            avg_confidence = max_confidence = min_confidence = 0

        print(f"\n📊 탐지 통계:")
        print(f"   - 총 탐지 수: {total_detections}")
        print(f"   - 탐지된 프레임: {frames_with_detection}/{self.total_frames} ({100*frames_with_detection/self.total_frames:.1f}%)")
        print(f"   - 평균 신뢰도: {avg_confidence:.4f}")
        print(f"   - 최대 신뢰도: {max_confidence:.4f}")
        print(f"   - 최소 신뢰도: {min_confidence:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO 볼 탐지 결과 시각화 및 저장",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 프레임으로 저장
  python visualize_ball_detection.py results/yolo11n/video_ball.csv input_video.mp4 --frames

  # 비디오로 저장
  python visualize_ball_detection.py results/yolo11n/video_ball.csv input_video.mp4 --video

  # 둘 다 저장
  python visualize_ball_detection.py results/yolo11n/video_ball.csv input_video.mp4 --frames --video
        """
    )

    parser.add_argument("csv_file", help="탐지 결과 CSV 파일 경로")
    parser.add_argument("video_file", help="입력 비디오 파일 경로")
    parser.add_argument("--output-dir", default="frames", help="출력 디렉토리 (기본값: frames)")
    parser.add_argument("--frames", action="store_true", help="프레임으로 저장")
    parser.add_argument("--video", action="store_true", help="비디오로 저장")
    parser.add_argument("--both", action="store_true", help="프레임과 비디오 모두 저장")
    parser.add_argument("--confidence", type=float, default=0.0, help="최소 신뢰도 임계값 (기본값: 0.0)")
    parser.add_argument("--format", default="png", choices=["png", "jpg"], help="프레임 저장 포맷")

    args = parser.parse_args()

    # --both 옵션 처리
    if args.both:
        args.frames = True
        args.video = True

    # 기본값: 둘 다 저장
    if not args.frames and not args.video:
        args.frames = True
        args.video = True

    return args

def main():
    args = parse_args()

    print("="*60)
    print("YOLO Ball Detection Visualization")
    print("="*60)

    try:
        # 시각화 객체 생성
        visualizer = BallDetectionVisualizer(
            args.video_file,
            args.csv_file,
            args.output_dir
        )

        # 통계 출력
        visualizer.create_summary_stats()

        # 프레임 저장
        if args.frames:
            visualizer.visualize_frames(
                output_format=args.format,
                confidence_threshold=args.confidence
            )

        # 비디오 생성
        if args.video:
            visualizer.create_output_video(
                confidence_threshold=args.confidence
            )

        print("\n" + "="*60)
        print("✅ 시각화 완료!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
