#!/usr/bin/env python3
"""
통합 파이프라인
==============

축구 분석의 전체 파이프라인을 한 번에 실행:
1. 선수 탐지
2. 볼 탐지
3. 유니폼 색상 분류
4. 추적 (선택사항)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from detect_player import detect_players
from detect_ball import detect_balls
from classify_uniform_color import classify_colors


class Pipeline:
    """축구 분석 통합 파이프라인"""

    def __init__(
        self,
        video_path: str,
        player_model: str,
        ball_model: Optional[str] = None,
        image_folder: str = "images",
        output_folder: str = "results",
        batch_size: int = 32,
        detection_interval: int = 1
    ):
        """
        Args:
            video_path: 입력 비디오 경로
            player_model: 선수 탐지 모델 경로
            ball_model: 볼 탐지 모델 경로 (선택사항)
            image_folder: 추출된 이미지 폴더
            output_folder: 출력 폴더
            batch_size: 배치 크기
            detection_interval: 탐지 간격
        """
        self.video_path = video_path
        self.player_model = player_model
        self.ball_model = ball_model
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.batch_size = batch_size
        self.detection_interval = detection_interval

        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.model_name = os.path.splitext(os.path.basename(player_model))[0]
        self.output_path = os.path.join(output_folder, self.model_name)

    def _print_header(self, title: str):
        """섹션 헤더 출력"""
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)

    def run_player_detection(self) -> str:
        """
        1단계: 선수 탐지

        Returns:
            선수 탐지 CSV 경로
        """
        self._print_header("1. Player Detection")

        player_csv = os.path.join(self.output_path, f"{self.video_name}.csv")

        detect_players(
            video_path=self.video_path,
            model_path=self.player_model,
            output_folder=self.output_folder,
            detection_interval=self.detection_interval,
            batch_size=self.batch_size
        )

        return player_csv

    def run_ball_detection(self) -> Optional[str]:
        """
        2단계: 볼 탐지 (선택사항)

        Returns:
            볼 탐지 CSV 경로 (없으면 None)
        """
        if not self.ball_model:
            print("\nSkipping ball detection (no model provided)")
            return None

        self._print_header("2. Ball Detection")

        ball_csv = os.path.join(self.output_path, f"{self.video_name}_ball.csv")

        detect_balls(
            video_path=self.video_path,
            model_path=self.ball_model,
            output_folder=self.output_folder,
            detection_interval=self.detection_interval,
            batch_size=self.batch_size
        )

        return ball_csv

    def run_color_classification(self, player_csv: str) -> str:
        """
        3단계: 유니폼 색상 분류

        Args:
            player_csv: 선수 탐지 CSV 경로

        Returns:
            색상 분류 결과 CSV 경로
        """
        self._print_header("3. Uniform Color Classification")

        if not os.path.exists(self.image_folder):
            print(f"Warning: Image folder not found: {self.image_folder}")
            print("Skipping color classification")
            return None

        output_csv = os.path.join(self.output_path, f"{self.video_name}_color.csv")

        classify_colors(
            detection_csv=player_csv,
            image_folder=self.image_folder,
            output_csv=output_csv
        )

        return output_csv

    def print_summary(self, results: dict):
        """파이프라인 실행 요약 출력"""
        self._print_header("Pipeline Summary")

        print("\nGenerated files:")
        for step, result in results.items():
            if result:
                status = "✓" if os.path.exists(result) else "✗"
                print(f"  {status} {step}: {result}")
            else:
                print(f"  - {step}: Skipped")

        print(f"\nOutput folder: {self.output_path}")

    def run(self) -> dict:
        """
        전체 파이프라인 실행

        Returns:
            생성된 파일들의 경로 딕셔너리
        """
        try:
            print(f"\nStarting Soccer Analysis Pipeline")
            print(f"Video: {self.video_path}")
            print(f"Player Model: {self.player_model}")
            if self.ball_model:
                print(f"Ball Model: {self.ball_model}")
            print(f"Output: {self.output_path}")

            results = {}

            # 1. 선수 탐지
            player_csv = self.run_player_detection()
            results["Player Detection"] = player_csv

            # 2. 볼 탐지
            ball_csv = self.run_ball_detection()
            results["Ball Detection"] = ball_csv

            # 3. 색상 분류
            color_csv = self.run_color_classification(player_csv)
            results["Color Classification"] = color_csv

            # 요약 출력
            self.print_summary(results)

            print("\nPipeline completed successfully!")
            return results

        except Exception as e:
            print(f"\nError during pipeline execution: {e}")
            import traceback
            traceback.print_exc()
            return {}


def main():
    parser = argparse.ArgumentParser(
        description='Soccer analysis pipeline - detect players, balls, and classify colors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (player detection only)
  python pipeline.py yolo11x.pt video.mp4

  # With ball detection
  python pipeline.py yolo11x.pt video.mp4 --ball_model ball_model.pt

  # With custom parameters
  python pipeline.py yolo11x.pt video.mp4 \\
    --image_folder extracted_frames/ \\
    --output results/ \\
    --batch 64 \\
    --interval 2

  # Full pipeline
  python pipeline.py yolo11x.pt video.mp4 \\
    --ball_model ball_model.pt \\
    --image_folder images/ \\
    --output results/
        """
    )

    parser.add_argument("player_model", help="Player detection model path (e.g., yolo11x.pt)")
    parser.add_argument("video_file", nargs="?", default="test.mp4", help="Input video path")
    parser.add_argument("--ball_model", type=str, default=None,
                        help="Ball detection model path (optional)")
    parser.add_argument("--image_folder", type=str, default="images",
                        help="Extracted images folder (default: images)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output folder (default: results)")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--interval", type=int, default=1,
                        help="Detection interval in frames (default: 1)")

    args = parser.parse_args()

    # 입력 검증
    if not os.path.exists(args.player_model):
        print(f"Error: Player model not found: {args.player_model}")
        sys.exit(1)

    if not os.path.exists(args.video_file):
        print(f"Error: Video file not found: {args.video_file}")
        sys.exit(1)

    if args.ball_model and not os.path.exists(args.ball_model):
        print(f"Error: Ball model not found: {args.ball_model}")
        sys.exit(1)

    # 파이프라인 실행
    pipeline = Pipeline(
        video_path=args.video_file,
        player_model=args.player_model,
        ball_model=args.ball_model,
        image_folder=args.image_folder,
        output_folder=args.output,
        batch_size=args.batch,
        detection_interval=args.interval
    )

    pipeline.run()


if __name__ == "__main__":
    main()
