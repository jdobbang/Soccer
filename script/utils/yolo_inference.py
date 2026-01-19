"""
YOLO 배치 추론 공통 유틸리티
===========================

YOLO 기반 탐지 작업을 위한 배치 추론 기능 제공
- 배치 처리로 GPU 효율 최적화
- Resume 기능으로 중단 복구 지원
- 이어쓰기 CSV 처리
"""

import os
import csv
import cv2
from typing import Optional, Callable, List
from tqdm import tqdm
from ultralytics import YOLO


class YOLOBatchInference:
    """YOLO 배치 추론을 수행하는 클래스"""

    def __init__(
        self,
        model_path: str,
        batch_size: int = 16,
        conf_threshold: float = 0.1,
        classes: Optional[List[int]] = None,
        verbose: bool = False
    ):
        """
        Args:
            model_path: YOLO 모델 경로
            batch_size: 배치 크기 (default: 16)
            conf_threshold: 신뢰도 임계값 (default: 0.1)
            classes: 필터링할 클래스 리스트 (None이면 모든 클래스)
            verbose: 모델 로딩 중 상세 출력 여부
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.classes = classes
        self.verbose = verbose

        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

    def get_last_processed_frame(self, csv_path: str, interval: int = 1) -> int:
        """
        CSV 파일에서 마지막 처리된 프레임 번호 반환

        Args:
            csv_path: CSV 파일 경로
            interval: 탐지 간격

        Returns:
            마지막 처리된 프레임 번호 (없으면 -1)
        """
        if not os.path.exists(csv_path):
            return -1

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                last_frame = -1

                for row in reader:
                    if row:
                        last_frame = max(last_frame, int(row[0]))

                if last_frame >= 0:
                    next_frame = last_frame + interval
                    print(f"Resuming from frame {next_frame} (last processed: {last_frame})")
                    return next_frame
        except Exception as e:
            print(f"Warning: Could not read existing CSV ({e}). Starting from 0.")

        return 0

    def write_csv_header(self, csv_path: str, header: List[str]) -> bool:
        """
        CSV 파일이 없으면 헤더 작성

        Args:
            csv_path: CSV 파일 경로
            header: 헤더 리스트

        Returns:
            새로 생성되었으면 True
        """
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            return False

        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

        return True

    def process_video(
        self,
        video_path: str,
        csv_path: str,
        detection_interval: int = 1,
        result_writer: Optional[Callable] = None,
        header: Optional[List[str]] = None
    ):
        """
        비디오를 배치로 처리하여 탐지 결과를 CSV로 저장

        Args:
            video_path: 비디오 파일 경로
            csv_path: 출력 CSV 파일 경로
            detection_interval: 탐지 간격 (기본값: 1)
            result_writer: 탐지 결과를 CSV에 쓰는 콜백 함수
                          Signature: (writer, frame_idx, results) -> None
            header: CSV 헤더 (기본값: ['frame', 'object_id', ...])
        """
        if header is None:
            header = ['frame', 'object_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'width', 'height']

        if result_writer is None:
            result_writer = self._default_result_writer

        # Resume 로직
        start_frame = self.get_last_processed_frame(csv_path, detection_interval)

        # CSV 파일 준비
        self.write_csv_header(csv_path, header)

        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        print(f"Processing: {video_path}")
        print(f"  - Interval: {detection_interval} frames")
        print(f"  - Batch Size: {self.batch_size}")
        print(f"  - Output CSV: {csv_path}")

        frame_count = start_frame
        batch_frames = []
        batch_indices = []

        # CSV 파일 추가 모드로 열기
        csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)

        pbar = tqdm(total=total_frames, initial=start_frame, desc="Processing", unit="frame")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 지정된 간격인 경우에만 배치에 추가
                if frame_count % detection_interval == 0:
                    batch_frames.append(frame)
                    batch_indices.append(frame_count)

                # 배치가 가득 차면 추론 실행
                if len(batch_frames) == self.batch_size:
                    results = self.model.predict(
                        batch_frames,
                        conf=self.conf_threshold,
                        verbose=False,
                        classes=self.classes,
                        stream=False
                    )

                    # 콜백 함수로 결과 처리
                    for i, result in enumerate(results):
                        result_writer(csv_writer, batch_indices[i], result)

                    # 배치 초기화
                    batch_frames = []
                    batch_indices = []

                frame_count += 1
                pbar.update(1)

            # 남은 프레임 처리
            if batch_frames:
                results = self.model.predict(
                    batch_frames,
                    conf=self.conf_threshold,
                    verbose=False,
                    classes=self.classes,
                    stream=False
                )

                for i, result in enumerate(results):
                    result_writer(csv_writer, batch_indices[i], result)

        finally:
            pbar.close()
            cap.release()
            csv_file.close()

        print("Processing Complete.")

    def _default_result_writer(self, writer, frame_idx: int, result):
        """기본 결과 기록 함수"""
        boxes = result.boxes

        for box_idx, box in enumerate(boxes):
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1

            writer.writerow([
                frame_idx, box_idx, x1, y1, x2, y2, f"{conf:.4f}", width, height
            ])
