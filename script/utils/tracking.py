"""
추적 모듈 공통 유틸리티
====================

track.py에서 반복되는 로직 모듈화
- 이미지 로드 및 처리
- 좌표 클립핑
- 변환(Transform) 정의
- Tracklet 데이터 구조
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import torchvision.transforms as T


@dataclass
class Tracklet:
    """Represents a single tracklet (sequence of detections with same ID)"""
    id: int
    frames: List[int] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list)  # [[x1,y1,x2,y2], ...]
    confidences: List[float] = field(default_factory=list)
    reid_features: List[np.ndarray] = field(default_factory=list)  # 512-dim vectors

    def __len__(self):
        return len(self.frames)

    def add_detection(self, frame: int, bbox: List[float], confidence: float):
        """Add a detection to this tracklet"""
        self.frames.append(frame)
        self.bboxes.append(bbox)
        self.confidences.append(confidence)

    def get_bbox_at_frame(self, frame: int) -> Optional[List[float]]:
        """Get bbox at specific frame"""
        try:
            idx = self.frames.index(frame)
            return self.bboxes[idx]
        except ValueError:
            return None

    def get_first_frame(self) -> int:
        """Get first frame number"""
        return self.frames[0] if self.frames else -1

    def get_last_frame(self) -> int:
        """Get last frame number"""
        return self.frames[-1] if self.frames else -1

    def get_time_span(self) -> int:
        """Get time span (last - first)"""
        if len(self.frames) < 2:
            return 0
        return self.frames[-1] - self.frames[0]


class FramePathHandler:
    """프레임 경로 처리 유틸리티"""

    @staticmethod
    def get_frame_path(frames_dir: str, frame_num: int) -> Optional[str]:
        """
        프레임 경로 찾기 (여러 패턴 지원)

        Args:
            frames_dir: 프레임 폴더 경로
            frame_num: 프레임 번호

        Returns:
            프레임 경로 (없으면 None)
        """
        # 패턴 1: frame_000001.jpg
        frame_path = os.path.join(frames_dir, f"frame_{frame_num:06d}.jpg")
        if os.path.exists(frame_path):
            return frame_path

        # 패턴 2: frame_1.jpg
        frame_path = os.path.join(frames_dir, f"frame_{frame_num}.jpg")
        if os.path.exists(frame_path):
            return frame_path

        # 패턴 3: 000001.jpg
        frame_path = os.path.join(frames_dir, f"{frame_num:06d}.jpg")
        if os.path.exists(frame_path):
            return frame_path

        return None

    @staticmethod
    def detect_frame_dimensions(frames_dir: str, start_frame: int = 0) -> Tuple[int, int]:
        """
        프레임 폴더에서 이미지 크기 감지

        Args:
            frames_dir: 프레임 폴더 경로
            start_frame: 시작 프레임 번호

        Returns:
            (width, height) 튜플
        """
        for frame_num in range(start_frame, start_frame + 10):
            frame_path = FramePathHandler.get_frame_path(frames_dir, frame_num)

            if frame_path:
                img = cv2.imread(frame_path)
                if img is not None:
                    height, width = img.shape[:2]
                    print(f"Detected frame dimensions: {width}x{height} from {frame_path}")
                    return width, height

        print("Warning: Could not detect frame size, using default 1920x1080")
        return 1920, 1080


class ImageProcessor:
    """이미지 처리 유틸리티"""

    @staticmethod
    def safe_load_image(image_path: str) -> Optional[np.ndarray]:
        """
        안전한 이미지 로드

        Args:
            image_path: 이미지 경로

        Returns:
            로드된 이미지 (실패하면 None)
        """
        if not os.path.exists(image_path):
            return None

        img = cv2.imread(image_path)
        if img is None:
            return None

        return img

    @staticmethod
    def clip_bbox(bbox: List[float], img_shape: Tuple[int, int]) -> List[float]:
        """
        이미지 경계 내로 bbox 클립핑

        Args:
            bbox: [x1, y1, x2, y2]
            img_shape: (height, width)

        Returns:
            클립된 bbox
        """
        height, width = img_shape
        x1, y1, x2, y2 = bbox

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))

        return [x1, y1, x2, y2]

    @staticmethod
    def crop_bbox(image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        bbox 영역 크롭

        Args:
            image: 원본 이미지
            bbox: [x1, y1, x2, y2]

        Returns:
            크롭된 이미지
        """
        clipped = ImageProcessor.clip_bbox(bbox, image.shape[:2])
        x1, y1, x2, y2 = clipped
        return image[y1:y2, x1:x2]

    @staticmethod
    def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
        """bbox 중심 좌표"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def get_bbox_area(bbox: List[float]) -> float:
        """bbox 면적"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)


class TransformProvider:
    """변환(Transform) 제공자"""

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    @staticmethod
    def get_reid_transform(
        height: int = 256,
        width: int = 128,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ) -> T.Compose:
        """
        Re-ID 모델용 이미지 전처리 Transform

        Args:
            height: 리사이즈 높이
            width: 리사이즈 너비
            mean: 정규화 평균 (None이면 ImageNet 기본값)
            std: 정규화 표준편차 (None이면 ImageNet 기본값)

        Returns:
            torchvision.transforms.Compose 객체
        """
        if mean is None:
            mean = TransformProvider._IMAGENET_MEAN
        if std is None:
            std = TransformProvider._IMAGENET_STD

        return T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    @staticmethod
    def get_detection_transform() -> T.Compose:
        """탐지용 기본 Transform (리사이즈만)"""
        return T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])


class GeometryUtils:
    """기하학적 계산 유틸리티"""

    @staticmethod
    def euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """두 점 사이의 유클리드 거리"""
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    @staticmethod
    def bbox_center_distance(bbox1: List[float], bbox2: List[float]) -> float:
        """두 bbox 중심 사이의 거리"""
        center1 = ImageProcessor.get_bbox_center(bbox1)
        center2 = ImageProcessor.get_bbox_center(bbox2)
        return GeometryUtils.euclidean_distance(center1, center2)

    @staticmethod
    def iou(bbox1: List[float], bbox2: List[float]) -> float:
        """
        Intersection over Union (IoU) 계산

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]

        Returns:
            IoU 값 (0~1)
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # 교집합 계산
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        if xi_max < xi_min or yi_max < yi_min:
            return 0.0

        intersection = (xi_max - xi_min) * (yi_max - yi_min)

        # 합집합 계산
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def linear_interpolate(v1: float, v2: float, alpha: float) -> float:
        """선형 보간"""
        return (1 - alpha) * v1 + alpha * v2

    @staticmethod
    def interpolate_bbox(bbox1: List[float], bbox2: List[float], alpha: float) -> List[float]:
        """
        두 bbox 사이의 선형 보간

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            alpha: 보간 가중치 (0~1)

        Returns:
            보간된 bbox
        """
        return [
            GeometryUtils.linear_interpolate(bbox1[i], bbox2[i], alpha)
            for i in range(4)
        ]
