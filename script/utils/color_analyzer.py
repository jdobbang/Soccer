"""
색상 분석 공통 유틸리티
====================

거리 기반 HSV 색상 분류 및 분석 기능 제공 (Euclidean distance 기반)
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


class ColorAnalyzer:
    """거리 기반 색상 분석 클래스 (Euclidean distance in HSV space)"""

    def __init__(self, team_colors: Optional[Dict[str, Dict]] = None, exclude_colors: Optional[Dict] = None):
        """
        Args:
            team_colors: 팀 색상 정의 딕셔너리
                        e.g., {
                            'orange': {'hsv_lower': ..., 'hsv_upper': ..., 'display_bgr': ...},
                            'black': {...}
                        }
            exclude_colors: 제외할 색상 정의
                           e.g., {'grass': {...}, 'skin': {...}}
        """
        # 기본 팀 색상 정의
        self.team_colors = team_colors or {
            'orange': {
                'hsv_lower': np.array([5, 100, 100]),
                'hsv_upper': np.array([25, 255, 255]),
                'display_bgr': (0, 165, 255),
            },
            'black': {
                'hsv_lower': np.array([0, 0, 0]),
                'hsv_upper': np.array([179, 255, 80]),
                'display_bgr': (0, 0, 0),
            },
        }

        # 기본 제외 색상
        self.exclude_colors = exclude_colors or {
            'grass': {
                'hsv_lower': np.array([35, 40, 40]),
                'hsv_upper': np.array([85, 255, 255]),
            },
            'skin': {
                'hsv_lower': np.array([0, 20, 70]),
                'hsv_upper': np.array([20, 150, 255]),
            },
        }

        # 색상 중심점 미리 계산 (HSV 범위의 중심)
        self.team_color_centers = {}
        for color_name, color_def in self.team_colors.items():
            self.team_color_centers[color_name] = self._compute_color_center(
                color_def['hsv_lower'], color_def['hsv_upper']
            )

    def _compute_color_center(
        self,
        hsv_lower: np.ndarray,
        hsv_upper: np.ndarray
    ) -> np.ndarray:
        """
        HSV 범위의 중심점 계산

        Args:
            hsv_lower: HSV 범위의 하한
            hsv_upper: HSV 범위의 상한

        Returns:
            중심점 HSV 값 (float32 배열)
        """
        return ((hsv_lower.astype(np.float32) + hsv_upper.astype(np.float32)) / 2.0)

    def get_upper_body_region(
        self,
        image: np.ndarray,
        bbox: Dict[str, float],
        upper_ratio: float = 0.5
    ) -> np.ndarray:
        """
        bbox에서 상체 영역만 추출

        Args:
            image: 원본 이미지 (BGR)
            bbox: bounding box {'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}
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

    def create_exclude_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """
        제외할 색상(잔디, 피부)에 대한 마스크 생성

        Args:
            hsv_image: HSV 색공간의 이미지

        Returns:
            유효한 픽셀만 255, 나머지는 0인 마스크
        """
        mask = np.ones(hsv_image.shape[:2], dtype=np.uint8) * 255

        for color_name, color_range in self.exclude_colors.items():
            exclude_mask = cv2.inRange(hsv_image, color_range['hsv_lower'], color_range['hsv_upper'])
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_mask))

        return mask

    def classify_color(
        self,
        crop: np.ndarray,
        confidence_threshold: float = 0.0
    ) -> Tuple[str, float]:
        """
        상체 영역에서 유니폼 색상 분류 (Euclidean distance 기반)

        Args:
            crop: 상체 영역 이미지 (BGR)
            confidence_threshold: 신뢰도 임계값

        Returns:
            (color_name, confidence) 튜플
        """
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return 'unknown', 0.0

        # BGR to HSV 변환
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # 제외 색상 마스킹
        valid_mask = self.create_exclude_mask(hsv)
        valid_pixels = cv2.countNonZero(valid_mask)

        if valid_pixels < 50:  # 유효 픽셀이 너무 적으면
            return 'unknown', 0.0

        # 유효한 픽셀만 추출
        hsv_flat = hsv.reshape(-1, 3).astype(np.float32)
        valid_mask_flat = valid_mask.flatten() > 0
        valid_pixels_hsv = hsv_flat[valid_mask_flat]  # Shape: (N, 3)

        # 각 팀 색상까지의 거리 계산
        scores = {}
        for team_name, center in self.team_color_centers.items():
            # HSV 거리 계산 (H는 원형 구조 고려)
            h_diff = np.abs(valid_pixels_hsv[:, 0] - center[0])
            h_diff = np.minimum(h_diff, 180 - h_diff)  # H는 원형이므로 최단 거리 사용

            sv_diff = valid_pixels_hsv[:, 1:] - center[1:]

            # 정규화된 거리 계산 (H: 0~180, S,V: 0~255)
            h_normalized = h_diff / 180.0
            sv_normalized = np.linalg.norm(sv_diff, axis=1) / 255.0

            distances = np.sqrt(h_normalized**2 + sv_normalized**2)

            # 평균 거리를 역수로 변환하여 점수 계산
            # 거리 0 -> score 1.0, 거리 증가 -> score 감소
            mean_distance = np.mean(distances)
            scores[team_name] = 1.0 / (1.0 + mean_distance * 2)

        # 가장 높은 점수의 색상 선택
        best_color = max(scores, key=scores.get)
        best_score = scores[best_color]
        print(best_color, best_score)
        # 임계값 적용
        if best_score < confidence_threshold:
            return 'unknown', best_score

        return best_color, best_score

    def analyze_region(
        self,
        image: np.ndarray,
        bbox: Dict[str, float],
        upper_ratio: float = 0.5
    ) -> Tuple[str, float]:
        """
        이미지의 bbox 영역의 색상 분석

        Args:
            image: 원본 이미지 (BGR)
            bbox: bounding box {'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}
            upper_ratio: 상체 영역 비율

        Returns:
            (color_name, confidence)
        """
        upper_body = self.get_upper_body_region(image, bbox, upper_ratio)
        return self.classify_color(upper_body)

    def get_color_display_bgr(self, color_name: str) -> Tuple[int, int, int]:
        """
        색상명에 대응하는 시각화용 BGR 값 반환

        Args:
            color_name: 색상명

        Returns:
            BGR 튜플
        """
        if color_name in self.team_colors:
            return self.team_colors[color_name]['display_bgr']
        return (128, 128, 128)  # 기본: 회색
