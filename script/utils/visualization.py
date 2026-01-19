"""
시각화 공통 유틸리티
==================

Bounding box, 트랙 ID, 텍스트 등을 이미지에 그리는 기능 제공
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional


class Visualizer:
    """이미지에 주석을 추가하는 시각화 클래스"""

    def __init__(self, font_size: float = 0.5, line_width: int = 1):
        """
        Args:
            font_size: 텍스트 크기
            line_width: 선 두께
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_size = font_size
        self.line_width = line_width

    def draw_bbox(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        label: Optional[str] = None
    ) -> np.ndarray:
        """
        이미지에 bounding box 그리기

        Args:
            image: 입력 이미지 (BGR)
            x1, y1, x2, y2: bbox 좌표
            color: BGR 색상
            thickness: 선 두께
            label: 선택적 라벨 텍스트

        Returns:
            주석이 추가된 이미지
        """
        image = image.copy()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        if label:
            # 라벨 배경
            text_size = cv2.getTextSize(label, self.font, self.font_size, 1)[0]
            text_x = x1
            text_y = max(y1 - 5, text_size[1] + 5)

            # 배경 사각형
            cv2.rectangle(
                image,
                (text_x, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                color,
                -1
            )

            # 텍스트
            cv2.putText(
                image,
                label,
                (text_x + 2, text_y - 2),
                self.font,
                self.font_size,
                (255, 255, 255),
                1
            )

        return image

    def draw_bboxes(
        self,
        image: np.ndarray,
        detections: List[Dict],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        label_key: Optional[str] = None
    ) -> np.ndarray:
        """
        이미지에 여러 bounding box 그리기

        Args:
            image: 입력 이미지 (BGR)
            detections: 탐지 정보 리스트
                       [{'x1': ..., 'y1': ..., 'x2': ..., 'y2': ..., 'track_id': ..., ...}, ...]
            color: BGR 색상
            thickness: 선 두께
            label_key: 라벨로 사용할 키 (e.g., 'track_id')

        Returns:
            주석이 추가된 이미지
        """
        for det in detections:
            x1, y1 = int(det['x1']), int(det['y1'])
            x2, y2 = int(det['x2']), int(det['y2'])

            label = None
            if label_key and label_key in det:
                label = f"ID:{det[label_key]}"

            image = self.draw_bbox(image, x1, y1, x2, y2, color, thickness, label)

        return image

    def draw_track(
        self,
        image: np.ndarray,
        track_id: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Optional[Tuple[int, int, int]] = None,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Track ID와 함께 bbox 그리기

        Args:
            image: 입력 이미지 (BGR)
            track_id: 트랙 ID
            x1, y1, x2, y2: bbox 좌표
            color: BGR 색상 (None이면 track_id로 생성)
            thickness: 선 두께

        Returns:
            주석이 추가된 이미지
        """
        if color is None:
            color = self.get_color_by_id(track_id)

        label = f"ID:{track_id}"
        return self.draw_bbox(image, x1, y1, x2, y2, color, thickness, label)

    def draw_text(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int] = (255, 255, 255),
        font_size: float = 0.5,
        thickness: int = 1,
        bg_color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        이미지에 텍스트 그리기

        Args:
            image: 입력 이미지 (BGR)
            text: 텍스트
            x, y: 텍스트 위치
            color: 텍스트 색상
            font_size: 폰트 크기
            thickness: 텍스트 두께
            bg_color: 배경 색상 (None이면 배경 없음)

        Returns:
            주석이 추가된 이미지
        """
        image = image.copy()

        if bg_color:
            text_size = cv2.getTextSize(text, self.font, font_size, thickness)[0]
            cv2.rectangle(
                image,
                (x - 3, y - text_size[1] - 3),
                (x + text_size[0] + 3, y + 3),
                bg_color,
                -1
            )

        cv2.putText(image, text, (x, y), self.font, font_size, color, thickness)

        return image

    def draw_line(
        self,
        image: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        이미지에 선 그리기

        Args:
            image: 입력 이미지 (BGR)
            pt1: 시작점
            pt2: 끝점
            color: BGR 색상
            thickness: 선 두께

        Returns:
            주석이 추가된 이미지
        """
        image = image.copy()
        cv2.line(image, pt1, pt2, color, thickness)
        return image

    def draw_circle(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        이미지에 원 그리기

        Args:
            image: 입력 이미지 (BGR)
            center: 원의 중심
            radius: 반지름
            color: BGR 색상
            thickness: 선 두께 (-1이면 채우기)

        Returns:
            주석이 추가된 이미지
        """
        image = image.copy()
        cv2.circle(image, center, radius, color, thickness)
        return image

    @staticmethod
    def get_color_by_id(id_value: int) -> Tuple[int, int, int]:
        """
        ID 값으로부터 고정된 색상 생성

        Args:
            id_value: ID 값

        Returns:
            BGR 튜플
        """
        np.random.seed(id_value)
        return tuple(map(int, np.random.randint(0, 256, 3)))

    @staticmethod
    def draw_trajectory(
        image: np.ndarray,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        radius: int = 3
    ) -> np.ndarray:
        """
        이미지에 궤적(trajectory) 그리기

        Args:
            image: 입력 이미지 (BGR)
            points: 좌표 포인트 리스트
            color: BGR 색상
            thickness: 선 두께
            radius: 포인트 반지름

        Returns:
            주석이 추가된 이미지
        """
        image = image.copy()

        for i in range(1, len(points)):
            cv2.line(image, points[i - 1], points[i], color, thickness)

        for pt in points:
            cv2.circle(image, pt, radius, color, -1)

        return image

    def add_confidence_bar(
        self,
        image: np.ndarray,
        confidence: float,
        x: int = 10,
        y: int = 30,
        width: int = 200,
        height: int = 20
    ) -> np.ndarray:
        """
        이미지의 우측 상단에 신뢰도 바 추가

        Args:
            image: 입력 이미지 (BGR)
            confidence: 신뢰도 값 (0-1)
            x, y: 바의 위치
            width: 바의 너비
            height: 바의 높이

        Returns:
            주석이 추가된 이미지
        """
        image = image.copy()

        # 배경 (회색)
        cv2.rectangle(image, (x, y), (x + width, y + height), (128, 128, 128), -1)

        # 신뢰도 바 (녹색)
        bar_width = int(width * confidence)
        cv2.rectangle(image, (x, y), (x + bar_width, y + height), (0, 255, 0), -1)

        # 테두리
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 1)

        # 텍스트
        text = f"{confidence:.2%}"
        text_size = cv2.getTextSize(text, self.font, 0.4, 1)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2

        cv2.putText(image, text, (text_x, text_y), self.font, 0.4, (255, 255, 255), 1)

        return image
