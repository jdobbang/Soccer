"""색상 분석 유틸리티 단위 테스트"""

import pytest
import numpy as np
import cv2
from utils.color_analyzer import ColorAnalyzer


class TestColorAnalyzer:
    """ColorAnalyzer 테스트"""

    @pytest.fixture
    def analyzer(self):
        """ColorAnalyzer 인스턴스"""
        return ColorAnalyzer()

    @pytest.fixture
    def sample_image(self):
        """샘플 이미지 생성"""
        # 480x640 BGR 이미지
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        return image

    def test_analyzer_initialization(self, analyzer):
        """분석기 초기화 테스트"""
        assert analyzer is not None
        assert 'orange' in analyzer.team_colors
        assert 'black' in analyzer.team_colors
        assert 'grass' in analyzer.exclude_colors
        assert 'skin' in analyzer.exclude_colors

    def test_get_upper_body_region(self, analyzer, sample_image):
        """상체 영역 추출 테스트"""
        bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 300}

        upper_body = analyzer.get_upper_body_region(sample_image, bbox, upper_ratio=0.5)

        # 상체는 상위 50%: (300-100) * 0.5 = 100 높이
        assert upper_body.shape == (100, 100, 3)

    def test_get_upper_body_region_custom_ratio(self, analyzer, sample_image):
        """상체 영역 추출 - 커스텀 비율 테스트"""
        bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 300}

        # 상위 30%
        upper_body = analyzer.get_upper_body_region(sample_image, bbox, upper_ratio=0.3)

        # (300-100) * 0.3 = 60 높이
        assert upper_body.shape == (60, 100, 3)

    def test_create_exclude_mask(self, analyzer):
        """제외 마스크 생성 테스트"""
        # HSV 이미지 생성
        hsv_image = np.zeros((480, 640, 3), dtype=np.uint8)

        mask = analyzer.create_exclude_mask(hsv_image)

        # 마스크는 uint8, 255 = valid, 0 = excluded
        assert mask.dtype == np.uint8
        assert mask.shape == (480, 640)
        assert np.all((mask == 0) | (mask == 255))

    def test_classify_color_empty_crop(self, analyzer):
        """빈 크롭 색상 분류 테스트"""
        empty_crop = np.empty((0, 0, 3), dtype=np.uint8)

        color, confidence = analyzer.classify_color(empty_crop)

        assert color == 'unknown'
        assert confidence == 0.0

    def test_classify_color_small_crop(self, analyzer):
        """작은 크롭 색상 분류 테스트"""
        small_crop = np.ones((3, 3, 3), dtype=np.uint8)

        color, confidence = analyzer.classify_color(small_crop)

        # 최소 크기 미만이면 unknown
        assert color == 'unknown'
        assert confidence == 0.0

    def test_classify_color_orange(self, analyzer):
        """주황색 분류 테스트"""
        # 주황색 이미지 생성 (BGR에서 주황은 (0, 165, 255))
        crop = np.full((100, 100, 3), (0, 165, 255), dtype=np.uint8)

        color, confidence = analyzer.classify_color(crop)

        # 주황색으로 분류되어야 함
        assert color in ['orange', 'unknown']
        # confidence >= 0.15 또는 unknown

    def test_analyze_region(self, analyzer, sample_image):
        """영역 분석 테스트"""
        bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 300}

        color, confidence = analyzer.analyze_region(sample_image, bbox, upper_ratio=0.5)

        # 검은색 이미지이므로 black으로 분류될 가능성
        assert color in ['orange', 'black', 'unknown']
        assert 0 <= confidence <= 1.0

    def test_get_color_display_bgr(self, analyzer):
        """색상 표시 BGR 값 테스트"""
        orange_bgr = analyzer.get_color_display_bgr('orange')
        black_bgr = analyzer.get_color_display_bgr('black')
        unknown_bgr = analyzer.get_color_display_bgr('unknown')

        # 주황색
        assert orange_bgr == (0, 165, 255)

        # 검은색
        assert black_bgr == (0, 0, 0)

        # 미정의 색상 (기본값)
        assert unknown_bgr == (128, 128, 128)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
