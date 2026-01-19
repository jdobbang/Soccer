"""추적 유틸리티 단위 테스트"""

import pytest
import numpy as np
from utils.tracking import (
    Tracklet, FramePathHandler, ImageProcessor,
    TransformProvider, GeometryUtils
)


class TestTracklet:
    """Tracklet 데이터 구조 테스트"""

    def test_tracklet_creation(self):
        """Tracklet 생성 테스트"""
        tracklet = Tracklet(id=1)

        assert tracklet.id == 1
        assert len(tracklet) == 0
        assert tracklet.get_first_frame() == -1
        assert tracklet.get_last_frame() == -1

    def test_tracklet_add_detection(self):
        """탐지 추가 테스트"""
        tracklet = Tracklet(id=1)

        tracklet.add_detection(0, [10, 20, 100, 150], 0.95)
        tracklet.add_detection(1, [12, 22, 102, 152], 0.96)

        assert len(tracklet) == 2
        assert tracklet.get_first_frame() == 0
        assert tracklet.get_last_frame() == 1
        assert tracklet.get_time_span() == 1

    def test_tracklet_get_bbox_at_frame(self):
        """특정 프레임의 bbox 조회 테스트"""
        tracklet = Tracklet(id=1)
        tracklet.add_detection(0, [10, 20, 100, 150], 0.95)
        tracklet.add_detection(2, [15, 25, 105, 155], 0.96)

        assert tracklet.get_bbox_at_frame(0) == [10, 20, 100, 150]
        assert tracklet.get_bbox_at_frame(2) == [15, 25, 105, 155]
        assert tracklet.get_bbox_at_frame(1) is None


class TestImageProcessor:
    """이미지 처리 유틸리티 테스트"""

    def test_clip_bbox(self):
        """bbox 클립핑 테스트"""
        img_shape = (480, 640)  # height, width
        bbox = [-10, -20, 650, 500]

        clipped = ImageProcessor.clip_bbox(bbox, img_shape)

        assert clipped == [0, 0, 640, 480]

    def test_clip_bbox_inside_bounds(self):
        """경계 내 bbox 테스트"""
        img_shape = (480, 640)
        bbox = [100, 100, 300, 300]

        clipped = ImageProcessor.clip_bbox(bbox, img_shape)

        assert clipped == [100, 100, 300, 300]

    def test_get_bbox_center(self):
        """bbox 중심 계산 테스트"""
        bbox = [100, 100, 300, 300]

        center = ImageProcessor.get_bbox_center(bbox)

        assert center == (200, 200)

    def test_get_bbox_area(self):
        """bbox 면적 계산 테스트"""
        bbox = [100, 100, 300, 300]

        area = ImageProcessor.get_bbox_area(bbox)

        assert area == 40000  # 200 * 200


class TestGeometryUtils:
    """기하학적 계산 유틸리티 테스트"""

    def test_euclidean_distance(self):
        """유클리드 거리 테스트"""
        dist = GeometryUtils.euclidean_distance((0, 0), (3, 4))

        assert abs(dist - 5.0) < 1e-6

    def test_bbox_center_distance(self):
        """bbox 중심 거리 테스트"""
        bbox1 = [0, 0, 100, 100]  # center (50, 50)
        bbox2 = [100, 100, 200, 200]  # center (150, 150)

        dist = GeometryUtils.bbox_center_distance(bbox1, bbox2)

        expected = np.sqrt((150 - 50) ** 2 + (150 - 50) ** 2)
        assert abs(dist - expected) < 1e-6

    def test_iou_perfect_overlap(self):
        """완전 겹치는 bbox IoU 테스트"""
        bbox1 = [0, 0, 100, 100]
        bbox2 = [0, 0, 100, 100]

        iou = GeometryUtils.iou(bbox1, bbox2)

        assert abs(iou - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        """겹치지 않는 bbox IoU 테스트"""
        bbox1 = [0, 0, 100, 100]
        bbox2 = [200, 200, 300, 300]

        iou = GeometryUtils.iou(bbox1, bbox2)

        assert abs(iou - 0.0) < 1e-6

    def test_iou_partial_overlap(self):
        """부분 겹치는 bbox IoU 테스트"""
        bbox1 = [0, 0, 100, 100]  # area = 10000
        bbox2 = [50, 50, 150, 150]  # area = 10000
        # intersection = 50*50 = 2500
        # union = 10000 + 10000 - 2500 = 17500
        # IoU = 2500/17500 ≈ 0.1429

        iou = GeometryUtils.iou(bbox1, bbox2)

        expected = 2500 / 17500
        assert abs(iou - expected) < 1e-4

    def test_linear_interpolate(self):
        """선형 보간 테스트"""
        v1, v2 = 0, 100

        # alpha=0: 전부 v1
        assert GeometryUtils.linear_interpolate(v1, v2, 0) == 0

        # alpha=0.5: 중점
        assert GeometryUtils.linear_interpolate(v1, v2, 0.5) == 50

        # alpha=1: 전부 v2
        assert GeometryUtils.linear_interpolate(v1, v2, 1) == 100

    def test_interpolate_bbox(self):
        """bbox 선형 보간 테스트"""
        bbox1 = [0, 0, 100, 100]
        bbox2 = [100, 100, 200, 200]

        # alpha=0.5: 중점
        interp = GeometryUtils.interpolate_bbox(bbox1, bbox2, 0.5)

        assert interp == [50, 50, 150, 150]


class TestTransformProvider:
    """변환 제공자 테스트"""

    def test_get_reid_transform(self):
        """Re-ID Transform 생성 테스트"""
        transform = TransformProvider.get_reid_transform()

        assert transform is not None
        assert callable(transform)

    def test_get_reid_transform_custom_size(self):
        """커스텀 크기 Re-ID Transform 테스트"""
        transform = TransformProvider.get_reid_transform(height=512, width=256)

        assert transform is not None
        # 실제 이미지로 테스트하려면 PIL Image가 필요


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
