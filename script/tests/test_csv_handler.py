"""CSV 처리 유틸리티 단위 테스트"""

import pytest
import os
import csv
import tempfile
import pandas as pd
from utils.csv_handler import CSVHandler


class TestCSVHandler:
    """CSVHandler 테스트"""

    @pytest.fixture
    def sample_csv(self):
        """샘플 CSV 파일 생성"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'image_name', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])
            writer.writerow([0, 'frame_000000.jpg', 1, 10.0, 20.0, 100.0, 150.0, 0.95])
            writer.writerow([1, 'frame_000001.jpg', 1, 12.0, 22.0, 102.0, 152.0, 0.96])
            writer.writerow([2, 'frame_000002.jpg', 2, 15.0, 25.0, 105.0, 155.0, 0.92])
            writer.writerow([3, 'frame_000003.jpg', 2, 17.0, 27.0, 107.0, 157.0, 0.93])

            temp_path = f.name

        yield temp_path

        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_load_detection_csv(self, sample_csv):
        """CSV 로드 테스트"""
        frame_data = CSVHandler.load_detection_csv(sample_csv)

        assert len(frame_data) == 4  # 4 frames
        assert len(frame_data[0]) == 1  # 1 detection in frame 0
        assert len(frame_data[2]) == 1  # 1 detection in frame 2

        # 첫 번째 탐지 확인
        det = frame_data[0][0]
        assert det['track_id'] == 1
        assert det['x1'] == 10.0
        assert det['confidence'] == 0.95

    def test_load_detection_csv_with_frame_range(self, sample_csv):
        """프레임 범위 필터링 테스트"""
        frame_data = CSVHandler.load_detection_csv(sample_csv, frame_range=(1, 2))

        assert len(frame_data) == 2  # Only frames 1 and 2
        assert 0 not in frame_data
        assert 1 in frame_data
        assert 2 in frame_data
        assert 3 not in frame_data

    def test_load_as_dataframe(self, sample_csv):
        """DataFrame 로드 테스트"""
        df = CSVHandler.load_as_dataframe(sample_csv)

        assert len(df) == 4  # 4 rows
        assert 'track_id' in df.columns
        assert df['track_id'].tolist() == [1, 1, 2, 2]

    def test_filter_by_track_id(self, sample_csv):
        """Track ID 필터링 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'filtered.csv')

            CSVHandler.filter_by_track_id(sample_csv, track_id=1, output_path=output_path)

            # 필터링된 파일 확인
            assert os.path.exists(output_path)

            filtered_df = pd.read_csv(output_path)
            assert len(filtered_df) == 2  # 2 rows for track_id=1
            assert all(filtered_df['track_id'] == 1)

    def test_get_frame_statistics(self, sample_csv):
        """프레임 통계 테스트"""
        stats = CSVHandler.get_frame_statistics(sample_csv)

        assert stats['total_rows'] == 4
        assert stats['frame_count'] == 4
        assert stats['min_frame'] == 0
        assert stats['max_frame'] == 3
        assert stats['avg_detections_per_frame'] == 1.0

    def test_save_csv(self):
        """CSV 저장 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.csv')

            data = [
                [0, 1, 10, 20, 100, 150, 0.95],
                [1, 1, 12, 22, 102, 152, 0.96]
            ]
            header = ['frame', 'object_id', 'x1', 'y1', 'x2', 'y2', 'confidence']

            CSVHandler.save_csv(output_path, data, header)

            # 저장된 파일 확인
            assert os.path.exists(output_path)

            df = pd.read_csv(output_path)
            assert len(df) == 2
            assert list(df.columns) == header


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
