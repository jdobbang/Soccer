"""
CSV 처리 공통 유틸리티
====================

탐지, 추적 결과 CSV 파일 처리 관련 기능 제공
"""

import os
import csv
import pandas as pd
from typing import Dict, List, Optional, Any
from collections import defaultdict


class CSVHandler:
    """CSV 파일 처리를 위한 유틸리티 클래스"""

    @staticmethod
    def load_detection_csv(
        csv_path: str,
        image_pattern: Optional[str] = None,
        frame_range: Optional[tuple] = None
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        탐지 CSV 파일 로드

        Args:
            csv_path: CSV 파일 경로
            image_pattern: 이미지 이름 패턴 (e.g., "frame_{:06d}.jpg")
            frame_range: 프레임 범위 (start, end) - None이면 전체

        Returns:
            프레임별 탐지 정보 딕셔너리
            {frame_num: [{'track_id': ..., 'x1': ..., ...}, ...]}
        """
        frame_data = defaultdict(list)

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            # 형식 감지
            has_object_id = 'object_id' in fieldnames
            has_image_name = 'image_name' in fieldnames

            for row in reader:
                frame = int(row['frame'])

                # 프레임 범위 필터링
                if frame_range:
                    start, end = frame_range
                    if not (start <= frame <= end):
                        continue

                # 컬럼명 정규화
                if has_object_id and not has_image_name:
                    # detection 형식 (tracking 정보 없음)
                    if image_pattern:
                        image_name = image_pattern.format(frame)
                    else:
                        image_name = f"frame_{frame:06d}.jpg"
                    track_id = -1  # detection에는 track_id 없음
                else:
                    # tracking 형식 (tracking 정보 포함)
                    image_name = row.get('image_name', f"frame_{frame:06d}.jpg")
                    track_id = int(row.get('track_id', row.get('object_id', -1)))

                frame_data[frame].append({
                    'image_name': image_name,
                    'track_id': track_id,
                    'x1': float(row['x1']),
                    'y1': float(row['y1']),
                    'x2': float(row['x2']),
                    'y2': float(row['y2']),
                    'confidence': float(row['confidence'])
                })

        return frame_data

    @staticmethod
    def load_as_dataframe(
        csv_path: str,
        frame_range: Optional[tuple] = None
    ) -> pd.DataFrame:
        """
        CSV를 pandas DataFrame으로 로드

        Args:
            csv_path: CSV 파일 경로
            frame_range: 프레임 범위 (start, end) - None이면 전체

        Returns:
            DataFrame
        """
        df = pd.read_csv(csv_path)

        if frame_range:
            start, end = frame_range
            df = df[(df['frame'] >= start) & (df['frame'] <= end)]

        return df

    @staticmethod
    def save_csv(
        output_path: str,
        data: List[List[Any]],
        header: List[str],
        mode: str = 'w'
    ):
        """
        데이터를 CSV 파일로 저장

        Args:
            output_path: 출력 파일 경로
            data: 저장할 데이터 리스트
            header: CSV 헤더
            mode: 파일 모드 ('w': 덮어쓰기, 'a': 추가)
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 'w' 모드이거나 파일이 비어있으면 헤더 작성
            if mode == 'w' or os.path.getsize(output_path) == 0:
                writer.writerow(header)

            writer.writerows(data)

    @staticmethod
    def filter_by_track_id(
        csv_path: str,
        track_id: int,
        output_path: str
    ):
        """
        특정 track_id로 필터링하여 새 CSV 생성

        Args:
            csv_path: 입력 CSV 경로
            track_id: 필터링할 track_id
            output_path: 출력 CSV 경로
        """
        df = pd.read_csv(csv_path)
        filtered_df = df[df['track_id'] == track_id]

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        filtered_df.to_csv(output_path, index=False)

        print(f"Filtered {len(filtered_df)} rows (track_id={track_id}) to {output_path}")

    @staticmethod
    def merge_csvs(
        csv_paths: List[str],
        output_path: str,
        sort_by: str = 'frame'
    ):
        """
        여러 CSV 파일을 병합

        Args:
            csv_paths: CSV 파일 경로 리스트
            output_path: 출력 파일 경로
            sort_by: 정렬 컬럼명
        """
        dfs = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            dfs.append(df)

        merged_df = pd.concat(dfs, ignore_index=True)

        if sort_by in merged_df.columns:
            merged_df = merged_df.sort_values(by=sort_by).reset_index(drop=True)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        merged_df.to_csv(output_path, index=False)

        print(f"Merged {len(csv_paths)} files -> {output_path} ({len(merged_df)} rows)")

    @staticmethod
    def get_frame_statistics(csv_path: str) -> Dict[str, Any]:
        """
        CSV 파일의 프레임 통계 정보 반환

        Args:
            csv_path: CSV 파일 경로

        Returns:
            {
                'total_rows': int,
                'frame_count': int,
                'min_frame': int,
                'max_frame': int,
                'avg_detections_per_frame': float,
                ...
            }
        """
        df = pd.read_csv(csv_path)

        frame_col = 'frame' if 'frame' in df.columns else df.columns[0]

        return {
            'total_rows': len(df),
            'frame_count': df[frame_col].nunique(),
            'min_frame': int(df[frame_col].min()),
            'max_frame': int(df[frame_col].max()),
            'avg_detections_per_frame': len(df) / df[frame_col].nunique(),
        }
