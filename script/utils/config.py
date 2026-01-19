"""
설정 관리 유틸리티
===============

YAML 설정 파일 로드 및 관리
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class SortConfig:
    """SORT 추적 설정"""
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


@dataclass
class InterpolationConfig:
    """보간 설정"""
    max_gap: int = 30
    method: str = "linear"


@dataclass
class ReidConfig:
    """Re-ID 설정"""
    model_name: str = "osnet_x1_0"
    pretrained: bool = True
    extract_features: bool = True
    similarity_threshold: float = 0.7
    max_time_gap: int = 150
    max_spatial_distance: float = 200
    spatial_velocity_ratio: float = 2.0
    batch_size: int = 32
    num_workers: int = 4
    max_samples_per_tracklet: int = 5
    device: str = "cuda"


@dataclass
class DetectionConfig:
    """탐지 설정"""
    player_model: str
    ball_model: Optional[str] = None
    confidence_threshold: float = 0.1
    interval: int = 1
    batch_size: int = 32


@dataclass
class TrackingConfig:
    """추적 전체 설정"""
    sort: SortConfig
    interpolation: InterpolationConfig
    reid: ReidConfig


@dataclass
class ColorConfig:
    """색상 분류 설정"""
    enabled: bool = True
    upper_body_ratio: float = 0.5
    confidence_threshold: float = 0.15
    team_colors: Dict[str, Dict[str, Any]] = None
    exclude_colors: Dict[str, Dict[str, Any]] = None


@dataclass
class VisualizationConfig:
    """시각화 설정"""
    enabled: bool = True
    save_annotated_frames: bool = True
    draw_track_ids: bool = True
    draw_confidence: bool = True
    bbox_thickness: int = 2
    font_size: float = 0.5


@dataclass
class PathsConfig:
    """경로 설정"""
    output_dir: str = "results"
    frames_dir: str = "images"
    cache_dir: str = "cache"
    save_intermediate: Dict[str, bool] = None


@dataclass
class FilteringConfig:
    """필터링 설정"""
    min_confidence: float = 0.1
    filter_by_uniform_color: Optional[str] = None
    remove_large_low_confidence: bool = True
    large_bbox_threshold: float = 0.1
    low_confidence_threshold: float = 0.5
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None


@dataclass
class LoggingConfig:
    """로깅 설정"""
    verbose: bool = True
    progress_bar: bool = True
    save_logs: bool = True
    log_dir: str = "logs"


class ConfigManager:
    """설정 관리자"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        self.config_dict = {}
        self.load()

    def load(self):
        """YAML 설정 파일 로드"""
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file not found: {self.config_path}")
            print("Using default configuration")
            self.config_dict = {}
            return

        print(f"Loading configuration from: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config_dict = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        설정값 조회

        Args:
            key: 설정 키 (dot notation 지원, e.g., "tracking.sort.max_age")
            default: 기본값

        Returns:
            설정값
        """
        keys = key.split('.')
        value = self.config_dict

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def get_detection_config(self) -> DetectionConfig:
        """탐지 설정 객체 반환"""
        det_dict = self.config_dict.get('detection', {})

        return DetectionConfig(
            player_model=det_dict.get('player_model', 'yolo11x.pt'),
            ball_model=det_dict.get('ball_model'),
            confidence_threshold=det_dict.get('confidence_threshold', 0.1),
            interval=det_dict.get('interval', 1),
            batch_size=det_dict.get('batch_size', 32)
        )

    def get_tracking_config(self) -> TrackingConfig:
        """추적 설정 객체 반환"""
        track_dict = self.config_dict.get('tracking', {})

        sort_dict = track_dict.get('sort', {})
        sort_config = SortConfig(
            max_age=sort_dict.get('max_age', 30),
            min_hits=sort_dict.get('min_hits', 3),
            iou_threshold=sort_dict.get('iou_threshold', 0.3)
        )

        interp_dict = track_dict.get('interpolation', {})
        interp_config = InterpolationConfig(
            max_gap=interp_dict.get('max_gap', 30),
            method=interp_dict.get('method', 'linear')
        )

        reid_dict = track_dict.get('reid', {})
        reid_feat_dict = reid_dict.get('feature_extraction', {})
        reid_merge_dict = reid_dict.get('merging', {})

        reid_config = ReidConfig(
            model_name=reid_dict.get('model_name', 'osnet_x1_0'),
            pretrained=reid_dict.get('pretrained', True),
            extract_features=reid_dict.get('extract_features', True),
            similarity_threshold=reid_merge_dict.get('similarity_threshold', 0.7),
            max_time_gap=reid_merge_dict.get('max_time_gap', 150),
            max_spatial_distance=reid_merge_dict.get('max_spatial_distance', 200),
            spatial_velocity_ratio=reid_merge_dict.get('spatial_velocity_ratio', 2.0),
            batch_size=reid_feat_dict.get('batch_size', 32),
            num_workers=reid_feat_dict.get('num_workers', 4),
            max_samples_per_tracklet=reid_feat_dict.get('max_samples_per_tracklet', 5),
            device=reid_feat_dict.get('device', 'cuda')
        )

        return TrackingConfig(
            sort=sort_config,
            interpolation=interp_config,
            reid=reid_config
        )

    def get_color_config(self) -> ColorConfig:
        """색상 분류 설정 객체 반환"""
        color_dict = self.config_dict.get('color_classification', {})

        return ColorConfig(
            enabled=color_dict.get('enabled', True),
            upper_body_ratio=color_dict.get('upper_body_ratio', 0.5),
            confidence_threshold=color_dict.get('confidence_threshold', 0.15),
            team_colors=color_dict.get('team_colors', {}),
            exclude_colors=color_dict.get('exclude_colors', {})
        )

    def get_visualization_config(self) -> VisualizationConfig:
        """시각화 설정 객체 반환"""
        viz_dict = self.config_dict.get('visualization', {})

        return VisualizationConfig(
            enabled=viz_dict.get('enabled', True),
            save_annotated_frames=viz_dict.get('save_annotated_frames', True),
            draw_track_ids=viz_dict.get('draw_track_ids', True),
            draw_confidence=viz_dict.get('draw_confidence', True),
            bbox_thickness=viz_dict.get('bbox_thickness', 2),
            font_size=viz_dict.get('font_size', 0.5)
        )

    def get_paths_config(self) -> PathsConfig:
        """경로 설정 객체 반환"""
        paths_dict = self.config_dict.get('paths', {})

        return PathsConfig(
            output_dir=paths_dict.get('output_dir', 'results'),
            frames_dir=paths_dict.get('frames_dir', 'images'),
            cache_dir=paths_dict.get('cache_dir', 'cache'),
            save_intermediate=paths_dict.get('save_intermediate', {})
        )

    def get_filtering_config(self) -> FilteringConfig:
        """필터링 설정 객체 반환"""
        filt_dict = self.config_dict.get('filtering', {})
        det_filt = filt_dict.get('detection', {})
        frame_range = filt_dict.get('frame_range', {})

        return FilteringConfig(
            min_confidence=det_filt.get('min_confidence', 0.1),
            filter_by_uniform_color=det_filt.get('filter_by_uniform_color'),
            remove_large_low_confidence=det_filt.get('remove_large_low_confidence', True),
            large_bbox_threshold=det_filt.get('large_bbox_threshold', 0.1),
            low_confidence_threshold=det_filt.get('low_confidence_threshold', 0.5),
            frame_start=frame_range.get('start'),
            frame_end=frame_range.get('end')
        )

    def get_logging_config(self) -> LoggingConfig:
        """로깅 설정 객체 반환"""
        log_dict = self.config_dict.get('logging', {})

        return LoggingConfig(
            verbose=log_dict.get('verbose', True),
            progress_bar=log_dict.get('progress_bar', True),
            save_logs=log_dict.get('save_logs', True),
            log_dir=log_dict.get('log_dir', 'logs')
        )

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 반환"""
        return self.config_dict

    def save(self, output_path: str):
        """설정을 YAML 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config_dict, f, default_flow_style=False, allow_unicode=True)

        print(f"Configuration saved to: {output_path}")
