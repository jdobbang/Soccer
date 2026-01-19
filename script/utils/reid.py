"""
Re-ID (Person Re-Identification) 유틸리티
========================================

OSNet 기반 특징 추출 및 유사도 계산
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from .tracking import Tracklet, ImageProcessor, TransformProvider


class ReidModelHandler:
    """Re-ID 모델 로드 및 관리"""

    @staticmethod
    def load_reid_model(model_name: str = "osnet_x1_0", pretrained: bool = True, device: str = "cuda"):
        """
        Re-ID 모델 로드

        Args:
            model_name: 모델명 (기본: osnet_x1_0)
            pretrained: 사전학습 가중치 사용 여부
            device: 장치 (cuda, cpu)

        Returns:
            로드된 모델
        """
        try:
            import torchreid
        except ImportError:
            raise ImportError("torchreid is required. Install with: pip install torchreid")

        print(f"Loading Re-ID model: {model_name} (pretrained={pretrained})")

        # 모델 로드
        model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            loss="softmax",
            pretrained=pretrained
        )

        model = model.to(device)
        model.eval()

        print(f"Model loaded on {device}")
        return model


class TrackletReidDataset(Dataset):
    """Re-ID 특징 추출용 Tracklet 데이터셋"""

    def __init__(
        self,
        tracklets: List[Tracklet],
        frames_dir: str,
        transform=None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            tracklets: Tracklet 리스트
            frames_dir: 프레임 폴더 경로
            transform: 이미지 전처리 변환
            max_samples: 각 tracklet에서 최대 샘플 수 (None이면 모두)
        """
        self.tracklets = tracklets
        self.frames_dir = frames_dir
        self.transform = transform or TransformProvider.get_reid_transform()
        self.max_samples = max_samples

        # 모든 (이미지, 메타데이터) 샘플 평탄화
        self.samples = []
        for tracklet in tracklets:
            for i, (frame_num, bbox) in enumerate(zip(tracklet.frames, tracklet.bboxes)):
                if max_samples and i >= max_samples:
                    break

                self.samples.append({
                    'tracklet_id': tracklet.id,
                    'frame_num': frame_num,
                    'bbox': bbox,
                    'confidence': tracklet.confidences[i] if i < len(tracklet.confidences) else 1.0
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 이미지 로드 및 크롭
        from .tracking import FramePathHandler

        frame_path = FramePathHandler.get_frame_path(self.frames_dir, sample['frame_num'])
        if frame_path is None:
            return None

        image = ImageProcessor.safe_load_image(frame_path)
        if image is None:
            return None

        # bbox 크롭
        crop = ImageProcessor.crop_bbox(image, sample['bbox'])
        if crop.size == 0:
            return None

        # PIL 변환 및 전처리
        crop = Image.fromarray(crop[:, :, ::-1])  # BGR to RGB
        if self.transform:
            crop = self.transform(crop)

        return {
            'image': crop,
            'tracklet_id': sample['tracklet_id'],
            'frame_num': sample['frame_num'],
            'bbox': sample['bbox']
        }


def collate_fn_reid(batch):
    """None 샘플 필터링 배치 합성 함수"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    return {
        'image': torch.stack([item['image'] for item in batch]),
        'tracklet_id': [item['tracklet_id'] for item in batch],
        'frame_num': [item['frame_num'] for item in batch],
        'bbox': [item['bbox'] for item in batch]
    }


class ReidFeatureExtractor:
    """Re-ID 특징 추출기"""

    def __init__(self, model, device: str = "cuda"):
        """
        Args:
            model: Re-ID 모델
            device: 장치 (cuda, cpu)
        """
        self.model = model
        self.device = device

    def extract_features(self, tracklets: List[Tracklet], frames_dir: str, batch_size: int = 32) -> Dict[int, np.ndarray]:
        """
        모든 tracklet에서 Re-ID 특징 추출

        Args:
            tracklets: Tracklet 리스트
            frames_dir: 프레임 폴더 경로
            batch_size: 배치 크기

        Returns:
            {tracklet_id: 특징 배열} 딕셔너리
        """
        print(f"\n=== Extracting Re-ID Features ===")
        print(f"Tracklets: {len(tracklets)}")
        print(f"Batch size: {batch_size}")

        # 데이터셋 및 DataLoader 생성
        dataset = TrackletReidDataset(
            tracklets=tracklets,
            frames_dir=frames_dir,
            transform=TransformProvider.get_reid_transform(),
            max_samples=5  # 각 tracklet에서 최대 5개 샘플 (빠른 추출)
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn_reid
        )

        # 특징 추출
        features_by_tracklet = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                if batch is None:
                    continue

                images = batch['image'].to(self.device)

                # 모델 추론
                feature = self.model(images)  # (batch_size, 512)
                feature = feature.cpu().numpy()

                # Tracklet별로 특징 저장
                for i, tracklet_id in enumerate(batch['tracklet_id']):
                    features_by_tracklet[tracklet_id].append(feature[i])

        print(f"Extracted features for {len(features_by_tracklet)} tracklets")

        return dict(features_by_tracklet)

    @staticmethod
    def compute_tracklet_avg_feature(features: List[np.ndarray]) -> np.ndarray:
        """
        Tracklet의 평균 특징 계산

        Args:
            features: 특징 배열 리스트 (각 512차원)

        Returns:
            평균 특징 (512차원)
        """
        if not features:
            return np.zeros(512)

        avg_feature = np.mean(np.array(features), axis=0)
        # 정규화
        avg_feature = avg_feature / (np.linalg.norm(avg_feature) + 1e-8)
        return avg_feature


class SimilarityCalculator:
    """유사도 계산"""

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        코사인 유사도 계산

        Args:
            vec1: 특징 벡터 1
            vec2: 특징 벡터 2

        Returns:
            유사도 (0~1)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    @staticmethod
    def compute_similarity_matrix(features_list: List[np.ndarray]) -> np.ndarray:
        """
        특징 리스트에 대한 유사도 행렬 계산

        Args:
            features_list: 특징 벡터 리스트

        Returns:
            유사도 행렬 (NxN)
        """
        n = len(features_list)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = SimilarityCalculator.cosine_similarity(features_list[i], features_list[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        return similarity_matrix
