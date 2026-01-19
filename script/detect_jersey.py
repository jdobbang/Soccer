#!/usr/bin/env python3
"""
Jersey Number Detection for Soccer Players
===========================================

Orange 팀 선수들의 유니폼 번호를 OCR로 검출하여 CSV로 저장
"""

import argparse
import os
import csv
import cv2
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any

from utils.ocr_adapter import create_ocr_engine, OCREngine


# Configuration
CONFIG = {
    # Filtering
    'team_color': 'orange',
    'min_color_confidence': 0.15,
    'min_bbox_area': 500,  # px, 너무 작은 bbox 제외

    # Crop regions
    'chest_vertical': (0.20, 0.45),
    'chest_horizontal': (0.15, 0.85),
    'back_vertical': (0.15, 0.40),
    'back_horizontal': (0.10, 0.90),
    'crop_mode': 'adaptive',  # 'chest', 'back', 'adaptive'

    # OCR
    'use_gpu': True,
    'ocr_engine': 'paddleocr',       # 'paddleocr' or 'easyocr'
    'ocr_use_angle_cls': True,       # PaddleOCR angle classification
    'ocr_det': True,                 # Enable detection
    'ocr_rec': True,                 # Enable recognition
    'ocr_cls': True,                 # Enable classification
    'ocr_show_log': False,           # Show PaddleOCR logs
    'min_ocr_confidence': 0.5,
    'preprocessing': True,

    # Validation
    'valid_number_range': (1, 99),
    'max_number_length': 2,
    'false_positive_threshold': 0.7,

    # Sampling
    'sampling_strategy': 'quality',  # 'all', 'quality', 'uniform'
    'quality_percentile': 0.70,
    'uniform_interval': 5,

    # Consolidation
    'vote_weight_count': 0.4,
    'vote_weight_confidence': 0.6,
}


def load_orange_team_detections(csv_path, min_color_conf=0.15, team_color='orange'):
    """
    test_color.csv에서 orange 팀만 필터링

    Args:
        csv_path: test_color.csv 경로
        min_color_conf: 최소 color confidence threshold
        team_color: 필터링할 팀 색상

    Returns:
        frame_data: {frame_num: [detection_dict, ...]}
    """
    frame_data = defaultdict(list)
    total_count = 0
    filtered_count = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_count += 1

            # 팀 색상 필터링
            if row['uniform_color'] != team_color:
                continue

            # Color confidence 필터링
            color_conf = float(row['color_confidence'])
            if color_conf < min_color_conf:
                continue

            filtered_count += 1
            frame = int(row['frame'])

            frame_data[frame].append({
                'image_name': row['image_name'],
                'track_id': int(row['track_id']),
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2']),
                'confidence': float(row['confidence']),
                'uniform_color': row['uniform_color'],
                'color_confidence': color_conf
            })

    print(f"Loaded {total_count} total detections")
    print(f"Filtered to {filtered_count} {team_color} team detections (min_conf={min_color_conf})")
    print(f"Covering {len(frame_data)} frames")

    return frame_data


def extract_number_region(image, bbox, region_type='adaptive', config=CONFIG):
    """
    bbox에서 유니폼 번호가 있는 영역 crop

    Args:
        image: 원본 이미지
        bbox: bounding box dict (x1, y1, x2, y2)
        region_type: 'chest', 'back', 'adaptive'
        config: configuration dict

    Returns:
        If region_type != 'adaptive': (crop, metadata)
        If region_type == 'adaptive': [(crop1, meta1), (crop2, meta2)]
    """
    h, w = image.shape[:2]

    x1 = max(0, int(bbox['x1']))
    y1 = max(0, int(bbox['y1']))
    x2 = min(w, int(bbox['x2']))
    y2 = min(h, int(bbox['y2']))

    box_w = x2 - x1
    box_h = y2 - y1

    # Chest region (앞면)
    chest_y_start, chest_y_end = config['chest_vertical']
    chest_x_start, chest_x_end = config['chest_horizontal']

    chest_y1 = y1 + int(box_h * chest_y_start)
    chest_y2 = y1 + int(box_h * chest_y_end)
    chest_x1 = x1 + int(box_w * chest_x_start)
    chest_x2 = x1 + int(box_w * chest_x_end)

    # Back region (뒷면)
    back_y_start, back_y_end = config['back_vertical']
    back_x_start, back_x_end = config['back_horizontal']

    back_y1 = y1 + int(box_h * back_y_start)
    back_y2 = y1 + int(box_h * back_y_end)
    back_x1 = x1 + int(box_w * back_x_start)
    back_x2 = x1 + int(box_w * back_x_end)

    # Boundary check
    chest_y1, chest_y2 = max(0, chest_y1), min(h, chest_y2)
    chest_x1, chest_x2 = max(0, chest_x1), min(w, chest_x2)
    back_y1, back_y2 = max(0, back_y1), min(h, back_y2)
    back_x1, back_x2 = max(0, back_x1), min(w, back_x2)

    if region_type == 'chest':
        crop = image[chest_y1:chest_y2, chest_x1:chest_x2]
        metadata = {'type': 'chest', 'x1': chest_x1, 'y1': chest_y1, 'x2': chest_x2, 'y2': chest_y2}
        return crop, metadata

    elif region_type == 'back':
        crop = image[back_y1:back_y2, back_x1:back_x2]
        metadata = {'type': 'back', 'x1': back_x1, 'y1': back_y1, 'x2': back_x2, 'y2': back_y2}
        return crop, metadata

    else:  # adaptive
        crops = []

        chest_crop = image[chest_y1:chest_y2, chest_x1:chest_x2]
        chest_meta = {'type': 'chest', 'x1': chest_x1, 'y1': chest_y1, 'x2': chest_x2, 'y2': chest_y2}
        crops.append((chest_crop, chest_meta))

        back_crop = image[back_y1:back_y2, back_x1:back_x2]
        back_meta = {'type': 'back', 'x1': back_x1, 'y1': back_y1, 'x2': back_x2, 'y2': back_y2}
        crops.append((back_crop, back_meta))

        return crops


def preprocess_for_ocr(crop):
    """
    OCR 정확도 향상을 위한 전처리

    Args:
        crop: 원본 crop 이미지

    Returns:
        List[preprocessed_images]
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return []

    # 1. Resize if too small
    min_dimension = 100
    h, w = crop.shape[:2]
    if min(h, w) < min_dimension:
        scale = min_dimension / min(h, w)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # 3. CLAHE (대비 향상)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # 4. Sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # 5. Adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

    # Return multiple versions
    return [sharpened, thresh1, thresh2]


def validate_jersey_number(text, confidence, config=CONFIG):
    """
    검출된 번호 검증

    Args:
        text: OCR 검출 텍스트
        confidence: OCR confidence
        config: configuration dict

    Returns:
        bool: valid or not
    """
    # Rule 1: Numeric only
    if not text.isdigit():
        return False

    # Rule 2: Length check
    if len(text) < 1 or len(text) > config['max_number_length']:
        return False

    number = int(text)

    # Rule 3: Valid range
    if number < config['valid_number_range'][0] or number > config['valid_number_range'][1]:
        return False

    # Rule 4: Confidence threshold
    if confidence < config['min_ocr_confidence']:
        return False

    # Rule 5: Higher threshold for false positive prone numbers
    false_positive_prone = ['88', '11', '00', '99']
    if text in false_positive_prone and confidence < config['false_positive_threshold']:
        return False

    return True


def detect_number_with_ocr(crop, ocr_engine: OCREngine, crop_meta, config=CONFIG):
    """
    OCR로 번호 검출 (PaddleOCR 또는 EasyOCR)

    Args:
        crop: crop 이미지
        ocr_engine: OCREngine instance (PaddleOCR or EasyOCR adapter)
        crop_meta: crop metadata
        config: configuration dict

    Returns:
        (jersey_number: str, confidence: float, ocr_bbox: dict, region_type: str) or None
    """
    if crop is None or crop.size == 0:
        return None

    # Preprocess
    if config['preprocessing']:
        preprocessed_crops = preprocess_for_ocr(crop)
    else:
        preprocessed_crops = [crop]

    best_result = None
    best_confidence = 0.0

    for proc_crop in preprocessed_crops:
        if proc_crop is None or proc_crop.size == 0:
            continue

        try:
            # Run EasyOCR
            results = ocr_engine.readtext(proc_crop)

            if results is None or len(results) == 0:
                continue

            # Parse results
            for detection in results:
                bbox_coords, text, confidence = detection

                # Extract digits only
                cleaned_text = ''.join(c for c in text if c.isdigit())

                if cleaned_text and confidence > best_confidence:
                    # Validate
                    if validate_jersey_number(cleaned_text, confidence, config):
                        # OCR bbox coordinates
                        bbox_coords = np.array(bbox_coords, dtype=np.int32)
                        ocr_bbox = {
                            'x': int(bbox_coords[0][0]),
                            'y': int(bbox_coords[0][1]),
                            'w': int(bbox_coords[2][0] - bbox_coords[0][0]),
                            'h': int(bbox_coords[2][1] - bbox_coords[0][1])
                        }

                        best_result = (cleaned_text, confidence, ocr_bbox, crop_meta['type'])
                        best_confidence = confidence

        except Exception as e:
            continue

    return best_result


def sample_detections(detections_df, strategy='quality', config=CONFIG):
    """
    성능 최적화를 위한 detection sampling

    Args:
        detections_df: DataFrame with all detections
        strategy: 'all', 'quality', 'uniform'
        config: configuration dict

    Returns:
        sampled DataFrame
    """
    if strategy == 'all':
        return detections_df

    detections_df = detections_df.copy()
    detections_df['bbox_area'] = (detections_df['x2'] - detections_df['x1']) * \
                                   (detections_df['y2'] - detections_df['y1'])

    if strategy == 'quality':
        # Top 30% by bbox area
        threshold = detections_df['bbox_area'].quantile(config['quality_percentile'])
        sampled = detections_df[detections_df['bbox_area'] > threshold]
        print(f"Quality sampling: {len(sampled)}/{len(detections_df)} detections (top {int((1-config['quality_percentile'])*100)}%)")

    elif strategy == 'uniform':
        # Every Nth frame
        interval = config['uniform_interval']
        sampled = detections_df[detections_df['frame'] % interval == 0]
        print(f"Uniform sampling: {len(sampled)}/{len(detections_df)} detections (every {interval} frames)")

    else:
        sampled = detections_df

    return sampled


def has_tracking_info(detections_df):
    """
    Detection 데이터인지 Tracking 데이터인지 판단

    Args:
        detections_df: DataFrame with columns including track_id

    Returns:
        bool: True if tracking data (track_id != -1), False if detection data
    """
    # track_id가 모두 -1이면 detection 데이터
    return not (detections_df['track_id'] == -1).all()


def consolidate_detections_by_track(detections_df, config=CONFIG):
    """
    track_id별로 여러 프레임의 검출 결과를 통합 (Tracking CSV용)
    Detection CSV인 경우 None 반환

    Args:
        detections_df: DataFrame with columns [frame, track_id, jersey_number, number_confidence]
        config: configuration dict

    Returns:
        consolidated_df: DataFrame with one row per track_id, or None if detection data
    """
    # Detection 데이터인 경우 consolidation 스킵
    if not has_tracking_info(detections_df):
        return None

    results = []

    # Filter out unknown numbers
    valid_df = detections_df[detections_df['jersey_number'] != 'unknown'].copy()

    if len(valid_df) == 0:
        # No valid detections to consolidate
        return pd.DataFrame(columns=['track_id', 'jersey_number', 'consolidated_confidence',
                                     'detection_count', 'first_frame', 'last_frame', 'sample_frames'])

    for track_id in valid_df['track_id'].unique():
        track_data = valid_df[valid_df['track_id'] == track_id]

        # Voting with confidence weighting
        number_votes = {}
        for _, row in track_data.iterrows():
            num = row['jersey_number']
            conf = row['number_confidence']

            if num not in number_votes:
                number_votes[num] = {'count': 0, 'total_conf': 0.0, 'max_conf': 0.0}

            number_votes[num]['count'] += 1
            number_votes[num]['total_conf'] += conf
            number_votes[num]['max_conf'] = max(number_votes[num]['max_conf'], conf)

        # Calculate weighted score
        best_number = None
        best_score = 0.0

        for num, stats in number_votes.items():
            avg_conf = stats['total_conf'] / stats['count']
            score = (stats['count'] * config['vote_weight_count']) + (avg_conf * config['vote_weight_confidence'])

            if score > best_score:
                best_score = score
                best_number = num

        # Get frame range
        frames = track_data['frame'].tolist()
        sample_frames = ','.join(map(str, sorted(frames)[:10]))  # First 10 frames

        results.append({
            'track_id': track_id,
            'jersey_number': best_number,
            'consolidated_confidence': best_score,
            'detection_count': len(track_data),
            'first_frame': min(frames),
            'last_frame': max(frames),
            'sample_frames': sample_frames
        })

    return pd.DataFrame(results)


def process_sequence(
    color_csv,
    frames_dir,
    output_dir,
    ocr_engine,
    config=CONFIG
):
    """
    전체 시퀀스 처리

    Args:
        color_csv: test_color.csv 경로
        frames_dir: 프레임 이미지 폴더
        output_dir: 출력 디렉토리
        ocr_engine: PaddleOCR instance
        config: configuration dict
    """
    # Load orange team detections
    print("\n" + "="*70)
    print("Loading Orange Team Detections")
    print("="*70)
    frame_data = load_orange_team_detections(
        color_csv,
        min_color_conf=config['min_color_confidence'],
        team_color=config['team_color']
    )

    if not frame_data:
        print("No detections found!")
        return

    # Convert to DataFrame for sampling
    all_detections = []
    for frame_num, dets in frame_data.items():
        for det in dets:
            det['frame'] = frame_num
            all_detections.append(det)

    detections_df = pd.DataFrame(all_detections)

    # Apply sampling
    print("\n" + "="*70)
    print("Applying Sampling Strategy")
    print("="*70)
    sampled_df = sample_detections(detections_df, strategy=config['sampling_strategy'], config=config)

    # Group back by frame
    sampled_frame_data = defaultdict(list)
    for _, row in sampled_df.iterrows():
        sampled_frame_data[row['frame']].append(row.to_dict())

    # Process frames
    print("\n" + "="*70)
    print("Processing Frames with OCR")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)
    detailed_csv_path = os.path.join(output_dir, 'jersey_numbers_detailed.csv')

    detailed_results = []
    sorted_frames = sorted(sampled_frame_data.keys())

    for frame_num in tqdm(sorted_frames, desc="OCR Processing"):
        detections = sampled_frame_data[frame_num]
        image_name = detections[0]['image_name']
        image_path = os.path.join(frames_dir, image_name)

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        for det in detections:
            # Extract number region(s)
            crops = extract_number_region(image, det, region_type=config['crop_mode'], config=config)

            # If adaptive mode, crops is a list
            if config['crop_mode'] == 'adaptive':
                best_detection = None
                best_conf = 0.0

                for crop, crop_meta in crops:
                    result = detect_number_with_ocr(crop, ocr_engine, crop_meta, config)
                    if result and result[1] > best_conf:
                        best_detection = result
                        best_conf = result[1]

                if best_detection:
                    jersey_number, number_conf, ocr_bbox, region_type = best_detection
                else:
                    jersey_number, number_conf, ocr_bbox, region_type = 'unknown', 0.0, {'x': 0, 'y': 0, 'w': 0, 'h': 0}, 'none'

            else:
                # Single crop
                crop, crop_meta = crops
                result = detect_number_with_ocr(crop, ocr_engine, crop_meta, config)

                if result:
                    jersey_number, number_conf, ocr_bbox, region_type = result
                else:
                    jersey_number, number_conf, ocr_bbox, region_type = 'unknown', 0.0, {'x': 0, 'y': 0, 'w': 0, 'h': 0}, 'none'

            # Save detailed result
            detailed_results.append({
                'frame': frame_num,
                'image_name': image_name,
                'track_id': det['track_id'],
                'x1': det['x1'],
                'y1': det['y1'],
                'x2': det['x2'],
                'y2': det['y2'],
                'bbox_confidence': det['confidence'],
                'uniform_color': det['uniform_color'],
                'color_confidence': det['color_confidence'],
                'jersey_number': jersey_number,
                'number_confidence': number_conf,
                'crop_region_type': region_type,
                'ocr_text_bbox_x': ocr_bbox['x'],
                'ocr_text_bbox_y': ocr_bbox['y'],
                'ocr_text_bbox_w': ocr_bbox['w'],
                'ocr_text_bbox_h': ocr_bbox['h']
            })

    # Save detailed CSV
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"\nDetailed results saved to: {detailed_csv_path}")
    print(f"Total detections processed: {len(detailed_df)}")

    # Consolidate by track_id (only for tracking data)
    print("\n" + "="*70)
    print("Consolidating Detections by Track ID")
    print("="*70)
    consolidated_df = consolidate_detections_by_track(detailed_df, config)

    if consolidated_df is not None:
        # Tracking data: save consolidated results
        consolidated_csv_path = os.path.join(output_dir, 'jersey_numbers_consolidated.csv')
        consolidated_df.to_csv(consolidated_csv_path, index=False)
        print(f"Consolidated results saved to: {consolidated_csv_path}")
        print(f"Total unique tracks with numbers: {len(consolidated_df)}")
    else:
        # Detection data: consolidation skipped
        print("Skipping consolidation (Detection data, no tracking info)")
        print(f"All {len(detailed_df)} detections saved in detailed CSV")

    # Print statistics
    print("\n" + "="*70)
    print("Jersey Number Distribution")
    print("="*70)

    # Detailed distribution
    number_dist = detailed_df[detailed_df['jersey_number'] != 'unknown']['jersey_number'].value_counts()
    print("\nFrom Detailed CSV:")
    for number, count in number_dist.head(20).items():
        print(f"  #{number}: {count} detections")

    unknown_count = len(detailed_df[detailed_df['jersey_number'] == 'unknown'])
    print(f"\n  Unknown: {unknown_count} detections")

    # Consolidated distribution (only if tracking data)
    if consolidated_df is not None:
        print("\nFrom Consolidated CSV (by Track ID):")
        for _, row in consolidated_df.sort_values('jersey_number').iterrows():
            print(f"  Track ID {row['track_id']:3d} = #{row['jersey_number']} "
                  f"(conf: {row['consolidated_confidence']:.3f}, {row['detection_count']} detections)")


def discover_csv_files(args) -> List[Dict[str, str]]:
    """
    Discover CSV files based on input arguments.
    Returns list of dicts with keys: csv_path, csv_name
    """
    csv_files = []

    if args.color_csv:
        # Single file mode
        csv_files.append({
            'csv_path': os.path.abspath(args.color_csv),
            'csv_name': os.path.basename(args.color_csv)
        })
    elif args.color_csv_dir:
        # Directory mode: find all *_color.csv files
        pattern = os.path.join(args.color_csv_dir, '**', '*_color.csv')
        paths = glob.glob(pattern, recursive=True)
        for path in sorted(paths):
            csv_files.append({
                'csv_path': os.path.abspath(path),
                'csv_name': os.path.relpath(path, args.color_csv_dir)
            })
    elif args.color_csv_pattern:
        # Glob pattern mode
        paths = glob.glob(args.color_csv_pattern, recursive=True)
        base_dir = os.path.commonpath(paths) if paths else '.'
        for path in sorted(paths):
            csv_files.append({
                'csv_path': os.path.abspath(path),
                'csv_name': os.path.relpath(path, base_dir)
            })

    return csv_files


def resolve_frames_dir(csv_path: str, args) -> str:
    """
    Resolve frames directory for a CSV file.
    Priority: frames_dir_pattern > args.frames_dir > inference
    """
    csv_dir = os.path.dirname(csv_path)
    csv_parent = os.path.dirname(csv_dir)
    csv_grandparent = os.path.dirname(csv_parent)

    # Try pattern replacement
    if args.frames_dir_pattern:
        frames_dir = args.frames_dir_pattern.format(
            csv_dir=csv_dir,
            csv_parent=csv_parent,
            csv_grandparent=csv_grandparent
        )
        if os.path.isdir(frames_dir):
            return os.path.abspath(frames_dir)
        raise ValueError(f"Frames dir from pattern not found: {frames_dir}")

    # Use provided frames_dir
    if args.frames_dir:
        if os.path.isdir(args.frames_dir):
            return os.path.abspath(args.frames_dir)
        raise ValueError(f"Frames dir not found: {args.frames_dir}")

    # Try to infer from CSV location
    candidates = [
        os.path.join(csv_dir, 'frames'),
        os.path.join(csv_dir, 'images'),
        os.path.join(csv_parent, 'frames'),
        os.path.join(csv_parent, 'images'),
        os.path.join(csv_parent, 'original_frames'),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)

    raise ValueError(f"Cannot find frames directory for {csv_path}")


def generate_output_paths(csv_path: str, csv_name: str,
                         base_output_dir: str, is_batch: bool) -> Dict[str, str]:
    """
    Generate output directory and file paths.
    Batch mode creates subdirectories, single mode uses base dir.
    """
    if is_batch:
        # Create subdirectory mirroring input structure
        csv_subdir = os.path.dirname(csv_name)
        output_dir = os.path.join(base_output_dir, csv_subdir)
    else:
        output_dir = base_output_dir

    os.makedirs(output_dir, exist_ok=True)

    return {
        'output_dir': output_dir,
        'detailed_csv': os.path.join(output_dir, 'jersey_numbers_detailed.csv'),
        'consolidated_csv': os.path.join(output_dir, 'jersey_numbers_consolidated.csv')
    }


def should_skip_csv(output_paths: Dict[str, str], skip_existing: bool) -> bool:
    """
    Check if CSV should be skipped (already processed).
    """
    if not skip_existing:
        return False

    detailed = output_paths['detailed_csv']
    consolidated = output_paths['consolidated_csv']

    # Both files must exist and be non-empty
    return (os.path.exists(detailed) and os.path.getsize(detailed) > 0 and
            os.path.exists(consolidated) and os.path.getsize(consolidated) > 0)


def print_batch_summary(summary: Dict[str, Any]):
    """
    Print comprehensive batch processing summary.
    """
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"Total CSV files: {summary['total']}")
    print(f"  ✓ Successfully processed: {summary['processed']}")
    print(f"  ⊘ Skipped (existing): {summary['skipped']}")
    print(f"  ✗ Failed: {summary['failed']}")

    if summary['errors']:
        print("\nErrors:")
        for error in summary['errors']:
            print(f"  ✗ {error}")

    print("="*70)


def process_batch(csv_files: List[Dict], ocr_engine, args, config: dict) -> Dict[str, Any]:
    """
    Process multiple CSV files in batch mode.
    """
    summary = {
        'total': len(csv_files),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }

    is_batch = len(csv_files) > 1

    for csv_info in tqdm(csv_files, desc="Batch Processing", unit="csv", disable=not is_batch):
        csv_path = csv_info['csv_path']
        csv_name = csv_info['csv_name']

        try:
            # Resolve paths
            frames_dir = resolve_frames_dir(csv_path, args)
            output_paths = generate_output_paths(
                csv_path, csv_name, args.output_dir, is_batch
            )

            # Skip check
            if should_skip_csv(output_paths, args.skip_existing):
                print(f"[SKIP] {csv_name} (already processed)")
                summary['skipped'] += 1
                continue

            # Process
            print(f"\n{'='*70}")
            print(f"Processing: {csv_name}")
            print(f"{'='*70}")

            process_sequence(
                color_csv=csv_path,
                frames_dir=frames_dir,
                output_dir=output_paths['output_dir'],
                ocr_engine=ocr_engine,
                config=config
            )

            summary['processed'] += 1

            # Clear GPU cache between files
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            summary['failed'] += 1
            error_msg = f"{csv_name}: {str(e)}"
            summary['errors'].append(error_msg)

            print(f"\n[ERROR] Failed to process {csv_name}")
            print(f"  Error: {e}")

            if not args.continue_on_error:
                raise
            else:
                print("  Continuing with next file...")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Detect jersey numbers from soccer team color classification results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 실행
  python detect_jersey_numbers.py \\
      --color_csv results/player/yolo11x/test_color.csv \\
      --frames_dir original_frames/ \\
      --output_dir results/player/yolo11x/

  # Quality sampling (상위 30%만 처리)
  python detect_jersey_numbers.py \\
      --color_csv results/player/yolo11x/test_color.csv \\
      --frames_dir original_frames/ \\
      --output_dir results/player/yolo11x/ \\
      --sampling_strategy quality \\
      --quality_percentile 0.70

  # 모든 프레임 처리
  python detect_jersey_numbers.py \\
      --color_csv results/player/yolo11x/test_color.csv \\
      --frames_dir original_frames/ \\
      --output_dir results/player/yolo11x/ \\
      --sampling_strategy all
        """
    )

    # I/O arguments - Mutually exclusive input group
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--color_csv', type=str,
                        help='Path to test_color.csv (single file mode)')
    input_group.add_argument('--color_csv_dir', type=str,
                        help='Directory containing CSV files (batch mode)')
    input_group.add_argument('--color_csv_pattern', type=str,
                        help='Glob pattern for CSV files (e.g., "results/**/test_color.csv")')

    parser.add_argument('--frames_dir', type=str, default=None,
                        help='Path to frames directory (required for single mode, optional for batch)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for CSV results')

    # Batch-specific arguments
    parser.add_argument('--frames_dir_pattern', type=str, default=None,
                        help='Pattern to derive frames_dir: {csv_dir}, {csv_parent}, {csv_grandparent}')
    parser.add_argument('--continue_on_error', action='store_true', default=True,
                        help='Continue processing other files if one fails (default: True)')
    parser.add_argument('--no_continue_on_error', action='store_false', dest='continue_on_error',
                        help='Stop on first error')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip CSVs that already have output files')

    # Filtering
    parser.add_argument('--team_color', type=str, default='orange',
                        help='Team color to process (default: orange)')
    parser.add_argument('--min_color_confidence', type=float, default=0.15,
                        help='Minimum color confidence (default: 0.15)')
    parser.add_argument('--min_bbox_area', type=float, default=500,
                        help='Minimum bbox area in pixels (default: 500)')

    # OCR Engine Selection
    parser.add_argument('--ocr_engine', type=str, default='paddleocr',
                        choices=['paddleocr', 'easyocr'],
                        help='OCR engine to use (default: paddleocr)')
    parser.add_argument('--no_angle_cls', action='store_false', dest='use_angle_cls',
                        help='Disable PaddleOCR angle classification')
    parser.add_argument('--ocr_show_log', action='store_true',
                        help='Show PaddleOCR logs')

    # OCR settings
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU for OCR (default: True)')
    parser.add_argument('--no_gpu', action='store_false', dest='use_gpu',
                        help='Disable GPU, use CPU instead')
    parser.add_argument('--min_ocr_confidence', type=float, default=0.5,
                        help='Minimum OCR confidence (default: 0.5)')
    parser.add_argument('--no_preprocessing', action='store_false', dest='preprocessing',
                        help='Disable image preprocessing')

    # Crop settings
    parser.add_argument('--crop_mode', type=str, default='adaptive',
                        choices=['chest', 'back', 'adaptive'],
                        help='Crop region mode (default: adaptive)')

    # Sampling
    parser.add_argument('--sampling_strategy', type=str, default='quality',
                        choices=['all', 'quality', 'uniform'],
                        help='Sampling strategy (default: quality)')
    parser.add_argument('--quality_percentile', type=float, default=0.70,
                        help='Percentile threshold for quality sampling (default: 0.70)')
    parser.add_argument('--uniform_interval', type=int, default=5,
                        help='Frame interval for uniform sampling (default: 5)')

    args = parser.parse_args()

    # Validate arguments
    if args.color_csv and not args.frames_dir:
        parser.error("--frames_dir is required when using --color_csv")

    # Update config
    config = CONFIG.copy()
    config['team_color'] = args.team_color
    config['min_color_confidence'] = args.min_color_confidence
    config['min_bbox_area'] = args.min_bbox_area
    config['use_gpu'] = args.use_gpu
    config['ocr_engine'] = args.ocr_engine
    config['ocr_use_angle_cls'] = args.use_angle_cls
    config['ocr_show_log'] = getattr(args, 'ocr_show_log', False)
    config['min_ocr_confidence'] = args.min_ocr_confidence
    config['preprocessing'] = args.preprocessing
    config['crop_mode'] = args.crop_mode
    config['sampling_strategy'] = args.sampling_strategy
    config['quality_percentile'] = args.quality_percentile
    config['uniform_interval'] = args.uniform_interval

    # Initialize OCR Engine
    print("="*70)
    print(f"Initializing OCR Engine: {config['ocr_engine'].upper()}")
    print("="*70)
    print(f"GPU: {args.use_gpu}")

    ocr = create_ocr_engine(
        engine_type=config['ocr_engine'],
        lang='en',
        use_gpu=args.use_gpu,
        use_angle_cls=config['ocr_use_angle_cls'],
        det=config['ocr_det'],
        rec=config['ocr_rec'],
        cls=config['ocr_cls'],
        show_log=config['ocr_show_log']
    )

    print(f"{config['ocr_engine'].upper()} initialized successfully")

    # Discover CSV files
    csv_files = discover_csv_files(args)

    if not csv_files:
        print("No CSV files found!")
        return

    is_batch = len(csv_files) > 1

    if is_batch:
        print(f"\n{'='*70}")
        print(f"BATCH MODE: Processing {len(csv_files)} CSV files")
        print(f"{'='*70}")

    # Process
    summary = process_batch(csv_files, ocr, args, config)

    # Summary
    if is_batch:
        print_batch_summary(summary)

    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
