#!/usr/bin/env python3
"""
Filter tracking results by jersey number detection.

This script filters tracking CSV to only include tracklets where a specific
jersey number was detected at least once, using bbox+frame matching.
"""

import pandas as pd
import argparse
import os
import sys

# Default IoU threshold for bbox matching
DEFAULT_IOU_THRESHOLD = 0.9


def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        IoU value (0.0 ~ 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_tracking_csv(path):
    """
    Load tracking CSV file.

    Args:
        path: Path to tracking CSV file

    Returns:
        DataFrame with columns: frame, track_id, x1, y1, x2, y2, confidence
    """
    print(f"Loading tracking data from {path}...")

    if not os.path.exists(path):
        print(f"Error: Tracking CSV not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    # Validate required columns
    required_cols = ['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence']
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"Error: Tracking CSV is missing required columns: {missing}")
        sys.exit(1)

    print(f"Loaded {len(df)} tracking rows with {df['track_id'].nunique()} unique track_ids")
    print(f"Frame range: {df['frame'].min()} - {df['frame'].max()}")

    return df


def load_jersey_csv(path, target_number):
    """
    Load jersey detection CSV file and filter to target jersey number.

    Args:
        path: Path to jersey detection CSV file
        target_number: Target jersey number to filter

    Returns:
        DataFrame with jersey detections for target number only
    """
    print(f"\nLoading jersey detection data from {path}...")

    if not os.path.exists(path):
        print(f"Error: Jersey CSV not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    # Validate required columns
    required_cols = ['frame', 'x1', 'y1', 'x2', 'y2', 'jersey_number',
                     'number_confidence', 'uniform_color', 'color_confidence', 'crop_region_type']
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"Error: Jersey CSV is missing required columns: {missing}")
        sys.exit(1)

    total_rows = len(df)

    # Convert jersey_number to string for consistent comparison
    df['jersey_number'] = df['jersey_number'].astype(str)

    # Filter to target jersey number only
    target_df = df[df['jersey_number'] == target_number].copy()
    target_rows = len(target_df)

    print(f"Loaded {total_rows} jersey detections")
    print(f"Detections with jersey #{target_number}: {target_rows}")

    return target_df


def find_matching_track_ids(tracking_df, jersey_df, iou_threshold=DEFAULT_IOU_THRESHOLD):
    """
    Find track_ids that match jersey detections based on frame + bbox IoU matching.

    Args:
        tracking_df: Tracking DataFrame
        jersey_df: Jersey detection DataFrame (already filtered to target number)
        iou_threshold: Minimum IoU to consider a match (default: 0.9)

    Returns:
        Set of track_ids that have at least one matching jersey detection
    """
    print("\n" + "="*70)
    print(f"Finding Track IDs with Jersey Detections (IoU >= {iou_threshold})")
    print("="*70)

    matching_track_ids = set()
    match_details = []  # For logging

    # Group jersey detections by frame for faster lookup
    jersey_by_frame = {}
    for _, row in jersey_df.iterrows():
        frame = int(row['frame'])
        if frame not in jersey_by_frame:
            jersey_by_frame[frame] = []
        jersey_by_frame[frame].append({
            'bbox': (float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])),
            'jersey_number': row['jersey_number'],
            'number_confidence': row['number_confidence']
        })

    print(f"Jersey detections across {len(jersey_by_frame)} frames")

    # Find matching track_ids
    for _, row in tracking_df.iterrows():
        frame = int(row['frame'])

        # Skip if no jersey detections in this frame
        if frame not in jersey_by_frame:
            continue

        track_bbox = (float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2']))

        # Check IoU with each jersey detection in this frame
        for jersey_det in jersey_by_frame[frame]:
            iou = calculate_iou(track_bbox, jersey_det['bbox'])

            if iou >= iou_threshold:
                track_id = int(row['track_id'])
                if track_id not in matching_track_ids:
                    matching_track_ids.add(track_id)
                    match_details.append({
                        'track_id': track_id,
                        'frame': frame,
                        'bbox': track_bbox,
                        'iou': iou
                    })
                break  # Found a match for this tracking row

    print(f"Matching track_ids found: {len(matching_track_ids)}")

    if match_details:
        print("\nFirst detection per track_id:")
        for detail in sorted(match_details, key=lambda x: x['track_id']):
            print(f"  Track {detail['track_id']}: frame {detail['frame']}, "
                  f"bbox ({detail['bbox'][0]:.1f}, {detail['bbox'][1]:.1f}, "
                  f"{detail['bbox'][2]:.1f}, {detail['bbox'][3]:.1f}), "
                  f"IoU={detail['iou']:.3f}")

    return matching_track_ids


def merge_tracking_with_jersey(tracking_df, jersey_df, valid_track_ids, iou_threshold=DEFAULT_IOU_THRESHOLD):
    """
    Merge tracking data with jersey detection information using IoU matching.

    Args:
        tracking_df: Full tracking DataFrame
        jersey_df: Jersey detection DataFrame (filtered to target number)
        valid_track_ids: Set of track_ids to include
        iou_threshold: Minimum IoU to consider a match (default: 0.9)

    Returns:
        Merged DataFrame with tracking + jersey columns
    """
    print("\n" + "="*70)
    print("Merging Tracking and Jersey Data")
    print("="*70)

    # Filter tracking data to only valid track_ids
    filtered_tracking = tracking_df[tracking_df['track_id'].isin(valid_track_ids)].copy()
    print(f"Filtered tracking to {len(filtered_tracking)} rows for {len(valid_track_ids)} track_ids")

    # Group jersey detections by frame for faster lookup
    jersey_by_frame = {}
    for _, row in jersey_df.iterrows():
        frame = int(row['frame'])
        if frame not in jersey_by_frame:
            jersey_by_frame[frame] = []
        jersey_by_frame[frame].append({
            'bbox': (float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])),
            'uniform_color': row['uniform_color'],
            'color_confidence': row['color_confidence'],
            'jersey_number': row['jersey_number'],
            'number_confidence': row['number_confidence'],
            'crop_region_type': row['crop_region_type']
        })

    # Add jersey columns to tracking data
    jersey_numbers = []
    number_confidences = []
    uniform_colors = []
    color_confidences = []
    crop_region_types = []

    for _, row in filtered_tracking.iterrows():
        frame = int(row['frame'])
        track_bbox = (float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2']))

        # Default values
        best_match = None
        best_iou = 0.0

        # Find best matching jersey detection in this frame
        if frame in jersey_by_frame:
            for jersey_det in jersey_by_frame[frame]:
                iou = calculate_iou(track_bbox, jersey_det['bbox'])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match = jersey_det

        if best_match:
            jersey_numbers.append(best_match['jersey_number'])
            number_confidences.append(best_match['number_confidence'])
            uniform_colors.append(best_match['uniform_color'])
            color_confidences.append(best_match['color_confidence'])
            crop_region_types.append(best_match['crop_region_type'])
        else:
            jersey_numbers.append('unknown')
            number_confidences.append(0.0)
            uniform_colors.append('')
            color_confidences.append(0.0)
            crop_region_types.append('none')

    # Add columns to dataframe
    filtered_tracking['jersey_number'] = jersey_numbers
    filtered_tracking['number_confidence'] = number_confidences
    filtered_tracking['uniform_color'] = uniform_colors
    filtered_tracking['color_confidence'] = color_confidences
    filtered_tracking['crop_region_type'] = crop_region_types

    # Sort by frame, then track_id
    merged = filtered_tracking.sort_values(['frame', 'track_id']).reset_index(drop=True)

    # Statistics
    frames_with_jersey = (merged['jersey_number'] != 'unknown').sum()
    print(f"Frames with jersey info: {frames_with_jersey} ({100*frames_with_jersey/len(merged):.1f}%)")
    print(f"Frames without jersey info: {len(merged) - frames_with_jersey} "
          f"({100*(len(merged)-frames_with_jersey)/len(merged):.1f}%)")

    return merged


def save_filtered_results(merged_df, output_path):
    """
    Save merged results to CSV file.

    Args:
        merged_df: Merged DataFrame
        output_path: Output CSV file path
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Save to CSV
    merged_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Output rows: {len(merged_df)}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter tracking results by jersey number detection (bbox+frame matching)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter for player #10 (default)
  python filter_tracking_by_jersey.py \\
      --tracking tracking_results/result_0_45000/csv/step4_post_interpolated.csv \\
      --jersey detection_results/yolo11x/jersey_numbers_detailed.csv \\
      --number 10 \\
      --output tracking_results/result_0_45000/csv/player_10_filtered.csv

  # Filter for player #42
  python filter_tracking_by_jersey.py \\
      --tracking tracking.csv \\
      --jersey jersey.csv \\
      --number 42 \\
      --output player_42.csv
        """
    )

    # Required arguments
    parser.add_argument('--tracking', type=str, required=True,
                        help='Path to tracking CSV (e.g., step4_post_interpolated.csv)')
    parser.add_argument('--jersey', type=str, required=True,
                        help='Path to jersey detection CSV (e.g., jersey_numbers_detailed.csv)')
    parser.add_argument('--number', type=str, default='10',
                        help='Target jersey number to filter (default: "10")')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV path for filtered results')
    parser.add_argument('--iou', type=float, default=DEFAULT_IOU_THRESHOLD,
                        help=f'IoU threshold for bbox matching (default: {DEFAULT_IOU_THRESHOLD})')

    args = parser.parse_args()

    # Print header
    print("="*70)
    print(f"Jersey Number Filtering Pipeline (IoU >= {args.iou})")
    print("="*70)
    print(f"Input Files:")
    print(f"  - Tracking CSV: {args.tracking}")
    print(f"  - Jersey CSV: {args.jersey}")
    print(f"Target Jersey Number: {args.number}")
    print(f"IoU Threshold: {args.iou}")
    print("="*70)

    # Execute pipeline
    # 1. Load tracking CSV
    tracking_df = load_tracking_csv(args.tracking)

    # 2. Load jersey detection CSV (filtered to target number)
    jersey_df = load_jersey_csv(args.jersey, args.number)

    if len(jersey_df) == 0:
        print(f"\nWARNING: No detections found for jersey number '{args.number}'")
        print("Creating empty output file with headers only")

        # Create empty DataFrame with correct columns
        empty_df = pd.DataFrame(columns=[
            'frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence',
            'uniform_color', 'color_confidence', 'jersey_number',
            'number_confidence', 'crop_region_type'
        ])
        empty_df.to_csv(args.output, index=False)

        print(f"Empty output saved to: {args.output}")
        sys.exit(0)

    # 3. Find track_ids with matching jersey detections (bbox IoU matching)
    valid_track_ids = find_matching_track_ids(tracking_df, jersey_df, args.iou)

    if len(valid_track_ids) == 0:
        print(f"\nWARNING: No matching track_ids found for jersey number '{args.number}'")
        print("This might indicate bbox coordinate mismatch between files")
        print("Creating empty output file with headers only")

        # Create empty DataFrame with correct columns
        empty_df = pd.DataFrame(columns=[
            'frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence',
            'uniform_color', 'color_confidence', 'jersey_number',
            'number_confidence', 'crop_region_type'
        ])
        empty_df.to_csv(args.output, index=False)

        print(f"Empty output saved to: {args.output}")
        sys.exit(0)

    # 4. Print track details
    print("\n" + "="*70)
    print(f"Track IDs matching jersey #{args.number}: {len(valid_track_ids)}")
    print("="*70)
    print(f"Track IDs: {sorted(valid_track_ids)}")

    print("\nTrack Details:")
    for track_id in sorted(valid_track_ids):
        track_frames = tracking_df[tracking_df['track_id'] == track_id]
        frame_count = len(track_frames)
        frame_range = f"{track_frames['frame'].min()} - {track_frames['frame'].max()}"

        print(f"  Track {track_id}: {frame_count} frames (frame {frame_range})")

    # 5. Merge tracking with jersey data
    merged_df = merge_tracking_with_jersey(tracking_df, jersey_df, valid_track_ids, args.iou)

    # 6. Save results
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)
    save_filtered_results(merged_df, args.output)

    # Final statistics
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)
    print(f"Input tracking rows: {len(tracking_df)}")
    print(f"Output tracking rows: {len(merged_df)}")
    print(f"Reduction: {100*(1 - len(merged_df)/len(tracking_df)):.1f}%")
    print(f"Unique track_ids in output: {merged_df['track_id'].nunique()}")
    print("="*70)


if __name__ == '__main__':
    main()