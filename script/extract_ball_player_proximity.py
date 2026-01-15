#!/usr/bin/env python3
"""
Extract segments where the ball appears near a specific player.

This script filters player tracking data to keep only frames where:
1. Both the player and ball are detected in the same frame
2. The distance between their centers is within a threshold (default: 100 pixels)

Segments of consecutive frames are identified and tagged with segment IDs.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_data(player_csv, ball_csv):
    """Load and prepare player and ball tracking data."""
    print(f"Loading player tracking data from: {player_csv}")
    player_df = pd.read_csv(player_csv)

    print(f"Loading ball detection data from: {ball_csv}")
    ball_df = pd.read_csv(ball_csv)

    # Rename ball columns to avoid confusion during merge
    ball_df_renamed = ball_df.rename(columns={
        'x1': 'ball_x1',
        'y1': 'ball_y1',
        'x2': 'ball_x2',
        'y2': 'ball_y2',
        'confidence': 'ball_confidence',
        'width': 'ball_width',
        'height': 'ball_height'
    }).drop(columns=['object_id'])  # Drop object_id as it's not needed

    print(f"Player data: {len(player_df)} rows")
    print(f"Ball data: {len(ball_df)} detections")

    return player_df, ball_df_renamed


def merge_player_ball(player_df, ball_df):
    """Merge player and ball data on frame number."""
    print("\nMerging player and ball data on frame number...")
    merged = pd.merge(player_df, ball_df, on='frame', how='inner')
    print(f"After merge: {len(merged)} frames have both player and ball detection")

    return merged


def calculate_distance(df):
    """Calculate center-to-center distance between player and ball."""
    print("Calculating distance between player and ball centers...")

    # Calculate centers
    player_cx = (df['x1'] + df['x2']) / 2
    player_cy = (df['y1'] + df['y2']) / 2
    ball_cx = (df['ball_x1'] + df['ball_x2']) / 2
    ball_cy = (df['ball_y1'] + df['ball_y2']) / 2

    # Calculate Euclidean distance
    df['distance'] = np.sqrt((player_cx - ball_cx)**2 + (player_cy - ball_cy)**2)

    return df


def filter_by_distance(df, threshold):
    """Filter frames by distance threshold."""
    print(f"\nFiltering by distance threshold: {threshold} pixels")
    filtered = df[df['distance'] <= threshold].copy()
    print(f"After filtering: {len(filtered)} frames with distance <= {threshold} pixels")

    if len(filtered) == 0:
        print("Warning: No frames found within distance threshold!")
        return filtered

    print(f"Distance statistics:")
    print(f"  Min: {filtered['distance'].min():.2f} pixels")
    print(f"  Max: {filtered['distance'].max():.2f} pixels")
    print(f"  Mean: {filtered['distance'].mean():.2f} pixels")
    print(f"  Median: {filtered['distance'].median():.2f} pixels")

    return filtered


def identify_segments(df):
    """Identify continuous segments of frames and assign segment IDs."""
    if len(df) == 0:
        return df

    print("\nIdentifying continuous segments...")

    # Sort by frame to ensure correct ordering
    df = df.sort_values('frame').reset_index(drop=True)

    # Calculate frame gaps
    frame_diff = df['frame'].diff()

    # A new segment starts when frame gap > 1
    segment_boundaries = frame_diff > 1
    segment_id = segment_boundaries.cumsum()

    df['segment_id'] = segment_id + 1  # +1 to start from 1 instead of 0

    segment_counts = df['segment_id'].value_counts().sort_index()
    print(f"Found {len(segment_counts)} segments")
    print(f"Segment lengths (frames per segment):")
    for seg_id, count in segment_counts.items():
        frame_range = df[df['segment_id'] == seg_id]['frame']
        print(f"  Segment {seg_id}: {count} frames (frame {frame_range.min()}-{frame_range.max()})")

    return df


def select_output_columns(df):
    """Select and order output columns."""
    # Player columns (keep all original)
    player_cols = [
        'frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence',
        'jersey_number', 'number_confidence', 'uniform_color',
        'color_confidence', 'crop_region_type'
    ]

    # Ball columns
    ball_cols = ['ball_x1', 'ball_y1', 'ball_x2', 'ball_y2', 'ball_confidence']

    # Calculated columns
    calc_cols = ['distance', 'segment_id']

    # Combine and filter to only columns that exist
    all_cols = player_cols + ball_cols + calc_cols
    existing_cols = [col for col in all_cols if col in df.columns]

    return df[existing_cols]


def save_output(df, output_path):
    """Save filtered data to CSV."""
    print(f"\nSaving output to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} frames to {output_path}")


def print_summary(df, threshold):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total frames extracted: {len(df)}")
    print(f"Distance threshold: {threshold} pixels")
    print(f"Number of segments: {df['segment_id'].max() if len(df) > 0 else 0}")

    if len(df) > 0:
        avg_segment_length = len(df) / df['segment_id'].max()
        print(f"Average segment length: {avg_segment_length:.1f} frames")
        print(f"Frame range: {df['frame'].min()}-{df['frame'].max()}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract ball-player proximity segments from tracking data'
    )
    parser.add_argument(
        '--player-csv',
        default='/workspace/Soccer/tracking_results/result_0_45000/csv/player_10_filtered_step2.csv',
        help='Path to player tracking CSV'
    )
    parser.add_argument(
        '--ball-csv',
        default='/workspace/Soccer/results/yolo11x_1280_ball/test_ball.csv',
        help='Path to ball detection CSV'
    )
    parser.add_argument(
        '--output',
        default='/workspace/Soccer/tracking_results/result_0_45000/csv/player_10_ball_proximity.csv',
        help='Path to output CSV'
    )
    parser.add_argument(
        '--distance-threshold',
        type=float,
        default=100,
        help='Distance threshold in pixels (default: 100)'
    )

    args = parser.parse_args()

    print("="*70)
    print("Extract Ball-Player Proximity Segments")
    print("="*70 + "\n")

    # Process data
    player_df, ball_df = load_data(args.player_csv, args.ball_csv)
    merged_df = merge_player_ball(player_df, ball_df)
    merged_df = calculate_distance(merged_df)
    filtered_df = filter_by_distance(merged_df, args.distance_threshold)

    if len(filtered_df) > 0:
        filtered_df = identify_segments(filtered_df)
        output_df = select_output_columns(filtered_df)
        save_output(output_df, args.output)
    else:
        print("No frames found within distance threshold. Creating empty output file.")
        output_df = pd.DataFrame()
        save_output(output_df, args.output)

    print_summary(filtered_df, args.distance_threshold)


if __name__ == '__main__':
    main()
