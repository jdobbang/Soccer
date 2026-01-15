#!/usr/bin/env python3
"""
Visualize player and ball bounding boxes on video frames.

This script reads the proximity data CSV and draws bounding boxes for both
the player and ball on each frame, then saves the annotated frames.
"""

import argparse
import pandas as pd
import cv2
import numpy as np
from pathlib import Path


def load_proximity_data(csv_path):
    """Load proximity data from CSV."""
    print(f"Loading proximity data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} frames")
    return df


def get_frame_path(frame_num, frames_dir):
    """Get the frame file path for a given frame number."""
    frame_filename = f"frame_{frame_num:06d}.jpg"
    frame_path = frames_dir / frame_filename
    return frame_path


def load_frame(frame_path):
    """Load a frame image."""
    if not frame_path.exists():
        print(f"Warning: Frame not found at {frame_path}")
        return None
    return cv2.imread(str(frame_path))


def draw_box(image, x1, y1, x2, y2, color, label="", thickness=2):
    """Draw a bounding box on the image."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = color
        thickness_text = 1

        # Get text size to create background
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness_text)

        # Draw background for text
        cv2.rectangle(image,
                      (x1, y1 - text_height - baseline - 4),
                      (x1 + text_width + 4, y1),
                      color, -1)

        # Draw text
        cv2.putText(image, label, (x1 + 2, y1 - baseline - 2),
                   font, font_scale, (255, 255, 255), thickness_text)


def visualize_frame(frame, row, output_dir=None):
    """Draw bounding boxes and optionally save the frame."""
    frame_num = int(row['frame'])

    # Draw player bounding box (green)
    player_color = (0, 255, 0)  # Green in BGR
    player_label = f"Player {int(row['track_id'])} (conf: {row['confidence']:.2f})"
    draw_box(frame, row['x1'], row['y1'], row['x2'], row['y2'],
             player_color, player_label)

    # Draw ball bounding box (red)
    ball_color = (0, 0, 255)  # Red in BGR
    ball_label = f"Ball (conf: {row['ball_confidence']:.2f})"
    draw_box(frame, row['ball_x1'], row['ball_y1'], row['ball_x2'], row['ball_y2'],
             ball_color, ball_label)

    # Draw distance info
    distance = row['distance']
    segment_id = int(row['segment_id'])
    info_text = f"Frame: {frame_num} | Distance: {distance:.1f}px | Segment: {segment_id}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)  # White
    thickness = 1

    # Draw background for info text
    (text_width, text_height), baseline = cv2.getTextSize(info_text, font, font_scale, thickness)
    cv2.rectangle(frame, (5, 5), (10 + text_width, 15 + text_height), (0, 0, 0), -1)
    cv2.putText(frame, info_text, (10, 20), font, font_scale, font_color, thickness)

    # Save if output directory specified
    if output_dir:
        output_path = output_dir / f"frame_{frame_num:06d}_annotated.jpg"
        cv2.imwrite(str(output_path), frame)
        return output_path

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Visualize player and ball bounding boxes on frames'
    )
    parser.add_argument(
        '--proximity-csv',
        default='/workspace/Soccer/tracking_results/result_0_45000/csv/player_10_ball_proximity_200px.csv',
        help='Path to proximity CSV file'
    )
    parser.add_argument(
        '--frames-dir',
        default='/workspace/Soccer/frames',
        help='Path to frames directory'
    )
    parser.add_argument(
        '--output-dir',
        default='/workspace/Soccer/tracking_results/result_0_45000/visualized_frames',
        help='Output directory for annotated frames'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process (None = all)'
    )
    parser.add_argument(
        '--segment-id',
        type=int,
        default=None,
        help='Visualize only specific segment ID'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir)

    print("="*70)
    print("Visualize Ball-Player Proximity Frames")
    print("="*70 + "\n")

    # Load data
    df = load_proximity_data(args.proximity_csv)

    # Filter by segment if specified
    if args.segment_id:
        df = df[df['segment_id'] == args.segment_id]
        print(f"Filtered to segment {args.segment_id}: {len(df)} frames\n")

    # Limit frames if specified
    if args.max_frames:
        df = df.head(args.max_frames)
        print(f"Processing first {len(df)} frames\n")

    # Process frames
    processed = 0
    skipped = 0

    for idx, row in df.iterrows():
        frame_num = int(row['frame'])
        frame_path = get_frame_path(frame_num, frames_dir)

        # Load frame
        frame = load_frame(frame_path)
        if frame is None:
            skipped += 1
            continue

        # Visualize and save
        output_path = visualize_frame(frame, row, output_dir)
        processed += 1

        if processed % 50 == 0:
            print(f"Processed {processed} frames...")

        if args.max_frames and processed >= args.max_frames:
            break

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Successfully processed: {processed} frames")
    print(f"Skipped: {skipped} frames")
    print(f"Output directory: {output_dir}")
    print(f"Total files saved: {len(list(output_dir.glob('*.jpg')))}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
