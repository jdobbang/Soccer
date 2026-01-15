#!/usr/bin/env python3
"""
Visualize tracking results by drawing bounding boxes with track IDs on frames.
Reads tracking results from CSV and frames from disk, outputs visualized frames.
"""


import cv2
import csv
import os
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def generate_color(track_id):
    """Generate a consistent color for each track ID."""
    np.random.seed(track_id)
    return tuple(np.random.randint(0, 255, 3).tolist())


def visualize_tracking(tracking_csv, frames_dir, output_dir_base):
    """
    Visualize tracking results on frames.
    
    Args:
        tracking_csv: Path to tracking results CSV file
        frames_dir: Directory containing frame images
        output_dir: Directory to save visualized frames
    """
    # Read tracking results grouped by frame (do this before creating output dir)
    print("Reading tracking results...")
    tracks_by_frame = defaultdict(list)

    with open(tracking_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_num = int(row['frame'])
            track_id = int(row['track_id'])
            x1 = float(row['x1'])
            y1 = float(row['y1'])
            x2 = float(row['x2'])
            y2 = float(row['y2'])
            confidence = float(row['confidence'])
            tracks_by_frame[frame_num].append({
                'track_id': track_id,
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence
            })

    # Determine min/max frame number
    if tracks_by_frame:
        frame_numbers = sorted(tracks_by_frame.keys())
        min_frame = frame_numbers[0]
        max_frame = frame_numbers[-1]
        # Update output_dir to include frame range
        output_dir = os.path.join(output_dir_base, f"result_{min_frame}_{max_frame}/vis")
    else:
        frame_numbers = []
        min_frame = max_frame = None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ...existing code...
    
    # Get unique track IDs and assign colors
    all_track_ids = set()
    for tracks in tracks_by_frame.values():
        for track in tracks:
            all_track_ids.add(track['track_id'])
    
    track_colors = {track_id: generate_color(track_id) for track_id in all_track_ids}
    
    print(f"Found {len(tracks_by_frame)} frames with tracking data")
    print(f"Found {len(all_track_ids)} unique track IDs")
    
    # Process each frame
    frame_numbers = sorted(tracks_by_frame.keys())
    print("Visualizing frames...")
    
    for frame_num in tqdm(frame_numbers):
        # Find corresponding frame file
        frame_path = os.path.join(frames_dir, f"frame_{frame_num:06d}.jpg")
        
        if not os.path.exists(frame_path):
            print(f"Warning: Frame file not found: {frame_path}")
            continue
        
        # Read frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame: {frame_path}")
            continue
        
        # Draw bounding boxes and track IDs
        for track in tracks_by_frame[frame_num]:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            confidence = track['confidence']
            color = track_colors[track_id]
            
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and confidence
            label = f"ID:{track_id} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background for text
            cv2.rectangle(frame, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save visualized frame
        output_path = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
        cv2.imwrite(output_path, frame)
    
    print(f"Visualization complete! Saved {len(frame_numbers)} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize tracking results on frames"
    )
    parser.add_argument(
        "tracking_csv",
        type=str,
        help="Path to tracking results CSV file"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="frames",
        help="Directory containing frame images (default: frames)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tracking_results/",
        help="Output directory for visualized frames (default: tracking_visualization).\n"
             "최초/최종 프레임 번호가 자동으로 디렉토리명에 추가됩니다."
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.tracking_csv):
        print(f"Error: Tracking CSV file not found: {args.tracking_csv}")
        return
    
    if not os.path.exists(args.frames_dir):
        print(f"Error: Frames directory not found: {args.frames_dir}")
        return
    
    # Run visualization
    visualize_tracking(args.tracking_csv, args.frames_dir, args.output_dir)


if __name__ == "__main__":
    main()
