#!/usr/bin/env python3
"""
Run SORT tracking on detection results from CSV file.
Reads detection results and applies SORT tracking algorithm.
Outputs tracking results as CSV file compatible with visualize_tracking.py
"""

import csv
import os
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sort_tracker import Sort


def load_detections_from_csv(csv_path, start_frame=None, end_frame=None, conf_threshold=0.1):
    """
    Load detection results from CSV file.

    Args:
        csv_path: Path to detection CSV file
        start_frame: Optional start frame (inclusive)
        end_frame: Optional end frame (inclusive)
        conf_threshold: Minimum confidence threshold for detections

    Returns:
        detections_by_frame: Dictionary mapping frame numbers to detection arrays
    """
    print(f"Loading detections from {csv_path}...")
    if start_frame is not None or end_frame is not None:
        frame_range = f"frames {start_frame or 0} to {end_frame or 'end'}"
        print(f"Filtering {frame_range}")
    if conf_threshold > 0.0:
        print(f"Applying confidence threshold: {conf_threshold}")

    detections_by_frame = defaultdict(list)
    total_loaded = 0
    filtered_by_conf = 0

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_num = int(row['frame'])

            # Filter by frame range if specified
            if start_frame is not None and frame_num < start_frame:
                continue
            if end_frame is not None and frame_num > end_frame:
                continue

            x1 = float(row['x1'])
            y1 = float(row['y1'])
            x2 = float(row['x2'])
            y2 = float(row['y2'])
            confidence = float(row['confidence'])

            total_loaded += 1

            # Filter by confidence threshold
            if confidence < conf_threshold:
                filtered_by_conf += 1
                continue

            # Format: [x1, y1, x2, y2, confidence]
            detection = [x1, y1, x2, y2, confidence]
            detections_by_frame[frame_num].append(detection)

    # Convert lists to numpy arrays
    for frame_num in detections_by_frame:
        detections_by_frame[frame_num] = np.array(detections_by_frame[frame_num])

    total_detections = sum(len(dets) for dets in detections_by_frame.values())
    print(f"Loaded {total_detections} detections across {len(detections_by_frame)} frames")
    if filtered_by_conf > 0:
        print(f"Filtered out {filtered_by_conf} detections below confidence threshold")

    return detections_by_frame


def run_tracking(detections_by_frame, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Run SORT tracking on detections.

    Args:
        detections_by_frame: Dictionary mapping frame numbers to detection arrays
        max_age: Maximum number of frames to keep alive a track without detections
        min_hits: Minimum number of associated detections before track is confirmed
        iou_threshold: Minimum IoU for matching detections to tracks

    Returns:
        tracks_by_frame: Dictionary mapping frame numbers to tracking results
    """
    print(f"Running SORT tracker (max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold})...")

    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    tracks_by_frame = {}

    # Get all frame numbers and sort them
    frame_numbers = sorted(detections_by_frame.keys())

    if not frame_numbers:
        print("No detections found!")
        return tracks_by_frame

    # Process each frame
    for frame_num in tqdm(frame_numbers, desc="Tracking"):
        # Get detections for this frame
        detections = detections_by_frame.get(frame_num, np.empty((0, 5)))

        # Update tracker
        tracks = tracker.update(detections)

        # Store tracking results
        # tracks format: [[x1, y1, x2, y2, track_id, confidence], ...]
        tracks_by_frame[frame_num] = tracks

    total_tracks = sum(len(tracks) for tracks in tracks_by_frame.values())
    unique_track_ids = set()
    for tracks in tracks_by_frame.values():
        for track in tracks:
            unique_track_ids.add(int(track[4]))

    print(f"Tracking complete: {total_tracks} total tracks, {len(unique_track_ids)} unique IDs")

    return tracks_by_frame


def save_tracking_results(tracks_by_frame, output_csv):
    """
    Save tracking results to CSV file.

    Args:
        tracks_by_frame: Dictionary mapping frame numbers to tracking results
        output_csv: Path to output CSV file
    """
    print(f"Saving tracking results to {output_csv}...")

    # Create output directory if needed
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])

        # Write tracking results
        frame_numbers = sorted(tracks_by_frame.keys())
        for frame_num in frame_numbers:
            tracks = tracks_by_frame[frame_num]
            for track in tracks:
                x1, y1, x2, y2, track_id, confidence = track
                writer.writerow([
                    int(frame_num),
                    int(track_id),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(confidence)
                ])

    print(f"Saved tracking results: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SORT tracking on detection results from CSV"
    )
    parser.add_argument(
        "detection_csv",
        type=str,
        help="Path to detection results CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default='tracking_results/test_tracked.csv',
        help="Output CSV file path (default: detection_csv with '_tracked' suffix)"
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=150,
        help="Maximum frames to keep alive a track without detections (default: 150)"
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Minimum hits before track is confirmed (default: 3)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Minimum IoU for matching (default: 0.3)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum confidence for detections (default: 0.1)"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Start frame number (inclusive, optional)"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="End frame number (inclusive, optional)"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.detection_csv):
        print(f"Error: Detection CSV file not found: {args.detection_csv}")
        return

    # Determine output path
    if args.output is None:
        base_name = os.path.splitext(args.detection_csv)[0]
        output_csv = f"{base_name}_tracked.csv"
    else:
        output_csv = args.output

    # Load detections
    detections_by_frame = load_detections_from_csv(
        args.detection_csv,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        conf_threshold=args.confidence_threshold
    )

    # Run tracking
    tracks_by_frame = run_tracking(
        detections_by_frame,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold
    )

    # Save results
    save_tracking_results(tracks_by_frame, output_csv)

    print("\n" + "="*60)
    print("Tracking complete!")
    print(f"Input:  {args.detection_csv}")
    print(f"Output: {output_csv}")
    print("="*60)
    print(f"\nTo visualize results, run:")
    print(f"python visualize_tracking.py {output_csv} --frames-dir <frames_directory>")


if __name__ == "__main__":
    main()

"""
python run_tracking.py ./detection_results/yolo11x/test.csv --start-frame 1350 --end-frame 1499
"""