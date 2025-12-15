#!/usr/bin/env python3
"""
CSV Detection Visualization Script
Reads CSV file with detection results and visualizes bboxes on video frames
Confidence-based color coding:
- High (0.67-1.0): Green
- Medium (0.34-0.67): Orange  
- Low (0.1-0.34): Red
"""

import argparse
import os
import cv2
import csv
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def get_bbox_color(confidence):
    """
    Returns color based on confidence value
    Confidence ranges:
    - [0.67, 1.0]: Green (0, 255, 0)
    - [0.34, 0.67): Orange (0, 165, 255)
    - [0.1, 0.34): Red (0, 0, 255)
    """
    if confidence >= 0.67:
        return (0, 255, 0)  # Green
    elif confidence >= 0.34:
        return (0, 165, 255)  # Orange
    else:
        return (0, 0, 255)  # Red


def load_detections_from_csv(csv_path):
    """
    Load detections from CSV file
    Returns: dict with frame_number as key and list of detections as value
    Each detection: (x1, y1, x2, y2, confidence)
    """
    detections_by_frame = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_num = int(row['frame'])
            x1 = int(row['x1'])
            y1 = int(row['y1'])
            x2 = int(row['x2'])
            y2 = int(row['y2'])
            confidence = float(row['confidence'])
            
            detections_by_frame[frame_num].append((x1, y1, x2, y2, confidence))
    
    return detections_by_frame


def visualize_detections(video_path, csv_path, output_dir="frames"):
    """
    Visualize detections from CSV on video frames and save as images
    """
    # Load detections
    print(f"Loading detections from: {csv_path}")
    detections_by_frame = load_detections_from_csv(csv_path)
    print(f"Loaded detections for {len(detections_by_frame)} frames")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video ({video_path})")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    print("Processing frames...")
    
    frame_count = 0
    saved_count = 0
    
    pbar = tqdm(total=total_frames, desc="Visualizing", unit="frame")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw detections if available for this frame
        if frame_count in detections_by_frame:
            detections = detections_by_frame[frame_count]
            
            for x1, y1, x2, y2, confidence in detections:
                color = get_bbox_color(confidence)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Save frame
        output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(output_path, frame)
        saved_count += 1
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"\nVisualization Complete!")
    print(f"- Total frames processed: {frame_count}")
    print(f"- Frames with detections: {len(detections_by_frame)}")
    print(f"- Frames saved: {saved_count}")
    print(f"- Output directory: {output_dir}")
    
    # Print color legend
    print("\nColor Legend:")
    print("  🟢 Green:  High confidence (0.67-1.0)")
    print("  🟠 Orange: Medium confidence (0.34-0.67)")
    print("  🔴 Red:    Low confidence (0.1-0.34)")


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CSV detections on video frames')
    parser.add_argument('csv_path', help='Path to CSV file with detections')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--output', default='frames', help='Output directory for frame images (default: frames)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        exit(1)
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        exit(1)
    
    visualize_detections(args.video_path, args.csv_path, args.output)


"""
Example usage:
python visualize_detect_csv.py results/yolo11x/test.csv test.mp4 --output frames
"""
