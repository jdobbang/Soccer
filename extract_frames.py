#!/usr/bin/env python3
"""
Extract frames from video file
================================

Extract all frames from a video file and save as individual images.
"""

import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path, output_dir, frame_format="frame_{:06d}.jpg", start_frame=0, end_frame=None):
    """
    Extract frames from video file.

    Args:
        video_path: Path to input video file
        output_dir: Output directory for frames
        frame_format: Frame filename format (default: frame_000001.jpg)
        start_frame: Start frame number (default: 0)
        end_frame: End frame number (default: None = extract all)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")

    # Determine frame range
    if end_frame is None:
        end_frame = total_frames - 1
    else:
        end_frame = min(end_frame, total_frames - 1)

    print(f"\nExtracting frames {start_frame} to {end_frame}")
    print(f"Output directory: {output_dir}")

    # Skip to start frame if needed
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Extract frames
    frame_count = 0
    saved_count = 0

    with tqdm(total=end_frame - start_frame + 1, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            # Check if we've reached the end frame
            if current_frame > end_frame:
                break

            # Save frame
            frame_filename = frame_format.format(current_frame)
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)

            saved_count += 1
            frame_count = current_frame
            pbar.update(1)

    cap.release()

    print(f"\nExtraction complete!")
    print(f"Saved {saved_count} frames to {output_dir}")
    print(f"Frame range: {start_frame} to {frame_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from video file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all frames from test.mp4 to frames/ directory
  python extract_frames.py test.mp4 --output-dir frames

  # Extract frames 100-500
  python extract_frames.py test.mp4 --output-dir frames --start-frame 100 --end-frame 500

  # Custom frame naming format
  python extract_frames.py test.mp4 --output-dir frames --format "img_{:05d}.png"
        """
    )

    parser.add_argument('video', type=str,
                        help='Path to input video file (e.g., test.mp4)')
    parser.add_argument('--output-dir', type=str, default='frames',
                        help='Output directory for frames (default: frames)')
    parser.add_argument('--format', type=str, default='frame_{:06d}.jpg',
                        help='Frame filename format (default: frame_{:06d}.jpg)')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Start frame number (default: 0)')
    parser.add_argument('--end-frame', type=int, default=None,
                        help='End frame number (default: None = extract all)')

    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.output_dir,
        frame_format=args.format,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )


if __name__ == '__main__':
    main()
