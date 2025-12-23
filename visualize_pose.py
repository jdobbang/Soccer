#!/usr/bin/env python3
"""
Pose Estimation 결과 시각화
============================

pose_estimation.csv를 읽어서 keypoint와 skeleton을 이미지에 그리고 저장
"""

import argparse
import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict


# COCO Keypoint 정의 (17 keypoints)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Skeleton 연결 정의 (COCO format)
SKELETON = [
    # 얼굴
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    # 상체
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),
    # 몸통
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    # 하체
    ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle'),
]

# 색상 정의 (BGR)
COLORS = {
    'keypoint': (0, 255, 0),      # 초록
    'skeleton': (255, 255, 0),     # 청록
    'bbox': (0, 255, 255),         # 노랑
    'text': (255, 255, 255),       # 흰색
}

# 부위별 색상
LIMB_COLORS = {
    'face': (255, 200, 200),       # 연한 파랑
    'arm_left': (0, 255, 0),       # 초록
    'arm_right': (0, 200, 0),      # 진한 초록
    'torso': (255, 255, 0),        # 청록
    'leg_left': (0, 165, 255),     # 주황
    'leg_right': (0, 100, 255),    # 진한 주황
}


def get_limb_color(kp1_name: str, kp2_name: str) -> tuple:
    """skeleton 부위에 따른 색상 반환"""
    if 'eye' in kp1_name or 'ear' in kp1_name or 'nose' in kp1_name:
        return LIMB_COLORS['face']
    elif 'left_shoulder' in kp1_name or 'left_elbow' in kp1_name or 'left_wrist' in kp1_name:
        if 'right' not in kp2_name:
            return LIMB_COLORS['arm_left']
    elif 'right_shoulder' in kp1_name or 'right_elbow' in kp1_name or 'right_wrist' in kp1_name:
        return LIMB_COLORS['arm_right']
    elif 'left_hip' in kp1_name or 'left_knee' in kp1_name or 'left_ankle' in kp1_name:
        if 'right' not in kp2_name:
            return LIMB_COLORS['leg_left']
    elif 'right_hip' in kp1_name or 'right_knee' in kp1_name or 'right_ankle' in kp1_name:
        return LIMB_COLORS['leg_right']
    return LIMB_COLORS['torso']


def load_pose_csv(csv_path: str) -> dict:
    """
    pose_estimation.csv 로드

    Returns:
        dict: {frame_num: [person_data, ...]}
    """
    frame_data = defaultdict(list)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame = int(row['frame'])

            # bbox 정보
            person = {
                'image_name': row['image_name'],
                'track_id': int(row['track_id']),
                'bbox': {
                    'x1': float(row['bbox_x1']),
                    'y1': float(row['bbox_y1']),
                    'x2': float(row['bbox_x2']),
                    'y2': float(row['bbox_y2']),
                    'conf': float(row['bbox_conf'])
                },
                'keypoints': {}
            }

            # keypoint 정보
            for kp_name in KEYPOINT_NAMES:
                x = float(row[f'{kp_name}_x'])
                y = float(row[f'{kp_name}_y'])
                conf = float(row[f'{kp_name}_conf'])
                person['keypoints'][kp_name] = (x, y, conf)

            frame_data[frame].append(person)

    return frame_data


def draw_pose(
    image: np.ndarray,
    person: dict,
    conf_threshold: float = 0.3,
    draw_bbox: bool = True,
    draw_skeleton: bool = True,
    draw_keypoints: bool = True,
    keypoint_radius: int = 4,
    skeleton_thickness: int = 2,
    bbox_thickness: int = 2
) -> np.ndarray:
    """
    한 사람의 pose를 이미지에 그리기

    Args:
        image: 원본 이미지
        person: person 데이터 (bbox, keypoints)
        conf_threshold: keypoint confidence threshold
        draw_bbox: bbox 그리기 여부
        draw_skeleton: skeleton 그리기 여부
        draw_keypoints: keypoint 그리기 여부

    Returns:
        그려진 이미지
    """
    img = image.copy()
    keypoints = person['keypoints']
    bbox = person['bbox']
    track_id = person['track_id']

    # Bbox 그리기
    if draw_bbox:
        x1, y1 = int(bbox['x1']), int(bbox['y1'])
        x2, y2 = int(bbox['x2']), int(bbox['y2'])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS['bbox'], bbox_thickness)

        # Track ID 표시
        label = f"ID:{track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLORS['bbox'], -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Skeleton 그리기
    if draw_skeleton:
        for kp1_name, kp2_name in SKELETON:
            if kp1_name in keypoints and kp2_name in keypoints:
                x1, y1, c1 = keypoints[kp1_name]
                x2, y2, c2 = keypoints[kp2_name]

                if c1 >= conf_threshold and c2 >= conf_threshold:
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    color = get_limb_color(kp1_name, kp2_name)
                    cv2.line(img, pt1, pt2, color, skeleton_thickness)

    # Keypoints 그리기
    if draw_keypoints:
        for kp_name, (x, y, conf) in keypoints.items():
            if conf >= conf_threshold:
                pt = (int(x), int(y))
                cv2.circle(img, pt, keypoint_radius, COLORS['keypoint'], -1)
                cv2.circle(img, pt, keypoint_radius, (0, 0, 0), 1)  # 검정 테두리

    return img


def visualize_sequence(
    pose_csv: str,
    image_folder: str,
    output_folder: str,
    conf_threshold: float = 0.3,
    draw_bbox: bool = True,
    draw_skeleton: bool = True,
    draw_keypoints: bool = True,
    save_video: bool = False,
    video_fps: int = 30
):
    """
    시퀀스 전체 시각화

    Args:
        pose_csv: pose_estimation.csv 경로
        image_folder: 원본 이미지 폴더
        output_folder: 출력 폴더 (이미지 저장)
        conf_threshold: keypoint confidence threshold
        draw_bbox: bbox 그리기 여부
        draw_skeleton: skeleton 그리기 여부
        draw_keypoints: keypoint 그리기 여부
        save_video: 비디오로 저장 여부
        video_fps: 비디오 FPS
    """
    # 데이터 로드
    print(f"Loading pose data from: {pose_csv}")
    frame_data = load_pose_csv(pose_csv)

    if not frame_data:
        print("No pose data found!")
        return

    print(f"Loaded {len(frame_data)} frames")

    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 비디오 writer 초기화
    video_writer = None
    if save_video:
        video_path = os.path.join(output_folder, 'pose_visualization.mp4')

    # 프레임별 처리
    sorted_frames = sorted(frame_data.keys())
    output_images = []

    for frame_num in tqdm(sorted_frames, desc="Visualizing"):
        persons = frame_data[frame_num]

        # 이미지 이름 추출
        image_name = persons[0]['image_name']
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image not found - {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Cannot read image - {image_path}")
            continue

        # 각 사람에 대해 pose 그리기
        for person in persons:
            image = draw_pose(
                image, person,
                conf_threshold=conf_threshold,
                draw_bbox=draw_bbox,
                draw_skeleton=draw_skeleton,
                draw_keypoints=draw_keypoints
            )

        # 프레임 번호 표시
        cv2.putText(image, f"Frame: {frame_num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 이미지 저장
        output_path = os.path.join(output_folder, f"pose_{frame_num:06d}.jpg")
        cv2.imwrite(output_path, image)
        output_images.append(output_path)

        # 비디오 writer 초기화 (첫 프레임에서)
        if save_video and video_writer is None:
            h, w = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, (w, h))

        if save_video:
            video_writer.write(image)

    # 비디오 저장 완료
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {video_path}")

    print(f"Saved {len(output_images)} images to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Pose Estimation Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 시각화 (이미지 저장)
  python visualize_pose.py --pose_csv detection_results/yolo11x/pose_estimation.csv \\
                           --image_folder images/ \\
                           --output_folder pose_vis/

  # 비디오로 저장
  python visualize_pose.py --pose_csv detection_results/yolo11x/pose_estimation.csv \\
                           --image_folder images/ \\
                           --output_folder pose_vis/ \\
                           --save_video --fps 30

  # skeleton만 표시 (bbox 없이)
  python visualize_pose.py --pose_csv pose_estimation.csv \\
                           --image_folder images/ \\
                           --output_folder pose_vis/ \\
                           --no_bbox
        """
    )

    parser.add_argument('--pose_csv', type=str, required=True,
                        help='pose_estimation.csv 경로')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='원본 이미지 폴더 경로')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='출력 폴더 경로')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                        help='Keypoint confidence threshold (default: 0.3)')
    parser.add_argument('--no_bbox', action='store_true',
                        help='Bbox 표시 안함')
    parser.add_argument('--no_skeleton', action='store_true',
                        help='Skeleton 표시 안함')
    parser.add_argument('--no_keypoints', action='store_true',
                        help='Keypoints 표시 안함')
    parser.add_argument('--save_video', action='store_true',
                        help='MP4 비디오로 저장')
    parser.add_argument('--fps', type=int, default=30,
                        help='비디오 FPS (default: 30)')

    args = parser.parse_args()

    visualize_sequence(
        pose_csv=args.pose_csv,
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        conf_threshold=args.conf_threshold,
        draw_bbox=not args.no_bbox,
        draw_skeleton=not args.no_skeleton,
        draw_keypoints=not args.no_keypoints,
        save_video=args.save_video,
        video_fps=args.fps
    )


if __name__ == '__main__':
    main()
