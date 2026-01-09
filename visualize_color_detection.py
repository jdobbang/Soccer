import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Argument parser 설정
parser = argparse.ArgumentParser(description='Visualize color detection results with optional uniform color filtering')
parser.add_argument('--csv_path', type=str, default='detection_results/yolo11x/test_color.csv',
                    help='Path to detection CSV file with color information')
parser.add_argument('--input_dir', type=str, default='detection_results/yolo11x/detected_frames',
                    help='Input directory containing detection images')
parser.add_argument('--output_dir', type=str, default='detection_results/yolo11x/detected_frames_color',
                    help='Output directory for visualized images')
parser.add_argument('--uniform_color', type=str, default=None,
                    help='Filter by specific uniform color (e.g., orange, black). If not specified, use all detections.')

args = parser.parse_args()

# CSV 파일 읽기
csv_path = args.csv_path
print(f"Using CSV: {csv_path}")

df = pd.read_csv(csv_path)

# 유니폼 색상 필터링
if args.uniform_color:
    original_count = len(df)
    df = df[df['uniform_color'] == args.uniform_color]
    print(f"Filtering by uniform_color='{args.uniform_color}': {original_count} -> {len(df)} detections")

# 입력 및 출력 디렉토리 설정
input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Total rows: {len(df)}")
print(f"Total frames: {df['frame'].nunique()}")
print(f"Total tracks: {df['track_id'].nunique()}")
print("\nUniform color distribution:")
print(df['uniform_color'].value_counts())
print(f"\nInput directory: {input_dir}")
print(f"Output directory: {output_dir}")

# 색상 매핑 (BGR format for OpenCV)
color_map = {
    'black': (50, 50, 50),      # 어두운 회색
    'orange': (0, 140, 255),    # 주황색
    'white': (220, 220, 220),   # 밝은 회색
    'red': (0, 0, 255),         # 빨강
    'blue': (255, 0, 0),        # 파랑
    'green': (0, 200, 0),       # 초록
    'yellow': (0, 255, 255),    # 노랑
    'purple': (200, 0, 200),    # 보라
    'unknown': (128, 128, 128)  # 회색
}

# 텍스트 색상 (밝은 색상)
text_color_map = {
    'black': (200, 200, 200),   # 밝은 회색
    'orange': (0, 200, 255),    # 밝은 주황색
    'white': (255, 255, 255),   # 흰색
    'red': (100, 100, 255),     # 밝은 빨강
    'blue': (255, 150, 150),    # 밝은 파랑
    'green': (100, 255, 100),   # 밝은 초록
    'yellow': (150, 255, 255),  # 밝은 노랑
    'purple': (255, 100, 255),  # 밝은 보라
    'unknown': (255, 255, 255)  # 흰색
}

# 프레임별로 처리
for frame_num in tqdm(sorted(df['frame'].unique()), desc="Processing frames"):
    # 해당 프레임의 모든 detection 가져오기
    frame_data = df[df['frame'] == frame_num]

    # 이미지 파일 경로
    image_name = frame_data.iloc[0]['image_name']
    image_path = input_dir / image_name

    if not image_path.exists():
        print(f"Warning: Image not found - {image_path}")
        continue

    # 이미지 읽기
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Failed to read image - {image_path}")
        continue

    # 각 detection에 대해 바운딩 박스 그리기
    for _, row in frame_data.iterrows():
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        track_id = int(row['track_id'])
        confidence = row['confidence']
        uniform_color = row['uniform_color']
        color_confidence = row['color_confidence']

        # 색상 선택
        bbox_color = color_map.get(uniform_color, (128, 128, 128))
        text_color = text_color_map.get(uniform_color, (255, 255, 255))

        # 바운딩 박스 그리기 (두께 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, 3)

        # 텍스트 배경 준비 (유니폼 색상 텍스트 제거)
        label = f"ID:{track_id}"
        conf_label = f"Det:{confidence:.2f} Col:{color_confidence:.2f}"

        # 텍스트 크기 계산
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_width1, text_height1), _ = cv2.getTextSize(label, font, font_scale, thickness)
        (text_width2, text_height2), _ = cv2.getTextSize(conf_label, font, font_scale, thickness)

        max_text_width = max(text_width1, text_width2)
        total_text_height = text_height1 + text_height2 + 15  # 15는 줄 간격

        # 텍스트 배경 그리기 (반투명)
        overlay = img.copy()
        cv2.rectangle(overlay,
                     (x1, y1 - total_text_height - 10),
                     (x1 + max_text_width + 10, y1),
                     bbox_color,
                     -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # 텍스트 그리기
        cv2.putText(img, label,
                   (x1 + 5, y1 - total_text_height + text_height1),
                   font, font_scale, text_color, thickness)
        cv2.putText(img, conf_label,
                   (x1 + 5, y1 - 5),
                   font, font_scale, text_color, thickness)

    # 결과 이미지 저장
    output_path = output_dir / image_name
    cv2.imwrite(str(output_path), img)

print(f"\nProcessing complete!")
print(f"Visualized frames saved to: {output_dir}")
print(f"Total frames processed: {df['frame'].nunique()}")

# 통계 저장
stats_path = output_dir / "color_statistics.txt"
with open(stats_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("COLOR DETECTION VISUALIZATION STATISTICS\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Total detections: {len(df)}\n")
    f.write(f"Total frames processed: {df['frame'].nunique()}\n")
    f.write(f"Total unique tracks: {df['track_id'].nunique()}\n\n")

    f.write("-" * 60 + "\n")
    f.write("COLOR DISTRIBUTION\n")
    f.write("-" * 60 + "\n")
    color_dist = df['uniform_color'].value_counts()
    for color, count in color_dist.items():
        pct = count / len(df) * 100
        f.write(f"{color:15s}: {count:6d} ({pct:5.2f}%)\n")

    f.write("\n" + "-" * 60 + "\n")
    f.write("AVERAGE CONFIDENCES BY COLOR\n")
    f.write("-" * 60 + "\n")
    for color in sorted(df['uniform_color'].unique()):
        color_data = df[df['uniform_color'] == color]
        avg_det_conf = color_data['confidence'].mean()
        avg_col_conf = color_data['color_confidence'].mean()
        f.write(f"\n{color.upper()}:\n")
        f.write(f"  Avg Detection Confidence: {avg_det_conf:.4f}\n")
        f.write(f"  Avg Color Confidence:     {avg_col_conf:.4f}\n")
        f.write(f"  Count:                    {len(color_data)}\n")

print(f"Statistics saved to: {stats_path}")
