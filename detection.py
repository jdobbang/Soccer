import argparse
import os
import cv2
import csv
from ultralytics import YOLO
from tqdm import tqdm


# --- Main Processing Function ---
def process_video(input_path, model_name, base_output_folder="results", detection_interval=5):
    # 폴더 및 경로 설정
    folder_name = os.path.splitext(os.path.basename(model_name))[0]
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(base_output_folder, folder_name)
    os.makedirs(output_path, exist_ok=True)

    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # 1. Detection CSV 생성 (동영상명으로)
    det_csv_path = os.path.join(output_path, f"{video_name}.csv")
    
    # 기존 CSV에서 마지막 프레임 확인
    start_frame = 0
    det_file_exists = os.path.exists(det_csv_path)
    if det_file_exists:
        with open(det_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # 헤더 스킵
            last_frame = 0
            for row in reader:
                if row:
                    last_frame = max(last_frame, int(row[0]))
            start_frame = last_frame + detection_interval
            print(f"Resuming from frame {start_frame} (last processed: {last_frame})")
    
    det_file = open(det_csv_path, 'a', newline='', encoding='utf-8')
    det_writer = csv.writer(det_file)
    if not det_file_exists:
        det_writer.writerow(['frame', 'object_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'width', 'height'])
    
    print(f"Results will be saved to: {output_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open video ({input_path})")
        return

    # 총 프레임 수 가져오기
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 시작 프레임으로 이동
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    saved_count = 0
    print(f"Start processing every {detection_interval} frames: {input_path}")

    # tqdm 진행바 설정 (시작 프레임부터)
    pbar = tqdm(total=total_frames, initial=start_frame, desc="Processing frames", unit="frame")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detection (n프레임 간격으로만 수행)
        if frame_count % detection_interval == 0:
            # Object Detection
            results = model.predict(frame, conf=0.1, verbose=False, classes=[0])
            boxes = results[0].boxes
            
            # 검출 결과 CSV 저장
            for idx, box in enumerate(boxes):
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                
                det_writer.writerow([frame_count, idx, x1, y1, x2, y2, f"{conf:.4f}", width, height])
            
            saved_count += 1

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    det_file.close()
    
    print(f"\nProcessing Complete.")
    print(f"- Total Frames Processed: {saved_count}")
    print(f"- Detections CSV: {det_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="YOLO model filename or path, e.g. yolo11n.pt")
    parser.add_argument("video_file", nargs="?", default="test.mp4", help="Input video path")
    parser.add_argument("--interval", type=int, default=1, help="Run detection every N frames")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(args.video_file, model_name=args.model_name, detection_interval=args.interval)