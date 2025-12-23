import argparse
import os
import cv2
import csv
from ultralytics import YOLO
from tqdm import tqdm

def process_video(input_path, model_name, base_output_folder="results", detection_interval=1, batch_size=16):
    """
    YOLO 배치 추론을 적용하여 비디오 처리 속도를 최적화한 함수
    """
    # 1. 경로 및 폴더 설정
    folder_name = os.path.splitext(os.path.basename(model_name))[0]
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(base_output_folder, folder_name)
    os.makedirs(output_path, exist_ok=True)
    
    det_csv_path = os.path.join(output_path, f"{video_name}.csv")

    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    # 2. 이어쓰기(Resume) 로직 확인
    start_frame = 0
    det_file_exists = os.path.exists(det_csv_path)
    
    if det_file_exists:
        try:
            with open(det_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                last_frame = -1
                for row in reader:
                    if row:
                        last_frame = max(last_frame, int(row[0]))
                
                if last_frame >= 0:
                    start_frame = last_frame + detection_interval
                    print(f"Resuming from frame {start_frame} (last processed: {last_frame})")
        except Exception as e:
            print(f"Warning: Could not read existing CSV ({e}). Starting from 0.")
            start_frame = 0

    # 3. CSV 파일 열기
    det_file = open(det_csv_path, 'a', newline='', encoding='utf-8')
    det_writer = csv.writer(det_file)
    
    # 파일이 처음 생성되는 경우에만 헤더 작성
    if not det_file_exists or os.path.getsize(det_csv_path) == 0:
        det_writer.writerow(['frame', 'object_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'width', 'height'])

    # 4. 비디오 열기
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open video ({input_path})")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 시작 프레임으로 이동
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"Processing: {input_path}")
    print(f"- Interval: {detection_interval} frames")
    print(f"- Batch Size: {batch_size}")
    print(f"- Output CSV: {det_csv_path}")

    frame_count = start_frame
    batch_frames = []
    batch_indices = [] # 해당 프레임 번호를 기억하기 위함
    
    # 진행바 설정
    pbar = tqdm(total=total_frames, initial=start_frame, desc="Processing", unit="frame")

    # 5. 메인 루프
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 지정된 간격(interval)인 경우에만 배치에 추가
        if frame_count % detection_interval == 0:
            batch_frames.append(frame)
            batch_indices.append(frame_count)

        # 배치가 가득 차면 추론 실행 (GPU 효율 극대화)
        if len(batch_frames) == batch_size:
            # stream=False: 배치 전체 결과를 리스트로 받음 (속도 유리)
            results = model.predict(batch_frames, conf=0.1, verbose=False, classes=[0], stream=False)
            
            # 결과 CSV 기록
            for i, result in enumerate(results):
                current_frame_idx = batch_indices[i]
                boxes = result.boxes
                
                for box_idx, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width = x2 - x1
                    height = y2 - y1
                    
                    det_writer.writerow([
                        current_frame_idx, box_idx, x1, y1, x2, y2, f"{conf:.4f}", width, height
                    ])
            
            # 배치 초기화
            batch_frames = []
            batch_indices = []

        frame_count += 1
        pbar.update(1)

    # 6. 남은 자투리 프레임 처리 (Last Batch)
    if batch_frames:
        results = model.predict(batch_frames, conf=0.1, verbose=False, classes=[0], stream=False)
        for i, result in enumerate(results):
            current_frame_idx = batch_indices[i]
            boxes = result.boxes
            for box_idx, box in enumerate(boxes):
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                det_writer.writerow([
                    current_frame_idx, box_idx, x1, y1, x2, y2, f"{conf:.4f}", width, height
                ])

    # 정리
    pbar.close()
    cap.release()
    det_file.close()
    print("\nProcessing Complete.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="YOLO model path (e.g. yolo11x.pt)")
    parser.add_argument("video_file", nargs="?", default="test.mp4", help="Input video path")
    parser.add_argument("--interval", type=int, default=1, help="Detection interval (frames)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 16)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_video(
        args.video_file, 
        args.model_name, 
        detection_interval=args.interval, 
        batch_size=args.batch
    )