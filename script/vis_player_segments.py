import pandas as pd
import cv2
import os

# --- [설정 부분] ---
# 1. 파일 경로 설정
CSV_PATH = 'detection_results/yolo11x/jersey_numbers_detailed.csv'
INPUT_FRAME_DIR = 'frames'  # 원본 프레임 이미지들이 있는 폴더 경로
OUTPUT_DIR = 'highlighted_player_10'             # 결과 이미지가 저장될 폴더

# 2. 분석 대상 설정
TARGET_NUMBER = '10'
MAX_FRAME = 45000

def visualize_target_player(csv_path, input_dir, output_dir, target_no):
    """
    CSV 데이터를 기반으로 특정 등번호 선수에게 박스를 치고 이미지를 저장합니다.
    """
    # 결과 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[알림] '{output_dir}' 폴더를 생성했습니다.")

    # 1. CSV 데이터 로드
    try:
        df = pd.read_csv(csv_path)
        # 등번호를 문자열로 통일 (비교 오류 방지)
        df['jersey_number'] = df['jersey_number'].astype(str)
    except FileNotFoundError:
        print(f"[오류] CSV 파일을 찾을 수 없습니다: {csv_path}")
        return

    # 2. 대상 번호 데이터만 필터링
    target_df = df[df['jersey_number'] == target_no]
    
    if target_df.empty:
        print(f"[알림] 등번호 {target_no}번에 대한 탐지 데이터가 없습니다.")
        return

    # 3. 프레임 순회 및 시각화
    unique_frames = sorted(target_df['frame'].unique())
    print(f"--- 시각화 시작: 등번호 {target_no} (총 {len(unique_frames)} 프레임) ---")

    for frame_no in unique_frames:
        if frame_no > MAX_FRAME:
            break

        # 프레임 이미지 파일명 생성 (파일명 규칙에 맞게 수정 필요)
        # 예: frame_000001.jpg 형식을 시도
        img_name = f"frame_{int(frame_no):06d}.jpg" 
        img_path = os.path.join(input_dir, img_name)
        
        # 이미지 로드
        img = cv2.imread(img_path)
        
        # 파일명이 다를 경우 대비 (예: 123.jpg 형식 시도)
        if img is None:
            img_name = f"{int(frame_no)}.jpg"
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)

        if img is None:
            # 500프레임마다 로그 출력 (누락 확인용)
            if frame_no % 500 == 0:
                print(f"[경고] 프레임 {frame_no} 이미지를 찾을 수 없습니다. 경로: {img_path}")
            continue

        # 해당 프레임에 있는 타겟 번호의 모든 객체 추출
        detections = target_df[target_df['frame'] == frame_no]
        
        for _, row in detections.iterrows():
            # CSV 컬럼: x1, y1, x2, y2 (이미지 픽셀 좌표)
            try:
                x1, y1 = int(row['x1']), int(row['y1'])
                x2, y2 = int(row['x2']), int(row['y2'])
                
                # 시각화 색상 (BGR: 노란색) 및 두께
                line_color = (0, 255, 255)
                
                # [그리기 1] 바운딩 박스
                cv2.rectangle(img, (x1, y1), (x2, y2), line_color, 3)
                
                # [그리기 2] 등번호 라벨 배경
                label = f"No.{target_no}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (w, h), _ = cv2.getTextSize(label, font, 0.8, 2)
                cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), line_color, -1)
                
                # [그리기 3] 등번호 텍스트 (검은색)
                cv2.putText(img, label, (x1, y1 - 5), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            except KeyError as e:
                print(f"[오류] CSV에 좌표 컬럼이 없습니다: {e}")
                return

        # 강조된 이미지 저장
        output_file_name = f"highlight_{target_no}_{int(frame_no):06d}.jpg"
        cv2.imwrite(os.path.join(output_dir, output_file_name), img)
        
        if frame_no % 1000 == 0:
            print(f"[진행] {frame_no} 프레임 처리 완료...")

    print(f"\n--- 시각화 완료! ---")
    print(f"결과 저장 위치: {os.path.abspath(output_dir)}")

# --- [코드 실행] ---
if __name__ == "__main__":
    visualize_target_player(CSV_PATH, INPUT_FRAME_DIR, OUTPUT_DIR, TARGET_NUMBER)