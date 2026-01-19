import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_frame_intervals(df, jersey_number):
    """
    특정 등번호가 나타나는 프레임들을 연속된 구간으로 묶어서 반환합니다.
    """
    df_filtered = df[df['jersey_number'] == jersey_number].sort_values('frame')
    frames = df_filtered['frame'].tolist()
    intervals = []
    
    if frames:
        start = prev = frames[0]
        for f in frames[1:]:
            if f == prev + 1:
                prev = f
            else:
                intervals.append((start, prev))
                start = prev = f
        intervals.append((start, prev))
    return intervals

def save_presence_histogram_min(df, jersey_number, max_frames=450000, bin_size=30, fps=30, output_path='appearance_graph_min.png'):
    """
    30프레임 단위로 탐지 여부를 체크하고, X축을 '분(Minute)' 단위로 표시하여 저장합니다.
    """
    target_df = df[df['jersey_number'] == jersey_number].copy()
    
    if target_df.empty:
        print(f"No data found for jersey_number {jersey_number}")
        return

    # 1. 30프레임 단위 그룹화 (탐지 여부 바이너리화)
    target_df['bin_index'] = (target_df['frame'] - 1) // bin_size
    unique_bins = target_df['bin_index'].unique()
    
    # 2. 시각화 설정 (가로로 길게 설정하여 가독성 확보)
    plt.figure(figsize=(25, 6))
    
    # 실제 프레임 위치 계산
    x_positions_frame = unique_bins * bin_size + 1
    
    # 3. 막대 그래프 (Presence) 그리기
    plt.bar(x_positions_frame, [1] * len(x_positions_frame), width=bin_size, 
            color='royalblue', align='edge', edgecolor='none', alpha=0.8)
    
    # 4. X축 라벨을 '분' 단위로 설정 (10분 간격)
    interval_min = 10 
    tick_spacing = interval_min * 60 * fps  # 10분당 프레임 수
    ticks = np.arange(0, max_frames + tick_spacing, tick_spacing)
    tick_labels = [f"{int(t / (fps * 60))}m" for t in ticks]
    
    plt.xticks(ticks, tick_labels)
    
    # 그래프 스타일링
    plt.title(f'Detection Presence Timeline: Jersey Number {jersey_number}', fontsize=18, pad=20)
    plt.xlabel('Time (Minutes)', fontsize=14)
    plt.ylabel('Presence (1=Detected)', fontsize=14)
    
    plt.xlim(1, max_frames)
    plt.ylim(0, 1.3)
    plt.yticks([0, 1], ['Absent', 'Present'])
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\n[Success] Histogram saved to: {output_path}")

# --- 메인 실행 섹션 ---

# 1. CSV 파일 로드
file_path = 'detection_results/yolo11x/jersey_numbers_detailed.csv'
try:
    df = pd.read_csv(file_path)
    # 데이터 타입 통일
    df['jersey_number'] = df['jersey_number'].astype(str)
except FileNotFoundError:
    print(f"Error: {file_path}를 찾을 수 없습니다.")
    exit()

# 2. 분석 대상 설정 (예: 10번 선수)
target_no = "10"
fps_setting = 30

# 3. 시각화 실행 (X축 분 단위 표시)
save_presence_histogram_min(
    df, 
    target_no, 
    max_frames=45000, 
    bin_size=30, 
    fps=fps_setting, 
    output_path=f'presence_histogram_{target_no}.png'
)

# 4. 구간 상세 정보 출력 (프레임 & 분:초 변환)
intervals = get_frame_intervals(df, target_no)
print(f"\n--- Detailed Intervals for Jersey #{target_no} ---")
for s, e in intervals:
    # 초 단위 계산
    s_total_sec = s // fps_setting
    e_total_sec = e // fps_setting
    
    # 분:초 변환
    s_m, s_s = divmod(s_total_sec, 60)
    e_m, e_s = divmod(e_total_sec, 60)
    
    print(f"  Frame: {s:6d} ~ {e:6d}  |  Time: {int(s_m):02d}:{int(s_s):02d} ~ {int(e_m):02d}:{int(e_s):02d}")