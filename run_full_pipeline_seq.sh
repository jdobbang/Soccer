#!/bin/bash
# 순차적으로 프레임 범위를 나눠서 run_full_pipeline.py를 실행하는 스크립트

START=30000
END=39000
STEP=150
NUM_REID_WORKERS=10

for ((s=$START; s<=$END; s+=$STEP)); do
    e=$((s+STEP-1))
    if [ $e -gt $END ]; then
        e=$END
    fi
    echo "Processing frames $s to $e"
    python run_full_pipeline.py \
        --detections detection_results/yolo11x/test.csv \
        --frames-dir detection_results/yolo11x/detected_frames \
        --start-frame $s \
        --end-frame $e \
        --num-reid-workers $NUM_REID_WORKERS

    python visualize_tracking.py tracking_results/result_${s}_${e}/csv/step4_post_interpolated.csv --frames-dir detection_results/yolo11x/detected_frames
    # 에러 발생 시 중단하려면 아래 주석 해제
    # if [ $? -ne 0 ]; then exit 1; fi
done
