#!/bin/bash
START=0
END=45000
NUM_REID_WORKERS=10
UNIFORM_COLOR="orange"

echo "Processing frames $START to $END"
python run_full_pipeline.py \
    --detections detection_results/yolo11x/test_color.csv \
    --frames-dir frames \
    --start-frame $START \
    --end-frame $END \
    --num-reid-workers $NUM_REID_WORKERS \
    --uniform-color $UNIFORM_COLOR

python visualize_tracking.py tracking_results/result_${START}_${END}/csv/step4_post_interpolated.csv --frames-dir frames