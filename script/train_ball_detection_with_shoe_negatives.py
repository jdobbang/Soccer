#!/usr/bin/env python3
"""
Ball Detection Training with Shoe as Negative Class
====================================================

Shoe pseudo labels를 "ball이 아닌 것"으로 학습하여
ball detection의 false positive(발 → ball 오인식)을 줄임

Inference 시에는 ball class만 사용됨
"""

import argparse
import os
from ultralytics import YOLO


def train_ball_detection(
    data_yaml: str,
    model: str = 'yolo11x.pt',
    epochs: int = 200,
    imgsz: int = 1280,
    batch: int = 4,
    device: int = 0,
    project: str = '../runs/detect',
    name: str = 'ball_with_shoe_negatives',
    freeze_backbone: bool = False,
    pretrained_weights: str = None
):
    """
    Ball detection 모델 학습

    Args:
        data_yaml: Dataset config 경로
        model: 모델 (yolo11x.pt 등)
        epochs: 학습 epoch 수
        imgsz: 이미지 크기
        batch: 배치 크기
        device: GPU 번호
        project: 결과 저장 경로
        name: 실험 이름
        freeze_backbone: Backbone 동결 여부
        pretrained_weights: 사전학습 가중치 (없으면 model 파라미터 사용)
    """

    print("=" * 70)
    print("Ball Detection Training with Shoe Negatives")
    print("=" * 70)
    print(f"Dataset: {data_yaml}")
    print(f"Model: {model}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Project: {project}")
    print(f"Name: {name}")
    print("=" * 70)

    # 모델 로드
    if pretrained_weights:
        print(f"\nLoading pretrained weights: {pretrained_weights}")
        yolo_model = YOLO(pretrained_weights)
    else:
        print(f"\nLoading model: {model}")
        yolo_model = YOLO(model)

    # 학습
    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,

        # 기본 설정
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,

        # 옵티마이저
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # Loss 가중치
        # Shoe class(배경)를 감지하는 것에는 큰 의미 없지만,
        # Ball을 다른 것과 잘 구분하기 위해 분류 손실 약간 강화
        cls=1.0,  # 기본값
        box=7.5,
        dfl=1.5,

        # 데이터 증강 - 모두 비활성화
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        auto_augment=None,
        erasing=0.0,
        copy_paste=0.0,
        close_mosaic=100,

        # 콜백
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"Best weights: {os.path.join(project, name, 'weights', 'best.pt')}")
    print(f"Last weights: {os.path.join(project, name, 'weights', 'last.pt')}")
    print("\n[주의] Inference 시에는 shoe class는 무시하고 ball만 사용하세요:")
    print("  model = YOLO(best.pt)")
    print("  results = model.predict(image_path, conf=0.5)")
    print("  # Shoe 클래스는 post-processing으로 필터링")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train ball detection with shoe as negative class',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 설정
  python train_ball_detection_with_shoe_negatives.py

  # 커스텀 설정
  python train_ball_detection_with_shoe_negatives.py \\
    --data /workspace/Soccer/dataset/ball_shoe/merged/data.yaml \\
    --epochs 250 \\
    --batch 32 \\
    --name ball_shoe_v1

  # Fine-tuning (기존 ball 모델에서 시작)
  python train_ball_detection_with_shoe_negatives.py \\
    --data /workspace/Soccer/dataset/ball_shoe/merged/data.yaml \\
    --pretrained /workspace/Soccer/runs/detect/yolo11x/weights/best.pt \\
    --epochs 100 \\
    --name ball_shoe_finetuned
        """
    )

    parser.add_argument('--data', type=str,
                       default='/workspace/Soccer/dataset/ball_shoe/merged/data.yaml',
                       help='Dataset yaml path')
    parser.add_argument('--model', type=str, default='yolo11x.pt',
                       help='Model to train (default: yolo11x.pt)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs (default: 200)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device (default: 0)')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='ball_with_shoe_negatives',
                       help='Experiment name')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Pretrained weights path for fine-tuning')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone layers')

    args = parser.parse_args()

    # 학습 실행
    train_ball_detection(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        freeze_backbone=args.freeze_backbone,
        pretrained_weights=args.pretrained
    )


if __name__ == '__main__':
    main()
