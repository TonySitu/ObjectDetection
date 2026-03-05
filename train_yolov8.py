"""
YOLOv8 Training Script for MTSD Traffic Sign Dataset
Train a YOLOv8 model on your organized YOLO dataset
"""

from ultralytics import YOLO
import os
from pathlib import Path

def train_yolov8_mtsd(
    data_yaml_path,
    model_size='n',
    epochs=50,
    imgsz=640,
    batch_size=16,
    device='0',
    project='runs/train',
    name='mtsd_no_other',
    resume=False
):
    """
    Train YOLOv8 model on MTSD dataset

    Args:
        data_yaml_path: Path to data.yaml file
        model_size: Model size (n=nano, s=small, m=medium, l=large, x=extra-large)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch_size: Batch size
        device: Device to train on ('0' for GPU, 'cpu' for CPU)
        project: Project directory
        name: Experiment name
        resume: Resume training from last checkpoint
    """

    print("=" * 70)
    print("YOLOv8 Training - MTSD Traffic Sign Detection")
    print("=" * 70)

    # Verify data.yaml exists
    if not os.path.exists(data_yaml_path):
        print(f"❌ Error: data.yaml not found at {data_yaml_path}")
        return None

    print(f"\n📁 Dataset: {data_yaml_path}")
    print(f"🤖 Model: YOLOv8{model_size}")
    print(f"📊 Epochs: {epochs}")
    print(f"📐 Image size: {imgsz}")
    print(f"📦 Batch size: {batch_size}")
    print(f"💻 Device: {device}")

    # Load a pretrained YOLOv8 model
    print(f"\n🔄 Loading YOLOv8{model_size} pretrained model...")
    model = YOLO(f'yolov8{model_size}.pt')

    # Train the model
    print("\n🚀 Starting training...")
    print("=" * 70)

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        resume=resume,

        # Training hyperparameters (optimized for traffic signs)
        patience=50,
        save=True,
        save_period=10,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.2,
        scale=0.5,
        fliplr=0.0,
        mosaic=1.0,
        mixup=0.2,

        # Optimization
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Validation
        val=True,
        plots=True,

        # Other
        verbose=True,
        workers=8,
        dropout=0.2,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)

    # Get the best model path
    best_model_path = Path(project) / name / 'weights' / 'best.pt'
    last_model_path = Path(project) / name / 'weights' / 'last.pt'

    print(f"\n📦 Best model saved to: {best_model_path}")
    print(f"📦 Last model saved to: {last_model_path}")

    # Validate the model
    print("\n🧪 Validating best model...")
    metrics = model.val()

    print("\n📊 Validation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")

    return str(best_model_path)


def main():
    """
    Main training function
    """

    # CONFIGURATION - UPDATE THIS PATH
    data_yaml_path = r"D:\canadian-traffic-signs\data.yaml"

    # Training configuration
    config = {
        'data_yaml_path': data_yaml_path,
        'model_size': 'n',      # Options: 'n', 's', 'm', 'l', 'x'
        'epochs': 100,          # Number of training epochs
        'imgsz': 640,           # Image size
        'batch_size': 8,       # Batch size (reduce if OOM errors)
        'device': '0',          # '0' for GPU, 'cpu' for CPU
        'project': 'runs/train',
        'name': 'canadian_signs_200',
    }

    print("\n🚦 MTSD Traffic Sign Detection - YOLOv8 Training")
    print("=" * 70)
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70)

    # Verify data.yaml exists
    if not os.path.exists(config['data_yaml_path']):
        print(f"\n❌ Error: data.yaml not found!")
        print(f"Expected location: {config['data_yaml_path']}")
        print("\nMake sure you've run organize_yolo_dataset.py first!")
        return

    # Ask for confirmation
    print("\n⚠️  Training will start with the above configuration.")
    print("   This may take several hours depending on your hardware.")
    response = input("\nContinue? (y/n): ").strip().lower()

    if response != 'y':
        print("Training cancelled.")
        return

    # Start training
    best_model = train_yolov8_mtsd(**config)

    if best_model:
        print("\n" + "=" * 70)
        print("🎉 Training completed successfully!")
        print("=" * 70)
        print(f"\n📍 Your trained model: {best_model}")
        print("\n📝 Next steps:")
        print("  1. Test on video: python test_video.py")
        print("  2. Export model: yolo export model={} format=onnx".format(best_model))


if __name__ == "__main__":
    # Install required packages if needed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics (YOLOv8)...")
        import subprocess
        subprocess.check_call(["pip", "install", "ultralytics"])
        print("✓ Installation complete! Please run the script again.")
        exit(0)

    main()