"""
YOLOv8 Training Script for Augmented Traffic Sign Dataset
Train a new YOLOv8 model on your augmented realistic dataset
"""

from ultralytics import YOLO
import os
from pathlib import Path
import yaml
import shutil
import random


def split_dataset(dataset_dir, train_ratio=0.8):
    """
    Split dataset into train and val sets

    Args:
        dataset_dir: Path to dataset (contains images/ and labels/ folders)
        train_ratio: Ratio of training data (0.8 = 80% train, 20% val)
    """

    print("=" * 70)
    print("SPLITTING DATASET INTO TRAIN/VAL")
    print("=" * 70)

    images_dir = Path(dataset_dir) / 'images'
    labels_dir = Path(dataset_dir) / 'labels'

    # Create train/val directories
    train_images_dir = Path(dataset_dir) / 'train' / 'images'
    train_labels_dir = Path(dataset_dir) / 'train' / 'labels'
    val_images_dir = Path(dataset_dir) / 'val' / 'images'
    val_labels_dir = Path(dataset_dir) / 'val' / 'labels'

    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

    if not image_files:
        print("[ERROR] No images found!")
        return False

    print(f"Found {len(image_files)} images")

    # Shuffle
    random.shuffle(image_files)

    # Split
    train_count = int(len(image_files) * train_ratio)
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]

    print(f"Train: {len(train_files)} images")
    print(f"Val: {len(val_files)} images")

    # Copy files
    print("\nCopying files...")

    for i, img_file in enumerate(train_files):
        if (i + 1) % 500 == 0:
            print(f"  Train: {i + 1}/{len(train_files)}")

        # Copy image
        shutil.copy2(img_file, train_images_dir / img_file.name)

        # Copy label
        label_file = labels_dir / (img_file.stem + '.txt')
        if label_file.exists():
            shutil.copy2(label_file, train_labels_dir / label_file.name)

    for i, img_file in enumerate(val_files):
        if (i + 1) % 500 == 0:
            print(f"  Val: {i + 1}/{len(val_files)}")

        shutil.copy2(img_file, val_images_dir / img_file.name)

        label_file = labels_dir / (img_file.stem + '.txt')
        if label_file.exists():
            shutil.copy2(label_file, val_labels_dir / label_file.name)

    print("Split complete!")
    return True


def create_data_yaml(dataset_dir, signs_pngs_dir):
    """
    Create data.yaml file for YOLOv8 training

    Args:
        dataset_dir: Path to dataset (contains train/ and val/ folders)
        signs_pngs_dir: Path to transparent PNGs (to extract class names)
    """

    print("\n" + "=" * 70)
    print("CREATING data.yaml")
    print("=" * 70)

    # Get class names from PNG filenames
    png_files = sorted(list(Path(signs_pngs_dir).glob('*.png')))
    class_names = [f.stem for f in png_files]

    print(f"Found {len(class_names)} classes")
    print(f"\nFirst 10 classes:")
    for i, name in enumerate(class_names[:10]):
        print(f"  {i}: {name}")

    if len(class_names) > 10:
        print(f"  ...")
        print(f"  {len(class_names)-1}: {class_names[-1]}")

    # Create data.yaml content
    data_yaml = {
        'path': str(Path(dataset_dir).absolute()),  # Root directory
        'train': 'train/images',  # Training images
        'val': 'val/images',      # Validation images
        'nc': len(class_names),   # Number of classes
        'names': class_names      # Class names
    }

    # Save data.yaml
    yaml_path = Path(dataset_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\ndata.yaml created at: {yaml_path}")

    return str(yaml_path)


def train_yolov8_augmented(
    data_yaml_path,
    model_size='s',
    epochs=200,
    imgsz=640,
    batch_size=8,
    device='0',
    project='runs/train',
    name='augmented_traffic_signs',
    resume=False
):
    """
    Train YOLOv8 model on augmented dataset

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

    print("\n" + "=" * 70)
    print("YOLOv8 Training - Augmented Traffic Sign Detection")
    print("=" * 70)

    # Verify data.yaml exists
    if not os.path.exists(data_yaml_path):
        print(f"[ERROR] data.yaml not found at {data_yaml_path}")
        return None

    print(f"\nDataset: {data_yaml_path}")
    print(f"Model: YOLOv8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")

    # Load a pretrained YOLOv8 model
    print(f"\nLoading YOLOv8{model_size} pretrained model...")
    model = YOLO(f'yolov8{model_size}.pt')

    # Train the model
    print("\nStarting training...")
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

        hsv_h=0.01,      # Minimal - we want to preserve colors
        hsv_s=0.3,       # Reduced - already have color variation
        hsv_v=0.2,       # Reduced - already have brightness variation
        degrees=5.0,     # Reduced - already have rotation
        translate=0.1,   # Reduced - already have position variation
        scale=0.3,       # Reduced - already have scale variation
        fliplr=0.0,      # NO flip - traffic signs shouldn't be flipped!
        mosaic=0.0,      # Disabled - dataset already has multi-sign images
        mixup=0.0,       # Disabled - can mess up sign colors

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
        workers=0,
        dropout=0.0,     # No dropout - we have enough data
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)

    # Get the best model path
    best_model_path = Path(project) / name / 'weights' / 'best.pt'
    last_model_path = Path(project) / name / 'weights' / 'last.pt'

    print(f"\nBest model saved to: {best_model_path}")
    print(f"Last model saved to: {last_model_path}")

    # Validate the model
    print("\nValidating best model...")
    metrics = model.val()

    print("\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")

    return str(best_model_path)


def main():
    """
    Main training function
    """

    print("\nAugmented Traffic Sign Detection - YOLOv8 Training")
    print("=" * 70)

    # STEP 1: Get paths
    print("\nSTEP 1: Setup")
    dataset_dir = input("Path to your augmented dataset: ").strip()
    signs_pngs_dir = input("Path to your transparent PNGs: ").strip()

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return

    if not os.path.exists(signs_pngs_dir):
        print(f"[ERROR] Signs directory not found: {signs_pngs_dir}")
        return

    # STEP 2: Check if already split
    print("\nSTEP 2: Dataset Split")
    train_dir = Path(dataset_dir) / 'train'
    val_dir = Path(dataset_dir) / 'val'

    if train_dir.exists() and val_dir.exists():
        print("Dataset already split into train/val")
    else:
        print("Splitting dataset into train (80%) and val (20%)...")
        if not split_dataset(dataset_dir, train_ratio=0.8):
            print("[ERROR] Failed to split dataset")
            return

    # STEP 3: Create data.yaml
    print("\nSTEP 3: Create data.yaml")
    data_yaml_path = Path(dataset_dir) / 'data.yaml'

    if data_yaml_path.exists():
        print(f"data.yaml already exists: {data_yaml_path}")
        recreate = input("Recreate? (y/n): ").strip().lower()
        if recreate == 'y':
            data_yaml_path = create_data_yaml(dataset_dir, signs_pngs_dir)
    else:
        data_yaml_path = create_data_yaml(dataset_dir, signs_pngs_dir)

    # STEP 4: Training configuration
    print("\nSTEP 4: Training Configuration")

    config = {
        'data_yaml_path': str(data_yaml_path),
        'model_size': input("Model size (n/s/m/l/x, default 's'): ").strip() or 's',
        'epochs': int(input("Epochs (default 200): ").strip() or '200'),
        'batch_size': int(input("Batch size (default 8): ").strip() or '8'),
        'device': '0',  # GPU
        'project': r'D:\ObjectDetection\runs\train',
        'name': input("Model name (default 'augmented_signs'): ").strip() or 'augmented_signs',
    }

    print("\n" + "=" * 70)
    print("Final Configuration:")
    print("=" * 70)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70)

    # Ask for confirmation
    print("\n[WARNING] Training will start with the above configuration.")
    print("          This may take several hours depending on your hardware.")
    response = input("\nContinue? (y/n): ").strip().lower()

    if response != 'y':
        print("Training cancelled.")
        return

    # STEP 5: Start training
    print("\nSTEP 5: Training")
    best_model = train_yolov8_augmented(**config)

    if best_model:
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print("=" * 70)
        print(f"\nYour trained model: {best_model}")
        print("\nNext steps:")
        print("  1. Test on video: python test_video.py")
        print("  2. Test on images: python test_images.py")
        print(f"  3. Export model: yolo export model={best_model} format=onnx")
        print("\nModel saved with name: " + config['name'])


if __name__ == "__main__":

    # Install required packages if needed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics (YOLOv8)...")
        import subprocess
        subprocess.check_call(["pip", "install", "ultralytics", "--break-system-packages"])
        print("Installation complete! Please run the script again.")
        exit(0)

    main()