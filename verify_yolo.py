import os
from pathlib import Path
from collections import Counter


def verify_yolo_labels(labels_dir, classes_file=None):
    """
    Verify YOLO label files and provide statistics
    """
    print("=" * 60)
    print("YOLO Labels Verification")
    print("=" * 60)

    if not os.path.exists(labels_dir):
        print(f"Error: Directory not found: {labels_dir}")
        return False

    # Find all .txt files (excluding classes.txt)
    label_files = [f for f in Path(labels_dir).glob('*.txt') if f.name != 'classes.txt']

    print(f"\nTotal label files: {len(label_files)}")

    if len(label_files) == 0:
        print("No label files found!")
        return False

    # Statistics
    total_objects = 0
    class_counts = Counter()
    files_with_errors = []
    empty_files = []

    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                empty_files.append(label_file.name)
                continue

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    files_with_errors.append((label_file.name, line_num, "Invalid format - expected 5 values"))
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Verify normalized coordinates (0-1)
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                            0 <= width <= 1 and 0 <= height <= 1):
                        files_with_errors.append((label_file.name, line_num, "Coordinates not normalized (0-1)"))

                    class_counts[class_id] += 1
                    total_objects += 1

                except ValueError as e:
                    files_with_errors.append((label_file.name, line_num, f"Invalid values: {e}"))

        except Exception as e:
            files_with_errors.append((label_file.name, 0, f"File error: {e}"))

    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"  Total objects: {total_objects}")
    print(f"  Empty files: {len(empty_files)}")
    print(f"  Files with errors: {len(files_with_errors)}")

    # Print class distribution
    if class_counts:
        print(f"\n📈 Class Distribution:")

        # Load class names if available
        class_names = {}
        if classes_file and os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                class_names = {i: name.strip() for i, name in enumerate(f)}

        for class_id, count in sorted(class_counts.items()):
            class_name = class_names.get(class_id, f"Unknown")
            percentage = (count / total_objects) * 100
            print(f"  Class {class_id} ({class_name}): {count} ({percentage:.1f}%)")

    # Print errors
    if files_with_errors:
        print(f"\n⚠️  Errors Found ({len(files_with_errors)}):")
        for filename, line_num, error in files_with_errors[:10]:  # Show first 10
            print(f"  {filename}:{line_num} - {error}")
        if len(files_with_errors) > 10:
            print(f"  ... and {len(files_with_errors) - 10} more errors")

    # Print empty files
    if empty_files:
        print(f"\n📭 Empty Files ({len(empty_files)}):")
        for filename in empty_files[:10]:  # Show first 10
            print(f"  {filename}")
        if len(empty_files) > 10:
            print(f"  ... and {len(empty_files) - 10} more empty files")

    print("\n" + "=" * 60)

    if files_with_errors:
        print("⚠️  Verification completed with errors")
        return False
    else:
        print("✓ Verification passed!")
        return True


def verify_yolo_dataset(dataset_dir):
    """
    Verify complete YOLO dataset structure
    """
    print("=" * 60)
    print("YOLO Dataset Verification")
    print("=" * 60)
    print(f"\nDataset directory: {dataset_dir}\n")

    # Check structure
    required_dirs = [
        'images/train',
        'images/val',
        'labels/train',
        'labels/val'
    ]

    print("📁 Directory Structure:")
    all_exist = True
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_dir, dir_path)
        exists = os.path.exists(full_path)
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {dir_path}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n⚠️  Missing required directories!")
        return False

    # Check data.yaml
    data_yaml = os.path.join(dataset_dir, 'data.yaml')
    if os.path.exists(data_yaml):
        print(f"  ✓ data.yaml")
    else:
        print(f"  ✗ data.yaml")

    # Count files in each split
    print("\n📊 File Counts:")
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(dataset_dir, 'images', split)
        lbl_dir = os.path.join(dataset_dir, 'labels', split)

        if os.path.exists(img_dir):
            img_count = len(list(Path(img_dir).glob('*.[jp][pn]g')))  # jpg, jpeg, png
            lbl_count = len(list(Path(lbl_dir).glob('*.txt'))) if os.path.exists(lbl_dir) else 0

            match_symbol = "✓" if img_count == lbl_count else "⚠️"
            print(f"  {split:5s}: {img_count} images, {lbl_count} labels {match_symbol}")

            if img_count != lbl_count:
                print(f"         Warning: Image/label count mismatch!")

    # Verify label format for train split
    train_labels = os.path.join(dataset_dir, 'labels', 'train')
    classes_file = os.path.join(dataset_dir, 'classes.txt')

    if not os.path.exists(classes_file):
        # Try to find it in parent directory
        classes_file = os.path.join(os.path.dirname(dataset_dir), 'yolo_labels', 'classes.txt')

    print("\n" + "=" * 60)
    print("Verifying label format...")
    print("=" * 60)

    verify_yolo_labels(train_labels, classes_file if os.path.exists(classes_file) else None)

    return True


def main():
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Default paths - UPDATE THIS
        path = r"C:\Downloads\Capstone\Capstone\yolo_labels"

    if not os.path.exists(path):
        print(f"Error: Path not found: {path}")
        print("\nUsage:")
        print("  Verify labels only:   python verify_yolo.py <labels_dir>")
        print("  Verify full dataset:  python verify_yolo.py <dataset_dir>")
        sys.exit(1)

    # Determine if it's a labels directory or dataset directory
    if 'labels' in os.listdir(path) and 'images' in os.listdir(path):
        # It's a dataset directory
        verify_yolo_dataset(path)
    else:
        # It's a labels directory
        classes_file = os.path.join(path, 'classes.txt')
        verify_yolo_labels(path, classes_file if os.path.exists(classes_file) else None)


if __name__ == "__main__":
    main()