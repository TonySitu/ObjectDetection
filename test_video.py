"""
YOLOv8 Video Testing Script
Test your trained YOLOv8 model on MP4 videos
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import time


def test_on_video(
        model_path,
        video_path,
        output_path=None,
        conf_threshold=0.25,
        iou_threshold=0.45,
        show_labels=True,
        show_conf=True,
        line_width=2,
        save_video=True
):
    """
    Run YOLOv8 detection on a video file

    Args:
        model_path: Path to trained YOLOv8 model (.pt file)
        video_path: Path to input video file
        output_path: Path to save output video (None = auto generate)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        show_labels: Show class labels
        show_conf: Show confidence scores
        line_width: Bounding box line width
        save_video: Save output video
    """

    print("=" * 70)
    print("YOLOv8 Video Testing - Traffic Sign Detection")
    print("=" * 70)

    # Verify model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return

    # Verify video exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at {video_path}")
        return

    # Generate output path if not provided
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"output_{video_name}_detected.mp4"

    print(f"\n📦 Model: {model_path}")
    print(f"🎬 Input video: {video_path}")
    print(f"💾 Output video: {output_path}")
    print(f"🎯 Confidence threshold: {conf_threshold}")
    print(f"📊 IOU threshold: {iou_threshold}")

    # Load model
    print("\n🔄 Loading model...")
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Could not open video")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n📹 Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / fps:.2f} seconds")

    # Setup video writer
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("\n🚀 Processing video...")
    print("=" * 70)

    frame_count = 0
    detection_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Run inference
            results = model(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )

            # Get detections for this frame
            result = results[0]
            detections = len(result.boxes)
            detection_count += detections

            # Draw results on frame
            annotated_frame = result.plot(
                conf=show_conf,
                labels=show_labels,
                line_width=line_width
            )

            # Save frame
            if save_video:
                out.write(annotated_frame)

            # Show progress
            if frame_count % 30 == 0:  # Update every 30 frames
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                progress = (frame_count / total_frames) * 100
                eta = (total_frames - frame_count) / fps_current if fps_current > 0 else 0

                print(f"  Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                      f"FPS: {fps_current:.1f} | "
                      f"Detections: {detections} | "
                      f"ETA: {eta:.0f}s")

    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")

    finally:
        # Cleanup
        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()

    # Print summary
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("✅ Processing complete!")
    print("=" * 70)
    print(f"\n📊 Statistics:")
    print(f"  Frames processed: {frame_count}/{total_frames}")
    print(f"  Total detections: {detection_count}")
    print(f"  Average detections per frame: {detection_count / frame_count:.2f}")
    print(f"  Processing time: {elapsed_time:.2f} seconds")
    print(f"  Average FPS: {frame_count / elapsed_time:.2f}")

    if save_video:
        print(f"\n💾 Output saved to: {output_path}")
        print(f"   File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")


def main():
    """
    Main function - Update these paths for your system
    """

    # CONFIGURATION - UPDATE THESE PATHS

    # Path to your trained model
    model_path = r"D:\ObjectDetection\runs\models\best.pt"

    # Path to your test video
    video_path = r"C:\Users\Tony\Downloads\airport highway.mp4"

    # Output video path (None = auto generate)
    output_path = None

    # Detection settings
    config = {
        'model_path': model_path,
        'video_path': video_path,
        'output_path': output_path,
        'conf_threshold': 0.25,  # Confidence threshold (0-1)
        'iou_threshold': 0.45,  # IOU threshold for NMS
        'show_labels': True,  # Show class labels
        'show_conf': True,  # Show confidence scores
        'line_width': 2,  # Bounding box line width
        'save_video': True,  # Save output video
    }

    print("\n🚦 MTSD Traffic Sign Detection - Video Testing")
    print("=" * 70)
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70)

    # Verify files exist
    if not os.path.exists(config['model_path']):
        print(f"\n❌ Error: Model not found!")
        print(f"Expected location: {config['model_path']}")
        print("\nMake sure you've trained the model first with train_yolov8.py")
        print("Or update the model_path to point to your trained model.")
        return

    if not os.path.exists(config['video_path']):
        print(f"\n❌ Error: Video not found!")
        print(f"Expected location: {config['video_path']}")
        print("\nPlease update video_path to point to your test video.")
        return

    # Run detection
    test_on_video(**config)

    print("\n✨ Done! Check the output video for results.")


if __name__ == "__main__":
    # Install required packages if needed
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("\nInstalling required packages...")
        import subprocess

        subprocess.check_call(["pip", "install", "ultralytics", "opencv-python"])
        print("✓ Installation complete! Please run the script again.")
        exit(0)

    main()