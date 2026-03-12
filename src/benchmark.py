"""
Model Benchmark Tool

Runs all detection models on the same video source and compares:
    - Average FPS
    - Average inference time (ms)
    - Total persons detected
    - Average confidence score
    - Detection consistency (std dev of detections per frame)

Usage:
    python src/benchmark.py                         # webcam, 200 frames
    python src/benchmark.py --source video.mp4      # video file
    python src/benchmark.py --frames 500            # more frames for accuracy
    python src/benchmark.py --confidence 0.4        # lower threshold
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import get_detector, list_models


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark all detection models")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index or path to video file")
    parser.add_argument("--frames", type=int, default=200,
                        help="Number of frames to process per model")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for all models")
    parser.add_argument("--save", action="store_true",
                        help="Save results chart to benchmark_results/")
    return parser.parse_args()


def capture_frames(source, num_frames: int) -> list[np.ndarray]:
    """Pre-capture frames so every model runs on the exact same data."""
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source {source}")
        sys.exit(1)

    frames = []
    print(f"Capturing {num_frames} frames from source '{source}'...")
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            if len(frames) == 0:
                print("ERROR: Could not read any frames")
                sys.exit(1)
            break
        frames.append(frame)
        if (i + 1) % 50 == 0:
            print(f"  Captured {i + 1}/{num_frames}")

    cap.release()
    print(f"  Done. Got {len(frames)} frames.\n")
    return frames


def benchmark_model(model_name: str, frames: list[np.ndarray],
                    confidence: float) -> dict:
    print(f"  Loading {model_name}...")
    try:
        detector = get_detector(model_name, confidence=confidence)
    except Exception as e:
        print(f"  SKIP {model_name}: {e}\n")
        return None

    # Warmup (3 frames)
    for f in frames[:3]:
        detector.detect(f)

    inference_times = []
    detections_per_frame = []
    all_confidences = []

    print(f"  Running inference on {len(frames)} frames...")
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        t1 = time.perf_counter()

        inference_times.append((t1 - t0) * 1000)  # ms
        detections_per_frame.append(len(detections))
        for det in detections:
            all_confidences.append(det["confidence"])

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(frames)} frames processed")

    avg_time = np.mean(inference_times)
    fps = 1000.0 / avg_time if avg_time > 0 else 0
    total_detections = sum(detections_per_frame)
    avg_detections = np.mean(detections_per_frame)
    std_detections = np.std(detections_per_frame)
    avg_conf = np.mean(all_confidences) if all_confidences else 0
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)

    result = {
        "model": model_name,
        "avg_fps": round(fps, 1),
        "avg_time_ms": round(avg_time, 1),
        "min_time_ms": round(min_time, 1),
        "max_time_ms": round(max_time, 1),
        "total_detections": total_detections,
        "avg_detections_per_frame": round(avg_detections, 2),
        "detection_std": round(std_detections, 2),
        "avg_confidence": round(avg_conf * 100, 1),
        "frames_processed": len(frames),
    }

    print(f"  {model_name}: {fps:.1f} FPS | {avg_time:.1f}ms avg | "
          f"{total_detections} total detections\n")
    return result


def print_results_table(results: list[dict]):
    print("=" * 90)
    print(f"{'Model':<16} {'FPS':>7} {'Avg ms':>8} {'Min ms':>8} {'Max ms':>8} "
          f"{'Detections':>11} {'Avg/Frame':>10} {'Avg Conf':>9}")
    print("-" * 90)
    for r in results:
        print(f"{r['model']:<16} {r['avg_fps']:>7.1f} {r['avg_time_ms']:>8.1f} "
              f"{r['min_time_ms']:>8.1f} {r['max_time_ms']:>8.1f} "
              f"{r['total_detections']:>11} {r['avg_detections_per_frame']:>10.2f} "
              f"{r['avg_confidence']:>8.1f}%")
    print("=" * 90)


def save_charts(results: list[dict], output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(exist_ok=True, parents=True)
    models = [r["model"] for r in results]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"][:len(models)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Benchmark Comparison", fontsize=16, fontweight="bold")

    # FPS
    ax = axes[0][0]
    bars = ax.bar(models, [r["avg_fps"] for r in results], color=colors)
    ax.set_title("Average FPS (higher is better)")
    ax.set_ylabel("Frames per Second")
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{r["avg_fps"]}', ha="center", fontweight="bold")

    # Inference time
    ax = axes[0][1]
    bars = ax.bar(models, [r["avg_time_ms"] for r in results], color=colors)
    ax.set_title("Average Inference Time (lower is better)")
    ax.set_ylabel("Milliseconds")
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{r["avg_time_ms"]}ms', ha="center", fontweight="bold")

    # Avg detections per frame
    ax = axes[1][0]
    bars = ax.bar(models, [r["avg_detections_per_frame"] for r in results], color=colors)
    ax.errorbar(models, [r["avg_detections_per_frame"] for r in results],
                yerr=[r["detection_std"] for r in results],
                fmt="none", ecolor="black", capsize=5)
    ax.set_title("Avg Detections per Frame (with std dev)")
    ax.set_ylabel("Persons Detected")

    # Avg confidence
    ax = axes[1][1]
    bars = ax.bar(models, [r["avg_confidence"] for r in results], color=colors)
    ax.set_title("Average Confidence Score")
    ax.set_ylabel("Confidence %")
    ax.set_ylim(0, 100)
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{r["avg_confidence"]}%', ha="center", fontweight="bold")

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = output_dir / f"benchmark_{timestamp}.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nChart saved to {chart_path}")


def save_csv(results: list[dict], output_dir: Path):
    import csv
    output_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"benchmark_{timestamp}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved to {csv_path}")


def main():
    args = parse_args()
    output_dir = Path(__file__).resolve().parent.parent / "benchmark_results"

    frames = capture_frames(args.source, args.frames)

    model_names = list_models()
    results = []

    print(f"Benchmarking {len(model_names)} models on {len(frames)} frames "
          f"(confidence={args.confidence})\n")

    for name in model_names:
        result = benchmark_model(name, frames, args.confidence)
        if result:
            results.append(result)

    if not results:
        print("No models completed successfully.")
        return

    print_results_table(results)
    save_csv(results, output_dir)

    if args.save:
        save_charts(results, output_dir)

    # Determine best model per category
    print("\n--- SUMMARY ---")
    fastest = max(results, key=lambda r: r["avg_fps"])
    most_detections = max(results, key=lambda r: r["avg_detections_per_frame"])
    most_confident = max(results, key=lambda r: r["avg_confidence"])
    most_consistent = min(results, key=lambda r: r["detection_std"])

    print(f"  Fastest:          {fastest['model']} ({fastest['avg_fps']} FPS)")
    print(f"  Most detections:  {most_detections['model']} ({most_detections['avg_detections_per_frame']} avg/frame)")
    print(f"  Highest confidence: {most_confident['model']} ({most_confident['avg_confidence']}%)")
    print(f"  Most consistent:  {most_consistent['model']} (std={most_consistent['detection_std']})")
    print()


if __name__ == "__main__":
    main()
