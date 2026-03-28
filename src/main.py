"""
Canteen Monitoring System - Main Entry Point

Real-time person detection + crowd counting + buyer estimation using TFLite models.

Usage:
    python src/main.py                                  # default (efficientdet-lite0)
    python src/main.py --model mobilenet-ssd            # use MobileNet SSD
    python src/main.py --source 1                       # different camera
    python src/main.py --no-display                     # headless mode
    python src/main.py --log                            # enable CSV logging
    python src/main.py --skip-frames 1                  # process every 2nd frame
    python src/main.py --cam-res 320x240                # lower camera resolution
"""

import argparse
import json
import sys
import time
import csv
import platform
import threading
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.roi_manager import ROIManager
from models import get_detector, list_models
from models.base import BaseDetector

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class ThreadedCamera:
    """Continuously grabs frames in a background thread so detection never waits for I/O."""

    def __init__(self, source, width=640, height=480):
        if isinstance(source, int) and platform.system() == "Windows":
            self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(source)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._stopped = False

        # Read first frame synchronously to confirm camera works
        self._ret, self._frame = self._cap.read()
        if not self._ret:
            return

        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while not self._stopped:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame
            if not ret:
                break

    def read(self):
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None

    def isOpened(self):
        return self._cap.isOpened() and self._ret

    def release(self):
        self._stopped = True
        self._cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Canteen Monitoring System")
    parser.add_argument("--model", type=str, default="efficientdet-lite0",
                        choices=list_models(),
                        help=f"Detection model ({', '.join(list_models())})")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "config" / "roi_config.json"),
                        help="Path to ROI config JSON")
    parser.add_argument("--source", type=str, default=None,
                        help="Camera index or video file path (overrides config)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without GUI window (headless)")
    parser.add_argument("--log", action="store_true",
                        help="Enable CSV logging of detections")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Skip N frames between detections (0=process every frame)")
    parser.add_argument("--cam-res", type=str, default=None,
                        help="Camera resolution WxH (e.g. 320x240, 640x480)")
    return parser.parse_args()


def setup_logger(enabled: bool):
    if not enabled:
        return None, None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"detections_{timestamp}.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["timestamp", "roi_name", "person_count", "total_persons"])
    print(f"Logging to {log_path}")
    return log_file, writer


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    if args.source is not None:
        cam_source = int(args.source) if args.source.isdigit() else args.source
    else:
        cam_source = config.get("camera_source", 0)

    confidence = config.get("confidence_threshold", 0.5)

    print(f"Loading model: {args.model}...")
    detector = get_detector(args.model, confidence=confidence)
    print(f"Model loaded: {detector.name}")

    roi_mgr = ROIManager(args.config)
    if not roi_mgr.rois:
        print("WARNING: No ROIs defined. Run calibrate_roi.py first.")

    if args.cam_res:
        cw, ch = map(int, args.cam_res.split("x"))
    else:
        cw = config.get("frame_width", 640)
        ch = config.get("frame_height", 480)

    cap = ThreadedCamera(cam_source, width=cw, height=ch)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera source {cam_source}")
        return

    log_file, csv_writer = setup_logger(args.log)

    skip_info = f" | skip={args.skip_frames}" if args.skip_frames else ""
    res_info = f" | res={cw}x{ch}"
    print(f"\nMonitoring started | Model: {detector.name} | Camera: {cam_source} | ROIs: {len(roi_mgr.rois)}{res_info}{skip_info}")
    print("Press Q to quit.\n")

    fps_counter = 0
    fps_timer = time.time()
    display_fps = 0.0
    frame_count = 0
    skip = args.skip_frames
    last_detections = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if isinstance(cam_source, str):
                    break
                print("ERROR: Failed to read frame")
                break

            frame_count += 1
            if skip > 0 and (frame_count % (skip + 1)) != 1:
                detections = last_detections
            else:
                detections = detector.detect(frame)
                last_detections = detections

            total_persons = len(detections)

            roi_labels = []
            roi_counts = {roi.name: 0 for roi in roi_mgr.rois}

            for det in detections:
                cx, cy = det["center"]
                matched = roi_mgr.check_point(cx, cy)
                roi_labels.append(matched)
                for name in matched:
                    roi_counts[name] += 1

            buyer_count = sum(roi_counts.values())

            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                display_fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()

            if fps_counter == 1:
                status_parts = [f"Total: {total_persons}"]
                for roi_name, count in roi_counts.items():
                    if count > 0:
                        status_parts.append(f"{roi_name}: {count}")
                    else:
                        status_parts.append(f"{roi_name}: empty")

                timestamp = datetime.now().strftime("%H:%M:%S")
                status = " | ".join(status_parts)
                print(f"[{timestamp}] FPS: {display_fps:.1f} | {status} | Est. Buyers: {buyer_count}")

                if csv_writer:
                    for roi_name, count in roi_counts.items():
                        csv_writer.writerow([
                            datetime.now().isoformat(),
                            roi_name,
                            count,
                            total_persons
                        ])

            if not args.no_display:
                roi_mgr.draw_all(frame)
                BaseDetector.draw_detections(frame, detections, roi_labels)

                info_line1 = f"Model: {detector.name} | FPS: {display_fps:.1f}"
                info_line2 = f"Persons: {total_persons} | Est. Buyers: {buyer_count}"
                cv2.putText(frame, info_line1, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                cv2.putText(frame, info_line2, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                y_offset = 75
                for roi_name, count in roi_counts.items():
                    color = (0, 255, 0) if count > 0 else (128, 128, 128)
                    label = f"{roi_name}: {count}"
                    cv2.putText(frame, label, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    y_offset += 25

                cv2.imshow("Canteen Monitor", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()
        print("Monitoring stopped.")


if __name__ == "__main__":
    main()
