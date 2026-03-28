"""
ROI Calibration Tool

Run this script to visually define ROI polygons on your camera feed.

Controls:
    Left Click  - Place a polygon vertex
    R           - Finish current ROI (closes the polygon)
    U           - Undo last placed vertex
    C           - Clear all ROIs and start over
    S           - Save ROIs to config file
    Q / ESC     - Quit without saving
"""

import cv2
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.roi_manager import ROIManager

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config" / "roi_config.json")

COLORS = [
    (0, 255, 0),
    (255, 165, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 255, 0),
]

current_points = []
roi_manager = ROIManager()
roi_count = 0


def mouse_callback(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])


def main():
    global current_points, roi_count

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    cam_source = config.get("camera_source", 0)
    if isinstance(cam_source, int):
        cap = cv2.VideoCapture(cam_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_source)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera source {cam_source}")
        return

    window_name = "ROI Calibration - Click to place vertices"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n=== ROI CALIBRATION TOOL ===")
    print("Left Click : Place vertex")
    print("R          : Finish current ROI")
    print("U          : Undo last vertex")
    print("C          : Clear all ROIs")
    print("S          : Save and quit")
    print("Q / ESC    : Quit without saving\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break

        roi_manager.draw_all(frame, thickness=2)

        if len(current_points) > 0:
            pts_array = [current_points]
            import numpy as np
            cv2.polylines(frame, [np.array(current_points, dtype=np.int32)],
                          isClosed=False, color=(0, 200, 255), thickness=2)
            for pt in current_points:
                cv2.circle(frame, tuple(pt), 6, (0, 200, 255), -1)

        instructions = "LClick: vertex | R: finish ROI | U: undo | C: clear | S: save | Q: quit"
        cv2.putText(frame, instructions, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        roi_info = f"Saved ROIs: {len(roi_manager.rois)} | Current vertices: {len(current_points)}"
        cv2.putText(frame, roi_info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            if len(current_points) >= 3:
                roi_count += 1
                name = input(f"Enter name for ROI #{roi_count}: ").strip()
                if not name:
                    name = f"ROI_{roi_count}"
                color = COLORS[(roi_count - 1) % len(COLORS)]
                roi_manager.add_roi(name, current_points, color)
                current_points = []
                print(f"  -> ROI '{name}' saved with {len(roi_manager.rois)} total ROIs")
            else:
                print("  Need at least 3 vertices to form an ROI. Keep clicking.")

        elif key == ord("u"):
            if current_points:
                removed = current_points.pop()
                print(f"  Undo vertex {removed}")

        elif key == ord("c"):
            roi_manager.clear()
            current_points = []
            roi_count = 0
            print("  All ROIs cleared.")

        elif key == ord("s"):
            roi_manager.config_path = CONFIG_PATH
            roi_manager.save()
            print(f"\n  ROIs saved to {CONFIG_PATH}")
            break

        elif key in (ord("q"), 27):
            print("  Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
