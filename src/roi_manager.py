import json
import numpy as np
import cv2
from pathlib import Path


class ROI:
    def __init__(self, name: str, points: list, color: tuple = (0, 255, 0)):
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.color = tuple(color)

    def contains_point(self, x: int, y: int) -> bool:
        result = cv2.pointPolygonTest(self.points, (float(x), float(y)), False)
        return result >= 0

    def draw(self, frame: np.ndarray, thickness: int = 2, show_label: bool = True):
        cv2.polylines(frame, [self.points], isClosed=True, color=self.color, thickness=thickness)
        if show_label:
            top_left = self.points.min(axis=0)
            label_pos = (int(top_left[0]), int(top_left[1]) - 10)
            cv2.putText(frame, self.name, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, self.color, 2)


class ROIManager:
    def __init__(self, config_path: str = None):
        self.rois: list[ROI] = []
        self.config_path = config_path
        if config_path and Path(config_path).exists():
            self.load(config_path)

    def load(self, config_path: str):
        with open(config_path, "r") as f:
            data = json.load(f)
        self.rois = []
        for roi_data in data.get("rois", []):
            self.rois.append(ROI(
                name=roi_data["name"],
                points=roi_data["points"],
                color=roi_data.get("color", [0, 255, 0])
            ))

    def save(self, config_path: str = None):
        path = config_path or self.config_path
        if not path:
            raise ValueError("No config path specified")

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        data["rois"] = []
        for roi in self.rois:
            data["rois"].append({
                "name": roi.name,
                "points": roi.points.tolist(),
                "color": list(roi.color)
            })

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def add_roi(self, name: str, points: list, color: tuple = (0, 255, 0)):
        self.rois.append(ROI(name, points, color))

    def clear(self):
        self.rois = []

    def check_point(self, x: int, y: int) -> list[str]:
        """Return names of all ROIs that contain the given point."""
        return [roi.name for roi in self.rois if roi.contains_point(x, y)]

    def draw_all(self, frame: np.ndarray, thickness: int = 2):
        for roi in self.rois:
            roi.draw(frame, thickness)
