from abc import ABC, abstractmethod
import numpy as np
import cv2


class BaseDetector(ABC):
    """Shared interface for all person detection models."""

    name: str = "base"

    def __init__(self, confidence: float = 0.5):
        self.confidence = confidence

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a frame.

        Returns list of dicts:
            bbox: (x1, y1, x2, y2)
            center: (cx, cy)
            confidence: float
        """
        ...

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: list[dict],
                        roi_labels: list[list[str]] = None):
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center"]
            conf = det["confidence"]

            in_roi = roi_labels and i < len(roi_labels) and len(roi_labels[i]) > 0
            box_color = (0, 255, 0) if in_roi else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

            label = f"Person {conf:.0%}"
            if in_roi:
                label += f" [{', '.join(roi_labels[i])}]"

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
