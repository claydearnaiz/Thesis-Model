import numpy as np
import cv2
from ultralytics import YOLO


class PersonDetector:
    PERSON_CLASS_ID = 0  # COCO class index for "person"

    def __init__(self, model_name: str = "yolov8s.pt", confidence: float = 0.5):
        self.confidence = confidence
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLOv8 on a frame and return person detections.

        Returns list of dicts with keys:
            bbox: (x1, y1, x2, y2)
            center: (cx, cy)
            confidence: float
        """
        results = self.model(frame, conf=self.confidence,
                             classes=[self.PERSON_CLASS_ID], verbose=False)
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "confidence": float(box.conf[0])
            })

        return detections

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: list[dict],
                        roi_labels: list[list[str]] = None):
        """Draw bounding boxes and center dots on frame."""
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
