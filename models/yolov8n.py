import numpy as np
from ultralytics import YOLO
from models.base import BaseDetector


class YOLOv8nDetector(BaseDetector):
    name = "yolov8n"

    def __init__(self, confidence: float = 0.5):
        super().__init__(confidence)
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame: np.ndarray) -> list[dict]:
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
