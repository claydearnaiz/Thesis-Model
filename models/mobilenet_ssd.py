import cv2
import numpy as np
from pathlib import Path
from models.base import BaseDetector

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
PROTOTXT_PATH = WEIGHTS_DIR / "mobilenet_ssd_deploy.prototxt"
CAFFEMODEL_PATH = WEIGHTS_DIR / "mobilenet_ssd.caffemodel"

INPUT_SIZE = (300, 300)
PERSON_CLASS_ID = 15  # VOC class index for "person"


class MobileNetSSDDetector(BaseDetector):
    """MobileNet SSD V1 — lightweight, fast, 300x300 input (OpenCV DNN)."""

    name = "mobilenet-ssd"

    def __init__(self, confidence: float = 0.5):
        super().__init__(confidence)
        if not PROTOTXT_PATH.exists() or not CAFFEMODEL_PATH.exists():
            raise FileNotFoundError(
                f"MobileNet-SSD weights not found.\n"
                f"Run: python download_weights.py mobilenet-ssd\n"
                f"Expected:\n  {PROTOTXT_PATH}\n  {CAFFEMODEL_PATH}"
            )
        self.net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))

        # Warmup inference to avoid cold-start lag on first real frame
        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        self.detect(dummy)

    def detect(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, INPUT_SIZE), 0.007843, INPUT_SIZE, 127.5
        )
        self.net.setInput(blob)
        raw = self.net.forward()

        detections = []
        for i in range(raw.shape[2]):
            conf = float(raw[0, 0, i, 2])
            class_id = int(raw[0, 0, i, 1])

            if class_id != PERSON_CLASS_ID or conf < self.confidence:
                continue

            x1 = max(0, int(raw[0, 0, i, 3] * w))
            y1 = max(0, int(raw[0, 0, i, 4] * h))
            x2 = min(w, int(raw[0, 0, i, 5] * w))
            y2 = min(h, int(raw[0, 0, i, 6] * h))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "confidence": conf
            })

        return detections
