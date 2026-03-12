import cv2
import numpy as np
from pathlib import Path
from models.base import BaseDetector

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
CAFFEMODEL_URL = "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"

PROTOTXT_PATH = WEIGHTS_DIR / "mobilenet_ssd_deploy.prototxt"
CAFFEMODEL_PATH = WEIGHTS_DIR / "mobilenet_ssd.caffemodel"

COCO_PERSON_IDX = 15  # MobileNet-SSD uses VOC classes; "person" is index 15


class MobileNetSSDDetector(BaseDetector):
    name = "mobilenet-ssd"

    def __init__(self, confidence: float = 0.5):
        super().__init__(confidence)

        if not PROTOTXT_PATH.exists() or not CAFFEMODEL_PATH.exists():
            raise FileNotFoundError(
                f"MobileNet-SSD weights not found.\n"
                f"Run: python download_weights.py mobilenet-ssd\n"
                f"Expected files:\n"
                f"  {PROTOTXT_PATH}\n"
                f"  {CAFFEMODEL_PATH}"
            )

        self.net = cv2.dnn.readNetFromCaffe(
            str(PROTOTXT_PATH), str(CAFFEMODEL_PATH)
        )

    def detect(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        raw_detections = self.net.forward()

        detections = []
        for i in range(raw_detections.shape[2]):
            confidence = raw_detections[0, 0, i, 2]
            class_id = int(raw_detections[0, 0, i, 1])

            if class_id != COCO_PERSON_IDX or confidence < self.confidence:
                continue

            x1 = max(0, int(raw_detections[0, 0, i, 3] * w))
            y1 = max(0, int(raw_detections[0, 0, i, 4] * h))
            x2 = min(w, int(raw_detections[0, 0, i, 5] * w))
            y2 = min(h, int(raw_detections[0, 0, i, 6] * h))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "confidence": float(confidence)
            })

        return detections
