import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from models.base import BaseDetector

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
ONNX_PATH = WEIGHTS_DIR / "yolox_nano.onnx"

INPUT_SIZE = (416, 416)


def _generate_grids_and_strides(input_size, strides=(8, 16, 32)):
    grids = []
    expanded_strides = []
    for stride in strides:
        h = input_size[0] // stride
        w = input_size[1] // stride
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        grids.append(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1))
        expanded_strides.append(np.full((h * w, 1), stride))
    return np.concatenate(grids, axis=0), np.concatenate(expanded_strides, axis=0)


def _nms(boxes, scores, iou_threshold=0.45):
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=iou_threshold
    )
    if len(indices) == 0:
        return np.array([], dtype=int)
    return indices.flatten()


class YOLOXNanoDetector(BaseDetector):
    name = "yolox-nano"

    def __init__(self, confidence: float = 0.5):
        super().__init__(confidence)

        if not ONNX_PATH.exists():
            raise FileNotFoundError(
                f"YOLOX-Nano ONNX weights not found.\n"
                f"Run: python download_weights.py yolox-nano\n"
                f"Expected: {ONNX_PATH}"
            )

        self.session = ort.InferenceSession(str(ONNX_PATH))
        self.input_name = self.session.get_inputs()[0].name
        self.grids, self.strides = _generate_grids_and_strides(INPUT_SIZE)

    def _preprocess(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        ratio = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)
        new_h, new_w = int(h * ratio), int(w * ratio)

        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((INPUT_SIZE[0], INPUT_SIZE[1], 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        img = padded.astype(np.float32)
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        return img, ratio

    def detect(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        img, ratio = self._preprocess(frame)

        outputs = self.session.run(None, {self.input_name: img})[0]  # (1, N, 85)
        predictions = outputs[0]

        # Decode: center_x, center_y, w, h, obj_conf, class_scores...
        predictions[:, :2] = (predictions[:, :2] + self.grids) * self.strides
        predictions[:, 2:4] = np.exp(predictions[:, 2:4]) * self.strides

        obj_conf = predictions[:, 4]
        class_scores = predictions[:, 5:]
        class_ids = class_scores.argmax(axis=1)
        class_conf = class_scores[np.arange(len(class_scores)), class_ids]
        scores = obj_conf * class_conf

        person_mask = (class_ids == 0) & (scores >= self.confidence)
        if not person_mask.any():
            return []

        preds = predictions[person_mask]
        scores_filtered = scores[person_mask]

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = (cx - bw / 2) / ratio
        y1 = (cy - bh / 2) / ratio
        x2 = (cx + bw / 2) / ratio
        y2 = (cy + bh / 2) / ratio

        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)  # xywh for NMS
        keep = _nms(boxes, scores_filtered)

        detections = []
        for idx in keep:
            bx1 = max(0, int(x1[idx]))
            by1 = max(0, int(y1[idx]))
            bx2 = min(w, int(x2[idx]))
            by2 = min(h, int(y2[idx]))
            bcx = (bx1 + bx2) // 2
            bcy = (by1 + by2) // 2

            detections.append({
                "bbox": (bx1, by1, bx2, by2),
                "center": (bcx, bcy),
                "confidence": float(scores_filtered[idx])
            })

        return detections
