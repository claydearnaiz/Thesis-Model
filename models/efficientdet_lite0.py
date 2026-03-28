import cv2
import numpy as np
from pathlib import Path
from models.base import BaseDetector

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
MODEL_PATH = WEIGHTS_DIR / "efficientdet_lite0.tflite"

INPUT_SIZE = 320
PERSON_CLASS_ID = 0
STRIDES = [8, 16, 32, 64, 128]
NUM_SCALES = 3
ASPECT_RATIOS = [1.0, 2.0, 0.5]
ANCHOR_SCALE = 4.0

# Detect which runtime is available
_BACKEND = None
try:
    from tflite_runtime.interpreter import Interpreter
    _BACKEND = "tflite_runtime"
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        _BACKEND = "tensorflow"
    except ImportError:
        try:
            import mediapipe
            _BACKEND = "mediapipe"
        except ImportError:
            raise ImportError(
                "No TFLite backend found. Install one of:\n"
                "  pip install tflite-runtime   (Raspberry Pi)\n"
                "  pip install mediapipe         (PC)"
            )


def _generate_anchors():
    """Generate EfficientDet-Lite0 anchor boxes for 320x320 input."""
    anchors = []
    scales = [2 ** (i / NUM_SCALES) for i in range(NUM_SCALES)]

    for stride in STRIDES:
        grid_h = int(np.ceil(INPUT_SIZE / stride))
        grid_w = int(np.ceil(INPUT_SIZE / stride))
        for y in range(grid_h):
            for x in range(grid_w):
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                for scale in scales:
                    for ratio in ASPECT_RATIOS:
                        w = ANCHOR_SCALE * stride * scale * np.sqrt(ratio)
                        h = ANCHOR_SCALE * stride * scale / np.sqrt(ratio)
                        anchors.append([cy, cx, h, w])

    return np.array(anchors, dtype=np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class EfficientDetLite0Detector(BaseDetector):
    """EfficientDet-Lite0 — 320x320, INT8 quantized. Auto-selects backend."""

    name = "efficientdet-lite0"

    def __init__(self, confidence: float = 0.5):
        super().__init__(confidence)
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"EfficientDet-Lite0 weights not found.\n"
                f"Run: python download_weights.py efficientdet-lite0\n"
                f"Expected: {MODEL_PATH}"
            )

        if _BACKEND == "mediapipe":
            self._init_mediapipe()
        else:
            self._init_tflite()

    def _init_mediapipe(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            ObjectDetector, ObjectDetectorOptions, RunningMode
        )
        self._mp = mp
        options = ObjectDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODEL_PATH),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.IMAGE,
            max_results=20,
            score_threshold=self.confidence,
            category_allowlist=["person"],
        )
        self._detector = ObjectDetector.create_from_options(options)

        dummy = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy)
        self._detector.detect(mp_img)

    def _init_tflite(self):
        self._interpreter = Interpreter(model_path=str(MODEL_PATH))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._anchors = _generate_anchors()

        dummy = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        self._interpreter.set_tensor(self._input_details[0]["index"], dummy)
        self._interpreter.invoke()

    def detect(self, frame: np.ndarray) -> list[dict]:
        if _BACKEND == "mediapipe":
            return self._detect_mediapipe(frame)
        else:
            return self._detect_tflite(frame)

    def _detect_mediapipe(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        result = self._detector.detect(mp_image)

        detections = []
        for det in result.detections:
            bbox = det.bounding_box
            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)
            x2 = min(w, bbox.origin_x + bbox.width)
            y2 = min(h, bbox.origin_y + bbox.height)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            conf = det.categories[0].score
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "confidence": float(conf)
            })
        return detections

    def _detect_tflite(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]

        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img, axis=0).astype(np.uint8)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()

        # Raw outputs: boxes offsets [1,N,4] and class scores [1,N,90]
        box_outputs = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        class_outputs = self._interpreter.get_tensor(self._output_details[1]["index"])[0]

        person_scores = _sigmoid(class_outputs[:, PERSON_CLASS_ID])
        mask = person_scores >= self.confidence
        if not mask.any():
            return []

        scores_filtered = person_scores[mask]
        box_filtered = box_outputs[mask]
        anchors_filtered = self._anchors[mask]

        # Decode boxes: offsets relative to anchors [cy, cx, h, w]
        a_cy, a_cx, a_h, a_w = (
            anchors_filtered[:, 0], anchors_filtered[:, 1],
            anchors_filtered[:, 2], anchors_filtered[:, 3],
        )
        ty, tx, th, tw = (
            box_filtered[:, 0], box_filtered[:, 1],
            box_filtered[:, 2], box_filtered[:, 3],
        )

        pred_cy = a_cy + ty * a_h
        pred_cx = a_cx + tx * a_w
        pred_h = a_h * np.exp(th)
        pred_w = a_w * np.exp(tw)

        # Convert to pixel coords [0, INPUT_SIZE] then normalize to frame
        y1s = np.clip((pred_cy - pred_h / 2) / INPUT_SIZE * h, 0, h).astype(int)
        x1s = np.clip((pred_cx - pred_w / 2) / INPUT_SIZE * w, 0, w).astype(int)
        y2s = np.clip((pred_cy + pred_h / 2) / INPUT_SIZE * h, 0, h).astype(int)
        x2s = np.clip((pred_cx + pred_w / 2) / INPUT_SIZE * w, 0, w).astype(int)

        # NMS
        boxes_xywh = np.stack([
            x1s, y1s, (x2s - x1s), (y2s - y1s)
        ], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores_filtered.tolist(), self.confidence, 0.45)
        if len(indices) == 0:
            return []

        detections = []
        for idx in indices.flatten():
            x1, y1, x2, y2 = int(x1s[idx]), int(y1s[idx]), int(x2s[idx]), int(y2s[idx])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "confidence": float(scores_filtered[idx])
            })

        return detections
