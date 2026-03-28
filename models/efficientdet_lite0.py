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


def _dequantize(tensor, details):
    """Dequantize INT8 tensor to float using scale and zero_point."""
    if details["dtype"] == np.float32:
        return tensor.astype(np.float32)
    quant = details["quantization_parameters"]
    scale = quant["scales"]
    zero_point = quant["zero_points"]
    return (tensor.astype(np.float32) - zero_point) * scale


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
        self._interpreter = Interpreter(
            model_path=str(MODEL_PATH),
            num_threads=4,
        )
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._anchors = _generate_anchors()

        # Identify which output is boxes (shape [...,4]) vs scores (shape [...,90])
        self._box_idx = 0
        self._score_idx = 1
        for i, det in enumerate(self._output_details):
            if det["shape"][-1] == 4:
                self._box_idx = i
            elif det["shape"][-1] == 90:
                self._score_idx = i

        # Warmup
        dummy = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        self._interpreter.set_tensor(self._input_details[0]["index"], dummy)
        self._interpreter.invoke()

    def detect(self, frame: np.ndarray) -> list[dict]:
        if _BACKEND == "mediapipe":
            return self._detect_mediapipe(frame)
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

        # Get raw tensors and dequantize if INT8
        raw_boxes = self._interpreter.get_tensor(
            self._output_details[self._box_idx]["index"])
        raw_scores = self._interpreter.get_tensor(
            self._output_details[self._score_idx]["index"])

        box_outputs = _dequantize(raw_boxes, self._output_details[self._box_idx])[0]
        class_outputs = _dequantize(raw_scores, self._output_details[self._score_idx])[0]

        person_scores = _sigmoid(class_outputs[:, PERSON_CLASS_ID])
        mask = person_scores >= self.confidence
        if not mask.any():
            return []

        scores_filtered = person_scores[mask]
        box_filtered = box_outputs[mask]
        anchors_filtered = self._anchors[mask]

        a_cy = anchors_filtered[:, 0]
        a_cx = anchors_filtered[:, 1]
        a_h = anchors_filtered[:, 2]
        a_w = anchors_filtered[:, 3]
        ty, tx, th, tw = box_filtered[:, 0], box_filtered[:, 1], box_filtered[:, 2], box_filtered[:, 3]

        pred_cy = a_cy + ty * a_h
        pred_cx = a_cx + tx * a_w
        pred_h = a_h * np.exp(np.clip(th, -5, 5))
        pred_w = a_w * np.exp(np.clip(tw, -5, 5))

        y1s = np.clip((pred_cy - pred_h / 2) / INPUT_SIZE * h, 0, h).astype(int)
        x1s = np.clip((pred_cx - pred_w / 2) / INPUT_SIZE * w, 0, w).astype(int)
        y2s = np.clip((pred_cy + pred_h / 2) / INPUT_SIZE * h, 0, h).astype(int)
        x2s = np.clip((pred_cx + pred_w / 2) / INPUT_SIZE * w, 0, w).astype(int)

        boxes_xywh = np.stack([x1s, y1s, (x2s - x1s), (y2s - y1s)], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, scores_filtered.tolist(), self.confidence, 0.45
        )
        if len(indices) == 0:
            return []

        detections = []
        for idx in indices.flatten():
            x1, y1 = int(x1s[idx]), int(y1s[idx])
            x2, y2 = int(x2s[idx]), int(y2s[idx])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "confidence": float(scores_filtered[idx])
            })

        return detections
