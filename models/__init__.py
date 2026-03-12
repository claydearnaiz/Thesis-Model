from models.yolov8s import YOLOv8sDetector
from models.yolov8n import YOLOv8nDetector
from models.mobilenet_ssd import MobileNetSSDDetector
from models.yolox_nano import YOLOXNanoDetector

MODEL_REGISTRY = {
    "yolov8s": YOLOv8sDetector,
    "yolov8n": YOLOv8nDetector,
    "mobilenet-ssd": MobileNetSSDDetector,
    "yolox-nano": YOLOXNanoDetector,
}


def get_detector(model_name: str, confidence: float = 0.5):
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return MODEL_REGISTRY[model_name](confidence=confidence)


def list_models():
    return list(MODEL_REGISTRY.keys())
