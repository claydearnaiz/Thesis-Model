from models.efficientdet_lite0 import EfficientDetLite0Detector
from models.mobilenet_ssd import MobileNetSSDDetector

MODEL_REGISTRY = {
    "efficientdet-lite0": EfficientDetLite0Detector,
    "mobilenet-ssd": MobileNetSSDDetector,
}


def get_detector(model_name: str, confidence: float = 0.5):
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return MODEL_REGISTRY[model_name](confidence=confidence)


def list_models():
    return list(MODEL_REGISTRY.keys())
