"""
Download pretrained model weights.

Usage:
    python download_weights.py efficientdet-lite0
    python download_weights.py mobilenet-ssd
    python download_weights.py all
"""

import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)


def download(url: str, dest: Path, desc: str):
    if dest.exists():
        print(f"  [SKIP] {desc} already exists: {dest.name}")
        return
    print(f"  Downloading {desc}...")
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  Saved to {dest} ({size_mb:.1f} MB)")


def download_efficientdet_lite0():
    print("\n=== EfficientDet-Lite0 (TFLite, INT8) ===")
    download(
        "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite",
        WEIGHTS_DIR / "efficientdet_lite0.tflite",
        "efficientdet_lite0.tflite"
    )


def download_mobilenet_ssd():
    print("\n=== MobileNet SSD V1 (Caffe) ===")
    download(
        "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.prototxt",
        WEIGHTS_DIR / "mobilenet_ssd_deploy.prototxt",
        "mobilenet_ssd_deploy.prototxt"
    )
    download(
        "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.caffemodel",
        WEIGHTS_DIR / "mobilenet_ssd.caffemodel",
        "mobilenet_ssd.caffemodel"
    )


DOWNLOADS = {
    "efficientdet-lite0": download_efficientdet_lite0,
    "mobilenet-ssd": download_mobilenet_ssd,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_weights.py <model|all>")
        print(f"Available: {', '.join(DOWNLOADS.keys())}, all")
        return

    target = sys.argv[1].lower()

    if target == "all":
        for fn in DOWNLOADS.values():
            fn()
    elif target in DOWNLOADS:
        DOWNLOADS[target]()
    else:
        print(f"Unknown model: {target}")
        print(f"Available: {', '.join(DOWNLOADS.keys())}, all")

    print("\nDone.")


if __name__ == "__main__":
    main()
