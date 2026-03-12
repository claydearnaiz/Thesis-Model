"""
Download model weights that aren't auto-fetched.

Usage:
    python download_weights.py mobilenet-ssd
    python download_weights.py yolox-nano
    python download_weights.py all
"""

import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)


def download_url(url: str, dest: Path, desc: str):
    if dest.exists():
        print(f"  [SKIP] {desc} already exists: {dest.name}")
        return
    print(f"  Downloading {desc}...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Saved to {dest}")


def download_gdrive(file_id: str, dest: Path, desc: str):
    if dest.exists():
        print(f"  [SKIP] {desc} already exists: {dest.name}")
        return
    import gdown
    print(f"  Downloading {desc} from Google Drive...")
    gdown.download(id=file_id, output=str(dest), quiet=False)
    print(f"  Saved to {dest}")


def download_mobilenet_ssd():
    print("\n=== MobileNet-SSD ===")
    download_url(
        "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.prototxt",
        WEIGHTS_DIR / "mobilenet_ssd_deploy.prototxt",
        "deploy.prototxt"
    )
    download_url(
        "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.caffemodel",
        WEIGHTS_DIR / "mobilenet_ssd.caffemodel",
        "mobilenet_ssd.caffemodel (~23MB)"
    )


def download_yolox_nano():
    print("\n=== YOLOX-Nano ===")
    download_url(
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
        WEIGHTS_DIR / "yolox_nano.onnx",
        "yolox_nano.onnx (~4MB)"
    )


DOWNLOADS = {
    "mobilenet-ssd": download_mobilenet_ssd,
    "yolox-nano": download_yolox_nano,
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
