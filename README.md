# Canteen Monitoring System

Real-time crowd counting and buyer estimation using TFLite person detection models.
Designed for deployment on Raspberry Pi 4B (8GB).

## Models

| Model | Input Size | Framework | Size |
|-------|-----------|-----------|------|
| EfficientDet-Lite0 | 320x320 | TensorFlow Lite | ~13MB |
| SSD MobileNet V2 | 300x300 | TensorFlow Lite | ~18MB |

Both models are pretrained on COCO (person class) — no training required.

## Setup

### 1. Clone & Create Virtual Environment

```bash
git clone https://github.com/claydearnaiz/Thesis-Model.git
cd Thesis-Model
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Raspberry Pi
```

### 2. Install Dependencies

**PC (development):**
```bash
pip install -r requirements.txt
```

**Raspberry Pi (deployment):**
```bash
pip install opencv-python-headless numpy tflite-runtime matplotlib flask
```

### 3. Download Model Weights

```bash
python download_weights.py all
```

### 4. Calibrate ROI Zones

```bash
python src/calibrate_roi.py
```

**Controls:**
- **Left Click** — Place polygon vertex
- **R** — Finish current ROI
- **U** — Undo last vertex
- **C** — Clear all ROIs
- **S** — Save and quit
- **Q / ESC** — Quit without saving

### 5. Run the Monitor

```bash
python src/main.py
```

**Options:**

| Flag | Description |
|------|-------------|
| `--model NAME` | `efficientdet-lite0` (default) or `mobilenet-ssd` |
| `--source N` | Camera index (0, 1) or path to video file |
| `--no-display` | Headless mode (no GUI) |
| `--log` | Enable CSV logging to `logs/` |

### 6. Benchmark Models

```bash
python src/benchmark.py --frames 200 --save
```

Outputs comparison table, CSV, and chart to `benchmark_results/`.

## Project Structure

```
Thesis Model/
├── config/
│   └── roi_config.json         # ROI definitions + camera settings
├── logs/                        # Detection CSV logs
├── models/
│   ├── __init__.py              # Model registry
│   ├── base.py                  # TFLite base detector interface
│   ├── efficientdet_lite0.py    # EfficientDet-Lite0
│   └── mobilenet_ssd.py         # SSD MobileNet V2
├── weights/                     # TFLite model files (downloaded)
├── benchmark_results/           # Benchmark outputs
├── src/
│   ├── benchmark.py             # Model comparison tool
│   ├── calibrate_roi.py         # Visual ROI calibration
│   ├── main.py                  # Main monitoring entry point
│   └── roi_manager.py           # ROI loading, saving, hit-testing
├── download_weights.py
├── requirements.txt
└── README.md
```

## How It Works

1. Camera captures frames continuously
2. TFLite model detects all persons in each frame (crowd count)
3. Center point of each bounding box is tested against ROI polygons
4. Persons inside ROI zones are counted as estimated buyers
5. Results displayed on-screen and optionally logged to CSV
