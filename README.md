# Canteen Monitoring System

Real-time person detection within ROI zones using YOLOv5n.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Raspberry Pi
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Calibrate ROI Zones

Open your camera feed and draw ROI polygons interactively:

```bash
python src/calibrate_roi.py
```

**Controls:**
- **Left Click** — Place a polygon vertex
- **R** — Finish current ROI (names it via terminal prompt)
- **U** — Undo last vertex
- **C** — Clear all ROIs
- **S** — Save and quit
- **Q / ESC** — Quit without saving

ROIs are saved to `config/roi_config.json`.

### 4. Run the Monitor

```bash
python src/main.py
```

**Options:**

| Flag | Description |
|------|-------------|
| `--config PATH` | Custom config file path |
| `--source N` | Camera index (0, 1, etc.) |
| `--no-display` | Headless mode (no GUI window) |
| `--log` | Enable CSV logging to `logs/` |

Press **Q** to quit the monitor.

## Project Structure

```
Thesis Model/
├── config/
│   └── roi_config.json       # ROI definitions + camera settings
├── logs/                      # Detection CSV logs
├── src/
│   ├── calibrate_roi.py       # Visual ROI calibration tool
│   ├── detector.py            # YOLOv5n person detection wrapper
│   ├── main.py                # Main monitoring entry point
│   └── roi_manager.py         # ROI loading, saving, hit-testing
├── implementation.md
├── requirements.txt
└── README.md
```

## How It Works

1. Camera captures frames continuously
2. YOLOv5n detects all "person" class objects in each frame
3. Center point of each bounding box is computed
4. Each center point is tested against all defined ROI polygons
5. If a center falls inside an ROI, that person is counted for that zone
6. Results are displayed on-screen and optionally logged to CSV
