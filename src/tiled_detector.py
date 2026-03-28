"""
Tiled detection for wide-angle / high-resolution cameras.

Splits the full frame into overlapping tiles, processes one tile per call
(round-robin), and merges cached results from all tiles with cross-tile NMS.

Grid auto-selection:
  - <= 800px wide  : 1x1 (no tiling, passthrough)
  - <= 1400px wide : 2x2 (4 tiles, full scan every ~200ms @ 20fps)
  - > 1400px wide  : 3x2 (6 tiles, full scan every ~300ms @ 20fps)
"""

import cv2
import numpy as np


def build_tile_grid(frame_w: int, frame_h: int) -> list[tuple]:
    """Return list of (x1, y1, x2, y2) tile regions with ~20% overlap."""
    if frame_w <= 800:
        return [(0, 0, frame_w, frame_h)]

    if frame_w <= 1400:
        cols, rows = 2, 2
    else:
        cols, rows = 3, 2

    tile_w = min(int(frame_w / cols * 1.2), frame_w)
    tile_h = min(int(frame_h / rows * 1.2), frame_h)

    stride_x = (frame_w - tile_w) / max(1, cols - 1) if cols > 1 else 0
    stride_y = (frame_h - tile_h) / max(1, rows - 1) if rows > 1 else 0

    tiles = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(c * stride_x)
            y1 = int(r * stride_y)
            x2 = min(x1 + tile_w, frame_w)
            y2 = min(y1 + tile_h, frame_h)
            tiles.append((x1, y1, x2, y2))
    return tiles


class TiledDetector:
    """
    Wraps any BaseDetector with round-robin tiled inference.

    Each detect() call processes ONE tile and returns the merged result
    from all tiles (using cached detections for tiles not processed this frame).
    """

    def __init__(self, detector, frame_w: int, frame_h: int):
        self.detector = detector
        self.name = f"{detector.name} (tiled)"
        self.confidence = detector.confidence
        self.tiles = build_tile_grid(frame_w, frame_h)
        self.tile_idx = 0
        self._cache = [[] for _ in self.tiles]

        grid_cols = {1: "1x1", 4: "2x2", 6: "3x2"}.get(len(self.tiles), f"{len(self.tiles)}")
        print(f"  [Tiling] {grid_cols} grid = {len(self.tiles)} tiles "
              f"| tile size ~{self.tiles[0][2]-self.tiles[0][0]}x{self.tiles[0][3]-self.tiles[0][1]}px "
              f"| full scan every {len(self.tiles)} frames")

    def detect(self, frame: np.ndarray) -> list[dict]:
        tx1, ty1, tx2, ty2 = self.tiles[self.tile_idx]
        tile_crop = frame[ty1:ty2, tx1:tx2]

        raw_dets = self.detector.detect(tile_crop)

        mapped = []
        for d in raw_dets:
            bx1, by1, bx2, by2 = d["bbox"]
            mapped.append({
                "bbox": (bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1),
                "center": (d["center"][0] + tx1, d["center"][1] + ty1),
                "confidence": d["confidence"],
            })

        self._cache[self.tile_idx] = mapped
        self.tile_idx = (self.tile_idx + 1) % len(self.tiles)

        all_dets = []
        for dets in self._cache:
            all_dets.extend(dets)

        if len(all_dets) <= 1:
            return all_dets

        boxes_xywh = [(d["bbox"][0], d["bbox"][1],
                        d["bbox"][2] - d["bbox"][0],
                        d["bbox"][3] - d["bbox"][1]) for d in all_dets]
        scores = [d["confidence"] for d in all_dets]
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.confidence, 0.45)

        if len(indices) == 0:
            return []
        return [all_dets[i] for i in indices.flatten()]
