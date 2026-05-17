"""
Visualisation and output writing.

draw_tracks()  — overlays bounding boxes and ID labels on a frame copy.
OutputWriter   — context manager that writes annotated JPEGs, CSV, and JSON.
"""
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

_BOX_COLOR  = (0, 255, 255)      # yellow (BGR)
_TEXT_COLOR = (0, 0, 0)          # black text on yellow background
_BOX_THICK  = 2
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICK = 2
_TEXT_BG    = (0, 255, 255)      # yellow label background


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Tuple[int, int, int, int, int]],  # (id, x, y, w, h)
    frame_idx: int,
    label_map: Optional[dict] = None,  # {track_id: class_label_string}
) -> np.ndarray:
    """
    Returns a copy of frame with yellow bounding boxes, ID + class labels,
    a center dot, and a frame-index overlay.

    label_map: optional dict mapping track_id → YOLO class name (e.g. 'person').
    """
    out = frame.copy()

    for (tid, x, y, w, h) in tracks:
        cv2.rectangle(out, (x, y), (x + w, y + h), _BOX_COLOR, _BOX_THICK)

        # Build label: "person  ID:2" or just "ID:2"
        cls_name = (label_map or {}).get(tid, "")
        label = f"{cls_name}  ID:{tid}" if cls_name else f"ID:{tid}"

        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICK)
        lx = x
        ly = max(y - 4, th + baseline)
        cv2.rectangle(out, (lx, ly - th - baseline), (lx + tw, ly), _TEXT_BG, -1)
        cv2.putText(out, label, (lx, ly - baseline // 2),
                    _FONT, _FONT_SCALE, _TEXT_COLOR, _FONT_THICK, cv2.LINE_AA)

        # Center dot
        cv2.circle(out, (x + w // 2, y + h // 2), 4, _BOX_COLOR, -1)

    frame_label = f"Frame {frame_idx:06d}  objects:{len(tracks)}"
    cv2.putText(out, frame_label, (8, 22),
                _FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, frame_label, (8, 22),
                _FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return out


class OutputWriter:
    """
    Writes annotated frames, a streaming CSV, and a final JSON summary.

    Output layout:
        output_dir/
        ├── frames/frame_000000.jpg
        └── detections.csv
        └── detections.json   (written on close)
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        source_name: str = "",
        total_frames: int = 0,
        jpeg_quality: int = 90,
    ) -> None:
        self._root        = Path(output_dir)
        self._frames_dir  = self._root / "frames"
        self._jpeg_q      = jpeg_quality
        self._source_name = source_name
        self._total       = total_frames
        self._json_frames: list = []

        self._root.mkdir(parents=True, exist_ok=True)
        self._frames_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self._root / "detections.csv"
        self._csv_fh   = self._csv_path.open("w", newline="", encoding="utf-8")
        self._csv_w    = csv.writer(self._csv_fh)
        self._csv_w.writerow(["frame_idx", "track_id", "x", "y", "w", "h"])

    # ── public API ─────────────────────────────────────────────────────────────

    def write_frame(
        self,
        frame_idx: int,
        annotated_frame: Optional[np.ndarray],
        tracks: List[Tuple[int, int, int, int, int]],
    ) -> None:
        """
        Save annotated JPEG and append detection rows to CSV + in-memory JSON list.
        annotated_frame may be None when --no-annotate is active.
        """
        if annotated_frame is not None:
            out_path = self._frames_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), annotated_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_q])

        dets_json = []
        for (tid, x, y, w, h) in tracks:
            self._csv_w.writerow([frame_idx, tid, x, y, w, h])
            dets_json.append({"track_id": tid, "x": x, "y": y, "w": w, "h": h})

        if tracks:
            self._json_frames.append({"frame_idx": frame_idx, "detections": dets_json})

    def close(self) -> None:
        """Flush and close CSV; write JSON summary."""
        self._csv_fh.close()

        json_path = self._root / "detections.json"
        payload = {
            "source":       self._source_name,
            "total_frames": self._total,
            "frames":       self._json_frames,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def __enter__(self) -> "OutputWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
