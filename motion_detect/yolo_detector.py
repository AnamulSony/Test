"""
YOLO-based object detector using ultralytics YOLOv8.

Pipeline role:
  1. YOLODetector.detect(frame)          → all objects YOLO sees this frame
  2. YOLODetector.filter_by_motion(dets, mask) → keep only moving ones
  3. KalmanTracker.update(filtered_dets) → stable IDs across frames

Why combine YOLO + motion mask:
  - YOLO eliminates false positives from noise (it knows what real objects look like)
  - Motion mask eliminates stationary objects YOLO detects (background furniture, etc.)
  - Together: only real objects that are actually moving get reported
"""
from typing import List, Optional

import cv2
import numpy as np

from motion_detect.diff_engine import Detection

# Fraction of a YOLO bounding box that must overlap the motion mask
# for the object to be considered "moving"
MOTION_OVERLAP_THRESH = 0.20


class YOLODetector:
    """
    Wraps ultralytics YOLO for per-frame object detection.

    Parameters
    ----------
    model_name : str
        YOLO model weights — 'yolov8n.pt' (fastest) … 'yolov8x.pt' (most accurate).
        Downloaded automatically on first use if not cached.
    conf : float
        Minimum detection confidence to keep a box.
    classes : list[int] | None
        YOLO class IDs to detect (None = all 80 COCO classes).
        Common IDs: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck, 15=cat, 16=dog.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf:        float = 0.35,
        classes:     Optional[List[int]] = None,
    ) -> None:
        from ultralytics import YOLO  # deferred import — allows syntax-check without ultralytics
        self._model   = YOLO(model_name)
        self._conf    = conf
        self._classes = classes

    # ── public API ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO on one BGR frame.
        Returns a Detection for every box above the confidence threshold.
        """
        results = self._model(
            frame,
            conf=self._conf,
            classes=self._classes,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x = max(0, x1);  y = max(0, y1)
            w = max(1, x2 - x)
            h = max(1, y2 - y)
            cls_id = int(box.cls[0])
            label  = results.names[cls_id]
            conf   = float(box.conf[0])
            detections.append(Detection(x=x, y=y, w=w, h=h, area=w * h,
                                        label=label, conf=conf))
        return detections

    def filter_by_motion(
        self,
        detections:    List[Detection],
        motion_mask:   Optional[np.ndarray],
        overlap_thresh: float = MOTION_OVERLAP_THRESH,
    ) -> List[Detection]:
        """
        Discard YOLO detections whose bounding box has less than overlap_thresh
        fraction of pixels set in the motion mask.

        If motion_mask is None (e.g., not enough frames yet), all detections pass.
        """
        if motion_mask is None or not detections:
            return detections

        mh, mw = motion_mask.shape[:2]
        moving: List[Detection] = []

        for det in detections:
            # Clamp box to frame bounds
            x1 = max(0, det.x);           y1 = max(0, det.y)
            x2 = min(mw, det.x + det.w);  y2 = min(mh, det.y + det.h)
            roi = motion_mask[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            if np.count_nonzero(roi) / roi.size >= overlap_thresh:
                moving.append(det)

        return moving
