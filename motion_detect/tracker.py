"""
Kalman-filter multi-object tracker with greedy IoU association.

Each track owns a cv2.KalmanFilter instance with:
  State (6D):       [cx, cy, w, h, vx, vy]  — constant-velocity model
  Measurement (4D): [cx, cy, w, h]           — directly from detections

Update cycle per frame:
  1. Predict  — advance every active track one step forward
  2. Match    — IoU between predicted boxes and new detections (greedy)
  3. Correct  — feed matched detection into kf.correct()
  4. Age      — increment age on unmatched tracks; deactivate if > max_age
  5. Spawn    — create new tracks for unmatched detections
  6. Return   — confirmed tracks (hit_streak >= min_hits)

Advantage over pure-IoU: the Kalman prediction smooths box motion and lets the
tracker bridge 1-2 frames of missed detection (occlusion, false negative) without
the bounding box jumping.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from motion_detect.diff_engine import Detection


# ── Per-track Kalman filter ────────────────────────────────────────────────────

def _make_kalman() -> cv2.KalmanFilter:
    """
    6-state / 4-measurement constant-velocity filter.
    State:       [cx, cy, w, h, vx, vy]
    Measurement: [cx, cy, w, h]
    """
    kf = cv2.KalmanFilter(6, 4)

    # x(k+1) = F·x(k)  —  position += velocity, size constant
    kf.transitionMatrix = np.array([
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)

    # z(k) = H·x(k)  —  observe cx, cy, w, h
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ], dtype=np.float32)

    kf.processNoiseCov     = np.eye(6, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 5e-1
    kf.errorCovPost        = np.eye(6, dtype=np.float32)

    return kf


@dataclass
class KalmanTrack:
    id:         int
    kf:         cv2.KalmanFilter
    bbox:       Tuple[int, int, int, int]   # (x, y, w, h) — last corrected
    label:      str  = ""    # YOLO class name of the most recent matched detection
    age:        int  = 0
    hit_streak: int  = 0
    active:     bool = True

    @staticmethod
    def from_detection(track_id: int, det: Detection) -> "KalmanTrack":
        kf  = _make_kalman()
        cx  = det.x + det.w / 2
        cy  = det.y + det.h / 2
        kf.statePost = np.array(
            [cx, cy, det.w, det.h, 0, 0], dtype=np.float32
        ).reshape(6, 1)
        return KalmanTrack(
            id=track_id, kf=kf,
            bbox=(det.x, det.y, det.w, det.h),
            label=det.label,
            hit_streak=1,
        )

    def predict(self) -> Tuple[int, int, int, int]:
        """Advance the filter one step; return predicted (x, y, w, h)."""
        state = self.kf.predict()
        cx, cy, w, h = state[:4].flatten()
        w = max(w, 1)
        h = max(h, 1)
        return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))

    def correct(self, det: Detection) -> None:
        """Feed a matched detection into the filter."""
        cx = det.x + det.w / 2
        cy = det.y + det.h / 2
        meas = np.array([cx, cy, det.w, det.h], dtype=np.float32).reshape(4, 1)
        self.kf.correct(meas)
        self.bbox  = (det.x, det.y, det.w, det.h)
        if det.label:
            self.label = det.label


# ── Multi-object tracker ───────────────────────────────────────────────────────

class KalmanTracker:
    """
    Multi-object tracker: one KalmanFilter per track, greedy IoU association.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU (against predicted box) to match a detection to a track.
    max_age : int
        Deactivate a track after this many consecutive unmatched frames.
    min_hits : int
        Require this many consecutive matches before reporting a track publicly.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_age:       int   = 5,
        min_hits:      int   = 2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age       = max_age
        self.min_hits      = min_hits
        self._tracks:  Dict[int, KalmanTrack] = {}
        self._next_id: int = 1

    def update(
        self, detections: List[Detection]
    ) -> List[Tuple[int, int, int, int, int]]:
        """
        Process one frame of detections.
        Returns list of (track_id, x, y, w, h) for confirmed active tracks.
        """
        active = [t for t in self._tracks.values() if t.active]

        # Step 1 — predict every active track
        predicted: Dict[int, Tuple[int, int, int, int]] = {
            t.id: t.predict() for t in active
        }

        # Step 2 & 3 — greedy IoU match against predicted boxes
        matched_tids, matched_dis = self._match(active, predicted, detections)

        # Step 4a — update matched tracks
        for tid, di in zip(matched_tids, matched_dis):
            t = self._tracks[tid]
            t.correct(detections[di])
            t.age        = 0
            t.hit_streak += 1

        # Step 4b — age unmatched tracks
        matched_tid_set = set(matched_tids)
        for t in active:
            if t.id not in matched_tid_set:
                t.age        += 1
                t.hit_streak  = 0
                if t.age > self.max_age:
                    t.active = False

        # Step 5 — spawn new tracks for unmatched detections
        matched_di_set = set(matched_dis)
        for di, det in enumerate(detections):
            if di not in matched_di_set:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = KalmanTrack.from_detection(tid, det)

        # Prune dead tracks
        self._tracks = {tid: t for tid, t in self._tracks.items() if t.active}

        # Step 6 — return confirmed tracks
        return [
            (t.id, *t.bbox)
            for t in self._tracks.values()
            if t.active and t.hit_streak >= self.min_hits
        ]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def get_label(self, track_id: int) -> str:
        """Return the YOLO class label of a track (empty string if not known)."""
        t = self._tracks.get(track_id)
        return t.label if t else ""

    # ── internals ──────────────────────────────────────────────────────────────

    def _match(
        self,
        active:     List[KalmanTrack],
        predicted:  Dict[int, Tuple[int, int, int, int]],
        detections: List[Detection],
    ) -> Tuple[List[int], List[int]]:
        if not active or not detections:
            return [], []

        triples = []
        for t in active:
            pred_box = predicted[t.id]
            for di, det in enumerate(detections):
                iou = _iou(pred_box, (det.x, det.y, det.w, det.h))
                if iou >= self.iou_threshold:
                    triples.append((iou, t.id, di))

        triples.sort(key=lambda x: x[0], reverse=True)

        matched_tids: List[int] = []
        matched_dis:  List[int] = []
        used_tids: set = set()
        used_dis:  set = set()

        for iou, tid, di in triples:
            if tid in used_tids or di in used_dis:
                continue
            matched_tids.append(tid)
            matched_dis.append(di)
            used_tids.add(tid)
            used_dis.add(di)

        return matched_tids, matched_dis


def _iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1 = max(ax, bx);  iy1 = max(ay, by)
    ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


# ── Utility ────────────────────────────────────────────────────────────────────

def most_central_track(
    tracks: List[Tuple[int, int, int, int, int]],
    frame_w: int,
    frame_h: int,
) -> List[Tuple[int, int, int, int, int]]:
    """Return only the track whose bbox centre is closest to the frame centre."""
    if not tracks:
        return []
    cx, cy = frame_w / 2, frame_h / 2
    best = min(
        tracks,
        key=lambda t: (t[1] + t[3] / 2 - cx) ** 2 + (t[2] + t[4] / 2 - cy) ** 2,
    )
    return [best]
