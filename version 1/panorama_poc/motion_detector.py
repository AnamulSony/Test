"""
Moving-object detection using affine-compensated frame differencing.

Camera motion is removed by warping frame_a into frame_b's coordinate system
using the VO-derived affine matrix (rotation + scale + translation).
"""
import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_DIFF_THRESH   = 25
_WARMUP_FRAMES = 5
_HISTORY       = 8
_VAR_THRESH    = 20.0
_DETECT_SHAD   = False
_OPEN_K  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_CLOSE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
_MIN_BLOB_AREA = 600


class MotionDetector:
    """
    Step 3 detector — finds objects moving independently of the camera.

    Primary: detect_between(frame_a, frame_b, M_a_to_b)
      Warps frame_a into frame_b's coordinate system, then diffs.

    Fallback: detect(frame, affine_mat) — MOG2 with affine compensation.
    """

    def __init__(self):
        self._subtractor  = self._new_mog2()
        self._warmup_left = _WARMUP_FRAMES
        self._prev_frame: Optional[np.ndarray] = None

    def detect_between(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        M_a_to_b: np.ndarray,
    ) -> np.ndarray:
        """
        Detect independently moving objects between frame_a and frame_b.
        M_a_to_b: 2×3 affine that maps frame_a → frame_b (from VO).
        Returns binary uint8 mask (255 = moving region).
        """
        mask = _affine_diff(frame_a, frame_b, M_a_to_b, _DIFF_THRESH, _MIN_BLOB_AREA)
        log.debug("detect_between: nonzero_px=%d", int(np.count_nonzero(mask)))
        return mask

    def detect(
        self,
        frame: np.ndarray,
        affine_mat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fallback — MOG2 with optional affine compensation."""
        if affine_mat is not None and self._prev_frame is not None:
            mask = _affine_diff(self._prev_frame, frame, affine_mat,
                                _DIFF_THRESH, _MIN_BLOB_AREA)
            self._prev_frame = frame.copy()
            return mask

        if self._warmup_left > 0:
            self._subtractor.apply(frame, learningRate=1.0)
            self._warmup_left -= 1
            self._prev_frame = frame.copy()
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        raw  = self._subtractor.apply(frame)
        raw  = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  _OPEN_K)
        raw  = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, _CLOSE_K)
        mask = _remove_small_blobs(raw, _MIN_BLOB_AREA)
        self._prev_frame = frame.copy()
        return mask

    def reset(self) -> None:
        self._subtractor  = self._new_mog2()
        self._warmup_left = _WARMUP_FRAMES
        self._prev_frame  = None

    @staticmethod
    def _new_mog2():
        return cv2.createBackgroundSubtractorMOG2(
            history=_HISTORY, varThreshold=_VAR_THRESH,
            detectShadows=_DETECT_SHAD)


# ── helpers ───────────────────────────────────────────────────────────────────

def _affine_diff(
    prev: np.ndarray,
    curr: np.ndarray,
    M: np.ndarray,
    threshold: int,
    min_blob: int,
) -> np.ndarray:
    h, w = curr.shape[:2]
    aligned = cv2.warpAffine(prev, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    diff = cv2.absdiff(curr, aligned)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    border_mask = (aligned.any(axis=2).astype(np.uint8)) * 255
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, border_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _OPEN_K)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE_K)
    return _remove_small_blobs(mask, min_blob)


def _remove_small_blobs(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def get_object_leading_edge(mask: np.ndarray) -> Optional[int]:
    """Return leftmost x-coordinate of foreground region, or None if empty."""
    cols = np.where(mask.any(axis=0))[0]
    return int(cols[0]) if len(cols) > 0 else None
