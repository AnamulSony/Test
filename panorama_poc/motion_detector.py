"""
Moving-object detection using affine-compensated frame differencing.

Camera motion is removed by warping frame i-1 into frame i's coordinate system
using the VO-derived affine matrix (rotation + scale + translation).
This correctly handles panning, which is a rotation — not a pure translation.

MOG2 is kept as a fallback when no affine matrix is supplied.
"""
import logging
from typing import Optional

import cv2
import numpy as np

from panorama_poc.frame_selector import compute_pixel_displacement
from panorama_poc.imu_integrator import ImuSegment

log = logging.getLogger(__name__)

# ── tuning knobs ──────────────────────────────────────────────────────────────
_DIFF_THRESH    = 25       # abs-diff threshold (0-255) for motion pixels
_WARMUP_FRAMES  = 5        # frames fed silently to MOG2 after reset
_HISTORY        = 8
_VAR_THRESH     = 20.0
_DETECT_SHAD    = False
_OPEN_K  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_CLOSE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
_MIN_BLOB_AREA  = 600      # px²


class MotionDetector:
    """
    Step 3 detector — finds objects moving independently of the camera.

    Primary method: detect_between(frame_a, frame_b, M_a_to_b)
      Warps frame_a into frame_b's coordinate system using the VO affine matrix,
      then diffs. Works from frame 0 (no warm-up needed, no previous-frame state).

    Fallback: detect(frame, dx_px, dy_px)
      MOG2 subtraction with translation compensation (used when no affine is given).
    """

    def __init__(self, f_px: float, approx_depth: float):
        self._f_px         = f_px
        self._approx_depth = approx_depth
        self._subtractor   = self._new_mog2()
        self._warmup_left  = _WARMUP_FRAMES
        self._prev_frame: Optional[np.ndarray] = None

    # ── PRIMARY: look-ahead affine differencing ───────────────────────────────

    def detect_between(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        M_a_to_b: np.ndarray,
    ) -> np.ndarray:
        """
        Detect objects that moved independently between frame_a and frame_b.
        M_a_to_b: 2×3 affine that maps frame_a → frame_b (from VO).
        Returns binary uint8 mask (255 = independently moving region).
        No state needed — pure pair-wise comparison.
        """
        mask = _affine_diff(frame_a, frame_b, M_a_to_b, _DIFF_THRESH, _MIN_BLOB_AREA)
        log.debug("detect_between: nonzero_px=%d", int(np.count_nonzero(mask)))
        return mask

    # ── FALLBACK: MOG2 with translation compensation ──────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        dx_px: float = 0.0,
        dy_px: float = 0.0,
        affine_mat: Optional[np.ndarray] = None,
        imu_seg: Optional[ImuSegment] = None,
    ) -> np.ndarray:
        """Fallback path — used only when no affine matrix is available."""
        if affine_mat is not None and self._prev_frame is not None:
            mask = _affine_diff(self._prev_frame, frame, affine_mat,
                                _DIFF_THRESH, _MIN_BLOB_AREA)
            self._prev_frame = frame.copy()
            return mask

        tx, ty = self._resolve_shift(dx_px, dy_px, imu_seg)
        comp   = _translate(frame, tx, ty) if (tx or ty) else frame

        if self._warmup_left > 0:
            self._subtractor.apply(comp, learningRate=1.0)
            self._warmup_left -= 1
            self._prev_frame = frame.copy()
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        raw  = self._subtractor.apply(comp)
        raw  = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  _OPEN_K)
        raw  = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, _CLOSE_K)
        mask = _remove_small_blobs(raw, _MIN_BLOB_AREA)
        self._prev_frame = frame.copy()
        return mask

    def reset(self) -> None:
        """Reset MOG2 state when jumping to a new reference frame."""
        self._subtractor  = self._new_mog2()
        self._warmup_left = _WARMUP_FRAMES
        self._prev_frame  = None

    def _resolve_shift(self, dx_px, dy_px, imu_seg):
        if dx_px or dy_px:
            return dx_px, dy_px
        if imu_seg:
            tx = compute_pixel_displacement(self._f_px, abs(imu_seg.displacement_x),
                                            self._approx_depth)
            ty = compute_pixel_displacement(self._f_px, abs(imu_seg.displacement_y),
                                            self._approx_depth)
            return (tx if imu_seg.displacement_x >= 0 else -tx,
                    ty if imu_seg.displacement_y >= 0 else -ty)
        return 0.0, 0.0

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
    """
    Warp prev into curr's coordinate system using affine M (maps prev → curr).
    Compute abs-diff, threshold, and clean up blobs.
    """
    h, w = curr.shape[:2]
    aligned_prev = cv2.warpAffine(prev, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    diff = cv2.absdiff(curr, aligned_prev)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Ignore border pixels zeroed by warp padding.
    # Multiply by 255: bitwise_and needs 0/255, not 0/1.
    border_mask = (aligned_prev.any(axis=2).astype(np.uint8)) * 255

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, border_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _OPEN_K)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE_K)
    mask = _remove_small_blobs(mask, min_blob)
    return mask


def _translate(frame: np.ndarray, tx: float, ty: float) -> np.ndarray:
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


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
