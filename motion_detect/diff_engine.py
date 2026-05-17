"""
Three-Frame Differencing motion detector.

Formula:  M(t) = |F(t) − F(t-1)| ∩ |F(t+1) − F(t)|

Why this formula works:
  A moving object at position P in frame t was at P-Δ in t-1 and moves to P+Δ
  in t+1.  Both abs-diffs fire at P (the current location).  Static background
  regions and slow lighting drift cancel out in the intersection.

Camera motion is compensated before differencing via ORB homography alignment:
  prev and next are warped into curr's coordinate space so that background pixels
  subtract to near-zero even when the camera pans or shakes.

Additional false-positive rejection:
  - Max blob area  (>25 % of frame → background change, not an object)
  - Max aspect ratio (>6:1 → motion-blur streak or camera-swing edge)
  - Convex-hull solidity (<0.20 → fragmented arc/ring, not a solid object)
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Tuning ─────────────────────────────────────────────────────────────────────
DIFF_THRESH    = 25     # abs-diff threshold (0–255); same as reference implementation
MIN_BLOB_AREA  = 500    # px²
MAX_BLOB_RATIO = 0.25   # blobs covering >25 % of frame are background, not objects
MAX_ASPECT     = 6.0    # long-side / short-side ratio limit
BLUR_K         = 5      # Gaussian pre-blur kernel (kills JPEG 8×8 block artifacts)

OPEN_K   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
CLOSE_K  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
DILATE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,  7))

# ORB stabilisation
_ORB_N_FEAT    = 500
_MIN_INLIERS   = 10
_RANSAC_THRESH = 3.0


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    area: int
    label: str  = ""    # class name from YOLO; empty for diff-based detections
    conf:  float = 0.0  # YOLO confidence score; 0 for diff-based detections


# ── Camera alignment ───────────────────────────────────────────────────────────

def _align_to_ref(
    ref: np.ndarray,
    other: np.ndarray,
) -> np.ndarray:
    """
    Warp `other` into `ref`'s coordinate space using ORB + RANSAC homography.
    Falls back to returning `other` unchanged if alignment fails (< _MIN_INLIERS
    inliers, featureless sky, etc.).
    """
    h, w = ref.shape[:2]
    ref_g   = cv2.cvtColor(ref,   cv2.COLOR_BGR2GRAY)
    other_g = cv2.cvtColor(other, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=_ORB_N_FEAT)
    kp_r, des_r = orb.detectAndCompute(ref_g,   None)
    kp_o, des_o = orb.detectAndCompute(other_g, None)

    if des_r is None or des_o is None or len(des_r) < 4 or len(des_o) < 4:
        return other

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_r, des_o)
    if len(matches) < _MIN_INLIERS:
        return other

    matches = sorted(matches, key=lambda m: m.distance)[:100]
    src_pts = np.float32([kp_r[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp_o[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, _RANSAC_THRESH)
    if H is None or mask is None or int(mask.ravel().sum()) < _MIN_INLIERS:
        return other

    return cv2.warpPerspective(
        other, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ── Preprocessing ──────────────────────────────────────────────────────────────

def _preprocess(frame: np.ndarray) -> np.ndarray:
    """BGR → grayscale → Gaussian blur. Blur removes JPEG noise before thresholding."""
    return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (BLUR_K, BLUR_K), 0)


# ── Three-frame differencing ───────────────────────────────────────────────────

def _diff_mask(a: np.ndarray, b: np.ndarray, thresh: int) -> np.ndarray:
    """Binary mask: |a − b| ≥ thresh.  Both inputs are preprocessed (gray, blurred)."""
    diff = cv2.absdiff(a, b)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    # Dilate before intersection to bridge sub-pixel misalignment gaps
    return cv2.dilate(mask, DILATE_K)


# ── Morphological cleanup ──────────────────────────────────────────────────────

def _clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  OPEN_K)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_K)
    return _remove_small_blobs(mask, min_area)


def _remove_small_blobs(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


# ── Contour → Detection extraction ────────────────────────────────────────────

def extract_detections(mask: np.ndarray) -> List[Detection]:
    """
    Convert cleaned binary mask to Detection list.

    Rejection filters (in order):
      1. Area < MIN_BLOB_AREA              — noise specks
      2. Area > MAX_BLOB_RATIO × frame     — background / lighting change
      3. Bounding-box aspect ratio > MAX_ASPECT — camera-swing streaks
      4. Convex-hull solidity < 0.20       — fragmented arcs / reflections
    """
    fh, fw = mask.shape[:2]
    max_area = MAX_BLOB_RATIO * fh * fw

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Detection] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BLOB_AREA or area > max_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        # Also reject by bounding-box area (contour can be sparse inside a huge bbox)
        if bw * bh > max_area:
            continue
        if max(bw, bh) / max(min(bw, bh), 1) > MAX_ASPECT:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area > 0 and (area / hull_area) < 0.20:
            continue

        detections.append(Detection(x=bx, y=by, w=bw, h=bh, area=int(area)))

    return detections


# ── Engine ─────────────────────────────────────────────────────────────────────

class DiffEngine:
    """
    Three-frame differencing engine.
    All frames passed as raw BGR numpy arrays.
    """

    def __init__(
        self,
        diff_thresh:   int  = DIFF_THRESH,
        min_blob_area: int  = MIN_BLOB_AREA,
        use_stabilize: bool = True,
    ) -> None:
        self.diff_thresh   = diff_thresh
        self.min_blob_area = min_blob_area
        self.use_stabilize = use_stabilize

    def process_triple(
        self,
        prev: np.ndarray,
        curr: np.ndarray,
        nxt:  np.ndarray,
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        M(t) = |F(t) − F(t-1)| ∩ |F(t+1) − F(t)|

        Both neighbors are aligned to curr's coordinate space before differencing
        so that background pixels subtract to near-zero despite camera motion.
        Each mask is dilated before the AND to handle sub-pixel alignment residuals.
        """
        if self.use_stabilize:
            prev_aligned = _align_to_ref(curr, prev)
            next_aligned = _align_to_ref(curr, nxt)
        else:
            prev_aligned = prev
            next_aligned = nxt

        curr_g = _preprocess(curr)
        m1 = _diff_mask(_preprocess(prev_aligned), curr_g, self.diff_thresh)
        m2 = _diff_mask(curr_g, _preprocess(next_aligned), self.diff_thresh)

        motion = cv2.bitwise_and(m1, m2)
        motion = _clean_mask(motion, self.min_blob_area)

        return motion, extract_detections(motion)

    def process_pair(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> Tuple[np.ndarray, List[Detection]]:
        """Single pairwise diff — edge frames only (first and last)."""
        a_g = _preprocess(a)
        b_g = _preprocess(b)
        mask = _clean_mask(_diff_mask(a_g, b_g, self.diff_thresh), self.min_blob_area)
        return mask, extract_detections(mask)
