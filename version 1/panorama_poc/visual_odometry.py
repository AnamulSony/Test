"""
Visual odometry: SIFT feature matching between consecutive frames.

Returns:
  - cumulative x/y pixel offsets (for frame selection)
  - per-frame affine matrices M[i]: maps frame i-1 → frame i (for motion detection)
"""
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

_MAX_FEATURES  = 2000
_RATIO_THRESH  = 0.75
_RANSAC_THRESH = 3.0
_IDENTITY      = np.eye(2, 3, dtype=np.float32)


def _sift() -> cv2.SIFT:
    return cv2.SIFT_create(nfeatures=_MAX_FEATURES)


def _match_affine(
    img_a: np.ndarray,
    img_b: np.ndarray,
    detector: cv2.SIFT,
) -> tuple[np.ndarray, float, float]:
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    kp_a, des_a = detector.detectAndCompute(gray_a, None)
    kp_b, des_b = detector.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return _IDENTITY.copy(), 0.0, 0.0

    raw  = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
    good = [m for m, n in raw if m.distance < _RATIO_THRESH * n.distance]

    if len(good) < 8:
        return _IDENTITY.copy(), 0.0, 0.0

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

    M, inliers = cv2.estimateAffinePartial2D(
        pts_a, pts_b,
        method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_THRESH,
    )
    if M is None or inliers is None or int(inliers.sum()) < 8:
        return _IDENTITY.copy(), 0.0, 0.0

    tx = float(M[0, 2])
    ty = float(M[1, 2])
    log.debug("VO: tx=%.1fpx  ty=%.1fpx  inliers=%d", tx, ty, int(inliers.sum()))
    return M.astype(np.float32), tx, ty


def compute_vo_offsets(
    frame_paths: list[str],
) -> tuple[list[float], list[float], list[np.ndarray]]:
    """
    Compute VO data for all frames in one pass.

    Returns:
      x_offsets   : cumulative horizontal shift per frame (px, positive = right)
      y_offsets   : cumulative vertical shift (px)
      affine_mats : affine_mats[i] maps frame i-1 → frame i  (affine_mats[0] = identity)
    """
    n = len(frame_paths)
    x_offsets   = [0.0] * n
    y_offsets   = [0.0] * n
    affine_mats: list[np.ndarray] = [_IDENTITY.copy()] * n

    detector = _sift()
    prev = cv2.imread(frame_paths[0])
    if prev is None:
        raise ValueError(f"Cannot read frame: {frame_paths[0]}")

    log.info("Computing visual odometry for %d frames …", n)

    for i in range(1, n):
        curr = cv2.imread(frame_paths[i])
        if curr is None:
            log.warning("VO: cannot read frame %d; using identity", i)
            x_offsets[i]   = x_offsets[i - 1]
            y_offsets[i]   = y_offsets[i - 1]
            affine_mats[i] = _IDENTITY.copy()
            continue

        M, tx, ty = _match_affine(prev, curr, detector)
        affine_mats[i] = M

        # positive offset = camera panned right
        x_offsets[i] = x_offsets[i - 1] - tx
        y_offsets[i] = y_offsets[i - 1] - ty
        prev = curr

        if i % 20 == 0:
            log.info("  VO frame %d/%d  cumulative_x=%.1fpx", i, n - 1, x_offsets[i])

    log.info("VO done. Total x-shift frame0→%d: %.1fpx", n - 1, x_offsets[-1])
    return x_offsets, y_offsets, affine_mats
