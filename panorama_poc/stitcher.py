"""
Step 8 — Cylindrical warping of each selected frame.
Step 9 — Feather (linear alpha) blending left-to-right.

These are intentionally separate functions matching flowchart steps 8 and 9.
Does NOT use cv2.Stitcher — frames are already selected by the pipeline.
"""
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Step 8: Warping ───────────────────────────────────────────────────────────

def warp_frames(frames: list[np.ndarray], f_px: float) -> list[np.ndarray]:
    """
    Step 8 — Project each BGR frame onto a cylinder of focal length f_px (px).
    Returns a list of warped BGR images, same count as input.
    """
    if not frames:
        raise ValueError("warp_frames: empty frame list")
    log.info("Step 8 — Warping %d frames (cylindrical, f_px=%.1f)", len(frames), f_px)
    warped = [_cylindrical_warp(f, f_px) for f in frames]
    log.info("  Warping complete.")
    return warped


def _cylindrical_warp(img: np.ndarray, f: float) -> np.ndarray:
    """Project img onto a cylinder of focal length f (px)."""
    h, w = img.shape[:2]
    y_i, x_i = np.indices((h, w), dtype=np.float32)
    theta   = (x_i - w / 2) / f
    phi     = (y_i - h / 2) / f
    x_flat  = np.tan(theta) * f + w / 2
    y_flat  = (phi / np.cos(theta)) * f + h / 2
    return cv2.remap(img, x_flat, y_flat,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ── Step 9: Blending ──────────────────────────────────────────────────────────

def blend_frames(warped: list[np.ndarray], overlap_px: int) -> np.ndarray:
    """
    Step 9 — Feather-blend warped frames left-to-right.
    overlap_px: width of the linear-alpha transition zone in pixels.
    Returns the final panorama as a BGR uint8 image.
    """
    if not warped:
        raise ValueError("blend_frames: empty frame list")
    log.info("Step 9 — Blending %d warped frames (overlap=%dpx)", len(warped), overlap_px)
    panorama = warped[0]
    for idx, nxt in enumerate(warped[1:], start=1):
        panorama = _feather_blend(panorama, nxt, overlap_px)
        log.debug("  blended frame %d/%d", idx, len(warped) - 1)
    log.info("  Blending complete. Panorama size: %dx%d",
             panorama.shape[1], panorama.shape[0])
    return panorama


def _feather_blend(left: np.ndarray, right: np.ndarray,
                   overlap_px: int) -> np.ndarray:
    """Linear-alpha blend over the overlap strip."""
    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]
    assert h_l == h_r, "height mismatch in feather_blend"

    overlap_px = min(overlap_px, w_l, w_r)
    out_w = w_l + w_r - overlap_px
    out   = np.zeros((h_l, out_w, 3), dtype=np.float32)

    out[:, :w_l - overlap_px] = left[:, :w_l - overlap_px].astype(np.float32)

    alpha = np.linspace(1, 0, overlap_px, dtype=np.float32)[None, :, None]
    out[:, w_l - overlap_px: w_l] = (
        alpha * left[:, w_l - overlap_px:].astype(np.float32)
        + (1 - alpha) * right[:, :overlap_px].astype(np.float32)
    )
    out[:, w_l:] = right[:, overlap_px:].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
