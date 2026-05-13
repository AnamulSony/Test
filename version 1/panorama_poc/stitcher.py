"""
Step 8 — Cylindrical warping of each selected frame.
Step 9 — Multi-band (Laplacian pyramid) blending left-to-right.

Does NOT use cv2.Stitcher — frames are already selected by the pipeline.
"""
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

_PYRAMID_LEVELS = 5


# ── Step 8: Warping ───────────────────────────────────────────────────────────

def warp_frames(frames: list[np.ndarray], f_px: float) -> list[np.ndarray]:
    """Step 8 — Cylindrical warp then equalise gains across frames."""
    if not frames:
        raise ValueError("warp_frames: empty frame list")
    log.info("Step 8 — Warping %d frames (cylindrical, f_px=%.1f)", len(frames), f_px)
    warped = [_cylindrical_warp(f, f_px) for f in frames]
    warped = _gain_compensate_frames(warped)
    log.info("  Warping complete.")
    return warped


def _cylindrical_warp(img: np.ndarray, f: float) -> np.ndarray:
    h, w = img.shape[:2]
    y_i, x_i = np.indices((h, w), dtype=np.float32)
    theta  = (x_i - w / 2) / f
    phi    = (y_i - h / 2) / f
    x_flat = np.tan(theta) * f + w / 2
    y_flat = (phi / np.cos(theta)) * f + h / 2
    return cv2.remap(img, x_flat, y_flat,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _gain_compensate_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Normalise mean brightness across frames to suppress exposure seams."""
    means  = [float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()) for f in frames]
    target = float(np.mean(means))
    result = []
    for f, m in zip(frames, means):
        gain = float(np.clip(target / m if m > 0 else 1.0, 0.5, 2.0))
        result.append(np.clip(f.astype(np.float64) * gain, 0, 255).astype(np.uint8))
    return result


# ── Step 9: Blending ──────────────────────────────────────────────────────────

def blend_frames(warped: list[np.ndarray], overlap_px: int) -> np.ndarray:
    """Step 9 — Multi-band blend warped frames left-to-right."""
    if not warped:
        raise ValueError("blend_frames: empty frame list")
    log.info("Step 9 — Blending %d warped frames (overlap=%dpx)", len(warped), overlap_px)
    panorama = warped[0]
    for idx, nxt in enumerate(warped[1:], start=1):
        panorama = _multiband_feather(panorama, nxt, overlap_px)
        log.debug("  blended frame %d/%d", idx, len(warped) - 1)
    log.info("  Blending complete. Panorama size: %dx%d",
             panorama.shape[1], panorama.shape[0])
    return panorama


def _multiband_feather(left: np.ndarray, right: np.ndarray,
                       overlap_px: int) -> np.ndarray:
    """Laplacian-pyramid blend over the overlap zone; direct copy outside it."""
    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]
    assert h_l == h_r, "height mismatch in _multiband_feather"

    overlap_px = min(overlap_px, w_l, w_r)
    out_w = w_l + w_r - overlap_px
    out   = np.zeros((h_l, out_w, 3), dtype=np.float32)

    out[:, :w_l - overlap_px]  = left[:, :w_l - overlap_px].astype(np.float32)
    out[:, w_l:]               = right[:, overlap_px:].astype(np.float32)

    lf = left[:, w_l - overlap_px:].astype(np.float32)
    rf = right[:, :overlap_px].astype(np.float32)

    alpha = np.linspace(1.0, 0.0, overlap_px, dtype=np.float32)
    alpha = np.tile(alpha[None, :], (h_l, 1))

    lvls = min(_PYRAMID_LEVELS, max(1, int(np.log2(min(h_l, overlap_px) + 1)) - 1))
    out[:, w_l - overlap_px: w_l] = _lap_blend_pair(lf, rf, alpha, lvls)
    return np.clip(out, 0, 255).astype(np.uint8)


def _lap_blend_pair(img1: np.ndarray, img2: np.ndarray,
                    alpha: np.ndarray, levels: int) -> np.ndarray:
    """Two-image Laplacian pyramid blend. img1/img2 float32 (H,W,3), alpha float32 (H,W)."""
    gp1, gp2, ga = [img1], [img2], [alpha]
    for _ in range(levels):
        gp1.append(cv2.pyrDown(gp1[-1]))
        gp2.append(cv2.pyrDown(gp2[-1]))
        ga.append(cv2.pyrDown(ga[-1]))
    lp1: list = [gp1[levels]]
    lp2: list = [gp2[levels]]
    for i in range(levels - 1, -1, -1):
        h, w = gp1[i].shape[:2]
        lp1.insert(0, gp1[i] - cv2.pyrUp(gp1[i + 1], dstsize=(w, h)))
        lp2.insert(0, gp2[i] - cv2.pyrUp(gp2[i + 1], dstsize=(w, h)))
    blended = []
    for i in range(levels + 1):
        a = ga[i][:, :, None]
        blended.append(a * lp1[i] + (1.0 - a) * lp2[i])
    result = blended[levels]
    for i in range(levels - 1, -1, -1):
        h, w = blended[i].shape[:2]
        result = cv2.pyrUp(result, dstsize=(w, h)) + blended[i]
    return result
