"""
GhostFreePanorama v1 — orchestrator following the flowchart:

  Start
    ↓
  [1] Frame Acquisition
    ↓
  [2] Select Frame Fᵢ  (i = 0)
    ↓
  ┌─[3] Find Moving Object in Fᵢ
  │   ↓
  │  Moving object?
  │   ├─ Yes → SKIP / IGNORE frame, advance i += 1
  │   └─ No  → [4] Select next frame with minimum overlapping region (Fₛ)
  │             ↓
  │            [6] Record Fₛ in selected list
  │             ↓
  │            [7] Update Fᵢ = Fₛ
  │   ↓
  │  Last frame? (i == n-1)
  └───── No ──────────────────────┘
         Yes ↓
  [8] Warping  (cylindrical projection of all selected frames)
         ↓
  [9] Blending (feather blend left-to-right)
         ↓
  End

Frame positions are computed from visual odometry only — no IMU used.
"""
import csv
import json
import logging
import os

import cv2
import numpy as np

from panorama_poc.calibration import StereoRectifier
from panorama_poc.depth import DepthEstimator
from panorama_poc.frame_selector import compute_frame_position, select_minimum_overlap_frame
from panorama_poc.io_loader import load_capture, read_frame
from panorama_poc.motion_detector import MotionDetector, get_object_leading_edge, _WARMUP_FRAMES
from panorama_poc.stitcher import warp_frames, blend_frames
from panorama_poc.visual_odometry import compute_vo_offsets

log = logging.getLogger(__name__)

_BLEND_OVERLAP_RATIO = 0.15


def run_pipeline(capture_dir: str, output_dir: str) -> None:
    """Execute the ghost-free panorama flowchart end-to-end."""
    np.random.seed(0)

    debug_dir = os.path.join(output_dir, "debug")
    masks_dir = os.path.join(debug_dir, "moving_masks")
    for d in (output_dir, debug_dir, masks_dir):
        os.makedirs(d, exist_ok=True)

    # ── [1] Frame Acquisition ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("[1] Frame Acquisition")
    data    = load_capture(capture_dir)
    calib   = data.calib
    n       = len(data.primary_paths)
    f_px    = calib.primary.fx
    frame_w = calib.primary.width

    rectifier = StereoRectifier(calib)
    DepthEstimator(calib, rectifier)
    detector  = MotionDetector()

    log.info("  %d frames  f_px=%.1f", n, f_px)

    log.info("  Rectifying frames for visual odometry …")
    rect_paths = _rectify_to_tmp(data.primary_paths, rectifier, output_dir)
    log.info("  Computing visual odometry …")
    vo_x, vo_y, vo_affines = compute_vo_offsets(rect_paths)
    log.info("  VO total x-shift: %.1fpx over %d frames", vo_x[-1], n - 1)

    _warmup_detector(detector, data.primary_paths, rectifier, vo_affines)

    # ── [2] Select Frame Fᵢ ──────────────────────────────────────────────────
    log.info("[2] Select Frame Fᵢ — starting at frame 0")
    i = 0
    selected_indices: list[int] = [0]
    positions_log: list[dict]   = []

    # ── Main loop: steps 3 → 7 → last-frame check → repeat ──────────────────
    while True:

        # ── [3] Find Moving Object ────────────────────────────────────────────
        frame_i    = rectifier.rectify_primary(read_frame(data.primary_paths[i]))
        next_i     = min(i + 1, n - 1)
        frame_next = rectifier.rectify_primary(read_frame(data.primary_paths[next_i]))
        affine_fwd = vo_affines[next_i]
        mask       = detector.detect_between(frame_i, frame_next, affine_fwd)
        cv2.imwrite(os.path.join(masks_dir, f"{i:06d}.png"), mask)

        has_mover = get_object_leading_edge(mask) is not None

        log.info("[3] Frame %d — moving object: %s", i, "YES" if has_mover else "NO")

        if has_mover:
            # ── SKIP frame with moving object ─────────────────────────────────
            log.info("[SKIP] Frame %d has moving object — ignoring, advancing to %d",
                     i, i + 1)
            i += 1
            if i >= n - 1:
                log.info("Last frame reached after skip (i=%d). Proceeding to warp & blend.", i)
                break
            continue

        # ── [4] Select next frame with minimum overlapping region ─────────────
        log.info("[4] No mover — selecting frame with minimum overlap")
        s_idx = select_minimum_overlap_frame(
            ref_idx=i,
            n_frames=n,
            frame_width=frame_w,
            vo_offsets=vo_x,
            min_overlap=_BLEND_OVERLAP_RATIO,
        )
        P_sel = compute_frame_position(i, s_idx, vo_x)
        positions_log.append({
            "frame_idx":    s_idx,
            "ref_frame":    i,
            "pixel_offset": round(P_sel, 2),
        })
        log.info("[4] → Fₛ = frame %d  (P=%.1fpx)", s_idx, P_sel)

        # ── [6] Record Fₛ ────────────────────────────────────────────────────
        selected_indices.append(s_idx)
        log.info("[6] Selected frames so far: %s", selected_indices)

        # ── [7] Update Fᵢ = Fₛ ───────────────────────────────────────────────
        log.info("[7] Fᵢ = %d", s_idx)
        if s_idx - i > _WARMUP_FRAMES:
            detector.reset()
            _warmup_detector(detector, data.primary_paths, rectifier, vo_affines,
                             start=max(0, s_idx - _WARMUP_FRAMES), end=s_idx)
        i = s_idx

        # ── Last frame decision ───────────────────────────────────────────────
        if i >= n - 1:
            log.info("Last frame reached (i=%d). Proceeding to warp & blend.", i)
            break

    log.info("Frame selection complete. Selected %d frames: %s",
             len(selected_indices), selected_indices)

    # ── [8] Warping ───────────────────────────────────────────────────────────
    log.info("[8] Warping selected frames …")
    selected_bgr = [
        rectifier.rectify_primary(read_frame(data.primary_paths[idx]))
        for idx in selected_indices
    ]
    warped = warp_frames(selected_bgr, f_px=f_px)

    # ── [9] Blending ──────────────────────────────────────────────────────────
    log.info("[9] Blending warped frames …")
    overlap_px = max(1, int(frame_w * _BLEND_OVERLAP_RATIO))
    panorama   = blend_frames(warped, overlap_px=overlap_px)

    _save_outputs(panorama, selected_indices, positions_log,
                  vo_x, vo_y, selected_bgr, output_dir, debug_dir)

    log.info("=" * 60)
    log.info("Pipeline complete.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _warmup_detector(
    detector: MotionDetector,
    paths: list[str],
    rectifier: StereoRectifier,
    vo_affines: list,
    start: int = 0,
    end: int | None = None,
) -> None:
    if end is None:
        end = min(start + _WARMUP_FRAMES, len(paths))
    for j in range(start, min(end, len(paths))):
        frame  = rectifier.rectify_primary(read_frame(paths[j]))
        affine = vo_affines[j] if j > 0 else None
        detector.detect(frame, affine_mat=affine)


def _rectify_to_tmp(
    paths: list[str],
    rectifier: StereoRectifier,
    output_dir: str,
) -> list[str]:
    rect_dir = os.path.join(output_dir, "rectified")
    os.makedirs(rect_dir, exist_ok=True)
    out_paths = []
    for p in paths:
        fname = os.path.basename(p)
        out   = os.path.join(rect_dir, fname)
        if not os.path.exists(out):
            img = cv2.imread(p)
            if img is not None:
                cv2.imwrite(out, rectifier.rectify_primary(img))
        out_paths.append(out)
    return out_paths


def _save_outputs(
    panorama: np.ndarray,
    selected_indices: list[int],
    positions_log: list[dict],
    vo_x: list[float],
    vo_y: list[float],
    selected_bgr: list[np.ndarray],
    output_dir: str,
    debug_dir: str,
) -> None:
    pano_path = os.path.join(output_dir, "panorama.png")
    cv2.imwrite(pano_path, panorama)
    log.info("Saved panorama → %s  (%dx%d)",
             pano_path, panorama.shape[1], panorama.shape[0])

    with open(os.path.join(output_dir, "selected_frames.json"), "w") as f:
        json.dump(selected_indices, f, indent=2)

    with open(os.path.join(debug_dir, "positions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_idx", "ref_frame", "pixel_offset"])
        w.writeheader()
        w.writerows(positions_log)

    with open(os.path.join(debug_dir, "vo_offsets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "x_offset_px", "y_offset_px"])
        for fi, (xo, yo) in enumerate(zip(vo_x, vo_y)):
            w.writerow([fi, f"{xo:.2f}", f"{yo:.2f}"])

    if selected_bgr:
        seam_canvas = selected_bgr[0].copy()
        h, sw = seam_canvas.shape[:2]
        cv2.line(seam_canvas, (sw - 1, 0), (sw - 1, h - 1), (0, 0, 255), 3)
        cv2.imwrite(os.path.join(debug_dir, "seams.png"), seam_canvas)
