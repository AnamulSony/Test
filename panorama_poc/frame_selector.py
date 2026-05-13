"""
Frame selection logic: ComputeFramePosition + FindNextOptimal.
Uses IMU-derived displacement and approximate/stereo depth.
"""
import logging
from typing import Optional

import numpy as np

from panorama_poc.imu_integrator import ImuRow, integrate_imu

log = logging.getLogger(__name__)


def compute_pixel_displacement(f_px: float, D: float, Z: float) -> float:
    """Equation [1]: ΔP = (f_px × D) / Z  — returns pixels (float)."""
    return (f_px * D) / Z


def compute_frame_position(
    imu_rows: list[ImuRow],
    ref_idx: int,
    target_idx: int,
    f_px: float,
    depth_z: float,
    vo_offsets: Optional[list[float]] = None,
) -> float:
    """
    Pixel offset of frame target_idx relative to frame ref_idx (positive = rightward).
    Primary: visual-odometry cumulative offsets when available (accurate for panning).
    Fallback: IMU linear-acceleration integration via Equation [1].
    """
    if vo_offsets is not None:
        px = abs(vo_offsets[target_idx] - vo_offsets[ref_idx])
        log.debug("frame %d→%d  VO ΔP=%.1fpx", ref_idx, target_idx, px)
        return px

    seg = integrate_imu(imu_rows, ref_idx, target_idx)
    D_horiz = abs(seg.displacement_x)
    px = compute_pixel_displacement(f_px=f_px, D=D_horiz, Z=depth_z)
    log.debug("frame %d→%d  IMU D_horiz=%.4fm  Z=%.2fm  ΔP=%.1fpx",
              ref_idx, target_idx, D_horiz, depth_z, px)
    return px


def find_next_optimal_frame(
    object_edge_px: float,
    frame_positions: list[tuple[int, float]],
) -> Optional[int]:
    """
    Return the frame index whose pixel position P is the largest value ≤ object_edge_px.
    frame_positions: list of (frame_idx, pixel_offset_from_ref), ordered ascending.
    Returns None if the list is empty; returns first frame_idx if all overshoot.
    """
    if not frame_positions:
        return None

    best_idx: Optional[int] = None
    for frame_idx, P in frame_positions:
        if P <= object_edge_px:
            best_idx = frame_idx
        else:
            break   # positions are ascending; no point continuing

    if best_idx is None:
        # All frames overshoot — fallback: return the first (closest) frame
        best_idx = frame_positions[0][0]
        log.warning("All frames overshoot object edge %.1fpx; falling back to frame %d",
                    object_edge_px, best_idx)
    return best_idx


def select_minimum_overlap_frame(
    imu_rows: list[ImuRow],
    ref_idx: int,
    n_frames: int,
    f_px: float,
    depth_z: float,
    frame_width: int,
    min_overlap: float = 0.15,
    vo_offsets: Optional[list[float]] = None,
) -> int:
    """
    Step 4 — No moving object: advance as far as possible while keeping at least
    min_overlap fraction of the frame width as overlap with the reference frame.
    Target advance = frame_width × (1 − min_overlap).
    Returns the closest matching frame index > ref_idx.
    """
    target_px = frame_width * (1.0 - min_overlap)
    best_idx  = ref_idx + 1
    best_diff = float("inf")
    for o in range(ref_idx + 1, n_frames):
        P    = compute_frame_position(imu_rows, ref_idx, o, f_px, depth_z, vo_offsets)
        diff = abs(P - target_px)
        if diff < best_diff:
            best_diff = diff
            best_idx  = o
        if P > target_px * 1.5:
            break   # well past target — stop
    log.debug("Step4 min_overlap=%.0f%%  target_px=%.1f  → frame %d",
              min_overlap * 100, target_px, best_idx)
    return best_idx
