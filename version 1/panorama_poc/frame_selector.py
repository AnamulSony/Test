"""
Frame selection logic using visual odometry offsets only.
Step 4: select the next frame with minimum overlapping region.
"""
import logging
from typing import Optional

log = logging.getLogger(__name__)


def compute_frame_position(
    ref_idx: int,
    target_idx: int,
    vo_offsets: list[float],
) -> float:
    """Pixel offset of target_idx relative to ref_idx using VO (positive = rightward)."""
    return abs(vo_offsets[target_idx] - vo_offsets[ref_idx])


def select_minimum_overlap_frame(
    ref_idx: int,
    n_frames: int,
    frame_width: int,
    vo_offsets: list[float],
    min_overlap: float = 0.15,
) -> int:
    """
    Step 4 — advance as far as possible while keeping at least min_overlap
    fraction of the frame width as overlap with the reference frame.
    Target advance = frame_width × (1 − min_overlap).
    Returns the closest matching frame index > ref_idx.
    """
    target_px = frame_width * (1.0 - min_overlap)
    best_idx  = ref_idx + 1
    best_diff = float("inf")

    for o in range(ref_idx + 1, n_frames):
        P    = compute_frame_position(ref_idx, o, vo_offsets)
        diff = abs(P - target_px)
        if diff < best_diff:
            best_diff = diff
            best_idx  = o
        if P > target_px * 1.5:
            break

    log.debug("Step4 min_overlap=%.0f%%  target_px=%.1f  → frame %d",
              min_overlap * 100, target_px, best_idx)
    return best_idx
