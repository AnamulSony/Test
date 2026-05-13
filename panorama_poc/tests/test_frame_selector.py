"""Unit tests for frame_selector — T4, T5, and edge cases."""
import pytest
from panorama_poc.frame_selector import compute_pixel_displacement, find_next_optimal_frame


# T4 — numerical sanity
def test_compute_pixel_displacement_t4():
    """f_px=2000, D=0.02m, Z=2m → ΔP = 20.0 ± 0.1 px"""
    dp = compute_pixel_displacement(f_px=2000.0, D=0.02, Z=2.0)
    assert abs(dp - 20.0) < 0.1


# T5 — frame selector worked example
def test_find_next_optimal_frame_t5():
    """object_edge=80, positions [(2,20),(3,25),(4,55),(5,79),(6,83)] → frame 5"""
    positions = [(2, 20.0), (3, 25.0), (4, 55.0), (5, 79.0), (6, 83.0)]
    result = find_next_optimal_frame(object_edge_px=80.0, frame_positions=positions)
    assert result == 5


def test_find_next_optimal_exact_match():
    """Frame whose P equals object_edge exactly should be selected."""
    positions = [(1, 50.0), (2, 80.0), (3, 100.0)]
    result = find_next_optimal_frame(object_edge_px=80.0, frame_positions=positions)
    assert result == 2


def test_find_next_optimal_all_overshoot():
    """All frames overshoot → return first candidate (fallback)."""
    positions = [(1, 90.0), (2, 150.0)]
    result = find_next_optimal_frame(object_edge_px=80.0, frame_positions=positions)
    assert result == 1  # fallback: first in list


def test_find_next_optimal_empty():
    """Empty list → returns None."""
    result = find_next_optimal_frame(object_edge_px=80.0, frame_positions=[])
    assert result is None


def test_compute_pixel_displacement_zero_d():
    """Zero displacement → zero pixel shift."""
    assert compute_pixel_displacement(f_px=1000.0, D=0.0, Z=5.0) == 0.0


def test_compute_pixel_displacement_scales_with_depth():
    """Deeper scene → smaller pixel shift for same physical motion."""
    dp_near = compute_pixel_displacement(f_px=1000.0, D=0.1, Z=2.0)
    dp_far  = compute_pixel_displacement(f_px=1000.0, D=0.1, Z=10.0)
    assert dp_near > dp_far
