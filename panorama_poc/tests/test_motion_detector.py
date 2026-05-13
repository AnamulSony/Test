"""Unit tests for motion_detector."""
import numpy as np
import pytest
from panorama_poc.motion_detector import MotionDetector, get_object_leading_edge


def test_blank_frame_produces_empty_mask():
    """All-black frames → no foreground detected."""
    det = MotionDetector(f_px=600.0, approx_depth=3.0)
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    # Feed multiple identical frames to train the background model
    for _ in range(6):
        mask = det.detect(blank)
    assert np.count_nonzero(mask) == 0


def test_leading_edge_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    assert get_object_leading_edge(mask) is None


def test_leading_edge_known_position():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[40:60, 80:120] = 255   # object at x=80..120
    assert get_object_leading_edge(mask) == 80


def test_reset_clears_model():
    """After reset + warm-up on blank frames, a bright frame should be foreground."""
    from panorama_poc.motion_detector import _WARMUP_FRAMES
    det = MotionDetector(f_px=600.0, approx_depth=3.0)
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(6):
        det.detect(blank)
    det.reset()
    # Complete warm-up with blank frames so the model learns "blank = background"
    for _ in range(_WARMUP_FRAMES):
        det.detect(blank)
    bright = np.full((100, 100, 3), 200, dtype=np.uint8)
    mask = det.detect(bright)
    assert np.count_nonzero(mask) > 0
