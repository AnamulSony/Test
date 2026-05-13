"""
Smoke test: run pipeline on the real capture/ directory.
Verifies output files exist and selected_frames has ≥ 2 entries.
"""
import json
import os
import pytest

CAPTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "capture")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "output", "smoke_test")


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(CAPTURE_DIR, "primary")),
    reason="capture/primary not found — provide real data to run smoke test",
)
def test_pipeline_smoke():
    from panorama_poc.pipeline import run_pipeline
    run_pipeline(CAPTURE_DIR, OUTPUT_DIR)

    pano = os.path.join(OUTPUT_DIR, "panorama.png")
    sel  = os.path.join(OUTPUT_DIR, "selected_frames.json")
    pos  = os.path.join(OUTPUT_DIR, "debug", "positions.csv")

    assert os.path.isfile(pano), "panorama.png not created"
    assert os.path.isfile(sel),  "selected_frames.json not created"
    assert os.path.isfile(pos),  "debug/positions.csv not created"

    with open(sel) as f:
        indices = json.load(f)
    assert len(indices) >= 2, f"Expected ≥2 selected frames, got {len(indices)}"
