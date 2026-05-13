"""
Load and validate the capture/ directory.
Raises ValueError with a clear message for any missing or malformed field.
"""
import csv
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from panorama_poc.imu_integrator import ImuRow

log = logging.getLogger(__name__)


@dataclass
class CameraCalib:
    fx: float; fy: float; cx: float; cy: float
    width: int; height: int
    dist: np.ndarray   # (5,) k1,k2,p1,p2,k3


@dataclass
class Calibration:
    primary: CameraCalib
    secondary: Optional[CameraCalib]
    baseline_m: Optional[float]
    R_secondary_in_primary: Optional[np.ndarray]   # (3,3)
    t_secondary_in_primary: Optional[np.ndarray]   # (3,)
    R_imu_in_primary: np.ndarray                   # (3,3)
    approx_depth_m: float                          # fallback depth (m)


@dataclass
class CaptureData:
    primary_paths: list[str]    # ordered PNG paths
    secondary_paths: list[str]  # may be empty
    imu_rows: list[ImuRow]
    calib: Calibration


# ── calib ────────────────────────────────────────────────────────────────────

def _parse_camera(d: dict, name: str) -> CameraCalib:
    for key in ("fx", "fy", "cx", "cy", "width", "height", "dist"):
        if key not in d:
            raise ValueError(f"calib.json[{name}] missing field '{key}'")
    dist = np.array(d["dist"], dtype=np.float64)
    if dist.shape != (5,):
        raise ValueError(f"calib.json[{name}].dist must have 5 elements, got {len(d['dist'])}")
    return CameraCalib(
        fx=float(d["fx"]), fy=float(d["fy"]),
        cx=float(d["cx"]), cy=float(d["cy"]),
        width=int(d["width"]), height=int(d["height"]),
        dist=dist,
    )


def _parse_matrix(val, name: str, shape: tuple) -> Optional[np.ndarray]:
    if val is None:
        return None
    arr = np.array(val, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"calib.json[{name}] must be {shape}, got {arr.shape}")
    return arr


def load_calibration(calib_path: str) -> Calibration:
    """Load and validate calib.json → Calibration."""
    if not os.path.isfile(calib_path):
        raise ValueError(f"calib.json not found: {calib_path}")
    with open(calib_path) as f:
        raw = json.load(f)

    if "primary" not in raw or raw["primary"] is None:
        raise ValueError("calib.json missing 'primary' camera block")

    primary = _parse_camera(raw["primary"], "primary")
    secondary = _parse_camera(raw["secondary"], "secondary") if raw.get("secondary") else None

    R_imu = _parse_matrix(raw.get("R_imu_in_primary"), "R_imu_in_primary", (3, 3))
    if R_imu is None:
        R_imu = np.eye(3)

    approx_depth = float(raw.get("approx_depth_m", 3.0))
    log.info("Calibration loaded — approx_depth=%.1fm  stereo=%s",
             approx_depth, "yes" if secondary else "no (approximate depth mode)")

    return Calibration(
        primary=primary,
        secondary=secondary,
        baseline_m=raw.get("baseline_m"),
        R_secondary_in_primary=_parse_matrix(raw.get("R_secondary_in_primary"),
                                             "R_secondary_in_primary", (3, 3)),
        t_secondary_in_primary=_parse_matrix(raw.get("t_secondary_in_primary"),
                                             "t_secondary_in_primary", (3,)),
        R_imu_in_primary=R_imu,
        approx_depth_m=approx_depth,
    )


# ── IMU CSV ──────────────────────────────────────────────────────────────────

def load_imu(imu_path: str) -> list[ImuRow]:
    """Load imu.csv → list[ImuRow], indexed by frame_idx. Raises on missing rows."""
    if not os.path.isfile(imu_path):
        raise ValueError(f"imu.csv not found: {imu_path}")

    required = {"frame_idx", "timestamp_s", "ax", "ay", "az", "gx", "gy", "gz"}
    rows: list[ImuRow] = []
    prev_t = -1.0

    with open(imu_path, newline="") as f:
        reader = csv.DictReader(f)
        if not required.issubset(set(reader.fieldnames or [])):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"imu.csv missing columns: {missing}")
        for line_no, row in enumerate(reader, start=2):
            t = float(row["timestamp_s"])
            if t <= prev_t:
                raise ValueError(f"imu.csv line {line_no}: timestamp not monotonically increasing")
            prev_t = t
            rows.append(ImuRow(
                frame_idx=int(row["frame_idx"]),
                timestamp_s=t,
                ax=float(row["ax"]), ay=float(row["ay"]), az=float(row["az"]),
                gx=float(row["gx"]), gy=float(row["gy"]), gz=float(row["gz"]),
            ))

    log.info("Loaded %d IMU rows from %s", len(rows), imu_path)
    return rows


# ── frames ───────────────────────────────────────────────────────────────────

def _load_frame_paths(directory: str) -> list[str]:
    """Return sorted PNG paths from directory (may be empty list if dir absent)."""
    if not os.path.isdir(directory):
        return []
    paths = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".png")
    )
    return paths


# ── main entry ───────────────────────────────────────────────────────────────

def load_capture(capture_dir: str) -> CaptureData:
    """Load the full capture/ directory → CaptureData. Raises on any issue."""
    calib = load_calibration(os.path.join(capture_dir, "calib.json"))
    imu_rows = load_imu(os.path.join(capture_dir, "imu.csv"))

    primary_paths = _load_frame_paths(os.path.join(capture_dir, "primary"))
    if not primary_paths:
        raise ValueError(f"No PNG files found in {capture_dir}/primary/")

    secondary_paths = _load_frame_paths(os.path.join(capture_dir, "secondary"))

    # Validate frame count matches IMU
    if len(primary_paths) != len(imu_rows):
        raise ValueError(
            f"Frame count ({len(primary_paths)}) != IMU row count ({len(imu_rows)})"
        )

    log.info("Capture loaded: %d primary frames, %d secondary frames, %d IMU rows",
             len(primary_paths), len(secondary_paths), len(imu_rows))
    return CaptureData(
        primary_paths=primary_paths,
        secondary_paths=secondary_paths,
        imu_rows=imu_rows,
        calib=calib,
    )


def read_frame(path: str) -> np.ndarray:
    """Read a PNG frame → BGR uint8 ndarray. Raises on failure."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read frame: {path}")
    return img
