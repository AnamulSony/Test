"""
Load and validate the capture/ directory.
Frame positions are computed from visual odometry only — no IMU required.
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

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
    approx_depth_m: float


@dataclass
class CaptureData:
    primary_paths: list[str]
    secondary_paths: list[str]
    calib: Calibration


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
    if not os.path.isfile(calib_path):
        raise ValueError(f"calib.json not found: {calib_path}")
    with open(calib_path) as f:
        raw = json.load(f)

    if "primary" not in raw or raw["primary"] is None:
        raise ValueError("calib.json missing 'primary' camera block")

    primary   = _parse_camera(raw["primary"], "primary")
    secondary = _parse_camera(raw["secondary"], "secondary") if raw.get("secondary") else None
    approx_depth = float(raw.get("approx_depth_m", 3.0))

    log.info("Calibration loaded — approx_depth=%.1fm  stereo=%s",
             approx_depth, "yes" if secondary else "no")

    return Calibration(
        primary=primary,
        secondary=secondary,
        baseline_m=raw.get("baseline_m"),
        R_secondary_in_primary=_parse_matrix(raw.get("R_secondary_in_primary"),
                                             "R_secondary_in_primary", (3, 3)),
        t_secondary_in_primary=_parse_matrix(raw.get("t_secondary_in_primary"),
                                             "t_secondary_in_primary", (3,)),
        approx_depth_m=approx_depth,
    )


def _load_frame_paths(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".png")
    )


def load_capture(capture_dir: str) -> CaptureData:
    calib = load_calibration(os.path.join(capture_dir, "calib.json"))

    primary_paths = _load_frame_paths(os.path.join(capture_dir, "primary"))
    if not primary_paths:
        raise ValueError(f"No PNG files found in {capture_dir}/primary/")

    secondary_paths = _load_frame_paths(os.path.join(capture_dir, "secondary"))

    log.info("Capture loaded: %d primary frames, %d secondary frames",
             len(primary_paths), len(secondary_paths))
    return CaptureData(
        primary_paths=primary_paths,
        secondary_paths=secondary_paths,
        calib=calib,
    )


def read_frame(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read frame: {path}")
    return img
