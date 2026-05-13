"""
Split VID-20260512-WA0007.mp4 into per-frame PNGs and produce imu.csv
synchronized to frame timestamps.

Output structure:
  capture/
    primary/   000000.png … 000120.png
    imu.csv    frame_idx, timestamp_s, ax, ay, az, gx, gy, gz
"""

import csv
import logging
import os
import sys

import cv2
import numpy as np
import openpyxl

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)

np.random.seed(0)

DATA_DIR   = r"d:\Vedio\POC\data"
VIDEO_PATH = os.path.join(DATA_DIR, "VID-20260512-WA0007.mp4")
IMU_PATH   = os.path.join(DATA_DIR, "DOC-20260512-WA0008..xlsx")

OUT_ROOT   = r"d:\Vedio\POC\capture"
PRIMARY_DIR = os.path.join(OUT_ROOT, "primary")
IMU_CSV     = os.path.join(OUT_ROOT, "imu.csv")


# ── helpers ──────────────────────────────────────────────────────────────────

def load_imu_xlsx(path: str) -> list[tuple]:
    """Load IMU xlsx → list of (time_s, ax, ay, az) rows (header skipped)."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:          # skip header
            continue
        t, ax, ay, az, _ = row
        if t is None:
            continue
        rows.append((float(t), float(ax), float(ay), float(az)))
    wb.close()
    log.info("Loaded %d IMU rows from %s", len(rows), path)
    return rows


def nearest_imu(imu_rows: list[tuple], target_t: float) -> tuple:
    """Return IMU row whose timestamp is closest to target_t."""
    times = np.array([r[0] for r in imu_rows])
    idx = int(np.argmin(np.abs(times - target_t)))
    return imu_rows[idx]


def extract_frames(video_path: str, out_dir: str) -> list[tuple[int, float]]:
    """
    Extract every frame from video_path as %06d.png in out_dir.
    Returns list of (frame_idx, timestamp_s).
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log.info("Video: %d frames @ %.3f FPS  (%dx%d)  duration=%.2fs",
             total, fps, width, height, total / fps)

    frame_meta = []
    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            log.warning("Premature end at frame %d", idx)
            break
        ts = idx / fps
        out_path = os.path.join(out_dir, f"{idx:06d}.png")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        frame_meta.append((idx, ts))

        if idx % 20 == 0:
            log.info("  saved frame %d / %d  (t=%.3fs)", idx, total, ts)

    cap.release()
    log.info("Extracted %d frames to %s", len(frame_meta), out_dir)
    return frame_meta


def write_imu_csv(frame_meta: list[tuple], imu_rows: list[tuple], out_path: str) -> None:
    """
    Write imu.csv with columns:
      frame_idx, timestamp_s, ax, ay, az, gx, gy, gz
    gx/gy/gz are set to 0.0 — the source data has no gyroscope channel.
    """
    imu_times = np.array([r[0] for r in imu_rows])
    video_t0  = frame_meta[0][1]   # should be 0.0
    imu_t0    = imu_rows[0][0]     # first IMU timestamp

    log.info("Video t0=%.4fs  IMU t0=%.4fs  (assuming simultaneous start)",
             video_t0, imu_t0)
    log.warning(
        "No gyroscope data in source file — gx/gy/gz written as 0.0. "
        "Orientation integration will be unavailable."
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "timestamp_s",
                         "ax", "ay", "az",
                         "gx", "gy", "gz"])
        for frame_idx, ts in frame_meta:
            closest = nearest_imu(imu_rows, ts)
            t_imu, ax, ay, az = closest
            writer.writerow([frame_idx, f"{ts:.6f}",
                             f"{ax:.8f}", f"{ay:.8f}", f"{az:.8f}",
                             "0.0", "0.0", "0.0"])

    log.info("Wrote imu.csv (%d rows) → %s", len(frame_meta), out_path)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Frame extraction + IMU sync ===")

    # 1. Load IMU
    imu_rows = load_imu_xlsx(IMU_PATH)

    # 2. Extract frames
    frame_meta = extract_frames(VIDEO_PATH, PRIMARY_DIR)

    # 3. Write synchronized imu.csv
    write_imu_csv(frame_meta, imu_rows, IMU_CSV)

    # 4. Summary
    log.info("Done.")
    log.info("  Frames  : %s  (%d PNGs)", PRIMARY_DIR, len(frame_meta))
    log.info("  IMU CSV : %s", IMU_CSV)
    log.info("")
    log.info("NOTE: 'secondary/' directory is absent — only one video provided.")
    log.info("      Stereo depth will be unavailable until a secondary video is added.")


if __name__ == "__main__":
    main()
