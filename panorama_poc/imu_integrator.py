"""
IMU integration: high-pass filter → trapezoidal integrate → displacement (m).
Velocity is reset to zero at each new reference frame to bound drift.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt

log = logging.getLogger(__name__)


@dataclass
class ImuRow:
    frame_idx: int
    timestamp_s: float
    ax: float   # m/s²
    ay: float   # m/s²
    az: float   # m/s²
    gx: float   # rad/s
    gy: float   # rad/s
    gz: float   # rad/s


@dataclass
class ImuSegment:
    """Integrated result from frame F_ref to F_target."""
    displacement_x: float = 0.0   # metres, horizontal
    displacement_y: float = 0.0   # metres, vertical
    displacement_z: float = 0.0   # metres, depth axis
    delta_yaw:   float = 0.0      # rad
    delta_pitch: float = 0.0      # rad
    delta_roll:  float = 0.0      # rad


def _highpass(signal: np.ndarray, timestamps: np.ndarray,
              cutoff_hz: float = 0.1) -> np.ndarray:
    """Remove low-frequency gravity bias with a 2nd-order Butterworth high-pass."""
    dt_mean = float(np.mean(np.diff(timestamps)))
    fs = 1.0 / dt_mean
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        log.warning("High-pass cutoff %.2f Hz >= Nyquist %.2f Hz; skipping filter", cutoff_hz, nyq)
        return signal
    b, a = butter(2, cutoff_hz / nyq, btype="high")
    return filtfilt(b, a, signal)


def integrate_imu(rows: list[ImuRow], ref_idx: int, target_idx: int) -> ImuSegment:
    """
    Integrate IMU from rows[ref_idx] to rows[target_idx].
    Returns ImuSegment with displacement in metres and orientation deltas in rad.
    Velocity is reset to zero at ref_idx (drift bound).
    """
    if ref_idx >= target_idx:
        return ImuSegment()

    seg = rows[ref_idx: target_idx + 1]
    if len(seg) < 2:
        return ImuSegment()

    t  = np.array([r.timestamp_s for r in seg])
    ax = np.array([r.ax for r in seg])
    ay = np.array([r.ay for r in seg])
    az = np.array([r.az for r in seg])
    gx = np.array([r.gx for r in seg])
    gy = np.array([r.gy for r in seg])
    gz = np.array([r.gz for r in seg])

    # High-pass to remove gravity bias on each axis
    if len(seg) > 9:   # filtfilt requires length > 3 × filter order (=9)
        ax = _highpass(ax, t)
        ay = _highpass(ay, t)
        az = _highpass(az, t)

    # Trapezoidal integration: acceleration → velocity (v0 = 0 at ref)
    dt   = np.diff(t)
    vx   = np.concatenate(([0.0], np.cumsum(0.5 * (ax[:-1] + ax[1:]) * dt)))
    vy   = np.concatenate(([0.0], np.cumsum(0.5 * (ay[:-1] + ay[1:]) * dt)))
    vz   = np.concatenate(([0.0], np.cumsum(0.5 * (az[:-1] + az[1:]) * dt)))

    # Velocity → displacement
    dx = np.sum(0.5 * (vx[:-1] + vx[1:]) * dt)
    dy = np.sum(0.5 * (vy[:-1] + vy[1:]) * dt)
    dz = np.sum(0.5 * (vz[:-1] + vz[1:]) * dt)

    # Angular integration (gyro, no high-pass needed)
    dyaw   = float(np.sum(0.5 * (gz[:-1] + gz[1:]) * dt))
    dpitch = float(np.sum(0.5 * (gy[:-1] + gy[1:]) * dt))
    droll  = float(np.sum(0.5 * (gx[:-1] + gx[1:]) * dt))

    return ImuSegment(
        displacement_x=float(dx),
        displacement_y=float(dy),
        displacement_z=float(dz),
        delta_yaw=dyaw,
        delta_pitch=dpitch,
        delta_roll=droll,
    )
