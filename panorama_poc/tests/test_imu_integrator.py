"""Unit tests for imu_integrator."""
import pytest
from panorama_poc.imu_integrator import ImuRow, integrate_imu


def _make_rows(n: int, ax=0.0, ay=0.0, az=0.0,
               gx=0.0, gy=0.0, gz=0.0, dt=0.033) -> list[ImuRow]:
    return [
        ImuRow(frame_idx=i, timestamp_s=i * dt,
               ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz)
        for i in range(n)
    ]


def test_zero_accel_zero_displacement():
    rows = _make_rows(10, ax=0, ay=0, az=0)
    seg = integrate_imu(rows, 0, 9)
    assert abs(seg.displacement_x) < 1e-9
    assert abs(seg.displacement_y) < 1e-9


def test_same_ref_target_returns_zero():
    rows = _make_rows(5, ax=1.0)
    seg = integrate_imu(rows, 2, 2)
    assert seg.displacement_x == 0.0


def test_displacement_positive_for_positive_accel():
    """Time-varying positive ax should produce nonzero displacement_x.
    Uses < 9 rows to skip the high-pass filter (which strips DC-only signals).
    """
    rows = _make_rows(8, ax=1.0, dt=0.01)   # 8 rows → no high-pass
    seg = integrate_imu(rows, 0, 7)
    assert seg.displacement_x > 0


def test_gyro_integration():
    """Constant gz (yaw rate) should produce non-zero delta_yaw."""
    rows = _make_rows(10, gz=0.1, dt=0.033)
    seg = integrate_imu(rows, 0, 9)
    assert abs(seg.delta_yaw) > 0


def test_ref_gt_target_returns_zero():
    rows = _make_rows(5)
    seg = integrate_imu(rows, 4, 2)
    assert seg.displacement_x == 0.0
