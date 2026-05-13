"""
Undistort and stereo-rectify frames using loaded calibration.
When secondary is absent, only undistortion is applied.
"""
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from panorama_poc.io_loader import Calibration, CameraCalib

log = logging.getLogger(__name__)


def build_undistort_maps(cam: CameraCalib) -> Tuple[np.ndarray, np.ndarray]:
    K = np.array([[cam.fx, 0, cam.cx],
                  [0, cam.fy, cam.cy],
                  [0, 0, 1]], dtype=np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, cam.dist, None, K, (cam.width, cam.height), cv2.CV_32FC1
    )
    return map1, map2


class StereoRectifier:
    def __init__(self, calib: Calibration):
        self.calib = calib
        self._prim_map1 = self._prim_map2 = None
        self._sec_map1  = self._sec_map2  = None
        self._Q: Optional[np.ndarray] = None
        self._stereo_ready = False
        self._build(calib)

    def _build(self, calib: Calibration) -> None:
        p = calib.primary
        K_p = np.array([[p.fx, 0, p.cx], [0, p.fy, p.cy], [0, 0, 1]], dtype=np.float64)

        if calib.secondary is None or calib.R_secondary_in_primary is None:
            self._prim_map1, self._prim_map2 = build_undistort_maps(p)
            log.info("StereoRectifier: monocular mode")
            return

        s = calib.secondary
        K_s = np.array([[s.fx, 0, s.cx], [0, s.fy, s.cy], [0, 0, 1]], dtype=np.float64)
        R = calib.R_secondary_in_primary
        T = calib.t_secondary_in_primary.reshape(3, 1)
        img_size = (p.width, p.height)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K_p, p.dist, K_s, s.dist, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        self._prim_map1, self._prim_map2 = cv2.initUndistortRectifyMap(
            K_p, p.dist, R1, P1, img_size, cv2.CV_32FC1
        )
        self._sec_map1, self._sec_map2 = cv2.initUndistortRectifyMap(
            K_s, s.dist, R2, P2, img_size, cv2.CV_32FC1
        )
        self._Q = Q
        self._stereo_ready = True
        log.info("StereoRectifier: stereo mode ready  baseline=%.3fm", calib.baseline_m)

    def rectify_primary(self, frame: np.ndarray) -> np.ndarray:
        return cv2.remap(frame, self._prim_map1, self._prim_map2, cv2.INTER_LINEAR)

    def rectify_secondary(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self._stereo_ready:
            return None
        return cv2.remap(frame, self._sec_map1, self._sec_map2, cv2.INTER_LINEAR)

    @property
    def stereo_ready(self) -> bool:
        return self._stereo_ready

    @property
    def Q_matrix(self) -> Optional[np.ndarray]:
        return self._Q
