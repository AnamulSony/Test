"""
Scene depth estimation.
Stereo mode: StereoSGBM disparity → Z = (B × f) / disparity.
Approximate mode: returns calib.approx_depth_m for every query.
"""
import logging
from typing import Optional

import cv2
import numpy as np

from panorama_poc.calibration import StereoRectifier
from panorama_poc.io_loader import Calibration

log = logging.getLogger(__name__)

_SGBM_DEFAULTS = dict(
    minDisparity=0,
    numDisparities=64,
    blockSize=7,
    P1=8 * 3 * 7 ** 2,
    P2=32 * 3 * 7 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)


class DepthEstimator:
    def __init__(self, calib: Calibration, rectifier: StereoRectifier):
        self._approx    = calib.approx_depth_m
        self._rectifier = rectifier
        self._baseline  = calib.baseline_m
        self._fx        = calib.primary.fx

        if rectifier.stereo_ready and calib.baseline_m:
            self._sgbm = cv2.StereoSGBM_create(**_SGBM_DEFAULTS)
            self._mode = "stereo"
            log.info("DepthEstimator: stereo mode  B=%.3fm  fx=%.1f", self._baseline, self._fx)
        else:
            self._sgbm = None
            self._mode = "approximate"
            log.info("DepthEstimator: approximate mode  Z=%.1fm (fixed)", self._approx)

    def get_depth(
        self,
        left_rect: np.ndarray,
        right_rect: Optional[np.ndarray],
        mask: Optional[np.ndarray] = None,
    ) -> float:
        if self._mode == "approximate" or right_rect is None:
            return self._approx

        left_gray  = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disp = self._sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0

        valid = disp > 1.0
        if mask is not None:
            valid &= mask.astype(bool)

        if not np.any(valid):
            return self._approx

        depth_map = (self._baseline * self._fx) / np.where(valid, disp, np.nan)
        return float(np.nanmedian(depth_map[valid]))
