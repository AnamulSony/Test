"""
Frame provider — abstracts over JPG/PNG image directories and video files.
Yields (frame_index, bgr_ndarray) pairs in chronological order.
"""
import re
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_FRAME_RE = re.compile(r"frame_(\d+)_t\d+ms\.(jpg|jpeg|png)", re.IGNORECASE)


class FrameSource:
    """
    Unified sequential frame provider.

    Accepts either:
    - A directory path containing image files (sorted lexicographically)
    - A video file path (mp4, avi, mkv, …)
    """

    def __init__(self, source: Union[str, Path]) -> None:
        self._source = Path(source)
        self._cap: Optional[cv2.VideoCapture] = None
        self._image_paths: Optional[List[Path]] = None

        if self._source.is_dir():
            self._image_paths = sorted(
                p for p in self._source.iterdir()
                if p.suffix.lower() in _IMG_EXTENSIONS
            )
            if not self._image_paths:
                raise ValueError(f"No image files found in {self._source}")
        elif self._source.is_file():
            self._cap = cv2.VideoCapture(str(self._source))
            if not self._cap.isOpened():
                raise ValueError(f"Cannot open video file: {self._source}")
        else:
            raise FileNotFoundError(f"Source not found: {self._source}")

    # ── public interface ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        if self._image_paths is not None:
            return len(self._image_paths)
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        if self._image_paths is not None:
            yield from self._iter_images()
        else:
            yield from self._iter_video()

    @property
    def fps(self) -> float:
        if self._cap is not None:
            return float(self._cap.get(cv2.CAP_PROP_FPS)) or 1.0
        return 1.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        if self._image_paths is not None:
            frame = cv2.imread(str(self._image_paths[0]))
            if frame is None:
                raise RuntimeError(f"Cannot read {self._image_paths[0]}")
            h, w = frame.shape[:2]
            return w, h
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── private iterators ──────────────────────────────────────────────────────

    def _iter_images(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        for idx, path in enumerate(self._image_paths):
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Cannot read image: {path}")
            yield idx, frame

    def _iter_video(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
