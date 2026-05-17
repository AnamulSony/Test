"""
Moving object detection — YOLO + Three-Frame Differencing + Kalman Filter.

Pipeline per frame:
  1. YOLOv8 detects all objects in the current frame
  2. Three-frame differencing produces a motion mask (M(t) = |F(t)-F(t-1)| ∩ |F(t+1)-F(t)|)
  3. Only YOLO boxes that overlap the motion mask are kept  → real moving objects
  4. KalmanTracker assigns stable IDs across frames
  5. most_central_track selects the one closest to the frame centre
  6. Annotated frames + CSV + JSON are written to the output directory

Usage:
    python main.py --source <frames_dir_or_video> [options]
    python main.py --help
"""
import argparse
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from motion_detect.annotator import OutputWriter, draw_tracks
from motion_detect.diff_engine import DiffEngine
from motion_detect.frame_source import FrameSource
from motion_detect.tracker import KalmanTracker, most_central_track
from motion_detect.yolo_detector import YOLODetector


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect moving objects using YOLO + three-frame differencing + Kalman."
    )
    # Source / output
    p.add_argument("--source", required=True,
                   help="Path to a frames directory or video file.")
    p.add_argument("--output", default="output",
                   help="Output directory (default: ./output).")

    # YOLO
    p.add_argument("--yolo-model", default="yolov8n.pt", metavar="MODEL",
                   help="YOLO weights file (default: yolov8n.pt). "
                        "Options: yolov8n/s/m/l/x.pt  (n=fastest, x=most accurate).")
    p.add_argument("--yolo-conf", type=float, default=0.35, metavar="FLOAT",
                   help="YOLO confidence threshold (default: 0.35).")
    p.add_argument("--classes", type=int, nargs="+", metavar="ID", default=None,
                   help="YOLO class IDs to detect (default: all). "
                        "0=person 2=car 3=motorcycle 5=bus 7=truck.")
    p.add_argument("--motion-overlap", type=float, default=0.20, metavar="FLOAT",
                   help="Min fraction of YOLO box that must overlap motion mask "
                        "(default: 0.20). Lower = more detections, higher = stricter.")

    # Three-frame differencing
    p.add_argument("--diff-thresh", type=int, default=25, metavar="INT",
                   help="Abs-diff pixel threshold for motion mask (default: 25).")
    p.add_argument("--no-stabilize", action="store_true",
                   help="Skip ORB homography alignment (faster for static cameras).")

    # Kalman tracker
    p.add_argument("--iou-thresh", type=float, default=0.25, metavar="FLOAT",
                   help="IoU threshold for Kalman track matching (default: 0.25).")
    p.add_argument("--max-age", type=int, default=5, metavar="INT",
                   help="Frames before track deletion (default: 5).")
    p.add_argument("--min-hits", type=int, default=2, metavar="INT",
                   help="Consecutive matches to confirm a track (default: 2).")

    # Output
    p.add_argument("--no-annotate", action="store_true",
                   help="Skip writing annotated JPEGs (CSV/JSON still written).")
    p.add_argument("--show", action="store_true",
                   help="Display annotated frames in a live window.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG logging.")

    return p.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    log = logging.getLogger(__name__)

    source_path = Path(args.source)
    if not source_path.exists():
        log.error("Source not found: %s", source_path)
        return 1

    log.info("Loading YOLO model: %s", args.yolo_model)
    yolo    = YOLODetector(
        model_name = args.yolo_model,
        conf       = args.yolo_conf,
        classes    = args.classes,
    )
    engine  = DiffEngine(
        diff_thresh   = args.diff_thresh,
        use_stabilize = not args.no_stabilize,
    )
    tracker = KalmanTracker(
        iou_threshold = args.iou_thresh,
        max_age       = args.max_age,
        min_hits      = args.min_hits,
    )

    with FrameSource(source_path) as source:
        total        = len(source)
        frame_w, frame_h = source.frame_size
        log.info("Source: %s  |  %d frames  |  %dx%d", source_path, total, frame_w, frame_h)

        with OutputWriter(
            output_dir   = args.output,
            source_name  = str(source_path),
            total_frames = total,
        ) as writer:

            buf: Deque[Tuple[int, np.ndarray]] = deque(maxlen=3)

            for frame_idx, frame in source:
                buf.append((frame_idx, frame))

                if len(buf) == 1:
                    continue

                if len(buf) == 2:
                    # Frame 0 — forward pair only, no full motion mask
                    mask, _ = engine.process_pair(buf[0][1], buf[1][1])
                    _detect_and_write(
                        seq_idx=0, frame=buf[0][1], frame_src_idx=buf[0][0],
                        motion_mask=mask, yolo=yolo, tracker=tracker,
                        frame_w=frame_w, frame_h=frame_h,
                        motion_overlap=args.motion_overlap,
                        writer=writer, no_annotate=args.no_annotate, show=args.show,
                    )
                    continue

                # Middle frame — full tri-frame motion mask
                seq_idx = frame_idx - 1  # buf[1] is one frame behind the latest
                mask, _ = engine.process_triple(buf[0][1], buf[1][1], buf[2][1])
                _detect_and_write(
                    seq_idx=seq_idx, frame=buf[1][1], frame_src_idx=buf[1][0],
                    motion_mask=mask, yolo=yolo, tracker=tracker,
                    frame_w=frame_w, frame_h=frame_h,
                    motion_overlap=args.motion_overlap,
                    writer=writer, no_annotate=args.no_annotate, show=args.show,
                )

                if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("User quit.")
                    break

            # Last frame — backward pair only
            if len(buf) == 1:
                _detect_and_write(
                    seq_idx=0, frame=buf[0][1], frame_src_idx=buf[0][0],
                    motion_mask=None, yolo=yolo, tracker=tracker,
                    frame_w=frame_w, frame_h=frame_h,
                    motion_overlap=args.motion_overlap,
                    writer=writer, no_annotate=args.no_annotate, show=args.show,
                )
            elif len(buf) >= 2:
                mask, _ = engine.process_pair(buf[-2][1], buf[-1][1])
                _detect_and_write(
                    seq_idx=total - 1, frame=buf[-1][1], frame_src_idx=buf[-1][0],
                    motion_mask=mask, yolo=yolo, tracker=tracker,
                    frame_w=frame_w, frame_h=frame_h,
                    motion_overlap=args.motion_overlap,
                    writer=writer, no_annotate=args.no_annotate, show=args.show,
                )

    if args.show:
        cv2.destroyAllWindows()

    log.info("Done. Output written to: %s", Path(args.output).resolve())
    return 0


def _detect_and_write(
    seq_idx:       int,
    frame:         np.ndarray,
    frame_src_idx: int,
    motion_mask:   Optional[np.ndarray],
    yolo:          YOLODetector,
    tracker:       KalmanTracker,
    frame_w:       int,
    frame_h:       int,
    motion_overlap: float,
    writer:        OutputWriter,
    no_annotate:   bool,
    show:          bool,
) -> None:
    log = logging.getLogger(__name__)

    # Step 1 — YOLO detects all objects
    all_dets = yolo.detect(frame)

    # Step 2 — keep only detections overlapping the motion mask
    moving_dets = yolo.filter_by_motion(all_dets, motion_mask, motion_overlap)

    # Step 3 — Kalman tracker assigns stable IDs
    raw_tracks = tracker.update(moving_dets)

    # Step 4 — keep only the most central moving object
    tracks = most_central_track(raw_tracks, frame_w, frame_h)

    if tracks:
        label_map = {t[0]: tracker.get_label(t[0]) for t in tracks}
        log.debug("Frame %d: %s @ (%d,%d) %dx%d",
                  frame_src_idx,
                  label_map.get(tracks[0][0], "?"),
                  tracks[0][1], tracks[0][2], tracks[0][3], tracks[0][4])
    else:
        label_map = {}

    # Step 5 — annotate and write
    if no_annotate:
        annotated = None
    else:
        annotated = draw_tracks(frame, tracks, frame_src_idx, label_map=label_map)

    writer.write_frame(seq_idx, annotated, tracks)

    if show:
        display = annotated if annotated is not None else draw_tracks(
            frame, tracks, frame_src_idx, label_map=label_map
        )
        cv2.imshow("Moving Object Detection (YOLO + 3-Frame Diff)", display)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level   = logging.DEBUG if args.verbose else logging.INFO,
        format  = "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt = "%H:%M:%S",
    )
    sys.exit(_run(args))


if __name__ == "__main__":
    main()
