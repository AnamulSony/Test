"""
Ghost-free panorama pipeline — improved stitching, warping, and blending.

Improvements:
  - Homography-based warping: SIFT + RANSAC between consecutive selected
    frames handles rotation, scale, and perspective (not just translation)
  - Distance-transform blending: each pixel weighted by distance to frame
    boundary — seams fall at the most natural point automatically
  - Auto-crop: black borders removed from final panorama

Usage:
    python run_frames.py --input D:/Vedio/video.mp4   --output D:/Vedio/output
    python run_frames.py --input D:/Vedio/data/frames --output D:/Vedio/output
"""
import argparse
import csv
import json
import logging
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from panorama_poc.frame_selector import compute_frame_position, select_minimum_overlap_frame
from panorama_poc.visual_odometry import compute_vo_offsets

log = logging.getLogger(__name__)

_BLEND_OVERLAP_RATIO  = 0.5
_MIN_ADVANCE_PX       = 20
_FLANN_INDEX_KDTREE   = 1
_PYRAMID_LEVELS       = 5
_GHOST_DIFF_THRESH    = 25   # pixel diff (0-255) that signals a moving object
_GHOST_DILATE_PX      = 21   # grow conflict zone to cover object borders cleanly


# ── frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    existing = sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.lower().endswith(".jpg")
    )
    if existing:
        log.info("  Frames already extracted (%d) — skipping", len(existing))
        return existing

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    log.info("  Video: %d frames  %.1f fps", total, fps)

    paths, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)
        idx += 1
        if idx % 100 == 0:
            log.info("  Extracted %d / %d …", idx, total)

    cap.release()
    log.info("  Done — %d frames → %s", len(paths), out_dir)
    return paths


def _load_frame_paths(folder: str) -> list[str]:
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not paths:
        raise ValueError(f"No images found in {folder}")
    return paths


# ── step 8: homography warping ────────────────────────────────────────────────

def _match_homography(img1: np.ndarray, img2: np.ndarray):
    """SIFT + FLANN + RANSAC homography. Returns H mapping img2 → img1, or None."""
    sift = cv2.SIFT_create(nfeatures=6000)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return None

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=_FLANN_INDEX_KDTREE, trees=5), dict(checks=50)
    )
    raw  = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.70 * n.distance]

    if len(good) < 12:
        log.warning("  Homography: only %d good matches — using identity", len(good))
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, inliers = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0, confidence=0.995)
    if H is None or inliers is None or int(inliers.sum()) < 10:
        return None

    if not _validate_homography(H, img1.shape[:2]):
        log.warning("  Homography: rejected (degenerate transform)")
        return None

    log.info("  Homography: %d inliers / %d matches", int(inliers.sum()), len(good))
    return H


def _validate_homography(H: np.ndarray, shape: tuple) -> bool:
    """Reject homographies with extreme scale, perspective distortion, or canvas explosion."""
    det = np.linalg.det(H[:2, :2])
    if det < 0.05 or det > 20.0:
        return False
    h, w = shape
    # Perspective components (H[2,0], H[2,1]) cause the canvas to explode.
    # Scale by image dimensions so the check is resolution-independent.
    if abs(H[2, 0]) * w > 0.05 or abs(H[2, 1]) * h > 0.05:
        return False
    corners  = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    proj     = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    orig     = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    max_disp = np.linalg.norm(proj - orig, axis=1).max()
    return float(max_disp) < max(w, h) * 1.5


def _compose_vo_affine(vo_affines: list[np.ndarray],
                       start: int, end: int) -> np.ndarray:
    """
    Compose per-frame VO affines[start+1 .. end] into a single 3×3 homography
    that maps frame[end] → frame[start] (same convention as SIFT H).

    Each vo_affines[k] is a 2×3 affine mapping frame k-1 → frame k
    (rotation + scale + translation captured by the VO).
    We compose them forward (start→end) then invert.
    """
    H_fwd = np.eye(3, dtype=np.float64)
    for k in range(start + 1, end + 1):
        M = vo_affines[k].astype(np.float64)
        M3 = np.vstack([M, [0.0, 0.0, 1.0]])
        H_fwd = M3 @ H_fwd          # accumulate: start → k
    try:
        return np.linalg.inv(H_fwd)  # invert: end → start
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)


def _build_chain(frames: list[np.ndarray],
                 vo_x: list[float],
                 vo_y: list[float],
                 indices: list[int],
                 vo_affines: list[np.ndarray] | None = None) -> list[np.ndarray]:
    """
    Chain homographies so H[i] maps frame i → frame 0 coordinate space.
    Fallback priority when SIFT is rejected:
      1. Composed VO affine (rotation + scale + translation over all intermediate frames)
      2. Pure VO translation (dx, dy) if affines not available
    """
    chain = [np.eye(3, dtype=np.float64)]
    for i in range(len(frames) - 1):
        H = _match_homography(frames[i], frames[i + 1])
        if H is None:
            if vo_affines is not None:
                H = _compose_vo_affine(vo_affines, indices[i], indices[i + 1])
                log.warning("  VO affine fallback (composed %d frames) for frame %d→%d",
                            indices[i + 1] - indices[i], indices[i], indices[i + 1])
            else:
                dx = vo_x[indices[i + 1]] - vo_x[indices[i]]
                dy = vo_y[indices[i + 1]] - vo_y[indices[i]]
                H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
                log.warning("  VO translation fallback dx=%.1fpx dy=%.1fpx for frame %d→%d",
                            dx, dy, indices[i], indices[i + 1])
        chain.append(chain[-1] @ H)
    return chain


# ── step 9: gain compensation + multi-band blending ──────────────────────────

def _gain_compensate(warped_list: list[np.ndarray],
                     mask_list: list[np.ndarray]) -> list[np.ndarray]:
    """Equalise per-frame mean intensity to suppress brightness seams."""
    means = []
    for img, mask in zip(warped_list, mask_list):
        pix = img[mask > 0].astype(np.float64)
        means.append(float(pix.mean()) if pix.size else 128.0)
    target = float(np.mean(means))
    result = []
    for img, m in zip(warped_list, means):
        gain = float(np.clip(target / m if m > 0 else 1.0, 0.5, 2.0))
        result.append(np.clip(img.astype(np.float64) * gain, 0, 255).astype(np.uint8))
    return result


def _lap_blend_pair(img1: np.ndarray, img2: np.ndarray,
                    alpha: np.ndarray, levels: int) -> np.ndarray:
    """Laplacian pyramid blend. img1/img2 float32 (H,W,3), alpha float32 (H,W)."""
    gp1, gp2, ga = [img1], [img2], [alpha]
    for _ in range(levels):
        gp1.append(cv2.pyrDown(gp1[-1]))
        gp2.append(cv2.pyrDown(gp2[-1]))
        ga.append(cv2.pyrDown(ga[-1]))
    lp1: list = [gp1[levels]]
    lp2: list = [gp2[levels]]
    for i in range(levels - 1, -1, -1):
        h, w = gp1[i].shape[:2]
        lp1.insert(0, gp1[i] - cv2.pyrUp(gp1[i + 1], dstsize=(w, h)))
        lp2.insert(0, gp2[i] - cv2.pyrUp(gp2[i + 1], dstsize=(w, h)))
    blended = []
    for i in range(levels + 1):
        a = ga[i][:, :, None]
        blended.append(a * lp1[i] + (1.0 - a) * lp2[i])
    result = blended[levels]
    for i in range(levels - 1, -1, -1):
        h, w = blended[i].shape[:2]
        result = cv2.pyrUp(result, dstsize=(w, h)) + blended[i]
    return result


def _multiband_blend(warped_list: list[np.ndarray],
                     mask_list: list[np.ndarray],
                     canvas_w: int, canvas_h: int) -> np.ndarray:
    """
    Laplacian-pyramid weighted blend.
    Each frame is weighted by its distance-transform (distance to boundary),
    but blending is done per pyramid level so high-frequency content transitions
    sharply while low-frequency colour bleeds smoothly — no colour halos.
    """
    weight_maps = [
        cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
        for m in mask_list
    ]
    total  = np.sum(weight_maps, axis=0)
    total  = np.where(total > 0, total, 1.0).astype(np.float32)
    alphas = [w / total for w in weight_maps]

    imgs = [f.astype(np.float32) for f in warped_list]
    lvls = min(_PYRAMID_LEVELS, max(1, int(np.log2(min(canvas_h, canvas_w) + 1)) - 1))

    result_pyr: list | None = None
    for img, alpha in zip(imgs, alphas):
        gp = [img]
        ga = [alpha]
        for _ in range(lvls):
            gp.append(cv2.pyrDown(gp[-1]))
            ga.append(cv2.pyrDown(ga[-1]))
        lp: list = [gp[lvls]]
        for i in range(lvls - 1, -1, -1):
            h, w = gp[i].shape[:2]
            lp.insert(0, gp[i] - cv2.pyrUp(gp[i + 1], dstsize=(w, h)))
        if result_pyr is None:
            result_pyr = [lp[i] * ga[i][:, :, None] for i in range(lvls + 1)]
        else:
            for i in range(lvls + 1):
                result_pyr[i] = result_pyr[i] + lp[i] * ga[i][:, :, None]

    reconstructed = result_pyr[lvls]
    for i in range(lvls - 1, -1, -1):
        h, w = result_pyr[i].shape[:2]
        reconstructed = cv2.pyrUp(reconstructed, dstsize=(w, h)) + result_pyr[i]
    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def _autocrop(img: np.ndarray) -> np.ndarray:
    """
    Bounding-box crop: remove outer black borders.
    Uses threshold 8 to discard JPEG-artifact near-blacks at frame edges.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    rows = np.where((thresh > 0).any(axis=1))[0]
    cols = np.where((thresh > 0).any(axis=0))[0]
    if not rows.size or not cols.size:
        return img
    return img[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]


# ── combined warp + stitch + blend ────────────────────────────────────────────

def build_panorama(frames_bgr: list[np.ndarray],
                   vo_x: list[float] | None = None,
                   vo_y: list[float] | None = None,
                   vo_affines: list[np.ndarray] | None = None,
                   selected_indices: list[int] | None = None) -> np.ndarray:
    """
    [8] Warp all frames to a common canvas using chained homographies.
    [9] Gain-compensate, then multi-band pyramid blend, then auto-crop.
    vo_affines: composed VO affines used as fallback when SIFT is rejected.
    """
    n = len(frames_bgr)
    h, w = frames_bgr[0].shape[:2]

    _vo_x    = vo_x            if vo_x      else [float(i * w) for i in range(n)]
    _vo_y    = vo_y            if vo_y      else [0.0] * n
    _sel_idx = selected_indices if selected_indices else list(range(n))

    log.info("[8] Computing chained homographies …")
    H_chain = _build_chain(frames_bgr, _vo_x, _vo_y, _sel_idx, vo_affines)

    # Compute canvas bounds by projecting all frame corners
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    all_pts = np.concatenate(
        [cv2.perspectiveTransform(corners, H).reshape(-1, 2) for H in H_chain], axis=0
    )
    x_min = int(np.floor(all_pts[:, 0].min()))
    y_min = int(np.floor(all_pts[:, 1].min()))
    x_max = int(np.ceil(all_pts[:, 0].max()))
    y_max = int(np.ceil(all_pts[:, 1].max()))
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    log.info("[8] Canvas: %d × %d px", canvas_w, canvas_h)

    # Translate homographies so all coordinates are positive on canvas
    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1]], dtype=np.float64)

    log.info("[8] Warping frames onto canvas …")
    warped_list, mask_list = [], []
    for i, (frame, H) in enumerate(zip(frames_bgr, H_chain)):
        warped = cv2.warpPerspective(frame, T @ H, (canvas_w, canvas_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        warped_list.append(warped)
        mask_list.append(mask)
        log.info("  Frame %d warped", i)

    log.info("  Applying gain compensation …")
    warped_list = _gain_compensate(warped_list, mask_list)

    log.info("[9] Multi-band pyramid blending …")
    panorama = _multiband_blend(warped_list, mask_list, canvas_w, canvas_h)

    log.info("  Auto-cropping black borders …")
    panorama = _autocrop(panorama)
    return panorama


# ── pipeline ──────────────────────────────────────────────────────────────────

def run(input_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # ── [1] Frame Acquisition ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("[1] Frame Acquisition")

    if os.path.isfile(input_path):
        frames_dir = os.path.join(output_dir, "frames")
        paths = extract_frames(input_path, frames_dir)
    else:
        paths = _load_frame_paths(input_path)

    n       = len(paths)
    frame_w = cv2.imread(paths[0]).shape[1]
    log.info("  %d frames  width=%d", n, frame_w)

    # ── Visual Odometry (for frame selection + fallback alignment) ───────────
    log.info("  Computing visual odometry …")
    vo_x, vo_y, vo_affines = compute_vo_offsets(paths)
    log.info("  VO total shift: x=%.1fpx  y=%.1fpx", vo_x[-1], vo_y[-1])

    # ── [2] Frame selection ───────────────────────────────────────────────────
    log.info("[2] Selecting frames (minimum overlap) …")
    i = 0
    selected_indices: list[int] = [0]
    positions_log: list[dict]   = []

    while True:
        s_idx = select_minimum_overlap_frame(
            ref_idx=i,
            n_frames=n,
            frame_width=frame_w,
            vo_offsets=vo_x,
            min_overlap=_BLEND_OVERLAP_RATIO,
        )
        P_sel = compute_frame_position(i, s_idx, vo_x)

        if P_sel < _MIN_ADVANCE_PX and s_idx < n - 1:
            i = s_idx + 1
            if i >= n - 1:
                break
            continue

        positions_log.append({"frame_idx": s_idx, "ref_frame": i,
                               "pixel_offset": round(P_sel, 2)})
        log.info("  Frame %d → Fₛ=%d  (advance=%.1fpx)", i, s_idx, P_sel)
        selected_indices.append(s_idx)
        i = s_idx

        if i >= n - 1:
            break

    log.info("Selected %d frames: %s", len(selected_indices), selected_indices)

    # ── [8+9] Warp + Stitch + Blend ───────────────────────────────────────────
    selected_bgr = [cv2.imread(paths[idx]) for idx in selected_indices]
    panorama     = build_panorama(selected_bgr,
                                  vo_x=vo_x,
                                  vo_y=vo_y,
                                  vo_affines=vo_affines,
                                  selected_indices=selected_indices)

    # ── Save ──────────────────────────────────────────────────────────────────
    pano_path = os.path.join(output_dir, "panorama.jpg")
    cv2.imwrite(pano_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
    log.info("Saved → %s  (%dx%d)", pano_path, panorama.shape[1], panorama.shape[0])

    with open(os.path.join(output_dir, "selected_frames.json"), "w") as f:
        json.dump(selected_indices, f, indent=2)

    with open(os.path.join(debug_dir, "positions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_idx", "ref_frame", "pixel_offset"])
        w.writeheader()
        w.writerows(positions_log)

    log.info("=" * 60)
    log.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    run(args.input, args.output)


if __name__ == "__main__":
    main()
