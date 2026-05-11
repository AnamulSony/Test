"""
Traditional Panorama Image Creation Pipeline
=============================================
Step 4  : Overlap Check       — verify 25-40% overlap between adjacent frames
Step 5  : Feature Detection   — ORB keypoints + descriptors on each frame pair
Step 6  : Image Alignment     — BFMatcher + Lowe ratio test + RANSAC homography
Step 7  : Warping             — cumulative homographies, single-pass warpPerspective
Step 8  : Blending            — distance-transform feather blend (all frames at once)
Step 9  : Cropping            — remove black borders
Step 10 : Enhancement         — CLAHE contrast, unsharp sharpen, brightness
Step 11 : Save                — write panorama as JPG or PNG

KEY FIX over naive approach:
  Each frame is matched against the PREVIOUS frame only (not the full
  growing panorama), giving a clean local transform H_i.  All H_i are
  composed into cumulative H_cum[i] = H_cum[i-1] @ H_i so every frame
  is expressed in frame-0 coordinate space.  Then every frame is warped
  onto a single pre-computed canvas in one shot and blended together —
  no error accumulation, no canvas height drift.
"""

import cv2
import os
import glob
import argparse
import numpy as np


# =============================================================================
# STEP 5 — Feature Detection (ORB)
# =============================================================================
def detect_features(image, max_features=3000):
    """
    Detect keypoints and compute binary descriptors using ORB.

    ORB = Oriented FAST corner detector + Rotated BRIEF descriptor.
      - FAST marks pixels where intensity changes abruptly (corners/edges).
      - BRIEF encodes each keypoint as a 256-bit binary string from
        random pixel-pair intensity comparisons inside a local patch.
      - 'nlevels=8' builds an image pyramid so features are found at
        multiple scales (objects at different distances / zoom levels).
      - Harris score ranks keypoints by corner response strength.

    Works on grayscale only — colour channels carry no extra information
    for corner detection and would slow things down.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb  = cv2.ORB_create(
        nfeatures  = max_features,
        scaleFactor= 1.2,
        nlevels    = 8,
        edgeThreshold = 15,
        scoreType  = cv2.ORB_HARRIS_SCORE,
        patchSize  = 31,
    )
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


# =============================================================================
# STEP 6 — Image Alignment (curr frame → prev frame)
# =============================================================================
def align_pair(kp_curr, des_curr, kp_prev, des_prev, min_matches=10, ratio=0.75):
    """
    Compute homography H that maps a point in the current frame to its
    corresponding location in the previous frame.

    6a. Brute-Force Matching (Hamming distance)
        ORB descriptors are binary strings; Hamming distance counts the
        number of bit positions that differ.  knnMatch(k=2) returns each
        descriptor's two nearest neighbours so we can apply the ratio test.

    6b. Lowe's Ratio Test
        Accept match m only if  m.distance < ratio * n.distance,
        where n is the second-best match.  This filters out ambiguous
        matches where the query descriptor looks almost equally similar
        to two different database descriptors.

    6c. RANSAC Affine (Partial) — Rotation + Scale + Translation
        estimateAffinePartial2D fits a 4-DOF similarity transform:
          [ s*cos(θ)  -s*sin(θ)  tx ]
          [ s*sin(θ)   s*cos(θ)  ty ]
        This handles panning (tx), slight tilt (θ), and zoom (s) but does
        NOT fit perspective distortion.  For a panning video camera this
        is the correct model — using the full 8-DOF homography on near-
        identical frames lets tiny perspective coefficients accumulate and
        blow up the canvas after many compositions.

    src_pts come from kp_curr  (queryIdx → the image we passed first)
    dst_pts come from kp_prev  (trainIdx → the image we passed second)
    → H maps curr-frame coords → prev-frame coords  ✓
    """
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(des_curr, des_prev, k=2)

    good = [m for m, n in raw if m.distance < ratio * n.distance]

    if len(good) < min_matches:
        return None, 0, good

    src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_prev[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Affine partial: rotation + uniform scale + translation (4 DOF)
    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts,
                                           method=cv2.RANSAC,
                                           ransacReprojThreshold=5.0)
    if M is None:
        return None, 0, good

    inliers = int(mask.sum()) if mask is not None else 0
    # Promote 2×3 → 3×3 so it composes like a homography
    H = np.vstack([M, [0.0, 0.0, 1.0]])
    return H, inliers, good


# =============================================================================
# STEP 4 — Overlap Check
# =============================================================================
def estimate_overlap(H_curr_to_prev, w_c, h_c, w_p, h_p):
    """
    Project the four corners of the current frame through H into the
    previous frame's coordinate space.  Clip to prev-frame bounds and
    compute the intersection area as a fraction of the current frame area.

    25-40 % → healthy panning overlap (ideal)
    > 60 %  → camera barely moved (frames almost identical)
    < 15 %  → gap between frames (may not stitch)
    """
    corners = np.float32([[0,0],[w_c,0],[w_c,h_c],[0,h_c]]).reshape(-1, 1, 2)
    proj    = cv2.perspectiveTransform(corners, H_curr_to_prev).reshape(-1, 2)

    px = np.clip(proj[:, 0], 0, w_p)
    py = np.clip(proj[:, 1], 0, h_p)

    overlap_w    = max(0.0, px.max() - px.min())
    overlap_h    = max(0.0, py.max() - py.min())
    overlap_area = overlap_w * overlap_h
    return (overlap_area / (w_c * h_c)) * 100.0


# =============================================================================
# STEP 7 — Canvas Sizing (from cumulative homographies)
# =============================================================================
def compute_canvas(frames, H_cum):
    """
    Project every frame's four corners through its cumulative homography to
    find the total bounding box that covers the entire panorama.

    Returns (canvas_w, canvas_h, T) where T is the translation matrix that
    shifts all coordinates into positive (canvas) space.
    """
    all_corners = []
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(corners, H_cum[i]))

    pts   = np.concatenate(all_corners, axis=0)
    x_min = int(np.floor(pts[:, 0, 0].min()))
    y_min = int(np.floor(pts[:, 0, 1].min()))
    x_max = int(np.ceil (pts[:, 0, 0].max()))
    y_max = int(np.ceil (pts[:, 0, 1].max()))

    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0,      1]], dtype=np.float64)

    return x_max - x_min, y_max - y_min, T


# =============================================================================
# STEP 8 — Blending (distance-transform feather weights)
# =============================================================================
def blend_all_frames(frames, H_cum, canvas_w, canvas_h, T):
    """
    Warp every frame onto the canvas and blend them using distance-transform
    weights — the standard feathering technique in traditional panorama tools.

    For each warped frame:
      1. Build a binary mask of valid (non-black) pixels.
      2. Run distanceTransform: every valid pixel gets a weight equal to its
         Euclidean distance to the nearest border/black pixel.
         → Pixels near the frame centre have HIGH weight (they are reliable).
         → Pixels near the seam edge have LOW weight (they may be distorted).
      3. Accumulate   accum  += warped * weight
                      weight_sum += weight
      4. After all frames: final pixel = accum / weight_sum

    This weighted average automatically creates a smooth, natural transition
    across every seam — no hard edges, no visible stitching lines.
    """
    accum      = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    weight_sum = np.zeros((canvas_h, canvas_w),    dtype=np.float64)

    for i, frame in enumerate(frames):
        warped = cv2.warpPerspective(frame, T @ H_cum[i], (canvas_w, canvas_h))

        mask = (warped.sum(axis=2) > 0).astype(np.uint8) * 255
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float64)

        accum      += warped.astype(np.float64) * dist[:, :, np.newaxis]
        weight_sum += dist

        if (i + 1) % 5 == 0:
            print(f"    Blended {i+1}/{len(frames)} frames...")

    weight_sum = np.maximum(weight_sum, 1e-6)
    return (accum / weight_sum[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)


# =============================================================================
# STEP 9 — Cropping
# =============================================================================
def crop_black_borders(image):
    """
    Find the bounding rectangle of all non-black content and crop to it.
    Removes the black regions introduced by warpPerspective at canvas edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    return image[y:y+h, x:x+w]


# =============================================================================
# STEP 10 — Final Enhancement
# =============================================================================
def enhance_image(image):
    """
    10a. CLAHE on LAB luminance channel — improves local contrast without
         shifting colours (works on L only, leaves a and b untouched).
    10b. Unsharp mask — sharpened = 1.5 * image - 0.5 * blurred
         Amplifies high-frequency edges (fine detail, lines, text).
    10c. convertScaleAbs(alpha, beta) — gentle contrast stretch + brightness.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    enhanced  = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    blurred   = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    return cv2.convertScaleAbs(sharpened, alpha=1.05, beta=5)


# =============================================================================
# STEP 11 — Save
# =============================================================================
def save_panorama(image, output_path):
    ext    = os.path.splitext(output_path)[1].lower()
    params = ([cv2.IMWRITE_PNG_COMPRESSION, 3]  if ext == ".png"
              else [cv2.IMWRITE_JPEG_QUALITY, 95])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, image, params)
    h, w = image.shape[:2]
    print(f"  Saved  : {output_path}")
    print(f"  Size   : {w} x {h} px")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def build_panorama(frames_dir, output_path, step=3, max_features=3000, min_matches=10):
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No frames found in: {frames_dir}")

    paths = paths[::step]

    print(f"\n{'='*60}")
    print(f" Traditional Panorama Pipeline")
    print(f"{'='*60}")
    print(f" Frames    : {len(paths)}  (every {step}th frame)")
    print(f" Features  : {max_features}  min-matches={min_matches}")
    print(f"{'='*60}\n")

    # Load all frames into memory
    frames = [cv2.imread(p) for p in paths]
    frames = [f for f in frames if f is not None]
    N = len(frames)

    # ── Steps 5 & 6: pairwise H (frame[i] → frame[i-1]) ─────────────────────
    print("[ Phase 1 ] Feature detection & pairwise alignment")
    H_pair = [np.eye(3, dtype=np.float64)]   # frame 0 = reference (identity)

    for i in range(1, N):
        prev, curr = frames[i-1], frames[i]
        h_p, w_p   = prev.shape[:2]
        h_c, w_c   = curr.shape[:2]

        # Step 5 — detect on both frames
        kp_prev, des_prev = detect_features(prev, max_features)
        kp_curr, des_curr = detect_features(curr, max_features)

        if des_prev is None or des_curr is None:
            print(f"  [{i:03d}] No descriptors — identity used")
            H_pair.append(np.eye(3, dtype=np.float64))
            continue

        # Step 6 — align curr → prev
        H, inliers, good = align_pair(kp_curr, des_curr, kp_prev, des_prev, min_matches)

        if H is None:
            print(f"  [{i:03d}] Too few matches ({len(good)}) — identity used")
            H_pair.append(np.eye(3, dtype=np.float64))
            continue

        # Step 4 — overlap check
        ov  = estimate_overlap(H, w_c, h_c, w_p, h_p)
        tag = "OK" if 15 <= ov <= 70 else "warn"
        print(f"  [{i:03d}] matches={len(good):4d}  inliers={inliers:4d}  overlap={ov:5.1f}% [{tag}]")
        H_pair.append(H)

    # Cumulative homographies — all frames in frame-0 space
    H_cum = [np.eye(3, dtype=np.float64)]
    for i in range(1, N):
        H_cum.append(H_cum[i-1] @ H_pair[i])

    # Step 7 — compute canvas size from all projected corners
    print(f"\n[ Phase 2 ] Warping & blending")
    canvas_w, canvas_h, T = compute_canvas(frames, H_cum)
    print(f"  Canvas     : {canvas_w} x {canvas_h} px")

    # Guard against degenerate canvas (e.g. bad H composition)
    MAX_DIM = 50000
    if canvas_w > MAX_DIM or canvas_h > MAX_DIM:
        raise RuntimeError(
            f"Canvas {canvas_w}x{canvas_h} is too large — "
            "transforms may have diverged. Try a larger --step value."
        )

    # Step 8 — warp all frames + distance-transform feather blend
    panorama = blend_all_frames(frames, H_cum, canvas_w, canvas_h, T)
    print(f"  Blended    : {panorama.shape[1]} x {panorama.shape[0]} px")

    # Step 9 — crop black borders
    print(f"\n[ Phase 3 ] Post-processing")
    panorama = crop_black_borders(panorama)
    print(f"  Step 9  Cropped    : {panorama.shape[1]} x {panorama.shape[0]} px")

    # Step 10 — enhance
    panorama = enhance_image(panorama)
    print(f"  Step 10 Enhanced   : {panorama.shape[1]} x {panorama.shape[0]} px")

    # Step 11 — save
    print(f"  Step 11 Saving...")
    save_panorama(panorama, output_path)
    print(f"{'='*60}\n")
    return panorama


def main():
    parser = argparse.ArgumentParser(
        description="Traditional panorama — Steps 4-11"
    )
    parser.add_argument("frames_dir",
                        help="Directory of extracted frames")
    parser.add_argument("-o", "--output", default="panorama.jpg",
                        help="Output path (default: panorama.jpg)")
    parser.add_argument("-s", "--step", type=int, default=3,
                        help="Use every Nth frame (default: 3)")
    parser.add_argument("-f", "--features", type=int, default=3000,
                        help="Max ORB keypoints per image (default: 3000)")
    parser.add_argument("-m", "--min-matches", type=int, default=10,
                        help="Min inlier matches to accept a pair (default: 10)")
    args = parser.parse_args()

    build_panorama(
        frames_dir  = args.frames_dir,
        output_path = args.output,
        step        = args.step,
        max_features= args.features,
        min_matches = args.min_matches,
    )


if __name__ == "__main__":
    main()
