import cv2
import os
import argparse


def split_video_to_frames(video_path, output_dir, frame_interval=1, target_fps=None, image_format="jpg", quality=95):
    """
    Split a video into individual frames.

    Args:
        video_path:     Path to the input video file.
        output_dir:     Directory where extracted frames will be saved.
        frame_interval: Extract every Nth frame (1 = every frame). Ignored when target_fps is set.
        target_fps:     Desired output frame rate. Overrides frame_interval.
        image_format:   Output image format ('jpg', 'png', 'bmp').
        quality:        JPEG quality (0-100), ignored for PNG/BMP.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if target_fps is not None:
        if target_fps > fps:
            print(f"Warning: target FPS ({target_fps}) exceeds video FPS ({fps:.2f}). Using video FPS.")
            target_fps = fps
        frame_interval = max(1, round(fps / target_fps))

    print(f"Video info:")
    print(f"  File        : {video_path}")
    print(f"  Resolution  : {width}x{height}")
    print(f"  Native FPS  : {fps:.2f}")
    print(f"  Target FPS  : {target_fps if target_fps else fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration    : {total_frames / fps:.2f}s")
    print(f"  Interval    : every {frame_interval} frame(s)")
    print(f"  Output dir  : {output_dir}\n")

    encode_params = []
    if image_format.lower() == "jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif image_format.lower() == "png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]  # 0-9, lower = faster/larger

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            filename = f"frame_{frame_idx:06d}_t{timestamp_ms:.0f}ms.{image_format}"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame, encode_params)
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"  Saved {saved_count} frames (frame index {frame_idx})...")

        frame_idx += 1

    cap.release()
    print(f"\nDone. Saved {saved_count} frames to '{output_dir}'")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Split a video into frames using OpenCV")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("-o", "--output", default="frames", help="Output directory (default: ./frames)")
    parser.add_argument("-n", "--interval", type=int, default=1,
                        help="Extract every Nth frame (default: 1 = every frame)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Target output frame rate (e.g. 30). Overrides --interval.")
    parser.add_argument("-f", "--format", choices=["jpg", "png", "bmp"], default="jpg",
                        help="Output image format (default: jpg)")
    parser.add_argument("-q", "--quality", type=int, default=95,
                        help="JPEG quality 0-100 (default: 95, ignored for PNG/BMP)")
    args = parser.parse_args()

    split_video_to_frames(
        video_path=args.video,
        output_dir=args.output,
        frame_interval=args.interval,
        target_fps=args.fps,
        image_format=args.format,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
