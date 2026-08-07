"""
------------------------------------------------------------
StegaFusion Frame Extraction Module
------------------------------------------------------------
Extracts all frames from a video and stores them as PNG images.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path
import cv2

from config.config import PathConfig


# ==========================================================
# EXTRACT VIDEO FRAMES
# ==========================================================

def extract_frames(video_path: Path) -> tuple[int, float, tuple]:
    """
    Extracts every frame from a video.

    Args:
        video_path (Path): Input video.

    Returns:
        tuple:
            Total Frames
            FPS
            Resolution
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Unable to open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    resolution = (width, height)

    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_name = f"frame_{frame_count:05d}.png"

        frame_path = PathConfig.FRAME_DIR / frame_name

        cv2.imwrite(str(frame_path), frame)

        frame_count += 1

    cap.release()

    return frame_count, fps, resolution


# ==========================================================
# TESTING
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Frame Extraction Test")
    print("=" * 70)

    video = PathConfig.COVER_VIDEO_DIR / "sample.mp4"

    total_frames, fps, resolution = extract_frames(video)

    print(f"Video           : {video.name}")
    print(f"Frames          : {total_frames}")
    print(f"FPS             : {fps}")
    print(f"Resolution      : {resolution}")

    print("\nFrame Extraction Completed Successfully!")