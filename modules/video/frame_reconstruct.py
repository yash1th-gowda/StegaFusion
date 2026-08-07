"""
------------------------------------------------------------
StegaFusion Frame Reconstruction
------------------------------------------------------------
Rebuilds a video from processed frames.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path
import cv2

from config.config import PathConfig


# ==========================================================
# REBUILD VIDEO
# ==========================================================

def reconstruct_video(
    frame_folder: Path,
    output_video: Path,
    fps: float = 24.0
):
    """
    Reconstruct a video from image frames.
    """

    frames = sorted(frame_folder.glob("*.png"))

    if not frames:
        raise FileNotFoundError(
            "No frames found."
        )

    first = cv2.imread(str(frames[0]))

    height, width = first.shape[:2]

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    for frame in frames:

        image = cv2.imread(str(frame))

        writer.write(image)

    writer.release()

    return len(frames)


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Video Reconstruction")
    print("=" * 70)

    frame_dir = PathConfig.TEMP_DIR / "frames"

    output = (
        PathConfig.OUTPUT_DIR /
        "reconstructed_video.mp4"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    count = reconstruct_video(
        frame_dir,
        output,
        fps=24
    )

    print()

    print(f"Frames Written : {count}")

    print(f"Saved : {output}")

    print()

    print("Video Reconstruction Successful!")