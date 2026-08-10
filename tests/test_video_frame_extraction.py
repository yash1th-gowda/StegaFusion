"""
StegaFusion Video Frame Extraction Test

Validates that a complete video can be opened and all
frames can be extracted correctly.
"""

from pathlib import Path

import cv2

from modules.video.frame_extract import extract_frames
from config.config import PathConfig


VIDEO_FILE = (
    PathConfig.COVER_VIDEO_DIR /
    "sample.mp4"
)


def main():

    print("=" * 70)
    print("StegaFusion Video Frame Extraction Test")
    print("=" * 70)

    # ------------------------------------------------------
    # Validate video
    # ------------------------------------------------------

    if not VIDEO_FILE.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_FILE}"
        )

    print()
    print(
        f"Video : {VIDEO_FILE}"
    )

    # ------------------------------------------------------
    # Extract frames
    # ------------------------------------------------------

    total_frames, fps, resolution = extract_frames(
        VIDEO_FILE
    )

    # ------------------------------------------------------
    # Display information
    # ------------------------------------------------------

    print()
    print(
        f"Frames     : {total_frames}"
    )

    print(
        f"FPS        : {fps}"
    )

    print(
        f"Resolution : {resolution}"
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    if total_frames <= 0:
        raise RuntimeError(
            "No frames were extracted."
        )

    if fps <= 0:
        raise RuntimeError(
            "Invalid FPS detected."
        )

    if resolution[0] <= 0 or resolution[1] <= 0:
        raise RuntimeError(
            "Invalid video resolution."
        )

    # ------------------------------------------------------
    # Verify first and last frame
    # ------------------------------------------------------

    first_frame = (
        PathConfig.FRAME_DIR /
        "frame_00000.png"
    )

    last_frame = (
        PathConfig.FRAME_DIR /
        f"frame_{total_frames - 1:05d}.png"
    )

    if not first_frame.exists():
        raise RuntimeError(
            f"First frame was not created: {first_frame}"
        )

    if not last_frame.exists():
        raise RuntimeError(
            f"Last frame was not created: {last_frame}"
        )

    # ------------------------------------------------------
    # Read first frame
    # ------------------------------------------------------

    image = cv2.imread(
        str(first_frame),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            "Unable to read extracted first frame."
        )

    print()
    print(
        f"First Frame Shape : {image.shape}"
    )

    print(
        f"First Frame Dtype : {image.dtype}"
    )

    print(
        f"First Frame Range : "
        f"{image.min()} -> {image.max()}"
    )

    # ------------------------------------------------------
    # Final result
    # ------------------------------------------------------

    print()
    print(
        "VIDEO FRAME EXTRACTION TEST PASSED!"
    )


if __name__ == "__main__":
    main()