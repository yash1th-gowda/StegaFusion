"""
StegaFusion Video Frame Reconstruction Test

Validates that extracted PNG frames can be reconstructed
into a playable video while preserving:

    - Frame count
    - Resolution
    - FPS
"""

from pathlib import Path

import cv2

from modules.video.frame_reconstruct import reconstruct_video
from config.config import PathConfig


# ==========================================================
# PATHS
# ==========================================================

FRAME_DIR = PathConfig.FRAME_DIR

OUTPUT_VIDEO = (
    PathConfig.OUTPUT_DIR /
    "reconstructed_test.mp4"
)

SOURCE_VIDEO = (
    PathConfig.COVER_VIDEO_DIR /
    "sample.mp4"
)


# ==========================================================
# MAIN TEST
# ==========================================================

def main():

    print("=" * 70)
    print("StegaFusion Video Frame Reconstruction Test")
    print("=" * 70)

    # ------------------------------------------------------
    # Validate input video
    # ------------------------------------------------------

    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(
            f"Source video not found: {SOURCE_VIDEO}"
        )

    # ------------------------------------------------------
    # Read original video properties
    # ------------------------------------------------------

    source = cv2.VideoCapture(
        str(SOURCE_VIDEO)
    )

    if not source.isOpened():
        raise RuntimeError(
            "Unable to open source video."
        )

    source_fps = source.get(
        cv2.CAP_PROP_FPS
    )

    source_frame_count = int(
        source.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    source_width = int(
        source.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    source_height = int(
        source.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    source.release()

    print()
    print(
        f"Source Video      : "
        f"{SOURCE_VIDEO}"
    )

    print(
        f"Source Frames     : "
        f"{source_frame_count}"
    )

    print(
        f"Source FPS        : "
        f"{source_fps}"
    )

    print(
        f"Source Resolution : "
        f"{source_width} x "
        f"{source_height}"
    )

    # ------------------------------------------------------
    # Validate extracted frames
    # ------------------------------------------------------

    frame_files = sorted(
        FRAME_DIR.glob("*.png")
    )

    if not frame_files:
        raise FileNotFoundError(
            f"No PNG frames found in {FRAME_DIR}"
        )

    print()
    print(
        f"Extracted Frames  : "
        f"{len(frame_files)}"
    )

    if len(frame_files) != source_frame_count:
        raise RuntimeError(
            "Extracted frame count does not "
            "match the source video."
        )

    # ------------------------------------------------------
    # Reconstruct video
    # ------------------------------------------------------

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    written_frames = reconstruct_video(
        FRAME_DIR,
        OUTPUT_VIDEO,
        fps=source_fps
    )

    print()
    print(
        f"Frames Written    : "
        f"{written_frames}"
    )

    print(
        f"Output Video      : "
        f"{OUTPUT_VIDEO}"
    )

    # ------------------------------------------------------
    # Validate output file
    # ------------------------------------------------------

    if not OUTPUT_VIDEO.exists():
        raise RuntimeError(
            "Reconstructed video was not created."
        )

    if OUTPUT_VIDEO.stat().st_size == 0:
        raise RuntimeError(
            "Reconstructed video is empty."
        )

    # ------------------------------------------------------
    # Open reconstructed video
    # ------------------------------------------------------

    output = cv2.VideoCapture(
        str(OUTPUT_VIDEO)
    )

    if not output.isOpened():
        raise RuntimeError(
            "Unable to open reconstructed video."
        )

    output_fps = output.get(
        cv2.CAP_PROP_FPS
    )

    output_frame_count = int(
        output.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    output_width = int(
        output.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    output_height = int(
        output.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    # ------------------------------------------------------
    # Read first frame
    # ------------------------------------------------------

    success, first_frame = output.read()

    output.release()

    if not success:
        raise RuntimeError(
            "Unable to read first frame "
            "from reconstructed video."
        )

    # ------------------------------------------------------
    # Display output properties
    # ------------------------------------------------------

    print()
    print(
        f"Output Frames     : "
        f"{output_frame_count}"
    )

    print(
        f"Output FPS        : "
        f"{output_fps}"
    )

    print(
        f"Output Resolution : "
        f"{output_width} x "
        f"{output_height}"
    )

    print(
        f"First Frame Shape : "
        f"{first_frame.shape}"
    )

    print(
        f"First Frame Dtype : "
        f"{first_frame.dtype}"
    )

    # ------------------------------------------------------
    # Validate frame count
    # ------------------------------------------------------

    if output_frame_count != source_frame_count:
        raise RuntimeError(
            "Output frame count does not "
            "match source frame count."
        )

    # ------------------------------------------------------
    # Validate resolution
    # ------------------------------------------------------

    if (
        output_width != source_width
        or
        output_height != source_height
    ):
        raise RuntimeError(
            "Output resolution does not "
            "match source resolution."
        )

    # ------------------------------------------------------
    # Validate FPS
    # ------------------------------------------------------

    if abs(output_fps - source_fps) > 0.1:
        raise RuntimeError(
            "Output FPS differs significantly "
            "from source FPS."
        )

    # ------------------------------------------------------
    # Final result
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "VIDEO FRAME RECONSTRUCTION TEST PASSED!"
    )
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()