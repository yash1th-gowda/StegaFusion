"""
StegaFusion MP4 Spatial Signal Stability Diagnostic

Research-only diagnostic.

Tests whether a controlled spatial-domain intensity signal
survives MP4V encoding/decoding.

This does NOT modify production steganography code.
"""

from pathlib import Path

import cv2
import numpy as np

from config.config import PathConfig


# ==========================================================
# CONFIGURATION
# ==========================================================

FRAME_INDEX = 40

SIGNAL_SIZES = [
    8,
    16,
    32,
    64,
]

DELTAS = [
    2,
    4,
    8,
    16,
]

BLOCK_SIZE = 8

OUTPUT_DIR = (
    PathConfig.OUTPUT_DIR
    / "video_spatial_stability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD SOURCE FRAME
# ==========================================================

def load_frame():

    video_path = (
        PathConfig.COVER_VIDEO_DIR
        / "sample.mp4"
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Unable to open video: {video_path}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        FRAME_INDEX
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            f"Unable to read frame {FRAME_INDEX}"
        )

    return frame


# ==========================================================
# WRITE MP4
# ==========================================================

def write_video(
    frames,
    output_path,
    fps=23.976,
):

    height, width = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Unable to create MP4."
        )

    for frame in frames:
        writer.write(frame)

    writer.release()


# ==========================================================
# READ FIRST FRAME
# ==========================================================

def read_video_frame(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open {video_path}"
        )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            "Unable to decode MP4."
        )

    return frame


# ==========================================================
# CREATE TEST REGION
# ==========================================================

def create_signal(
    blue,
    size,
    delta,
):

    modified = blue.copy()

    height, width = blue.shape

    # Place the signal in the central region.
    y0 = (
        height // 2
        - size // 2
    )

    x0 = (
        width // 2
        - size // 2
    )

    y1 = y0 + size
    x1 = x0 + size

    modified[
        y0:y1,
        x0:x1
    ] = np.clip(
        modified[
            y0:y1,
            x0:x1
        ].astype(np.int16)
        + delta,
        0,
        255
    ).astype(np.uint8)

    return modified


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "StegaFusion MP4 Spatial Signal "
        "Stability Diagnostic"
    )
    print("=" * 70)

    source = load_frame()

    blue = source[:, :, 0]

    print()
    print(
        f"Frame       : {FRAME_INDEX}"
    )

    print(
        f"Blue Shape  : {blue.shape}"
    )

    print(
        f"Blue Range  : "
        f"{blue.min()} -> {blue.max()}"
    )

    print()

    print(
        "SIZE  DELTA  "
        "MEAN_RECOVERED  "
        "MEDIAN_RECOVERED  "
        "MIN  MAX  "
        "RECOVERY_%"
    )

    print("-" * 70)

    for size in SIGNAL_SIZES:

        for delta in DELTAS:

            modified_blue = create_signal(
                blue,
                size,
                delta,
            )

            stego = source.copy()

            stego[:, :, 0] = (
                modified_blue
            )

            # --------------------------------------------------
            # Spatial distortion
            # --------------------------------------------------

            spatial_diff = (
                modified_blue.astype(
                    np.int16
                )
                -
                blue.astype(
                    np.int16
                )
            )

            # --------------------------------------------------
            # Encode
            # --------------------------------------------------

            output_path = (
                OUTPUT_DIR
                /
                (
                    f"size_{size}"
                    f"_delta_{delta}.mp4"
                )
            )

            frames = [
                stego
                for _ in range(10)
            ]

            write_video(
                frames,
                output_path,
            )

            decoded = read_video_frame(
                output_path
            )

            decoded_blue = (
                decoded[:, :, 0]
            )

            # --------------------------------------------------
            # Measure only signal region
            # --------------------------------------------------

            height, width = blue.shape

            y0 = (
                height // 2
                - size // 2
            )

            x0 = (
                width // 2
                - size // 2
            )

            y1 = y0 + size
            x1 = x0 + size

            original_region = (
                blue[
                    y0:y1,
                    x0:x1
                ].astype(
                    np.float32
                )
            )

            recovered_region = (
                decoded_blue[
                    y0:y1,
                    x0:x1
                ].astype(
                    np.float32
                )
            )

            recovered_delta = (
                recovered_region
                -
                original_region
            )

            mean_recovered = np.mean(
                recovered_delta
            )

            median_recovered = np.median(
                recovered_delta
            )

            min_recovered = np.min(
                recovered_delta
            )

            max_recovered = np.max(
                recovered_delta
            )

            # A pixel counts as successfully
            # recovering the signal if at least
            # half the intended delta survives.

            successful = np.count_nonzero(
                recovered_delta
                >=
                (delta * 0.5)
            )

            total = (
                size * size
            )

            recovery_percent = (
                successful
                /
                total
                *
                100
            )

            print(
                f"{size:4d}  "
                f"{delta:5d}  "
                f"{mean_recovered:14.4f}  "
                f"{median_recovered:16.4f}  "
                f"{min_recovered:4.1f}  "
                f"{max_recovered:4.1f}  "
                f"{recovery_percent:9.2f}"
            )

    print()
    print("=" * 70)
    print(
        "SPATIAL MP4 DIAGNOSTIC COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No production embedding code was modified."
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()