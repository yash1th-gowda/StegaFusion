"""
StegaFusion Video Coefficient Group Stability Diagnostic

Purpose:
    Determine whether groups of DWT LH coefficients can preserve
    an embedding signal through lossy MP4V compression.

IMPORTANT:
    This is a diagnostic only.
    It does NOT modify production embedding code.
"""

from pathlib import Path

import cv2
import numpy as np

from config.config import PathConfig
from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

FRAME_INDEX = 40

GROUP_SIZES = [
    8,
    16,
    32,
    64,
    128,
]

DELTAS = [
    2.0,
    4.0,
    8.0,
]

NUM_GROUPS = 50

OUTPUT_DIR = (
    PathConfig.OUTPUT_DIR
    / "video_coefficient_group"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD FRAME
# ==========================================================

def load_source_frame():

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
# CREATE MP4
# ==========================================================

def write_mp4(
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
            "Unable to create MP4 writer."
        )

    for frame in frames:
        writer.write(frame)

    writer.release()


# ==========================================================
# READ FIRST FRAME
# ==========================================================

def read_first_frame(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open encoded video: "
            f"{video_path}"
        )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            "Unable to decode MP4."
        )

    return frame


# ==========================================================
# BUILD GROUPS
# ==========================================================

def build_groups(
    shape,
    group_size,
    count,
):

    height, width = shape

    total = height * width

    usable = (
        total // group_size
    ) * group_size

    indices = np.arange(
        usable
    )

    groups = []

    for i in range(count):

        start = (
            i * group_size
        )

        end = (
            start + group_size
        )

        if end > usable:
            break

        groups.append(
            indices[start:end]
        )

    return groups


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "StegaFusion Video Coefficient "
        "Group Stability Diagnostic"
    )
    print("=" * 70)

    source_frame = load_source_frame()

    blue = source_frame[:, :, 0]

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

    original_bands = apply_dwt(
        blue
    )

    original_lh = (
        original_bands["LH"]
        .copy()
    )

    print(
        f"LH Shape    : "
        f"{original_lh.shape}"
    )

    # ------------------------------------------------------
    # Test each group size
    # ------------------------------------------------------

    for group_size in GROUP_SIZES:

        print()
        print("-" * 70)

        print(
            f"GROUP SIZE : {group_size}"
        )

        groups = build_groups(
            original_lh.shape,
            group_size,
            NUM_GROUPS,
        )

        print(
            f"Groups     : {len(groups)}"
        )

        for delta in DELTAS:

            print()
            print(
                f"Testing Delta : {delta}"
            )

            # --------------------------------------------------
            # Embed into groups
            # --------------------------------------------------

            test_lh = (
                original_lh.copy()
            )

            embedded_positions = []

            flat = test_lh.ravel()

            for group in groups:

                flat[group] += delta

                embedded_positions.append(
                    group.copy()
                )

            test_lh = (
                flat.reshape(
                    original_lh.shape
                )
            )

            # --------------------------------------------------
            # Reconstruct spatial image
            # --------------------------------------------------

            bands = {
                key: value.copy()
                for key, value
                in original_bands.items()
            }

            bands["LH"] = test_lh

            reconstructed = (
                apply_inverse_dwt(
                    bands
                )
            )

            # Important:
            # rounded conversion, not truncation.

            reconstructed = np.round(
                reconstructed
            )

            reconstructed = np.clip(
                reconstructed,
                0,
                255,
            ).astype(
                np.uint8
            )

            stego_frame = (
                source_frame.copy()
            )

            stego_frame[:, :, 0] = (
                reconstructed
            )

            # --------------------------------------------------
            # Direct spatial distortion
            # --------------------------------------------------

            spatial_diff = (
                stego_frame[:, :, 0].astype(
                    np.int16
                )
                -
                source_frame[:, :, 0].astype(
                    np.int16
                )
            )

            changed_pixels = np.count_nonzero(
                spatial_diff
            )

            mean_spatial = np.mean(
                np.abs(
                    spatial_diff
                )
            )

            max_spatial = np.max(
                np.abs(
                    spatial_diff
                )
            )

            # --------------------------------------------------
            # Create video
            #
            # Repeat the same stego frame so that
            # temporal effects don't influence the
            # coefficient measurement.
            # --------------------------------------------------

            frames = [
                stego_frame
                for _ in range(10)
            ]

            output_path = (
                OUTPUT_DIR
                / (
                    f"group_{group_size}"
                    f"_delta_{int(delta)}.mp4"
                )
            )

            write_mp4(
                frames,
                output_path,
            )

            decoded = (
                read_first_frame(
                    output_path
                )
            )

            decoded_blue = (
                decoded[:, :, 0]
            )

            # --------------------------------------------------
            # Recover DWT
            # --------------------------------------------------

            decoded_bands = (
                apply_dwt(
                    decoded_blue
                )
            )

            recovered_lh = (
                decoded_bands["LH"]
            )

            # --------------------------------------------------
            # Measure group recovery
            # --------------------------------------------------

            recovered_deltas = []

            errors = []

            successful_groups = 0

            for group in (
                embedded_positions
            ):

                original_values = (
                    original_lh.ravel()[group]
                )

                recovered_values = (
                    recovered_lh.ravel()[group]
                )

                original_mean = np.mean(
                    original_values
                )

                recovered_mean = np.mean(
                    recovered_values
                )

                recovered_delta = (
                    recovered_mean
                    - original_mean
                )

                error = (
                    recovered_delta
                    - delta
                )

                recovered_deltas.append(
                    recovered_delta
                )

                errors.append(
                    error
                )

                # Count a group as successfully
                # recovered when at least half
                # of the intended delta survives.

                if (
                    recovered_delta
                    >= delta * 0.5
                ):

                    successful_groups += 1

            recovered_deltas = np.array(
                recovered_deltas
            )

            errors = np.array(
                errors
            )

            # --------------------------------------------------
            # Statistics
            # --------------------------------------------------

            mean_recovered = np.mean(
                recovered_deltas
            )

            median_recovered = np.median(
                recovered_deltas
            )

            mean_error = np.mean(
                np.abs(errors)
            )

            max_error = np.max(
                np.abs(errors)
            )

            success_rate = (
                successful_groups
                /
                len(groups)
                *
                100
            )

            print()
            print(
                f"Changed Pixels : "
                f"{changed_pixels}"
            )

            print(
                f"Max Spatial   : "
                f"{max_spatial}"
            )

            print(
                f"Mean Spatial  : "
                f"{mean_spatial:.8f}"
            )

            print(
                f"Mean Recovered Delta : "
                f"{mean_recovered:.4f}"
            )

            print(
                f"Median Recovered     : "
                f"{median_recovered:.4f}"
            )

            print(
                f"Mean Absolute Error  : "
                f"{mean_error:.4f}"
            )

            print(
                f"Maximum Error        : "
                f"{max_error:.4f}"
            )

            print(
                f"Groups Recovering "
                f">=50% : "
                f"{successful_groups}/"
                f"{len(groups)} "
                f"({success_rate:.1f}%)"
            )

    print()
    print("=" * 70)
    print(
        "COEFFICIENT GROUP DIAGNOSTIC COMPLETE"
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