"""
StegaFusion MP4 Coefficient Stability Diagnostic

Purpose:
    Determine which LH DWT coefficients remain stable after
    the current video encoding/decoding pipeline.

This is a diagnostic only.
It does NOT modify production embedding logic.

Pipeline:

    Source Frame
        ↓
    Blue Channel
        ↓
    DWT
        ↓
    Modify one LH coefficient at a time
        ↓
    Rounded inverse DWT
        ↓
    Video encoding
        ↓
    Video decoding
        ↓
    Blue Channel
        ↓
    DWT
        ↓
    Compare coefficient
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
# SETTINGS
# ==========================================================

FRAME_INDEX = 40

# Number of candidate coefficients to test.
# We deliberately keep this small for the first diagnostic.
CANDIDATE_COUNT = 100

# Test modification.
TEST_DELTA = 8

# Number of frames in the temporary test video.
# Using the same video pipeline structure as the existing tests.
TOTAL_FRAMES = 197

FPS = 23.976023976023978

OUTPUT_VIDEO = (
    PathConfig.OUTPUT_DIR /
    "video_coefficient_stability.mp4"
)

TEMP_DIR = (
    PathConfig.TEMP_DIR /
    "coefficient_stability"
)


# ==========================================================
# HELPERS
# ==========================================================

def reconstruct_uint8(bands):
    """
    Perform inverse DWT and convert the floating-point
    reconstruction safely to uint8.

    IMPORTANT:
    Round before clipping/conversion.
    """

    reconstructed = apply_inverse_dwt(bands)

    reconstructed = np.round(
        reconstructed
    ).clip(
        0,
        255
    ).astype(np.uint8)

    return reconstructed


def load_source_frame():
    """
    Load the selected source frame.
    """

    frame_path = (
        PathConfig.FRAME_DIR /
        f"frame_{FRAME_INDEX:05d}.png"
    )

    if not frame_path.exists():
        raise FileNotFoundError(
            f"Frame not found: {frame_path}"
        )

    frame = cv2.imread(
        str(frame_path),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise RuntimeError(
            f"Unable to read frame: {frame_path}"
        )

    return frame


def make_test_frame(
    source_frame,
    position,
    delta
):
    """
    Modify exactly one LH coefficient.
    """

    blue = source_frame[:, :, 0]

    bands = apply_dwt(blue)

    row, col = position

    modified_bands = {
        key: value.copy()
        for key, value in bands.items()
    }

    modified_bands["LH"][row, col] += delta

    reconstructed_blue = reconstruct_uint8(
        modified_bands
    )

    test_frame = source_frame.copy()

    test_frame[:, :, 0] = reconstructed_blue

    return test_frame


def write_test_video(
    source_frame,
    positions
):
    """
    Create one MP4 containing one test frame for each
    candidate coefficient.

    Each frame contains a modification at one candidate
    coefficient.
    """

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (
            source_frame.shape[1],
            source_frame.shape[0]
        )
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Unable to open video writer: "
            f"{OUTPUT_VIDEO}"
        )

    for position in positions:

        test_frame = make_test_frame(
            source_frame,
            position,
            TEST_DELTA
        )

        writer.write(test_frame)

    writer.release()


def read_decoded_frames():
    """
    Read all frames from the generated MP4.
    """

    cap = cv2.VideoCapture(
        str(OUTPUT_VIDEO)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open output video: "
            f"{OUTPUT_VIDEO}"
        )

    decoded = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        decoded.append(frame)

    cap.release()

    return decoded


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("StegaFusion MP4 Coefficient Stability Diagnostic")
    print("=" * 70)

    print()
    print(
        f"Testing Frame      : {FRAME_INDEX}"
    )

    print(
        f"Test Delta         : {TEST_DELTA}"
    )

    print(
        f"Candidate Count    : {CANDIDATE_COUNT}"
    )

    # ------------------------------------------------------
    # Load source
    # ------------------------------------------------------

    source_frame = load_source_frame()

    blue = source_frame[:, :, 0]

    print()
    print("Source Frame")

    print(
        f"Shape              : {blue.shape}"
    )

    print(
        f"Range              : "
        f"{blue.min()} -> {blue.max()}"
    )

    # ------------------------------------------------------
    # Original DWT
    # ------------------------------------------------------

    original_bands = apply_dwt(
        blue
    )

    lh = original_bands["LH"]

    print(
        f"LH Shape           : {lh.shape}"
    )

    # ------------------------------------------------------
    # Select candidate coefficients
    #
    # Start with the first 100 coefficients in scan order.
    # Later we can expand this diagnostic.
    # ------------------------------------------------------

    positions = []

    rows, cols = lh.shape

    for row in range(rows):

        for col in range(cols):

            positions.append(
                (row, col)
            )

            if len(positions) >= CANDIDATE_COUNT:
                break

        if len(positions) >= CANDIDATE_COUNT:
            break

    print()
    print(
        f"Candidates Selected: {len(positions)}"
    )

    # ------------------------------------------------------
    # Create MP4
    # ------------------------------------------------------

    print()
    print("Creating MP4 test video...")

    write_test_video(
        source_frame,
        positions
    )

    print(
        f"Video Saved        : {OUTPUT_VIDEO}"
    )

    # ------------------------------------------------------
    # Decode MP4
    # ------------------------------------------------------

    print()
    print("Decoding MP4...")

    decoded_frames = read_decoded_frames()

    print(
        f"Decoded Frames     : "
        f"{len(decoded_frames)}"
    )

    if len(decoded_frames) != len(positions):

        raise RuntimeError(
            "Decoded frame count does not match "
            "candidate count."
        )

    # ------------------------------------------------------
    # Analyze each coefficient
    # ------------------------------------------------------

    results = []

    print()
    print(
        "Analyzing coefficient stability..."
    )

    for index, position in enumerate(positions):

        row, col = position

        original_value = float(
            lh[row, col]
        )

        embedded_value = (
            original_value +
            TEST_DELTA
        )

        decoded_frame = (
            decoded_frames[index]
        )

        decoded_blue = (
            decoded_frame[:, :, 0]
        )

        decoded_bands = apply_dwt(
            decoded_blue
        )

        recovered_value = float(
            decoded_bands["LH"][row, col]
        )

        recovery_error = (
            recovered_value -
            embedded_value
        )

        recovered_delta = (
            recovered_value -
            original_value
        )

        results.append(
            {
                "position": position,
                "original": original_value,
                "embedded": embedded_value,
                "recovered": recovered_value,
                "recovered_delta": recovered_delta,
                "error": recovery_error,
            }
        )

    # ------------------------------------------------------
    # Sort by absolute recovery error
    # ------------------------------------------------------

    results.sort(
        key=lambda item:
        abs(item["error"])
    )

    # ------------------------------------------------------
    # Display results
    # ------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "TOP 30 MOST STABLE COEFFICIENTS"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Pos':>12} "
        f"{'Original':>10} "
        f"{'Embedded':>10} "
        f"{'Recovered':>10} "
        f"{'Delta':>10} "
        f"{'Error':>10}"
    )

    print(
        "-" * 100
    )

    for item in results[:30]:

        row, col = item["position"]

        print(
            f"({row:4d},{col:4d}) "
            f"{item['original']:10.3f} "
            f"{item['embedded']:10.3f} "
            f"{item['recovered']:10.3f} "
            f"{item['recovered_delta']:10.3f} "
            f"{item['error']:10.3f}"
        )

    # ------------------------------------------------------
    # Stability statistics
    # ------------------------------------------------------

    errors = np.array(
        [
            abs(item["error"])
            for item in results
        ],
        dtype=np.float64
    )

    recovered_deltas = np.array(
        [
            item["recovered_delta"]
            for item in results
        ],
        dtype=np.float64
    )

    print()
    print(
        "=" * 70
    )

    print(
        "STABILITY SUMMARY"
    )

    print(
        f"Minimum Absolute Error : "
        f"{errors.min():.6f}"
    )

    print(
        f"Maximum Absolute Error : "
        f"{errors.max():.6f}"
    )

    print(
        f"Mean Absolute Error    : "
        f"{errors.mean():.6f}"
    )

    print(
        f"Mean Recovered Delta   : "
        f"{recovered_deltas.mean():.6f}"
    )

    # ------------------------------------------------------
    # Count useful coefficients
    # ------------------------------------------------------

    for tolerance in (
        0.5,
        1.0,
        2.0,
        4.0,
    ):

        count = int(
            np.count_nonzero(
                errors <= tolerance
            )
        )

        print(
            f"Error <= {tolerance:3.1f}          : "
            f"{count}/{len(results)}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        "Coefficient Stability Diagnostic Complete"
    )

    print(
        "=" * 70
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()