"""
StegaFusion Distributed MP4 Coefficient Stability Diagnostic

Tests LH DWT coefficients distributed across the entire LH band.

This is diagnostic only.
It does NOT modify production embedding logic.

For each sampled coefficient:

    source frame
        ↓
    DWT
        ↓
    modify one LH coefficient
        ↓
    rounded inverse DWT
        ↓
    MP4 encoding
        ↓
    MP4 decoding
        ↓
    DWT
        ↓
    measure coefficient recovery
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

# Test modification.
TEST_DELTA = 8

# Number of coefficients to sample.
SAMPLE_COUNT = 100

FPS = 23.976023976023978

OUTPUT_VIDEO = (
    PathConfig.OUTPUT_DIR /
    "video_distributed_stability.mp4"
)


# ==========================================================
# RECONSTRUCTION
# ==========================================================

def reconstruct_uint8(bands):
    """
    Reconstruct an image using the production-safe
    round -> clip -> uint8 conversion.
    """

    reconstructed = apply_inverse_dwt(
        bands
    )

    reconstructed = np.round(
        reconstructed
    ).clip(
        0,
        255
    ).astype(np.uint8)

    return reconstructed


# ==========================================================
# LOAD FRAME
# ==========================================================

def load_frame():

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


# ==========================================================
# SAMPLE COEFFICIENTS
# ==========================================================

def sample_positions(shape):
    """
    Select coefficients distributed across the entire LH band.

    Includes:
        - corners
        - edges
        - center
        - regularly distributed positions
    """

    rows, cols = shape

    positions = []

    # ------------------------------------------------------
    # Important fixed positions
    # ------------------------------------------------------

    fixed = [
        (0, 0),
        (0, cols - 1),
        (rows - 1, 0),
        (rows - 1, cols - 1),
        (rows // 2, cols // 2),
        (0, cols // 2),
        (rows // 2, 0),
        (rows - 1, cols // 2),
        (rows // 2, cols - 1),
    ]

    for position in fixed:

        if position not in positions:
            positions.append(position)

    # ------------------------------------------------------
    # Grid sampling
    # ------------------------------------------------------

    grid_size = int(
        np.ceil(
            np.sqrt(
                SAMPLE_COUNT
            )
        )
    )

    row_values = np.linspace(
        0,
        rows - 1,
        grid_size,
        dtype=int
    )

    col_values = np.linspace(
        0,
        cols - 1,
        grid_size,
        dtype=int
    )

    for row in row_values:

        for col in col_values:

            position = (
                int(row),
                int(col)
            )

            if position not in positions:

                positions.append(
                    position
                )

            if len(positions) >= SAMPLE_COUNT:
                return positions

    return positions


# ==========================================================
# CREATE TEST FRAME
# ==========================================================

def create_test_frame(
    source_frame,
    position,
):
    """
    Modify exactly one LH coefficient.
    """

    blue = source_frame[:, :, 0]

    bands = apply_dwt(
        blue
    )

    original_value = float(
        bands["LH"][
            position[0],
            position[1]
        ]
    )

    modified_bands = {
        key: value.copy()
        for key, value in bands.items()
    }

    modified_bands["LH"][
        position[0],
        position[1]
    ] = (
        original_value +
        TEST_DELTA
    )

    reconstructed_blue = reconstruct_uint8(
        modified_bands
    )

    test_frame = source_frame.copy()

    test_frame[:, :, 0] = reconstructed_blue

    return (
        test_frame,
        original_value
    )


# ==========================================================
# WRITE MP4
# ==========================================================

def write_video(
    source_frame,
    positions
):

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    height, width = (
        source_frame.shape[:2]
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Unable to create MP4."
        )

    original_values = []

    for position in positions:

        test_frame, original_value = (
            create_test_frame(
                source_frame,
                position
            )
        )

        writer.write(
            test_frame
        )

        original_values.append(
            original_value
        )

    writer.release()

    return original_values


# ==========================================================
# READ MP4
# ==========================================================

def read_video():

    cap = cv2.VideoCapture(
        str(OUTPUT_VIDEO)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open generated MP4."
        )

    frames = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        frames.append(
            frame
        )

    cap.release()

    return frames


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "StegaFusion Distributed MP4 "
        "Coefficient Stability Diagnostic"
    )
    print("=" * 70)

    # ------------------------------------------------------
    # Load source
    # ------------------------------------------------------

    source_frame = load_frame()

    blue = source_frame[:, :, 0]

    print()
    print(
        f"Frame              : {FRAME_INDEX}"
    )

    print(
        f"Blue Shape         : {blue.shape}"
    )

    print(
        f"Blue Range         : "
        f"{blue.min()} -> {blue.max()}"
    )

    # ------------------------------------------------------
    # Original DWT
    # ------------------------------------------------------

    bands = apply_dwt(
        blue
    )

    lh = bands["LH"]

    print(
        f"LH Shape           : {lh.shape}"
    )

    # ------------------------------------------------------
    # Sample positions
    # ------------------------------------------------------

    positions = sample_positions(
        lh.shape
    )

    print()
    print(
        f"Sampled Coeffs     : "
        f"{len(positions)}"
    )

    print(
        f"Test Delta         : "
        f"{TEST_DELTA}"
    )

    # ------------------------------------------------------
    # Write MP4
    # ------------------------------------------------------

    print()
    print(
        "Creating MP4..."
    )

    original_values = write_video(
        source_frame,
        positions
    )

    print(
        f"Saved              : "
        f"{OUTPUT_VIDEO}"
    )

    # ------------------------------------------------------
    # Decode
    # ------------------------------------------------------

    print()
    print(
        "Decoding MP4..."
    )

    decoded_frames = read_video()

    print(
        f"Decoded Frames     : "
        f"{len(decoded_frames)}"
    )

    if len(decoded_frames) != len(positions):

        raise RuntimeError(
            "Decoded frame count does not match "
            "sample count."
        )

    # ------------------------------------------------------
    # Analyze
    # ------------------------------------------------------

    results = []

    for index, position in enumerate(
        positions
    ):

        row, col = position

        original = original_values[index]

        embedded = (
            original +
            TEST_DELTA
        )

        decoded = (
            decoded_frames[index]
        )

        decoded_blue = (
            decoded[:, :, 0]
        )

        decoded_bands = apply_dwt(
            decoded_blue
        )

        recovered = float(
            decoded_bands["LH"][
                row,
                col
            ]
        )

        recovered_delta = (
            recovered -
            original
        )

        error = (
            recovered -
            embedded
        )

        results.append(
            {
                "position": position,
                "original": original,
                "embedded": embedded,
                "recovered": recovered,
                "delta": recovered_delta,
                "error": error,
            }
        )

    # ------------------------------------------------------
    # Sort by recovery delta
    # ------------------------------------------------------

    results.sort(
        key=lambda item:
        abs(
            item["error"]
        )
    )

    # ------------------------------------------------------
    # Display
    # ------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "MOST STABLE COEFFICIENTS"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Position':>14} "
        f"{'Orig':>10} "
        f"{'Embed':>10} "
        f"{'Recover':>10} "
        f"{'Delta':>10} "
        f"{'Error':>10}"
    )

    print(
        "-" * 100
    )

    for item in results[:30]:

        row, col = (
            item["position"]
        )

        print(
            f"({row:4d},{col:4d}) "
            f"{item['original']:10.3f} "
            f"{item['embedded']:10.3f} "
            f"{item['recovered']:10.3f} "
            f"{item['delta']:10.3f} "
            f"{item['error']:10.3f}"
        )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    errors = np.array(
        [
            abs(item["error"])
            for item in results
        ]
    )

    deltas = np.array(
        [
            item["delta"]
            for item in results
        ]
    )

    print()
    print(
        "=" * 70
    )

    print(
        "STABILITY SUMMARY"
    )

    print(
        f"Minimum Error      : "
        f"{errors.min():.6f}"
    )

    print(
        f"Maximum Error      : "
        f"{errors.max():.6f}"
    )

    print(
        f"Mean Error         : "
        f"{errors.mean():.6f}"
    )

    print(
        f"Mean Recovered     : "
        f"{deltas.mean():.6f}"
    )

    print()

    for threshold in (
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    ):

        count = int(
            np.count_nonzero(
                errors <= threshold
            )
        )

        print(
            f"Error <= {threshold:4.1f} : "
            f"{count}/{len(results)}"
        )

    # ------------------------------------------------------
    # Strong recovery count
    # ------------------------------------------------------

    useful = int(
        np.count_nonzero(
            deltas >= (
                TEST_DELTA * 0.5
            )
        )
    )

    print()
    print(
        f"Recovered >= 50% of delta : "
        f"{useful}/{len(results)}"
    )

    print()
    print("=" * 70)
    print(
        "Distributed Coefficient Stability "
        "Diagnostic Complete"
    )
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()