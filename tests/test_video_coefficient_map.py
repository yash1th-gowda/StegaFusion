"""
StegaFusion MP4 DWT Coefficient Stability Map

Diagnostic only.

Measures how much individual LH coefficients survive
MP4 encoding/decoding.

No production code is modified.
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

TEST_DELTA = 8

GRID_ROWS = 20
GRID_COLS = 20

FPS = 23.976023976023978

OUTPUT_VIDEO = (
    PathConfig.OUTPUT_DIR /
    "video_coefficient_map.mp4"
)


# ==========================================================
# SAFE RECONSTRUCTION
# ==========================================================

def reconstruct_uint8(bands):

    reconstructed = apply_inverse_dwt(
        bands
    )

    return np.round(
        reconstructed
    ).clip(
        0,
        255
    ).astype(np.uint8)


# ==========================================================
# LOAD FRAME
# ==========================================================

def load_frame():

    path = (
        PathConfig.FRAME_DIR /
        f"frame_{FRAME_INDEX:05d}.png"
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    frame = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if frame is None:

        raise RuntimeError(
            f"Unable to read {path}"
        )

    return frame


# ==========================================================
# POSITIONS
# ==========================================================

def generate_positions(shape):

    rows, cols = shape

    row_values = np.linspace(
        0,
        rows - 1,
        GRID_ROWS,
        dtype=int
    )

    col_values = np.linspace(
        0,
        cols - 1,
        GRID_COLS,
        dtype=int
    )

    positions = []

    for r in row_values:

        for c in col_values:

            positions.append(
                (int(r), int(c))
            )

    return positions


# ==========================================================
# CREATE FRAME
# ==========================================================

def create_modified_frame(
    source,
    position
):

    blue = source[:, :, 0]

    bands = apply_dwt(
        blue
    )

    original = float(
        bands["LH"][
            position[0],
            position[1]
        ]
    )

    modified = {
        key: value.copy()
        for key, value in bands.items()
    }

    modified["LH"][
        position[0],
        position[1]
    ] = (
        original +
        TEST_DELTA
    )

    reconstructed = reconstruct_uint8(
        modified
    )

    frame = source.copy()

    frame[:, :, 0] = reconstructed

    return frame, original


# ==========================================================
# WRITE MP4
# ==========================================================

def create_video(
    source,
    positions
):

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    height, width = source.shape[:2]

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

    originals = []

    for position in positions:

        frame, original = (
            create_modified_frame(
                source,
                position
            )
        )

        writer.write(
            frame
        )

        originals.append(
            original
        )

    writer.release()

    return originals


# ==========================================================
# READ VIDEO
# ==========================================================

def read_video():

    cap = cv2.VideoCapture(
        str(OUTPUT_VIDEO)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Unable to open MP4."
        )

    frames = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        frames.append(frame)

    cap.release()

    return frames


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "StegaFusion MP4 DWT Coefficient Stability Map"
    )
    print("=" * 70)

    # ------------------------------------------------------
    # Load
    # ------------------------------------------------------

    source = load_frame()

    blue = source[:, :, 0]

    bands = apply_dwt(
        blue
    )

    lh = bands["LH"]

    print()
    print(
        f"Frame       : {FRAME_INDEX}"
    )

    print(
        f"Blue Shape  : {blue.shape}"
    )

    print(
        f"LH Shape    : {lh.shape}"
    )

    print(
        f"Test Delta  : {TEST_DELTA}"
    )

    # ------------------------------------------------------
    # Positions
    # ------------------------------------------------------

    positions = generate_positions(
        lh.shape
    )

    print(
        f"Positions   : {len(positions)}"
    )

    # ------------------------------------------------------
    # Encode
    # ------------------------------------------------------

    print()
    print(
        "Creating MP4..."
    )

    originals = create_video(
        source,
        positions
    )

    print(
        f"Saved       : {OUTPUT_VIDEO}"
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
        f"Decoded     : {len(decoded_frames)}"
    )

    if len(decoded_frames) != len(positions):

        raise RuntimeError(
            "Frame count mismatch."
        )

    # ------------------------------------------------------
    # Analyze
    # ------------------------------------------------------

    recoveries = []

    for index, position in enumerate(
        positions
    ):

        row, col = position

        original = originals[index]

        decoded = decoded_frames[index]

        decoded_blue = decoded[:, :, 0]

        recovered_bands = apply_dwt(
            decoded_blue
        )

        recovered = float(
            recovered_bands["LH"][
                row,
                col
            ]
        )

        recovered_delta = (
            recovered -
            original
        )

        error = abs(
            recovered_delta -
            TEST_DELTA
        )

        recoveries.append(
            (
                position,
                original,
                recovered_delta,
                error
            )
        )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    deltas = np.array(
        [
            item[2]
            for item in recoveries
        ]
    )

    errors = np.array(
        [
            item[3]
            for item in recoveries
        ]
    )

    print()
    print("=" * 70)
    print(
        "RECOVERY DISTRIBUTION"
    )
    print("=" * 70)

    print(
        f"Minimum Delta : {deltas.min():.3f}"
    )

    print(
        f"Maximum Delta : {deltas.max():.3f}"
    )

    print(
        f"Mean Delta    : {deltas.mean():.3f}"
    )

    print(
        f"Median Delta  : {np.median(deltas):.3f}"
    )

    print()

    for threshold in (
        0,
        0.5,
        1,
        1.5,
        2,
        3,
        4,
        6,
    ):

        count = int(
            np.count_nonzero(
                deltas >= threshold
            )
        )

        print(
            f"Delta >= {threshold:4.1f} : "
            f"{count}/{len(deltas)}"
        )

    # ------------------------------------------------------
    # Best positions
    # ------------------------------------------------------

    recoveries.sort(
        key=lambda item:
        item[3]
    )

    print()
    print("=" * 90)
    print(
        "BEST COEFFICIENT LOCATIONS"
    )
    print("=" * 90)

    print(
        f"{'Position':>14} "
        f"{'Original':>10} "
        f"{'Recovered':>12} "
        f"{'Error':>10}"
    )

    print(
        "-" * 90
    )

    for item in recoveries[:30]:

        position, original, delta, error = item

        print(
            f"{str(position):>14} "
            f"{original:10.3f} "
            f"{delta:12.3f} "
            f"{error:10.3f}"
        )

    # ------------------------------------------------------
    # Stable candidates
    # ------------------------------------------------------

    stable = [
        item
        for item in recoveries
        if item[2] >= 2.0
    ]

    print()
    print("=" * 70)

    print(
        f"Coefficients recovering >= 2 units : "
        f"{len(stable)}/{len(recoveries)}"
    )

    if stable:

        print()
        print(
            "Stable candidate locations:"
        )

        for item in stable[:20]:

            print(
                item[0],
                f"delta={item[2]:.3f}"
            )

    print()
    print("=" * 70)
    print(
        "Coefficient Stability Map Complete"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()