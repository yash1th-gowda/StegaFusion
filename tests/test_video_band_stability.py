"""
StegaFusion DWT Band Stability Diagnostic

Tests MP4 coefficient survival in:

    LH
    HL
    HH

This is diagnostic only.

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
# SETTINGS
# ==========================================================

FRAME_INDEX = 40

TEST_DELTA = 8

GRID_ROWS = 20
GRID_COLS = 20

FPS = 23.976023976023978

OUTPUT_DIR = (
    PathConfig.OUTPUT_DIR /
    "video_band_stability"
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
            f"Unable to read frame: {path}"
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

    for row in row_values:

        for col in col_values:

            positions.append(
                (
                    int(row),
                    int(col)
                )
            )

    return positions


# ==========================================================
# CREATE MODIFIED FRAME
# ==========================================================

def create_modified_frame(
    source,
    band_name,
    position
):

    blue = source[:, :, 0]

    bands = apply_dwt(
        blue
    )

    original = float(
        bands[band_name][
            position[0],
            position[1]
        ]
    )

    modified_bands = {
        key: value.copy()
        for key, value in bands.items()
    }

    modified_bands[band_name][
        position[0],
        position[1]
    ] = (
        original +
        TEST_DELTA
    )

    reconstructed = reconstruct_uint8(
        modified_bands
    )

    frame = source.copy()

    frame[:, :, 0] = reconstructed

    return frame, original


# ==========================================================
# CREATE MP4
# ==========================================================

def create_video(
    source,
    band_name,
    positions
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_video = (
        OUTPUT_DIR /
        f"{band_name}.mp4"
    )

    height, width = (
        source.shape[:2]
    )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height)
    )

    if not writer.isOpened():

        raise RuntimeError(
            f"Unable to create: "
            f"{output_video}"
        )

    originals = []

    for position in positions:

        frame, original = (
            create_modified_frame(
                source,
                band_name,
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

    return output_video, originals


# ==========================================================
# READ MP4
# ==========================================================

def read_video(path):

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Unable to open: {path}"
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
# ANALYZE BAND
# ==========================================================

def analyze_band(
    source,
    band_name,
    positions,
    originals,
    decoded_frames
):

    results = []

    for index, position in enumerate(
        positions
    ):

        row, col = position

        original = originals[index]

        decoded = decoded_frames[index]

        decoded_blue = (
            decoded[:, :, 0]
        )

        decoded_bands = apply_dwt(
            decoded_blue
        )

        recovered = float(
            decoded_bands[band_name][
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

        results.append(
            (
                position,
                original,
                recovered_delta,
                error
            )
        )

    return results


# ==========================================================
# BAND TEST
# ==========================================================

def test_band(
    source,
    band_name
):

    print()
    print("=" * 70)
    print(
        f"TESTING BAND: {band_name}"
    )
    print("=" * 70)

    blue = source[:, :, 0]

    bands = apply_dwt(
        blue
    )

    band = bands[band_name]

    print(
        f"Band Shape : {band.shape}"
    )

    positions = generate_positions(
        band.shape
    )

    print(
        f"Positions  : {len(positions)}"
    )

    output_video, originals = (
        create_video(
            source,
            band_name,
            positions
        )
    )

    print(
        f"MP4        : {output_video}"
    )

    decoded_frames = read_video(
        output_video
    )

    print(
        f"Decoded    : {len(decoded_frames)}"
    )

    if len(decoded_frames) != len(
        positions
    ):

        raise RuntimeError(
            f"{band_name}: frame count mismatch."
        )

    results = analyze_band(
        source,
        band_name,
        positions,
        originals,
        decoded_frames
    )

    deltas = np.array(
        [
            result[2]
            for result in results
        ]
    )

    errors = np.array(
        [
            result[3]
            for result in results
        ]
    )

    print()
    print(
        "RECOVERY"
    )

    print(
        f"Minimum Delta : "
        f"{deltas.min():.3f}"
    )

    print(
        f"Maximum Delta : "
        f"{deltas.max():.3f}"
    )

    print(
        f"Mean Delta    : "
        f"{deltas.mean():.3f}"
    )

    print(
        f"Median Delta  : "
        f"{np.median(deltas):.3f}"
    )

    print()

    counts = {}

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

        counts[threshold] = count

        print(
            f"Delta >= {threshold:4.1f} : "
            f"{count}/{len(deltas)}"
        )

    # ------------------------------------------------------
    # Best coefficients
    # ------------------------------------------------------

    results.sort(
        key=lambda item:
        item[3]
    )

    print()
    print(
        "BEST 10 COEFFICIENTS"
    )

    print(
        f"{'Position':>14} "
        f"{'Original':>10} "
        f"{'Delta':>10} "
        f"{'Error':>10}"
    )

    print(
        "-" * 60
    )

    for result in results[:10]:

        position, original, delta, error = (
            result
        )

        print(
            f"{str(position):>14} "
            f"{original:10.3f} "
            f"{delta:10.3f} "
            f"{error:10.3f}"
        )

    return {
        "band": band_name,
        "min": float(deltas.min()),
        "max": float(deltas.max()),
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "counts": counts,
        "results": results,
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "StegaFusion DWT Band Stability Diagnostic"
    )
    print("=" * 70)

    source = load_frame()

    blue = source[:, :, 0]

    print()
    print(
        f"Frame      : {FRAME_INDEX}"
    )

    print(
        f"Blue Shape : {blue.shape}"
    )

    print(
        f"Blue Range : "
        f"{blue.min()} -> {blue.max()}"
    )

    print(
        f"Test Delta : {TEST_DELTA}"
    )

    # ------------------------------------------------------
    # Test all high-frequency bands
    # ------------------------------------------------------

    results = []

    for band_name in (
        "LH",
        "HL",
        "HH",
    ):

        result = test_band(
            source,
            band_name
        )

        results.append(
            result
        )

    # ------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "FINAL BAND COMPARISON"
    )
    print("=" * 90)

    print(
        f"{'BAND':<8}"
        f"{'>=1':>10}"
        f"{'>=2':>10}"
        f"{'>=3':>10}"
        f"{'>=4':>10}"
        f"{'MAX':>12}"
        f"{'MEAN':>12}"
    )

    print(
        "-" * 90
    )

    for result in results:

        counts = result["counts"]

        print(
            f"{result['band']:<8}"
            f"{counts[1]:>10}"
            f"{counts[2]:>10}"
            f"{counts[3]:>10}"
            f"{counts[4]:>10}"
            f"{result['max']:>12.3f}"
            f"{result['mean']:>12.3f}"
        )

    # ------------------------------------------------------
    # Determine strongest band
    # ------------------------------------------------------

    strongest = max(
        results,
        key=lambda result:
        result["max"]
    )

    print()
    print("=" * 70)

    print(
        f"Strongest observed band : "
        f"{strongest['band']}"
    )

    print(
        f"Maximum recovered delta : "
        f"{strongest['max']:.3f}"
    )

    print(
        f"Mean recovered delta    : "
        f"{strongest['mean']:.3f}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This diagnostic does not modify "
        "production embedding."
    )

    print(
        "Do not change adaptive_lsb.py "
        "until this result is reviewed."
    )

    print()
    print("=" * 70)
    print(
        "Band Stability Diagnostic Complete"
    )
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()