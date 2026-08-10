"""
StegaFusion DWT Frame Suitability Diagnostic

Tests the same DWT/QIM round-trip on several frames from
the source video.

The purpose is to determine whether the failure is specific
to the extremely dark first frame or occurs throughout the
video.
"""

from pathlib import Path

import cv2
import numpy as np

from config.config import PathConfig

from modules.steganography.edge_detector import (
    generate_edge_map,
)

from modules.steganography.adaptive_lsb import (
    adaptive_embed,
    adaptive_extract,
)

from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)


PAYLOAD = (
    "00000000000000000000001000000000"
    "101100111000111100001111"
    "01010101010101010101010101010101"
    "11110000111100001111000011110000"
)


FRAME_INDICES = [
    0,
    1,
    5,
    10,
    20,
    40,
    60,
    80,
    100,
    150,
    196,
]


def first_mismatch(
    expected: str,
    recovered: str,
):

    limit = min(
        len(expected),
        len(recovered),
    )

    for index in range(limit):

        if expected[index] != recovered[index]:
            return index

    if len(expected) != len(recovered):
        return limit

    return None


def test_frame(
    frame_index: int,
):

    frame_path = (
        PathConfig.FRAME_DIR /
        f"frame_{frame_index:05d}.png"
    )

    if not frame_path.exists():

        return {
            "frame": frame_index,
            "status": "MISSING",
        }

    image = cv2.imread(
        str(frame_path)
    )

    if image is None:

        return {
            "frame": frame_index,
            "status": "READ ERROR",
        }

    blue = image[:, :, 0]

    original_min = int(
        blue.min()
    )

    original_max = int(
        blue.max()
    )

    original_mean = float(
        blue.mean()
    )

    # ------------------------------------------------------
    # DWT
    # ------------------------------------------------------

    bands = apply_dwt(
        blue
    )

    original_lh = (
        bands["LH"].copy()
    )

    # ------------------------------------------------------
    # Edge map
    # ------------------------------------------------------

    edge_map = generate_edge_map(
        original_lh
    )

    edge_pixels = int(
        np.count_nonzero(edge_map)
    )

    # ------------------------------------------------------
    # Embed
    # ------------------------------------------------------

    stego_lh, embedded = adaptive_embed(
        original_lh,
        edge_map,
        PAYLOAD
    )

    # ------------------------------------------------------
    # Direct extraction
    # ------------------------------------------------------

    direct = adaptive_extract(
        stego_lh,
        edge_map,
        len(PAYLOAD)
    )

    direct_mismatch = first_mismatch(
        PAYLOAD,
        direct
    )

    # ------------------------------------------------------
    # Replace LH
    # ------------------------------------------------------

    stego_bands = {
        key: value.copy()
        for key, value in bands.items()
    }

    stego_bands["LH"] = stego_lh

    # ------------------------------------------------------
    # Inverse DWT
    # ------------------------------------------------------

    reconstructed = apply_inverse_dwt(
        stego_bands
    )

    negative_pixels = int(
        np.count_nonzero(
            reconstructed < 0
        )
    )

    # ------------------------------------------------------
    # uint8 conversion
    # ------------------------------------------------------

    reconstructed_uint8 = np.clip(
        reconstructed,
        0,
        255
    ).astype(np.uint8)

    # ------------------------------------------------------
    # Spatial distortion
    # ------------------------------------------------------

    spatial_difference = np.abs(
        reconstructed_uint8.astype(
            np.int16
        )
        -
        blue.astype(
            np.int16
        )
    )

    spatial_changed = int(
        np.count_nonzero(
            spatial_difference
        )
    )

    spatial_max = int(
        spatial_difference.max()
    )

    # ------------------------------------------------------
    # DWT again
    # ------------------------------------------------------

    recovered_bands = apply_dwt(
        reconstructed_uint8
    )

    recovered_lh = (
        recovered_bands["LH"]
    )

    coefficient_difference = np.abs(
        recovered_lh -
        stego_lh
    )

    coefficient_changed = int(
        np.count_nonzero(
            coefficient_difference
        )
    )

    coefficient_max = float(
        coefficient_difference.max()
    )

    # ------------------------------------------------------
    # Extraction after round-trip
    # ------------------------------------------------------

    recovered_edge_map = generate_edge_map(
        recovered_lh
    )

    recovered = adaptive_extract(
        recovered_lh,
        recovered_edge_map,
        len(PAYLOAD)
    )

    roundtrip_mismatch = first_mismatch(
        PAYLOAD,
        recovered
    )

    return {
        "frame": frame_index,
        "status": "OK",
        "min": original_min,
        "max": original_max,
        "mean": original_mean,
        "edge": edge_pixels,
        "embedded": embedded,
        "direct": direct_mismatch is None,
        "direct_mismatch": direct_mismatch,
        "negative": negative_pixels,
        "changed_pixels": spatial_changed,
        "max_spatial": spatial_max,
        "changed_coeff": coefficient_changed,
        "max_coeff": coefficient_max,
        "roundtrip": roundtrip_mismatch is None,
        "roundtrip_mismatch": roundtrip_mismatch,
    }


def main():

    print("=" * 90)
    print(
        "StegaFusion DWT Frame Suitability Diagnostic"
    )
    print("=" * 90)

    # ------------------------------------------------------
    # Make sure frames exist
    # ------------------------------------------------------

    first_frame = (
        PathConfig.FRAME_DIR /
        "frame_00000.png"
    )

    if not first_frame.exists():

        print()
        print(
            "Frames are missing."
        )

        print(
            "Run this first:"
        )

        print(
            "python -m tests.test_video_frame_extraction"
        )

        return

    # ------------------------------------------------------
    # Test frames
    # ------------------------------------------------------

    results = []

    for frame_index in FRAME_INDICES:

        print()
        print(
            f"Testing frame {frame_index}..."
        )

        result = test_frame(
            frame_index
        )

        results.append(
            result
        )

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    print()
    print("=" * 90)

    print(
        "FRAME COMPARISON"
    )

    print("=" * 90)

    print()

    print(
        f"{'Frame':>6} "
        f"{'Range':>10} "
        f"{'Mean':>10} "
        f"{'Edges':>9} "
        f"{'Direct':>9} "
        f"{'RoundTrip':>11} "
        f"{'Mismatch':>10} "
        f"{'Neg':>7} "
        f"{'MaxCoeff':>10}"
    )

    print("-" * 90)

    for result in results:

        if result["status"] != "OK":

            print(
                f"{result['frame']:>6} "
                f"{result['status']}"
            )

            continue

        range_text = (
            f"{result['min']}"
            f"-"
            f"{result['max']}"
        )

        print(
            f"{result['frame']:>6} "
            f"{range_text:>10} "
            f"{result['mean']:>10.2f} "
            f"{result['edge']:>9} "
            f"{str(result['direct']):>9} "
            f"{str(result['roundtrip']):>11} "
            f"{str(result['roundtrip_mismatch']):>10} "
            f"{result['negative']:>7} "
            f"{result['max_coeff']:>10.2f}"
        )

    # ------------------------------------------------------
    # Final interpretation
    # ------------------------------------------------------

    print()
    print("=" * 90)

    passing_frames = [
        result
        for result in results
        if (
            result.get("status") == "OK"
            and result.get("roundtrip")
        )
    ]

    failing_frames = [
        result
        for result in results
        if (
            result.get("status") == "OK"
            and not result.get("roundtrip")
        )
    ]

    print(
        f"Round-trip PASS frames : "
        f"{len(passing_frames)}"
    )

    print(
        f"Round-trip FAIL frames : "
        f"{len(failing_frames)}"
    )

    print()

    if passing_frames:

        print(
            "RESULT:"
        )

        print(
            "The DWT/QIM round-trip works on at least "
            "some normal video frames."
        )

        print(
            "Frame selection is therefore an important "
            "part of the embedding design."
        )

    else:

        print(
            "RESULT:"
        )

        print(
            "No tested frame survived the DWT round-trip."
        )

        print(
            "Further DWT/QIM investigation is required."
        )

    print("=" * 90)


if __name__ == "__main__":
    main()