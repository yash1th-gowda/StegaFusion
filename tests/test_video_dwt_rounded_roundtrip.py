"""
StegaFusion Rounded DWT Round-Trip Diagnostic

Tests whether rounding the inverse-DWT spatial image before
uint8 conversion allows the embedded DWT coefficients to
survive the spatial round-trip.

This is diagnostic only.
It does NOT modify production code.
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
    expected,
    recovered,
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
    frame_index,
):

    frame_path = (
        PathConfig.FRAME_DIR /
        f"frame_{frame_index:05d}.png"
    )

    image = cv2.imread(
        str(frame_path)
    )

    if image is None:

        raise RuntimeError(
            f"Unable to read {frame_path}"
        )

    blue = image[:, :, 0]

    # ------------------------------------------------------
    # Original DWT
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
    # Reconstruct
    # ------------------------------------------------------

    stego_bands = {
        key: value.copy()
        for key, value in bands.items()
    }

    stego_bands["LH"] = stego_lh

    reconstructed = apply_inverse_dwt(
        stego_bands
    )

    negative_pixels = int(
        np.count_nonzero(
            reconstructed < 0
        )
    )

    # ------------------------------------------------------
    # CURRENT METHOD
    # ------------------------------------------------------

    clipped = np.clip(
        reconstructed,
        0,
        255
    ).astype(np.uint8)

    # ------------------------------------------------------
    # ROUNDED METHOD
    # ------------------------------------------------------

    rounded = np.round(
        reconstructed
    ).clip(
        0,
        255
    ).astype(np.uint8)

    # ------------------------------------------------------
    # Recover DWT - CURRENT
    # ------------------------------------------------------

    clipped_bands = apply_dwt(
        clipped
    )

    clipped_lh = (
        clipped_bands["LH"]
    )

    # ------------------------------------------------------
    # Recover DWT - ROUNDED
    # ------------------------------------------------------

    rounded_bands = apply_dwt(
        rounded
    )

    rounded_lh = (
        rounded_bands["LH"]
    )

    # ------------------------------------------------------
    # Extract - CURRENT
    # ------------------------------------------------------

    clipped_edge_map = generate_edge_map(
        clipped_lh
    )

    clipped_payload = adaptive_extract(
        clipped_lh,
        clipped_edge_map,
        len(PAYLOAD)
    )

    clipped_mismatch = first_mismatch(
        PAYLOAD,
        clipped_payload
    )

    # ------------------------------------------------------
    # Extract - ROUNDED
    # ------------------------------------------------------

    rounded_edge_map = generate_edge_map(
        rounded_lh
    )

    rounded_payload = adaptive_extract(
        rounded_lh,
        rounded_edge_map,
        len(PAYLOAD)
    )

    rounded_mismatch = first_mismatch(
        PAYLOAD,
        rounded_payload
    )

    # ------------------------------------------------------
    # Coefficient distortion
    # ------------------------------------------------------

    rounded_error = np.abs(
        rounded_lh -
        stego_lh
    )

    clipped_error = np.abs(
        clipped_lh -
        stego_lh
    )

    return {
        "frame": frame_index,
        "range": (
            int(blue.min()),
            int(blue.max()),
        ),
        "mean": float(blue.mean()),
        "embedded": embedded,
        "direct": direct_mismatch is None,
        "direct_mismatch": direct_mismatch,
        "negative": negative_pixels,
        "clipped_mismatch": clipped_mismatch,
        "rounded_mismatch": rounded_mismatch,
        "clipped_max_error": float(
            clipped_error.max()
        ),
        "rounded_max_error": float(
            rounded_error.max()
        ),
        "clipped_changed": int(
            np.count_nonzero(
                clipped_error > 1e-9
            )
        ),
        "rounded_changed": int(
            np.count_nonzero(
                rounded_error > 1e-9
            )
        ),
    }


def main():

    print("=" * 90)
    print(
        "StegaFusion Rounded DWT Round-Trip Diagnostic"
    )
    print("=" * 90)

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
        "RESULTS"
    )

    print("=" * 90)

    print()

    print(
        f"{'Frame':>6} "
        f"{'Range':>10} "
        f"{'Direct':>8} "
        f"{'Clip':>8} "
        f"{'Round':>8} "
        f"{'ClipErr':>9} "
        f"{'RoundErr':>9} "
        f"{'Negative':>10}"
    )

    print("-" * 90)

    for result in results:

        frame = result["frame"]

        low, high = result["range"]

        range_text = (
            f"{low}-{high}"
        )

        clipped_pass = (
            result["clipped_mismatch"]
            is None
        )

        rounded_pass = (
            result["rounded_mismatch"]
            is None
        )

        print(
            f"{frame:6d} "
            f"{range_text:>10} "
            f"{str(result['direct']):>8} "
            f"{str(clipped_pass):>8} "
            f"{str(rounded_pass):>8} "
            f"{result['clipped_max_error']:9.3f} "
            f"{result['rounded_max_error']:9.3f} "
            f"{result['negative']:10d}"
        )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    clipped_pass_count = sum(
        result["clipped_mismatch"] is None
        for result in results
    )

    rounded_pass_count = sum(
        result["rounded_mismatch"] is None
        for result in results
    )

    print()
    print("=" * 90)

    print(
        f"Clipped round-trip PASS : "
        f"{clipped_pass_count}/{len(results)}"
    )

    print(
        f"Rounded round-trip PASS : "
        f"{rounded_pass_count}/{len(results)}"
    )

    print()

    if rounded_pass_count == len(results):

        print(
            "RESULT: ROUNDED CONVERSION PASSES "
            "ALL TESTED FRAMES."
        )

        print()
        print(
            "This strongly supports replacing the "
            "current truncating uint8 conversion with:"
        )

        print(
            "np.round(...).clip(0, 255).astype(np.uint8)"
        )

    elif rounded_pass_count > clipped_pass_count:

        print(
            "RESULT: ROUNDING IMPROVES DWT "
            "ROUND-TRIP RECOVERY."
        )

        print(
            "Additional testing is required before "
            "changing production code."
        )

    else:

        print(
            "RESULT: ROUNDING DOES NOT SOLVE "
            "THE DWT ROUND-TRIP PROBLEM."
        )

        print(
            "Do not modify production conversion yet."
        )

    print("=" * 90)


if __name__ == "__main__":
    main()