"""
StegaFusion DWT Round-Trip Coefficient Diagnostic

Determines exactly how much the embedded LH coefficients change
after:

    Original uint8 image
        ↓
    DWT
        ↓
    Adaptive QIM embedding
        ↓
    Inverse DWT
        ↓
    uint8 conversion
        ↓
    DWT again

This test does NOT modify production code.
"""

from pathlib import Path

import numpy as np

from config.config import PathConfig

from modules.steganography.lsb_embed import load_frame
from modules.steganography.edge_detector import generate_edge_map
from modules.steganography.adaptive_lsb import (
    adaptive_embed,
    adaptive_extract,
)
from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)


# ==========================================================
# PATHS
# ==========================================================

FRAME_FILE = (
    PathConfig.FRAME_DIR /
    "frame_00000.png"
)


# ==========================================================
# PAYLOAD
# ==========================================================

PAYLOAD = (
    "00000000000000000000001000000000"
    "101100111000111100001111"
    "01010101010101010101010101010101"
    "11110000111100001111000011110000"
)


# ==========================================================
# FIRST MISMATCH
# ==========================================================

def first_mismatch(
    expected: str,
    recovered: str,
):
    limit = min(
        len(expected),
        len(recovered)
    )

    for index in range(limit):

        if expected[index] != recovered[index]:
            return index

    if len(expected) != len(recovered):
        return limit

    return None


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("StegaFusion DWT Round-Trip Coefficient Diagnostic")
    print("=" * 70)

    # ------------------------------------------------------
    # Load frame
    # ------------------------------------------------------

    if not FRAME_FILE.exists():

        raise FileNotFoundError(
            f"Frame not found: {FRAME_FILE}"
        )

    image = load_frame(
        FRAME_FILE
    )

    blue = image[:, :, 0]

    print()
    print("Original Blue Channel")

    print(
        f"Shape  : {blue.shape}"
    )

    print(
        f"Dtype  : {blue.dtype}"
    )

    print(
        f"Range  : "
        f"{blue.min()} -> {blue.max()}"
    )

    # ------------------------------------------------------
    # Original DWT
    # ------------------------------------------------------

    original_bands = apply_dwt(
        blue
    )

    original_lh = (
        original_bands["LH"].copy()
    )

    # ------------------------------------------------------
    # Edge map
    # ------------------------------------------------------

    edge_map = generate_edge_map(
        original_lh
    )

    print()
    print(
        f"Edge Pixels : "
        f"{np.count_nonzero(edge_map)}"
    )

    # ------------------------------------------------------
    # Embed
    # ------------------------------------------------------

    stego_lh, embedded = adaptive_embed(
        original_lh,
        edge_map,
        PAYLOAD
    )

    print(
        f"Embedded Bits : {embedded}"
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

    print()
    print(
        "Direct Extraction : "
        f"{'PASS' if direct_mismatch is None else 'FAIL'}"
    )

    if direct_mismatch is not None:

        print(
            f"First Mismatch : "
            f"{direct_mismatch}"
        )

        raise RuntimeError(
            "Embedding itself failed."
        )

    # ------------------------------------------------------
    # Compare original and embedded LH
    # ------------------------------------------------------

    lh_change = np.abs(
        stego_lh - original_lh
    )

    print()
    print("LH Changes After Embedding")
    print(
        f"Changed Coefficients : "
        f"{np.count_nonzero(lh_change)}"
    )

    print(
        f"Maximum Difference   : "
        f"{lh_change.max():.6f}"
    )

    print(
        f"Mean Absolute Diff   : "
        f"{lh_change.mean():.6f}"
    )

    # ------------------------------------------------------
    # Replace LH
    # ------------------------------------------------------

    stego_bands = {
        key: value.copy()
        for key, value in original_bands.items()
    }

    stego_bands["LH"] = stego_lh

    # ------------------------------------------------------
    # Inverse DWT
    # ------------------------------------------------------

    reconstructed = apply_inverse_dwt(
        stego_bands
    )

    print()
    print("Reconstructed Float Image")

    print(
        f"Min    : "
        f"{reconstructed.min():.6f}"
    )

    print(
        f"Max    : "
        f"{reconstructed.max():.6f}"
    )

    print(
        f"Mean   : "
        f"{reconstructed.mean():.6f}"
    )

    # ------------------------------------------------------
    # Measure clipping
    # ------------------------------------------------------

    below_zero = np.count_nonzero(
        reconstructed < 0
    )

    above_255 = np.count_nonzero(
        reconstructed > 255
    )

    print()
    print("Clipping")

    print(
        f"Below 0   : {below_zero}"
    )

    print(
        f"Above 255 : {above_255}"
    )

    # ------------------------------------------------------
    # Convert back to uint8
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
        reconstructed_uint8.astype(np.int16)
        -
        blue.astype(np.int16)
    )

    print()
    print("Spatial Distortion")

    print(
        f"Changed Pixels : "
        f"{np.count_nonzero(spatial_difference)}"
    )

    print(
        f"Maximum Diff   : "
        f"{spatial_difference.max()}"
    )

    print(
        f"Mean Abs Diff  : "
        f"{spatial_difference.mean():.10f}"
    )

    # ------------------------------------------------------
    # DWT AGAIN
    # ------------------------------------------------------

    recovered_bands = apply_dwt(
        reconstructed_uint8
    )

    recovered_lh = (
        recovered_bands["LH"]
    )

    # ------------------------------------------------------
    # Coefficient comparison
    # ------------------------------------------------------

    coefficient_difference = np.abs(
        recovered_lh - stego_lh
    )

    print()
    print("Embedded LH vs Recovered LH")

    print(
        f"Changed Coefficients : "
        f"{np.count_nonzero(coefficient_difference)}"
    )

    print(
        f"Maximum Difference   : "
        f"{coefficient_difference.max():.6f}"
    )

    print(
        f"Mean Absolute Diff   : "
        f"{coefficient_difference.mean():.6f}"
    )

    # ------------------------------------------------------
    # Find largest coefficient errors
    # ------------------------------------------------------

    flat = coefficient_difference.ravel()

    largest_indices = np.argsort(
        flat
    )[-10:]

    print()
    print("Largest LH Errors")

    for flat_index in reversed(
        largest_indices
    ):

        row, column = np.unravel_index(
            flat_index,
            coefficient_difference.shape
        )

        print(
            f"({row}, {column}) "
            f"Embedded={stego_lh[row, column]:.6f} "
            f"Recovered={recovered_lh[row, column]:.6f} "
            f"Diff={coefficient_difference[row, column]:.6f}"
        )

    # ------------------------------------------------------
    # Extract again
    # ------------------------------------------------------

    recovered_edge_map = generate_edge_map(
        recovered_lh
    )

    recovered_payload = adaptive_extract(
        recovered_lh,
        recovered_edge_map,
        len(PAYLOAD)
    )

    mismatch = first_mismatch(
        PAYLOAD,
        recovered_payload
    )

    print()
    print(
        "Extraction After DWT Round-Trip : "
        f"{'PASS' if mismatch is None else 'FAIL'}"
    )

    print(
        f"First Mismatch : {mismatch}"
    )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print()
    print("=" * 70)

    if mismatch is None:

        print(
            "DWT ROUND-TRIP TEST PASSED"
        )

    else:

        print(
            "DWT ROUND-TRIP COEFFICIENT "
            "DISTORTION CONFIRMED"
        )

    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()