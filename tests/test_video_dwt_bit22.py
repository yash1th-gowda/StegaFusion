"""
StegaFusion DWT Bit-22 Diagnostic

Tracks the exact DWT coefficient responsible for the first
payload mismatch.

Pipeline:

Original frame
    ↓
DWT
    ↓
LH
    ↓
QIM embedding
    ↓
Coefficient corresponding to payload bit 22
    ↓
Inverse DWT
    ↓
uint8 conversion
    ↓
DWT
    ↓
Recovered coefficient
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

from modules.steganography.lsb_utils import (
    coefficient_to_int,
    extract_one_bit,
)


PAYLOAD = (
    "00000000000000000000001000000000"
    "101100111000111100001111"
    "01010101010101010101010101010101"
    "11110000111100001111000011110000"
)

FRAME_INDEX = 40

MISMATCH_INDEX = 22


def find_embedding_position(
    edge_map,
    payload_length,
    mismatch_index,
):
    """
    Reproduce the adaptive embedding traversal and determine
    which LH coordinate carries the requested payload bit.

    This function assumes adaptive_lsb.py traverses the
    coefficients in row-major order while skipping positions
    according to its edge/smooth selection logic.
    """

    height, width = edge_map.shape

    bit_index = 0

    for row in range(height):

        for col in range(width):

            if edge_map[row, col] == 0:

                if bit_index == mismatch_index:

                    return row, col

                bit_index += 1

                if bit_index >= payload_length:

                    return None

    return None


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


def main():

    print("=" * 80)
    print(
        "StegaFusion DWT Bit-22 Diagnostic"
    )
    print("=" * 80)

    frame_path = (
        PathConfig.FRAME_DIR /
        f"frame_{FRAME_INDEX:05d}.png"
    )

    if not frame_path.exists():

        raise FileNotFoundError(
            f"Frame not found: {frame_path}"
        )

    image = cv2.imread(
        str(frame_path)
    )

    if image is None:

        raise RuntimeError(
            "Unable to read frame."
        )

    blue = image[:, :, 0]

    print()
    print(
        f"Frame : {FRAME_INDEX}"
    )

    print(
        f"Shape : {blue.shape}"
    )

    print(
        f"Range : "
        f"{blue.min()} -> {blue.max()}"
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

    print()
    print(
        "Original LH"
    )

    print(
        f"Shape : {original_lh.shape}"
    )

    # ------------------------------------------------------
    # Edge map
    # ------------------------------------------------------

    edge_map = generate_edge_map(
        original_lh
    )

    print(
        f"Edge Pixels : "
        f"{np.count_nonzero(edge_map)}"
    )

    # ------------------------------------------------------
    # Find coefficient
    # ------------------------------------------------------

    position = find_embedding_position(
        edge_map,
        len(PAYLOAD),
        MISMATCH_INDEX,
    )

    if position is None:

        raise RuntimeError(
            "Unable to determine embedding position."
        )

    row, col = position

    print()
    print(
        f"Payload Bit : "
        f"{MISMATCH_INDEX}"
    )

    print(
        f"Expected Bit : "
        f"{PAYLOAD[MISMATCH_INDEX]}"
    )

    print(
        f"LH Position : "
        f"({row}, {col})"
    )

    original_value = float(
        original_lh[row, col]
    )

    print(
        f"Original LH Value : "
        f"{original_value:.6f}"
    )

    # ------------------------------------------------------
    # Embed
    # ------------------------------------------------------

    stego_lh, embedded = adaptive_embed(
        original_lh,
        edge_map,
        PAYLOAD
    )

    embedded_value = float(
        stego_lh[row, col]
    )

    print()
    print(
        f"Embedded Bits : "
        f"{embedded}"
    )

    print(
        f"Embedded LH Value : "
        f"{embedded_value:.6f}"
    )

    print(
        f"Coefficient Difference : "
        f"{embedded_value - original_value:.6f}"
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
        f"Direct Extraction : "
        f"{'PASS' if direct_mismatch is None else 'FAIL'}"
    )

    print(
        f"Direct Mismatch : "
        f"{direct_mismatch}"
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

    print()
    print(
        "Reconstructed Float Image"
    )

    print(
        f"Min : "
        f"{reconstructed.min():.6f}"
    )

    print(
        f"Max : "
        f"{reconstructed.max():.6f}"
    )

    # ------------------------------------------------------
    # Pixel values around affected region
    # ------------------------------------------------------

    reconstructed_uint8 = np.clip(
        reconstructed,
        0,
        255
    ).astype(np.uint8)

    # ------------------------------------------------------
    # DWT again
    # ------------------------------------------------------

    recovered_bands = apply_dwt(
        reconstructed_uint8
    )

    recovered_lh = (
        recovered_bands["LH"]
    )

    recovered_value = float(
        recovered_lh[row, col]
    )

    print()
    print(
        "Recovered LH"
    )

    print(
        f"Recovered LH Value : "
        f"{recovered_value:.6f}"
    )

    print(
        f"Recovery Error : "
        f"{recovered_value - embedded_value:.6f}"
    )

    # ------------------------------------------------------
    # Quantized values
    # ------------------------------------------------------

    original_int = coefficient_to_int(
        original_value
    )

    embedded_int = coefficient_to_int(
        embedded_value
    )

    recovered_int = coefficient_to_int(
        recovered_value
    )

    print()
    print(
        "Integer Representation"
    )

    print(
        f"Original  : "
        f"{original_int}"
    )

    print(
        f"Embedded  : "
        f"{embedded_int}"
    )

    print(
        f"Recovered : "
        f"{recovered_int}"
    )

    # ------------------------------------------------------
    # Extract bit directly from coefficient
    # ------------------------------------------------------

    direct_bit = extract_one_bit(
        embedded_int
    )

    recovered_bit = extract_one_bit(
        recovered_int
    )

    print()
    print(
        "Bit Interpretation"
    )

    print(
        f"Expected  : "
        f"{PAYLOAD[MISMATCH_INDEX]}"
    )

    print(
        f"Embedded  : "
        f"{direct_bit}"
    )

    print(
        f"Recovered : "
        f"{recovered_bit}"
    )

    # ------------------------------------------------------
    # Final
    # ------------------------------------------------------

    print()
    print("=" * 80)

    if (
        direct_bit ==
        PAYLOAD[MISMATCH_INDEX]
        and
        recovered_bit !=
        PAYLOAD[MISMATCH_INDEX]
    ):

        print(
            "CONFIRMED:"
        )

        print(
            "The DWT round-trip is changing the "
            "quantized state of the coefficient."
        )

        print(
            "The failure occurs in coefficient "
            "reconstruction, not payload generation."
        )

    else:

        print(
            "The expected coefficient transition "
            "was not reproduced."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()