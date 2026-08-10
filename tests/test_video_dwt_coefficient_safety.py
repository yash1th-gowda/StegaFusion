"""
StegaFusion DWT Coefficient Safety Diagnostic

Examines the spatial pixels affected by the LH coefficient
responsible for the first payload mismatch.

The purpose is to prove that changing the coefficient from
0 -> 1 causes an out-of-range spatial value.
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

FRAME_INDEX = 40

TARGET_BIT = 22


def find_embedding_position(
    edge_map,
    payload_length,
    target_index,
):
    """
    Find the LH coordinate used for a particular payload bit.

    This mirrors the current smooth-pixel traversal used by
    the diagnostic pipeline.
    """

    height, width = edge_map.shape

    bit_index = 0

    for row in range(height):

        for col in range(width):

            if edge_map[row, col] == 0:

                if bit_index == target_index:

                    return row, col

                bit_index += 1

                if bit_index >= payload_length:

                    return None

    return None


def main():

    print("=" * 80)
    print(
        "StegaFusion DWT Coefficient Safety Diagnostic"
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
        f"Blue Range : "
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

    # ------------------------------------------------------
    # Edge map
    # ------------------------------------------------------

    edge_map = generate_edge_map(
        original_lh
    )

    # ------------------------------------------------------
    # Find coefficient
    # ------------------------------------------------------

    position = find_embedding_position(
        edge_map,
        len(PAYLOAD),
        TARGET_BIT
    )

    if position is None:

        raise RuntimeError(
            "Could not locate target coefficient."
        )

    row, col = position

    print()
    print(
        f"Payload Bit : {TARGET_BIT}"
    )

    print(
        f"Expected Bit : "
        f"{PAYLOAD[TARGET_BIT]}"
    )

    print(
        f"LH Position : "
        f"({row}, {col})"
    )

    print(
        f"Original LH : "
        f"{original_lh[row, col]:.6f}"
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

    original_value = float(
        original_lh[row, col]
    )

    print()
    print(
        f"Embedded Bits : {embedded}"
    )

    print(
        f"Embedded LH : "
        f"{embedded_value:.6f}"
    )

    print(
        f"LH Change : "
        f"{embedded_value - original_value:.6f}"
    )

    # ------------------------------------------------------
    # Construct stego bands
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

    print()
    print(
        "Reconstructed Image"
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
    # Compare original and reconstructed pixels
    # ------------------------------------------------------

    difference = (
        reconstructed.astype(np.float64)
        -
        blue.astype(np.float64)
    )

    changed = np.abs(
        difference
    ) > 1e-9

    changed_count = int(
        np.count_nonzero(changed)
    )

    print()
    print(
        f"Changed Spatial Pixels : "
        f"{changed_count}"
    )

    # ------------------------------------------------------
    # Show smallest pixels
    # ------------------------------------------------------

    print()
    print(
        "LOWEST RECONSTRUCTED PIXELS"
    )

    print("-" * 80)

    flat_indices = np.argsort(
        reconstructed.ravel()
    )[:20]

    for flat_index in flat_indices:

        pixel_row, pixel_col = np.unravel_index(
            flat_index,
            reconstructed.shape
        )

        original_pixel = float(
            blue[pixel_row, pixel_col]
        )

        reconstructed_pixel = float(
            reconstructed[
                pixel_row,
                pixel_col
            ]
        )

        delta = (
            reconstructed_pixel
            -
            original_pixel
        )

        print(
            f"({pixel_row:4d}, {pixel_col:4d}) "
            f"Original={original_pixel:8.3f} "
            f"Reconstructed={reconstructed_pixel:8.3f} "
            f"Delta={delta:8.3f}"
        )

    # ------------------------------------------------------
    # Count out-of-range values
    # ------------------------------------------------------

    below_zero = int(
        np.count_nonzero(
            reconstructed < 0
        )
    )

    above_255 = int(
        np.count_nonzero(
            reconstructed > 255
        )
    )

    print()
    print(
        "OUT-OF-RANGE ANALYSIS"
    )

    print(
        f"Below 0   : {below_zero}"
    )

    print(
        f"Above 255 : {above_255}"
    )

    # ------------------------------------------------------
    # Test two conversion methods
    # ------------------------------------------------------

    clipped = np.clip(
        reconstructed,
        0,
        255
    ).astype(np.uint8)

    rounded = np.round(
        reconstructed
    ).clip(
        0,
        255
    ).astype(np.uint8)

    print()
    print(
        "INTEGER CONVERSION"
    )

    print(
        f"Clipped Min : "
        f"{clipped.min()}"
    )

    print(
        f"Rounded Min : "
        f"{rounded.min()}"
    )

    # ------------------------------------------------------
    # DWT after clipped conversion
    # ------------------------------------------------------

    clipped_bands = apply_dwt(
        clipped
    )

    clipped_value = float(
        clipped_bands["LH"][row, col]
    )

    # ------------------------------------------------------
    # DWT after rounded conversion
    # ------------------------------------------------------

    rounded_bands = apply_dwt(
        rounded
    )

    rounded_value = float(
        rounded_bands["LH"][row, col]
    )

    print()
    print(
        "TARGET COEFFICIENT AFTER CONVERSION"
    )

    print(
        f"Embedded LH : "
        f"{embedded_value:.6f}"
    )

    print(
        f"After clip  : "
        f"{clipped_value:.6f}"
    )

    print(
        f"After round : "
        f"{rounded_value:.6f}"
    )

    # ------------------------------------------------------
    # Final conclusion
    # ------------------------------------------------------

    print()
    print("=" * 80)

    if (
        embedded_value != clipped_value
        and
        below_zero > 0
    ):

        print(
            "CONFIRMED:"
        )

        print(
            "The coefficient modification causes "
            "out-of-range spatial values."
        )

        print(
            "Converting the reconstructed image to "
            "uint8 destroys the coefficient change."
        )

        print()
        print(
            "The correct fix should prevent unsafe "
            "coefficient modifications."
        )

    else:

        print(
            "The coefficient safety problem was "
            "not reproduced."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()