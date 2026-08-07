"""
------------------------------------------------------------
StegaFusion Adaptive LSB Engine
------------------------------------------------------------
Implements adaptive embedding and extraction using
an edge map generated from DWT coefficients.

Author      : Yashwanth Gowda M
Version     : 2.0.0
Created     : August 2026
------------------------------------------------------------
"""

import numpy as np

from modules.steganography.lsb_utils import (
    coefficient_to_int,
    coefficient_to_float,
    embed_one_bit,
    embed_two_bits,
    extract_one_bit,
    extract_two_bits,
)


# ==========================================================
# EMBEDDING CAPACITY
# ==========================================================

def calculate_capacity(edge_map: np.ndarray) -> int:
    """
    Calculate embedding capacity.

    Smooth coefficient -> 1 bit
    Edge coefficient   -> 2 bits
    """

    edge_pixels = np.count_nonzero(edge_map)
    smooth_pixels = edge_map.size - edge_pixels

    return edge_pixels * 2 + smooth_pixels


# ==========================================================
# EMBEDDING
# ==========================================================

def adaptive_embed(
    coefficients: np.ndarray,
    edge_map: np.ndarray,
    payload_bits: str,
):
    """
    Adaptive LSB embedding.

    Returns
    -------
    stego_coefficients
    bits_embedded
    """

    stego = coefficients.copy()

    bit_index = 0

    rows, cols = stego.shape

    for r in range(rows):

        for c in range(cols):

            if bit_index >= len(payload_bits):
                return stego, bit_index

            value = coefficient_to_int(stego[r, c])

            if edge_map[r, c] == 255:

                bits = payload_bits[bit_index:bit_index + 2]

                if len(bits) < 2:
                    bits = bits.ljust(2, "0")

                value = embed_two_bits(value, bits)

                bit_index += 2

            else:

                bits = payload_bits[bit_index]

                value = embed_one_bit(value, bits)

                bit_index += 1

            stego[r, c] = coefficient_to_float(value)

    return stego, bit_index


# ==========================================================
# EXTRACTION
# ==========================================================

def adaptive_extract(
    coefficients: np.ndarray,
    edge_map: np.ndarray,
    total_bits: int,
):
    """
    Recover payload bits from coefficients.
    """

    recovered = ""

    rows, cols = coefficients.shape

    for r in range(rows):

        for c in range(cols):

            if len(recovered) >= total_bits:
                return recovered[:total_bits]

            value = coefficient_to_int(coefficients[r, c])

            if edge_map[r, c] == 255:

                recovered += extract_two_bits(value)

            else:

                recovered += extract_one_bit(value)

    return recovered[:total_bits]


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Adaptive LSB Engine Test")
    print("=" * 70)

    coeff = np.array(
        [
            [101.4, 88.7],
            [55.2, 199.8]
        ]
    )

    edge = np.array(
        [
            [255, 0],
            [255, 0]
        ],
        dtype=np.uint8
    )

    payload = "110010"

    print(f"Capacity : {calculate_capacity(edge)} bits")

    stego, embedded = adaptive_embed(
        coeff,
        edge,
        payload
    )

    recovered = adaptive_extract(
        stego,
        edge,
        embedded
    )

    print(f"Embedded : {embedded} bits")
    print(f"Recovered: {recovered}")

    if recovered == payload:
        print("\nAdaptive LSB Engine Passed!")
    else:
        print("\nAdaptive LSB Engine Failed!")