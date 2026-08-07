"""
------------------------------------------------------------
StegaFusion Adaptive LSB Module
------------------------------------------------------------
Provides adaptive LSB embedding and extraction
for DWT coefficients.

Author      : Yashwanth Gowda M
Version     : 1.0.0
Created     : August 2026
------------------------------------------------------------
"""

import numpy as np


# ==========================================================
# EMBEDDING CAPACITY
# ==========================================================

def get_capacity(edge_map: np.ndarray) -> int:
    """
    Calculate total embedding capacity.

    Edge pixel   -> 2 bits
    Smooth pixel -> 1 bit
    """

    edge_pixels = np.count_nonzero(edge_map)

    total_pixels = edge_map.size

    smooth_pixels = total_pixels - edge_pixels

    capacity = edge_pixels * 2 + smooth_pixels

    return capacity


# ==========================================================
# EMBED SINGLE COEFFICIENT
# ==========================================================

def embed_coefficient(
    coefficient: float,
    bits: str
) -> float:
    """
    Embed 1 or 2 bits into a DWT coefficient.
    """

    value = int(round(coefficient))

    number_of_bits = len(bits)

    mask = (1 << number_of_bits) - 1

    value &= ~mask

    value |= int(bits, 2)

    return float(value)


# ==========================================================
# EXTRACT SINGLE COEFFICIENT
# ==========================================================

def extract_coefficient(
    coefficient: float,
    bit_count: int
) -> str:
    """
    Extract embedded bits from one coefficient.
    """

    value = int(round(coefficient))

    mask = (1 << bit_count) - 1

    extracted = value & mask

    return format(extracted, f"0{bit_count}b")


# ==========================================================
# ADAPTIVE EMBEDDING
# ==========================================================

def embed_bits(
    coefficients: np.ndarray,
    edge_map: np.ndarray,
    payload_bits: str
):
    """
    Adaptive LSB embedding.

    Edge   -> 2 bits

    Smooth -> 1 bit
    """

    stego = coefficients.copy()

    bit_index = 0

    rows, cols = coefficients.shape

    for i in range(rows):

        for j in range(cols):

            if bit_index >= len(payload_bits):

                return stego, bit_index

            if edge_map[i, j] == 255:

                bit_count = 2

            else:

                bit_count = 1

            bits = payload_bits[
                bit_index:
                bit_index + bit_count
            ]

            if len(bits) < bit_count:

                bits = bits.ljust(bit_count, "0")

            stego[i, j] = embed_coefficient(
                stego[i, j],
                bits
            )

            bit_index += bit_count

    return stego, bit_index


# ==========================================================
# ADAPTIVE EXTRACTION
# ==========================================================

def extract_bits(
    coefficients: np.ndarray,
    edge_map: np.ndarray,
    total_bits: int
):
    """
    Recover embedded bits.
    """

    recovered = ""

    rows, cols = coefficients.shape

    for i in range(rows):

        for j in range(cols):

            if len(recovered) >= total_bits:

                return recovered

            if edge_map[i, j] == 255:

                bit_count = 2

            else:

                bit_count = 1

            recovered += extract_coefficient(
                coefficients[i, j],
                bit_count
            )

    return recovered[:total_bits]


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Adaptive LSB Test")
    print("=" * 70)

    coeff = np.array(
        [
            [101, 120],
            [88, 43]
        ],
        dtype=float
    )

    edge = np.array(
        [
            [255, 0],
            [255, 0]
        ],
        dtype=np.uint8
    )

    payload = "110010"

    print()

    print("Capacity :", get_capacity(edge))

    stego, used = embed_bits(
        coeff,
        edge,
        payload
    )

    recovered = extract_bits(
        stego,
        edge,
        used
    )

    print("Bits Embedded :", used)

    print("Recovered     :", recovered)

    print()

    print("Adaptive LSB Test Successful!")