"""
------------------------------------------------------------
StegaFusion LSB Utilities
------------------------------------------------------------
Provides low-level bit manipulation functions for
Adaptive LSB embedding.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

import numpy as np


# ==========================================================
# INTEGER CONVERSION
# ==========================================================

def coefficient_to_int(value: float) -> int:
    """
    Convert DWT coefficient to integer.
    """
    return int(round(value))


def coefficient_to_float(value: int) -> float:
    """
    Convert integer back to float.
    """
    return float(value)


# ==========================================================
# EMBED ONE BIT
# ==========================================================

def embed_one_bit(value: int, bit: str) -> int:

    value &= ~1

    value |= int(bit)

    return value


# ==========================================================
# EMBED TWO BITS
# ==========================================================

def embed_two_bits(value: int, bits: str) -> int:

    value &= ~3

    value |= int(bits, 2)

    return value


# ==========================================================
# EXTRACT ONE BIT
# ==========================================================

def extract_one_bit(value: int) -> str:

    return str(value & 1)


# ==========================================================
# EXTRACT TWO BITS
# ==========================================================

def extract_two_bits(value: int) -> str:

    return format(value & 3, "02b")


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("StegaFusion LSB Utility Test")
    print("=" * 60)

    value = 155

    print()

    print("Original :", value)

    one = embed_one_bit(value, "1")

    print("1-bit :", one)

    two = embed_two_bits(value, "10")

    print("2-bit :", two)

    print()

    print(
        "Extract 1:",
        extract_one_bit(one)
    )

    print(
        "Extract 2:",
        extract_two_bits(two)
    )

    print()

    print("LSB Utility Test Passed!")