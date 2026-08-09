"""
------------------------------------------------------------
StegaFusion LSB Utilities
------------------------------------------------------------
Provides robust quantization-based bit manipulation
functions for adaptive DWT-domain steganography.

The API is intentionally kept compatible with
adaptive_lsb.py.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

# ==========================================================
# QUANTIZATION SETTINGS
# ==========================================================

# Distance between adjacent embedding states.
#
# A larger value improves robustness against DWT/image
# reconstruction errors, at the cost of slightly larger
# coefficient modifications.
#
# Start conservatively with 1.
QUANTIZATION_STEP: int = 1


# ==========================================================
# INTEGER CONVERSION
# ==========================================================

def coefficient_to_int(value: float) -> int:
    """
    Convert a DWT coefficient to the nearest integer.
    """

    return int(round(value))


def coefficient_to_float(value: int) -> float:
    """
    Convert an integer embedding value back to float.
    """

    return float(value)


# ==========================================================
# INTERNAL QUANTIZATION
# ==========================================================

def _nearest_multiple(
    value: int,
    step: int
) -> int:
    """
    Return the nearest multiple of `step`.
    """

    return int(
        round(value / step)
    ) * step


# ==========================================================
# EMBED ONE BIT
# ==========================================================

def embed_one_bit(
    value: int,
    bit: str
) -> int:
    """
    Embed one bit using two quantization states.

    State 0:
        multiple of 8

    State 1:
        multiple of 8 + 4

    Therefore adjacent logical states are separated
    by QUANTIZATION_STEP.
    """

    if bit not in ("0", "1"):
        raise ValueError(
            "Bit must be '0' or '1'."
        )

    step = QUANTIZATION_STEP

    # Two states require a period of 2 * step.
    period = step * 2

    base = (
        round(value / period)
        * period
    )

    if bit == "0":
        return int(base)

    return int(base + step)


# ==========================================================
# EMBED TWO BITS
# ==========================================================

def embed_two_bits(
    value: int,
    bits: str
) -> int:
    """
    Embed two bits using four quantization states.

    States:

        00 -> 0
        01 -> step
        10 -> 2 * step
        11 -> 3 * step
    """

    if len(bits) != 2:
        raise ValueError(
            "Exactly two bits are required."
        )

    if any(bit not in "01" for bit in bits):
        raise ValueError(
            "Bits must contain only '0' and '1'."
        )

    state = int(
        bits,
        2
    )

    step = QUANTIZATION_STEP

    period = step * 4

    base = (
        round(value / period)
        * period
    )

    return int(
        base + state * step
    )


# ==========================================================
# EXTRACT ONE BIT
# ==========================================================

def extract_one_bit(
    value: int
) -> str:
    """
    Extract one bit from a quantized coefficient.

    The coefficient is mapped to the nearest quantization
    state before decoding.
    """

    step = QUANTIZATION_STEP

    state = int(
        round(value / step)
    )

    return str(
        state % 2
    )


# ==========================================================
# EXTRACT TWO BITS
# ==========================================================

def extract_two_bits(
    value: int
) -> str:
    """
    Extract two bits from a quantized coefficient.
    """

    step = QUANTIZATION_STEP

    state = int(
        round(value / step)
    )

    state %= 4

    return format(
        state,
        "02b"
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Robust LSB Utility Test")
    print("=" * 70)

    print()
    print(
        f"Quantization Step : "
        f"{QUANTIZATION_STEP}"
    )

    # ------------------------------------------------------
    # One-bit tests
    # ------------------------------------------------------

    test_values = [
        -20,
        -7,
        -1,
        0,
        3,
        10,
        17,
        25,
        100,
    ]

    print()
    print("One-Bit Tests")
    print("-" * 70)

    for value in test_values:

        for bit in ("0", "1"):

            embedded = embed_one_bit(
                value,
                bit
            )

            recovered = extract_one_bit(
                embedded
            )

            print(
                f"Value={value:4d} "
                f"Bit={bit} "
                f"Embedded={embedded:5d} "
                f"Recovered={recovered}"
            )

            if recovered != bit:

                raise AssertionError(
                    "One-bit embedding test failed."
                )

    # ------------------------------------------------------
    # Two-bit tests
    # ------------------------------------------------------

    print()
    print("Two-Bit Tests")
    print("-" * 70)

    for value in test_values:

        for bits in (
            "00",
            "01",
            "10",
            "11"
        ):

            embedded = embed_two_bits(
                value,
                bits
            )

            recovered = extract_two_bits(
                embedded
            )

            print(
                f"Value={value:4d} "
                f"Bits={bits} "
                f"Embedded={embedded:5d} "
                f"Recovered={recovered}"
            )

            if recovered != bits:

                raise AssertionError(
                    "Two-bit embedding test failed."
                )

    print()
    print(
        "Robust LSB Utility Test Passed!"
    )