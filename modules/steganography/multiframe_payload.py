"""
StegaFusion Multi-Frame Payload Utilities

Splits a binary payload into fixed-size chunks suitable for
distribution across multiple video frames.

This module does not perform:
    - encryption
    - steganographic embedding
    - video encoding
    - video decoding

It only handles payload chunking and reconstruction.
"""

from typing import List


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_CHUNK_BITS = 768


# ==========================================================
# VALIDATION
# ==========================================================

def _validate_payload(payload: str) -> None:
    """
    Validate that payload contains only binary characters.
    """

    if not isinstance(payload, str):
        raise TypeError(
            "payload must be a string."
        )

    if any(
        bit not in "01"
        for bit in payload
    ):
        raise ValueError(
            "Payload must contain only '0' and '1'."
        )


def _validate_chunk_size(
    chunk_size: int,
) -> None:
    """
    Validate chunk size.
    """

    if not isinstance(
        chunk_size,
        int,
    ):
        raise TypeError(
            "chunk_size must be an integer."
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )


# ==========================================================
# SPLIT PAYLOAD
# ==========================================================

def split_payload(
    payload: str,
    chunk_size: int = DEFAULT_CHUNK_BITS,
) -> List[str]:
    """
    Split a binary payload into fixed-size chunks.

    Parameters
    ----------
    payload:
        Binary string containing only 0 and 1.

    chunk_size:
        Maximum number of bits per chunk.

    Returns
    -------
    list[str]
        List of binary payload chunks.

    Example
    -------
    A 2000-bit payload with a 768-bit chunk size becomes:

        chunk 0 -> 768 bits
        chunk 1 -> 768 bits
        chunk 2 -> 464 bits
    """

    _validate_payload(
        payload
    )

    _validate_chunk_size(
        chunk_size
    )

    if not payload:
        return []

    return [
        payload[
            start:start + chunk_size
        ]
        for start in range(
            0,
            len(payload),
            chunk_size,
        )
    ]


# ==========================================================
# REASSEMBLE PAYLOAD
# ==========================================================

def combine_chunks(
    chunks: List[str],
) -> str:
    """
    Reassemble payload chunks.

    Parameters
    ----------
    chunks:
        List of binary strings.

    Returns
    -------
    str
        Original reconstructed binary payload.
    """

    if not isinstance(
        chunks,
        list,
    ):
        raise TypeError(
            "chunks must be a list."
        )

    for chunk in chunks:

        if not isinstance(
            chunk,
            str,
        ):
            raise TypeError(
                "Each chunk must be a string."
            )

        if any(
            bit not in "01"
            for bit in chunk
        ):
            raise ValueError(
                "Chunks must contain only "
                "'0' and '1'."
            )

    return "".join(
        chunks
    )


# ==========================================================
# CHUNK COUNT
# ==========================================================

def calculate_chunk_count(
    payload_bits: int,
    chunk_size: int = DEFAULT_CHUNK_BITS,
) -> int:
    """
    Calculate the number of chunks required.

    Parameters
    ----------
    payload_bits:
        Total payload length in bits.

    chunk_size:
        Maximum bits per chunk.

    Returns
    -------
    int
        Number of chunks required.
    """

    if not isinstance(
        payload_bits,
        int,
    ):
        raise TypeError(
            "payload_bits must be an integer."
        )

    if payload_bits < 0:
        raise ValueError(
            "payload_bits cannot be negative."
        )

    _validate_chunk_size(
        chunk_size
    )

    if payload_bits == 0:
        return 0

    return (
        payload_bits +
        chunk_size -
        1
    ) // chunk_size


# ==========================================================
# CHUNK INFORMATION
# ==========================================================

def get_chunk_sizes(
    payload_bits: int,
    chunk_size: int = DEFAULT_CHUNK_BITS,
) -> List[int]:
    """
    Return the size of every payload chunk.

    Example
    -------
    2000 bits with 768-bit chunks:

        [768, 768, 464]
    """

    count = calculate_chunk_count(
        payload_bits,
        chunk_size,
    )

    if count == 0:
        return []

    full_chunks = (
        payload_bits //
        chunk_size
    )

    remainder = (
        payload_bits %
        chunk_size
    )

    sizes = [
        chunk_size
        for _ in range(
            full_chunks
        )
    ]

    if remainder:
        sizes.append(
            remainder
        )

    return sizes


# ==========================================================
# MAIN TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "StegaFusion Multi-Frame Payload Utilities"
    )
    print("=" * 70)

    payload = (
        "101100111000111100001111"
        * 84
    )

    payload = payload[:2000]

    chunks = split_payload(
        payload,
        DEFAULT_CHUNK_BITS,
    )

    reconstructed = combine_chunks(
        chunks
    )

    print()
    print(
        f"Original Payload : "
        f"{len(payload)} bits"
    )

    print(
        f"Chunk Size       : "
        f"{DEFAULT_CHUNK_BITS} bits"
    )

    print(
        f"Chunk Count      : "
        f"{len(chunks)}"
    )

    print()

    for index, chunk in enumerate(
        chunks
    ):

        print(
            f"Chunk {index:2d} : "
            f"{len(chunk)} bits"
        )

    print()

    print(
        f"Reconstructed    : "
        f"{len(reconstructed)} bits"
    )

    print(
        f"Original Match   : "
        f"{reconstructed == payload}"
    )

    print()

    if reconstructed == payload:

        print(
            "RESULT: PAYLOAD CHUNKING PASS"
        )

    else:

        print(
            "RESULT: PAYLOAD CHUNKING FAIL"
        )