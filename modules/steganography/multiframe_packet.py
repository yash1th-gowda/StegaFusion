"""
StegaFusion Multi-Frame Packet Format

Packages one payload chunk with frame metadata.

Frame payload layout:

    [ 64-bit HEADER ][ CHUNK DATA ]

Header fields:

    Magic          : 16 bits
    Version        : 8 bits
    Total Chunks   : 12 bits
    Chunk Index    : 12 bits
    Chunk Length   : 16 bits

Total:
    16 + 8 + 12 + 12 + 16 = 64 bits

The packet layer does not perform:
    - AES encryption
    - steganographic embedding
    - video encoding
    - video decoding
"""

from dataclasses import dataclass


# ==========================================================
# CONFIGURATION
# ==========================================================

MAGIC = "1010101011001100"

VERSION = 1

HEADER_BITS = 64

MAX_FRAME_BITS = 768

MAX_CHUNK_DATA_BITS = (
    MAX_FRAME_BITS - HEADER_BITS
)

MAX_TOTAL_CHUNKS = (
    (1 << 12) - 1
)

MAX_CHUNK_INDEX = (
    (1 << 12) - 1
)

MAX_CHUNK_LENGTH = (
    (1 << 16) - 1
)


# ==========================================================
# DATA CLASS
# ==========================================================

@dataclass(frozen=True)
class FramePacket:
    """
    Decoded multi-frame packet metadata.
    """

    version: int
    total_chunks: int
    chunk_index: int
    chunk_length: int
    chunk_data: str


# ==========================================================
# INTEGER / BINARY HELPERS
# ==========================================================

def _int_to_bits(
    value: int,
    width: int,
) -> str:
    """
    Convert an integer to a fixed-width binary string.
    """

    if value < 0:
        raise ValueError(
            "value cannot be negative."
        )

    if value >= (1 << width):
        raise ValueError(
            f"value {value} does not fit "
            f"inside {width} bits."
        )

    return format(
        value,
        f"0{width}b",
    )


def _bits_to_int(
    bits: str,
) -> int:
    """
    Convert a binary string to an integer.
    """

    if not bits:
        return 0

    if any(
        bit not in "01"
        for bit in bits
    ):
        raise ValueError(
            "bits must contain only "
            "'0' and '1'."
        )

    return int(
        bits,
        2,
    )


def _validate_binary(
    value: str,
    name: str,
) -> None:
    """
    Validate a binary string.
    """

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{name} must be a string."
        )

    if any(
        bit not in "01"
        for bit in value
    ):
        raise ValueError(
            f"{name} must contain only "
            "'0' and '1'."
        )


# ==========================================================
# HEADER CREATION
# ==========================================================

def create_header(
    total_chunks: int,
    chunk_index: int,
    chunk_length: int,
) -> str:
    """
    Create a 64-bit frame header.
    """

    if total_chunks <= 0:
        raise ValueError(
            "total_chunks must be greater than zero."
        )

    if total_chunks > MAX_TOTAL_CHUNKS:
        raise ValueError(
            "total_chunks exceeds packet format limit."
        )

    if chunk_index < 0:
        raise ValueError(
            "chunk_index cannot be negative."
        )

    if chunk_index >= total_chunks:
        raise ValueError(
            "chunk_index must be smaller "
            "than total_chunks."
        )

    if chunk_index > MAX_CHUNK_INDEX:
        raise ValueError(
            "chunk_index exceeds packet format limit."
        )

    if chunk_length <= 0:
        raise ValueError(
            "chunk_length must be greater than zero."
        )

    if chunk_length > MAX_CHUNK_LENGTH:
        raise ValueError(
            "chunk_length exceeds packet format limit."
        )

    header = (
        MAGIC
        + _int_to_bits(
            VERSION,
            8,
        )
        + _int_to_bits(
            total_chunks,
            12,
        )
        + _int_to_bits(
            chunk_index,
            12,
        )
        + _int_to_bits(
            chunk_length,
            16,
        )
    )

    if len(header) != HEADER_BITS:
        raise RuntimeError(
            "Generated header has incorrect length."
        )

    return header


# ==========================================================
# PACKET CREATION
# ==========================================================

def create_packet(
    chunk_data: str,
    total_chunks: int,
    chunk_index: int,
) -> str:
    """
    Create a complete frame packet.

    Returns:

        64-bit header + chunk data
    """

    _validate_binary(
        chunk_data,
        "chunk_data",
    )

    if not chunk_data:
        raise ValueError(
            "chunk_data cannot be empty."
        )

    if len(chunk_data) > MAX_CHUNK_DATA_BITS:
        raise ValueError(
            f"chunk_data exceeds safe frame "
            f"limit of {MAX_CHUNK_DATA_BITS} bits."
        )

    header = create_header(
        total_chunks,
        chunk_index,
        len(chunk_data),
    )

    packet = (
        header +
        chunk_data
    )

    if len(packet) > MAX_FRAME_BITS:
        raise RuntimeError(
            "Packet exceeds maximum frame capacity."
        )

    return packet


# ==========================================================
# PACKET PARSING
# ==========================================================

def parse_packet(
    packet: str,
) -> FramePacket:
    """
    Parse a complete frame packet.
    """

    _validate_binary(
        packet,
        "packet",
    )

    if len(packet) < HEADER_BITS:
        raise ValueError(
            "Packet is shorter than the "
            "64-bit header."
        )

    header = packet[
        :HEADER_BITS
    ]

    if header[:16] != MAGIC:
        raise ValueError(
            "Invalid packet magic."
        )

    version = _bits_to_int(
        header[16:24]
    )

    if version != VERSION:
        raise ValueError(
            f"Unsupported packet version: "
            f"{version}"
        )

    total_chunks = _bits_to_int(
        header[24:36]
    )

    chunk_index = _bits_to_int(
        header[36:48]
    )

    chunk_length = _bits_to_int(
        header[48:64]
    )

    if total_chunks <= 0:
        raise ValueError(
            "Invalid total_chunks."
        )

    if chunk_index >= total_chunks:
        raise ValueError(
            "Invalid chunk_index."
        )

    if chunk_length <= 0:
        raise ValueError(
            "Invalid chunk_length."
        )

    chunk_data = packet[
        HEADER_BITS:
        HEADER_BITS + chunk_length
    ]

    if len(chunk_data) != chunk_length:
        raise ValueError(
            "Packet is truncated."
        )

    return FramePacket(
        version=version,
        total_chunks=total_chunks,
        chunk_index=chunk_index,
        chunk_length=chunk_length,
        chunk_data=chunk_data,
    )


# ==========================================================
# MAIN TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "StegaFusion Multi-Frame Packet Test"
    )
    print("=" * 70)

    chunk = (
        "101100111000111100001111"
        * 30
    )[:704]

    total_chunks = 3
    chunk_index = 1

    packet = create_packet(
        chunk,
        total_chunks,
        chunk_index,
    )

    decoded = parse_packet(
        packet
    )

    print()

    print(
        f"Header Bits          : "
        f"{HEADER_BITS}"
    )

    print(
        f"Maximum Frame Bits   : "
        f"{MAX_FRAME_BITS}"
    )

    print(
        f"Maximum Chunk Bits   : "
        f"{MAX_CHUNK_DATA_BITS}"
    )

    print()

    print(
        f"Total Chunks         : "
        f"{decoded.total_chunks}"
    )

    print(
        f"Chunk Index          : "
        f"{decoded.chunk_index}"
    )

    print(
        f"Chunk Length         : "
        f"{decoded.chunk_length}"
    )

    print(
        f"Packet Length        : "
        f"{len(packet)}"
    )

    print(
        f"Chunk Match          : "
        f"{decoded.chunk_data == chunk}"
    )

    print()

    if (
        decoded.total_chunks == total_chunks
        and
        decoded.chunk_index == chunk_index
        and
        decoded.chunk_data == chunk
        and
        len(packet) <= MAX_FRAME_BITS
    ):

        print(
            "RESULT: FRAME PACKET PASS"
        )

    else:

        print(
            "RESULT: FRAME PACKET FAIL"
        )