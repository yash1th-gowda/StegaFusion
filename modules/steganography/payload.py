"""
------------------------------------------------------------
StegaFusion Payload Module
------------------------------------------------------------
Handles reading, writing, and converting secret data into
binary format for encryption and steganography.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path


# ==========================================================
# READ SECRET FILE
# ==========================================================

def read_file(file_path: Path) -> bytes:
    """
    Reads a file in binary mode.

    Args:
        file_path (Path): Path to the secret file.

    Returns:
        bytes: File content.
    """

    with open(file_path, "rb") as file:
        return file.read()


# ==========================================================
# WRITE RECOVERED FILE
# ==========================================================

def write_file(file_path: Path, data: bytes) -> None:
    """
    Writes recovered data to a file.

    Args:
        file_path (Path): Output file path.
        data (bytes): Data to write.
    """

    with open(file_path, "wb") as file:
        file.write(data)


# ==========================================================
# BYTES TO BINARY
# ==========================================================

def bytes_to_binary(data: bytes) -> str:
    """
    Converts bytes into a binary string.

    Args:
        data (bytes)

    Returns:
        str
    """

    return "".join(format(byte, "08b") for byte in data)


# ==========================================================
# BINARY TO BYTES
# ==========================================================

def binary_to_bytes(binary_data: str) -> bytes:
    """
    Converts binary string back into bytes.

    Args:
        binary_data (str)

    Returns:
        bytes
    """

    return bytes(
        int(binary_data[i:i + 8], 2)
        for i in range(0, len(binary_data), 8)
    )


# ==========================================================
# PAYLOAD SIZE
# ==========================================================

def payload_size(data: bytes) -> int:
    """
    Returns payload size in bytes.

    Args:
        data (bytes)

    Returns:
        int
    """

    return len(data)

# ==========================================================
# CREATE PAYLOAD PACKET
# ==========================================================

def create_payload_packet(data: bytes) -> str:
    """
    Creates a binary payload packet containing:

        32-bit payload length
        +
        encrypted payload

    Args:
        data (bytes): Encrypted payload.

    Returns:
        str: Binary payload packet.
    """

    payload_bits = bytes_to_binary(data)

    payload_length = len(payload_bits)

    if payload_length > 0xFFFFFFFF:
        raise ValueError(
            "Payload is too large for the 32-bit length header."
        )

    length_header = format(
        payload_length,
        "032b"
    )

    return length_header + payload_bits


# ==========================================================
# PARSE PAYLOAD PACKET
# ==========================================================

def parse_payload_packet(
    packet: str
) -> tuple[int, str]:
    """
    Extracts the payload length and payload bits
    from a binary payload packet.

    Args:
        packet (str): Complete binary payload packet.

    Returns:
        tuple[int, str]:
            Payload length in bits.
            Payload binary data.
    """

    if len(packet) < 32:
        raise ValueError(
            "Payload packet is too small."
        )

    length_header = packet[:32]

    payload_length = int(
        length_header,
        2
    )

    payload_start = 32

    payload_end = (
        payload_start +
        payload_length
    )

    if len(packet) < payload_end:
        raise ValueError(
            "Incomplete payload packet."
        )

    payload_bits = packet[
        payload_start:payload_end
    ]

    return payload_length, payload_bits

# ==========================================================
# TESTING
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Payload Module Test")
    print("=" * 70)

    sample_data = b"Hello StegaFusion!"

    # ------------------------------------------------------
    # Bytes → Binary
    # ------------------------------------------------------

    binary = bytes_to_binary(
        sample_data
    )

    # ------------------------------------------------------
    # Binary → Bytes
    # ------------------------------------------------------

    recovered = binary_to_bytes(
        binary
    )

    # ------------------------------------------------------
    # Create Payload Packet
    # ------------------------------------------------------

    packet = create_payload_packet(
        sample_data
    )

    # ------------------------------------------------------
    # Parse Payload Packet
    # ------------------------------------------------------

    length, recovered_bits = parse_payload_packet(
        packet
    )

    packet_recovered = binary_to_bytes(
        recovered_bits
    )

    # ------------------------------------------------------
    # Display Results
    # ------------------------------------------------------

    print()

    print(
        f"Original Data       : {sample_data}"
    )

    print(
        f"Payload Size        : "
        f"{payload_size(sample_data)} bytes"
    )

    print(
        f"Binary Length       : "
        f"{len(binary)} bits"
    )

    print(
        f"Packet Length       : "
        f"{len(packet)} bits"
    )

    print(
        f"Header Length       : 32 bits"
    )

    print(
        f"Payload Length      : "
        f"{length} bits"
    )

    print(
        f"Recovered Data      : {recovered}"
    )

    print(
        f"Packet Recovered    : "
        f"{packet_recovered}"
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    if (
        sample_data == recovered
        and
        sample_data == packet_recovered
        and
        length == len(binary)
    ):

        print(
            "\nPayload Module Test Passed!"
        )

    else:

        print(
            "\nPayload Module Test Failed!"
        )