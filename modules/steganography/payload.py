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
# TESTING
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Payload Module Test")
    print("=" * 70)

    sample_data = b"Hello StegaFusion!"

    binary = bytes_to_binary(sample_data)

    recovered = binary_to_bytes(binary)

    print(f"Original Data      : {sample_data}")
    print(f"Payload Size       : {payload_size(sample_data)} bytes")
    print(f"Binary Length      : {len(binary)} bits")
    print(f"Recovered Data     : {recovered}")

    if sample_data == recovered:
        print("\nPayload Module Test Passed!")
    else:
        print("\nPayload Module Test Failed!")