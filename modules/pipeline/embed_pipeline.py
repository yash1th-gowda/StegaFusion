"""
------------------------------------------------------------
StegaFusion Embed Pipeline
------------------------------------------------------------
Coordinates the complete embedding workflow.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path

from modules.crypto.aes_encrypt import encrypt_file
from modules.steganography.payload import bytes_to_binary


def prepare_payload(
    input_file: Path,
    key_file: Path
):
    """
    Encrypt a file and convert it to a binary stream.

    Returns
    -------
    binary_payload : str
    """

    encrypted_bytes = encrypt_file(
        input_file,
        key_file
    )

    binary_payload = bytes_to_binary(
        encrypted_bytes
    )

    return binary_payload


if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Embed Pipeline Test")
    print("=" * 70)

    print()

    print("Pipeline Created Successfully!")

    print()

    print("Next Step : Connect AES -> Payload -> LSB")