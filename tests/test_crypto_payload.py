"""
------------------------------------------------------------
StegaFusion Crypto + Payload Integration Test
------------------------------------------------------------
Verifies that encrypted file data can be converted to a
binary stream and reconstructed without corruption.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path

from modules.crypto.aes_encrypt import encrypt_file
from modules.crypto.key_manager import generate_key, save_key
from modules.steganography.payload import (
    bytes_to_binary,
    binary_to_bytes,
)


# ==========================================================
# TEST
# ==========================================================

def main():

    print("=" * 70)
    print("StegaFusion Crypto + Payload Integration Test")
    print("=" * 70)

    # ------------------------------------------------------
    # Test paths
    # ------------------------------------------------------

    secret_file = Path(
        "input/secret_data/test_secret.txt"
    )

    key_file = Path(
        "input/keys/test_integration_key.bin"
    )

    # ------------------------------------------------------
    # Create test secret
    # ------------------------------------------------------

    secret_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    key_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    original_data = (
        b"StegaFusion integration test payload."
    )

    secret_file.write_bytes(
        original_data
    )

    # ------------------------------------------------------
    # Generate test AES key
    # ------------------------------------------------------

    key = generate_key()

    save_key(
        key,
        key_file.name
    )

    # ------------------------------------------------------
    # Encrypt file
    # ------------------------------------------------------

    encrypted_data = encrypt_file(
        secret_file,
        key_file
    )

    # ------------------------------------------------------
    # Convert encrypted bytes → binary
    # ------------------------------------------------------

    binary_payload = bytes_to_binary(
        encrypted_data
    )

    # ------------------------------------------------------
    # Convert binary → encrypted bytes
    # ------------------------------------------------------

    recovered_encrypted_data = binary_to_bytes(
        binary_payload
    )

    # ------------------------------------------------------
    # Display results
    # ------------------------------------------------------

    print()
    print(
        f"Original File Size       : "
        f"{len(original_data)} bytes"
    )

    print(
        f"Encrypted Data Size      : "
        f"{len(encrypted_data)} bytes"
    )

    print(
        f"Binary Payload Size      : "
        f"{len(binary_payload)} bits"
    )

    print(
        f"Recovered Encrypted Size : "
        f"{len(recovered_encrypted_data)} bytes"
    )

    print()

    # ------------------------------------------------------
    # Verify byte-for-byte equality
    # ------------------------------------------------------

    if encrypted_data != recovered_encrypted_data:

        print(
            "Crypto + Payload Integration Test FAILED!"
        )

        raise SystemExit(1)

    print(
        "Crypto + Payload Integration Test PASSED!"
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()