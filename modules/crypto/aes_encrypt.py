"""
------------------------------------------------------------
StegaFusion AES Encryption Module
------------------------------------------------------------
Provides AES-256 CBC encryption for text and binary files.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path

from Crypto.Cipher import AES

from modules.crypto.crypto_utils import (
    generate_iv,
    pad_data,
    encode_base64,
)
from modules.crypto.key_manager import load_key


# ==========================================================
# TEXT ENCRYPTION
# ==========================================================

def encrypt(
    plaintext: str,
    key: bytes,
    iv: bytes
) -> str:
    """
    Encrypt plaintext using AES-256 CBC.

    This function is retained for text-based testing
    and backward compatibility.
    """

    plaintext_bytes = plaintext.encode("utf-8")

    encrypted = encrypt_bytes(
        plaintext_bytes,
        key,
        iv
    )

    return encode_base64(encrypted)


# ==========================================================
# BINARY ENCRYPTION
# ==========================================================

def encrypt_bytes(
    data: bytes,
    key: bytes,
    iv: bytes
) -> bytes:
    """
    Encrypt arbitrary binary data using AES-256 CBC.

    Args:
        data: Plaintext bytes.
        key: 32-byte AES-256 key.
        iv: 16-byte initialization vector.

    Returns:
        Encrypted bytes.
    """

    cipher = AES.new(
        key,
        AES.MODE_CBC,
        iv
    )

    padded_data = pad_data(data)

    return cipher.encrypt(padded_data)


# ==========================================================
# FILE ENCRYPTION
# ==========================================================

def encrypt_file(
    input_file: Path,
    key_file: Path
) -> bytes:
    """
    Encrypt an arbitrary file.

    The returned byte stream contains:

        IV + Ciphertext

    This allows the extraction pipeline to recover the
    IV directly from the embedded payload.
    """

    input_file = Path(input_file)
    key_file = Path(key_file)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Secret file not found: {input_file}"
        )

    if not key_file.exists():
        raise FileNotFoundError(
            f"AES key not found: {key_file}"
        )

    key = load_key(
        key_file.name
    )

    if len(key) != 32:
        raise ValueError(
            "Invalid AES-256 key size."
        )

    plaintext = input_file.read_bytes()

    iv = generate_iv()

    ciphertext = encrypt_bytes(
        plaintext,
        key,
        iv
    )

    return iv + ciphertext


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion AES File Encryption Test")
    print("=" * 70)

    from modules.crypto.key_manager import (
        generate_key,
        save_key,
    )

    test_key = generate_key()

    save_key(
        test_key,
        "test_aes_key.bin"
    )

    test_file = Path("input/secret_data/test_secret.txt")

    test_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    test_file.write_text(
        "StegaFusion secure file encryption test.",
        encoding="utf-8"
    )

    encrypted = encrypt_file(
        test_file,
        Path("input/keys/test_aes_key.bin")
    )

    print()
    print(f"Original File : {test_file}")
    print(f"Encrypted Size: {len(encrypted)} bytes")
    print(f"IV Size       : 16 bytes")
    print()

    if len(encrypted) > 16:
        print("AES File Encryption Test Passed!")
    else:
        print("AES File Encryption Test Failed!")