"""
------------------------------------------------------------
StegaFusion AES Decryption Module
------------------------------------------------------------
Provides AES-256 CBC decryption for text and binary files.

Author     : Yashwanth Gowda M
Version    : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path

from Crypto.Cipher import AES

from modules.crypto.crypto_utils import (
    decode_base64,
    unpad_data,
)
from modules.crypto.key_manager import load_key


# ==========================================================
# TEXT DECRYPTION
# ==========================================================

def decrypt(
    ciphertext: str,
    key: bytes,
    iv: bytes
) -> str:
    """
    Decrypt Base64 encoded AES ciphertext.
    """

    encrypted_bytes = decode_base64(
        ciphertext
    )

    plaintext = decrypt_bytes(
        encrypted_bytes,
        key,
        iv
    )

    return plaintext.decode("utf-8")


# ==========================================================
# BINARY DECRYPTION
# ==========================================================

def decrypt_bytes(
    ciphertext: bytes,
    key: bytes,
    iv: bytes
) -> bytes:
    """
    Decrypt arbitrary binary data using AES-256 CBC.
    """

    cipher = AES.new(
        key,
        AES.MODE_CBC,
        iv
    )

    padded_plaintext = cipher.decrypt(
        ciphertext
    )

    return unpad_data(
        padded_plaintext
    )


# ==========================================================
# FILE PAYLOAD DECRYPTION
# ==========================================================

def decrypt_file(
    encrypted_data: bytes,
    key_file: Path
) -> bytes:
    """
    Decrypt an encrypted file payload.

    Expected format:

        IV + Ciphertext
    """

    key_file = Path(key_file)

    if not key_file.exists():
        raise FileNotFoundError(
            f"AES key not found: {key_file}"
        )

    if len(encrypted_data) <= 16:
        raise ValueError(
            "Encrypted payload is too small."
        )

    key = load_key(
        key_file.name
    )

    if len(key) != 32:
        raise ValueError(
            "Invalid AES-256 key size."
        )

    iv = encrypted_data[:16]

    ciphertext = encrypted_data[16:]

    return decrypt_bytes(
        ciphertext,
        key,
        iv
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion AES File Decryption Test")
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

    original_data = (
        b"StegaFusion binary encryption test."
    )

    from modules.crypto.aes_encrypt import (
        encrypt_bytes
    )
    from modules.crypto.crypto_utils import (
        generate_iv
    )

    iv = generate_iv()

    ciphertext = encrypt_bytes(
        original_data,
        test_key,
        iv
    )

    encrypted_payload = iv + ciphertext

    recovered = decrypt_file(
        encrypted_payload,
        Path("input/keys/test_aes_key.bin")
    )

    print()
    print(f"Original Size : {len(original_data)} bytes")
    print(f"Encrypted Size: {len(encrypted_payload)} bytes")
    print(f"Recovered Size: {len(recovered)} bytes")
    print()

    if recovered == original_data:
        print("AES File Decryption Test Passed!")
    else:
        print("AES File Decryption Test Failed!")