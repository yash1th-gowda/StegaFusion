"""
------------------------------------------------------------
StegaFusion AES Encryption Module
------------------------------------------------------------
Encrypts plaintext using AES-256 in CBC mode.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from Crypto.Cipher import AES

from modules.crypto.crypto_utils import (
    pad_data,
    encode_base64,
)


# ==========================================================
# AES ENCRYPTION
# ==========================================================

def encrypt(plaintext: str, key: bytes, iv: bytes) -> str:
    """
    Encrypt plaintext using AES-256 CBC mode.

    Args:
        plaintext (str): Message to encrypt.
        key (bytes): AES-256 encryption key.
        iv (bytes): Initialization Vector.

    Returns:
        str: Base64 encoded ciphertext.
    """

    # Create AES Cipher
    cipher = AES.new(
        key,
        AES.MODE_CBC,
        iv
    )

    # Convert plaintext into bytes
    plaintext_bytes = plaintext.encode("utf-8")

    # Apply PKCS7 Padding
    padded_data = pad_data(plaintext_bytes)

    # Encrypt
    ciphertext = cipher.encrypt(padded_data)

    # Convert encrypted bytes to Base64
    encoded_cipher = encode_base64(ciphertext)

    return encoded_cipher


# ==========================================================
# TESTING
# ==========================================================

if __name__ == "__main__":

    from modules.crypto.key_manager import (
        generate_key,
    )

    from modules.crypto.crypto_utils import (
        generate_iv,
    )

    print("=" * 70)
    print("StegaFusion AES Encryption Test")
    print("=" * 70)

    message = "Hello StegaFusion!"

    key = generate_key()

    iv = generate_iv()

    encrypted_text = encrypt(
        plaintext=message,
        key=key,
        iv=iv
    )

    print(f"Original Message : {message}")
    print()

    print(f"Encrypted Text   : {encrypted_text}")
    print()

    print(f"Cipher Length    : {len(encrypted_text)}")
    print()

    print("AES Encryption Successful!")