"""
------------------------------------------------------------
StegaFusion AES Decryption Module
------------------------------------------------------------
Decrypts AES-256 CBC encrypted ciphertext.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from Crypto.Cipher import AES

from modules.crypto.crypto_utils import (
    decode_base64,
    unpad_data,
)


# ==========================================================
# AES DECRYPTION
# ==========================================================

def decrypt(ciphertext: str, key: bytes, iv: bytes) -> str:
    """
    Decrypt AES-256 CBC encrypted ciphertext.

    Args:
        ciphertext (str): Base64 encoded ciphertext.
        key (bytes): AES-256 key.
        iv (bytes): Initialization Vector.

    Returns:
        str: Original plaintext.
    """

    # Decode Base64
    encrypted_bytes = decode_base64(ciphertext)

    # Create AES Cipher
    cipher = AES.new(
        key,
        AES.MODE_CBC,
        iv
    )

    # Decrypt
    padded_plaintext = cipher.decrypt(encrypted_bytes)

    # Remove PKCS7 Padding
    plaintext = unpad_data(padded_plaintext)

    return plaintext.decode("utf-8")


# ==========================================================
# TESTING
# ==========================================================

if __name__ == "__main__":

    from modules.crypto.key_manager import generate_key
    from modules.crypto.crypto_utils import generate_iv
    from modules.crypto.aes_encrypt import encrypt

    print("=" * 70)
    print("StegaFusion AES Decryption Test")
    print("=" * 70)

    original_message = "Hello StegaFusion!"

    key = generate_key()
    iv = generate_iv()

    encrypted = encrypt(
        plaintext=original_message,
        key=key,
        iv=iv
    )

    decrypted = decrypt(
        ciphertext=encrypted,
        key=key,
        iv=iv
    )

    print(f"Original Message : {original_message}")
    print()
    print(f"Encrypted Text   : {encrypted}")
    print()
    print(f"Decrypted Text   : {decrypted}")
    print()

    if original_message == decrypted:
        print("AES Encryption and Decryption Successful!")
    else:
        print("Decryption Failed!")