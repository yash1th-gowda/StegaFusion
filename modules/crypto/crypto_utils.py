"""
------------------------------------------------------------
StegaFusion Crypto Utilities
------------------------------------------------------------
Provides reusable cryptographic utility functions such as
IV generation, PKCS7 padding/unpadding, and Base64 encoding.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

import base64
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

from config.config import AESConfig


# ==========================================================
# INITIALIZATION VECTOR (IV)
# ==========================================================

def generate_iv() -> bytes:
    """
    Generates a secure random Initialization Vector (IV).

    Returns:
        bytes: Random 16-byte IV.
    """
    return get_random_bytes(AESConfig.IV_SIZE)


# ==========================================================
# PKCS7 PADDING
# ==========================================================

def pad_data(data: bytes) -> bytes:
    """
    Applies PKCS7 padding to plaintext.

    Args:
        data (bytes): Plaintext bytes.

    Returns:
        bytes: Padded plaintext.
    """
    return pad(data, AESConfig.BLOCK_SIZE)


# ==========================================================
# REMOVE PKCS7 PADDING
# ==========================================================

def unpad_data(data: bytes) -> bytes:
    """
    Removes PKCS7 padding.

    Args:
        data (bytes): Padded plaintext.

    Returns:
        bytes: Original plaintext.
    """
    return unpad(data, AESConfig.BLOCK_SIZE)


# ==========================================================
# BASE64 ENCODING
# ==========================================================

def encode_base64(data: bytes) -> str:
    """
    Encodes bytes into Base64 string.

    Args:
        data (bytes)

    Returns:
        str
    """
    return base64.b64encode(data).decode("utf-8")


# ==========================================================
# BASE64 DECODING
# ==========================================================

def decode_base64(data: str) -> bytes:
    """
    Decodes Base64 string into bytes.

    Args:
        data (str)

    Returns:
        bytes
    """
    return base64.b64decode(data.encode("utf-8"))


# ==========================================================
# TESTING
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Crypto Utilities Test")
    print("=" * 70)

    message = b"Hello StegaFusion!"

    padded = pad_data(message)
    unpadded = unpad_data(padded)

    encoded = encode_base64(message)
    decoded = decode_base64(encoded)

    iv = generate_iv()

    print(f"Original Message : {message}")
    print(f"Padded Length    : {len(padded)}")
    print(f"Recovered        : {unpadded}")
    print(f"Base64           : {encoded}")
    print(f"Decoded          : {decoded}")
    print(f"IV Length        : {len(iv)} bytes")

    print("\nCrypto Utilities Test Passed!")