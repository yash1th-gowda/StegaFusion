"""
------------------------------------------------------------
StegaFusion Key Manager
------------------------------------------------------------
Handles AES-256 key generation, storage and retrieval.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path
from Crypto.Random import get_random_bytes

from config.config import AESConfig, PathConfig


# ==========================================================
# KEY GENERATION
# ==========================================================

def generate_key() -> bytes:
    """
    Generates a secure AES-256 encryption key.

    Returns:
        bytes: A randomly generated 32-byte AES key.
    """
    return get_random_bytes(AESConfig.KEY_SIZE)


# ==========================================================
# SAVE KEY
# ==========================================================

def save_key(key: bytes, filename: str = "aes_key.bin") -> Path:
    """
    Saves the AES key to the input/keys directory.

    Args:
        key (bytes): AES encryption key.
        filename (str): Name of the key file.

    Returns:
        Path: Full path of the saved key file.
    """

    key_path = PathConfig.KEY_DIR / filename

    with open(key_path, "wb") as file:
        file.write(key)

    return key_path


# ==========================================================
# LOAD KEY
# ==========================================================

def load_key(filename: str = "aes_key.bin") -> bytes:
    """
    Loads an AES key from the input/keys directory.

    Args:
        filename (str): Name of the key file.

    Returns:
        bytes: Loaded AES key.
    """

    key_path = PathConfig.KEY_DIR / filename

    with open(key_path, "rb") as file:
        return file.read()


# ==========================================================
# INITIALIZATION VECTOR (IV)
# ==========================================================

def generate_iv() -> bytes:
    """
    Generates a secure Initialization Vector (IV).

    Returns:
        bytes: A randomly generated 16-byte IV.
    """
    return get_random_bytes(AESConfig.IV_SIZE)


# ==========================================================
# KEY VALIDATION
# ==========================================================

def validate_key(key: bytes) -> bool:
    """
    Validates whether the AES key is of the correct length.

    Args:
        key (bytes): AES key.

    Returns:
        bool: True if the key size is valid, otherwise False.
    """

    return len(key) == AESConfig.KEY_SIZE


# ==========================================================
# MAIN (FOR TESTING)
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Key Manager Test")
    print("=" * 70)

    # Generate a new AES key
    key = generate_key()
    print(f"Generated Key Size : {len(key)} bytes")

    # Save the key
    key_path = save_key(key)
    print(f"Key Saved At       : {key_path}")

    # Load the key
    loaded_key = load_key()
    print(f"Loaded Key Size    : {len(loaded_key)} bytes")

    # Generate IV
    iv = generate_iv()
    print(f"Generated IV Size  : {len(iv)} bytes")

    # Validate Key
    print(f"Key Valid          : {validate_key(loaded_key)}")

    print("\nKey Manager Test Completed Successfully!")