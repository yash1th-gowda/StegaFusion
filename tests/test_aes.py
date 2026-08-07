"""
------------------------------------------------------------
StegaFusion AES Test Module
------------------------------------------------------------
Tests the complete AES encryption and decryption workflow.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from modules.crypto.key_manager import generate_key
from modules.crypto.crypto_utils import generate_iv
from modules.crypto.aes_encrypt import encrypt
from modules.crypto.aes_decrypt import decrypt


def test_aes():

    print("=" * 70)
    print("StegaFusion AES Integration Test")
    print("=" * 70)

    original_message = (
        "This is a confidential message hidden inside a video using "
        "StegaFusion."
    )

    # Generate AES Key
    key = generate_key()

    # Generate IV
    iv = generate_iv()

    # Encrypt
    encrypted_message = encrypt(
        plaintext=original_message,
        key=key,
        iv=iv
    )

    # Decrypt
    decrypted_message = decrypt(
        ciphertext=encrypted_message,
        key=key,
        iv=iv
    )

    print(f"Original Message : {original_message}")
    print()
    print(f"Encrypted Message:\n{encrypted_message}")
    print()
    print(f"Decrypted Message : {decrypted_message}")
    print()

    assert original_message == decrypted_message

    print("=" * 70)
    print("AES TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_aes()