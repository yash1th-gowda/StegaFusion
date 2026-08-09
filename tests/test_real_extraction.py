"""
------------------------------------------------------------
StegaFusion Real Payload Extraction Test
------------------------------------------------------------
Tests complete single-frame recovery using the SAME
blue-channel DWT pipeline used during embedding.

Pipeline:

    Stego Frame
        ↓
    Blue Channel
        ↓
    Haar DWT
        ↓
    LH Band
        ↓
    Adaptive Edge Map
        ↓
    Adaptive Extraction
        ↓
    Payload Packet
        ↓
    Encrypted Data
        ↓
    AES-256 CBC Decryption
        ↓
    Original Secret Comparison

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path

import numpy as np

from modules.steganography.lsb_embed import load_frame
from modules.steganography.edge_detector import generate_edge_map
from modules.steganography.adaptive_lsb import adaptive_extract
from modules.steganography.payload import (
    parse_payload_packet,
    binary_to_bytes,
)
from modules.transform.wavelet_utils import apply_dwt
from modules.crypto.aes_decrypt import decrypt_file


# ==========================================================
# PATHS
# ==========================================================

STEGO_FRAME = Path(
    "temp/stego_frames/real_payload_frame.png"
)

EXPECTED_PAYLOAD_FILE = Path(
    "temp/test_payload.txt"
)

SECRET_FILE = Path(
    "input/secret_data/test_secret.txt"
)

KEY_FILE = Path(
    "input/keys/test_integration_key.bin"
)


# ==========================================================
# MAIN TEST
# ==========================================================

def main():

    print("=" * 70)
    print("StegaFusion Real Payload Extraction Test")
    print("=" * 70)

    # ------------------------------------------------------
    # Validate files
    # ------------------------------------------------------

    if not STEGO_FRAME.exists():
        raise FileNotFoundError(
            f"Stego frame not found: {STEGO_FRAME}"
        )

    if not EXPECTED_PAYLOAD_FILE.exists():
        raise FileNotFoundError(
            f"Expected payload not found: "
            f"{EXPECTED_PAYLOAD_FILE}"
        )

    if not SECRET_FILE.exists():
        raise FileNotFoundError(
            f"Secret file not found: {SECRET_FILE}"
        )

    if not KEY_FILE.exists():
        raise FileNotFoundError(
            f"AES key not found: {KEY_FILE}"
        )

    # ------------------------------------------------------
    # Load stego frame
    # ------------------------------------------------------

    image = load_frame(
        STEGO_FRAME
    )

    print()
    print(
        f"Stego Frame : {STEGO_FRAME}"
    )

    print(
        f"Shape       : {image.shape}"
    )

    print(
        f"Dtype       : {image.dtype}"
    )

    # ------------------------------------------------------
    # Extract BLUE channel
    #
    # IMPORTANT:
    # The embedding test uses image[:, :, 0].
    # Therefore extraction MUST use the same channel.
    # ------------------------------------------------------

    blue_channel = image[:, :, 0]

    print(
        f"Blue Min    : {blue_channel.min()}"
    )

    print(
        f"Blue Max    : {blue_channel.max()}"
    )

    # ------------------------------------------------------
    # Apply DWT
    # ------------------------------------------------------

    bands = apply_dwt(
        blue_channel
    )

    # ------------------------------------------------------
    # Generate SAME edge map
    # ------------------------------------------------------

    edge_map = generate_edge_map(
        bands["LH"]
    )

    edge_pixels = np.count_nonzero(
        edge_map
    )

    print()
    print(
        f"Edge Pixels : {edge_pixels}"
    )

    # ------------------------------------------------------
    # Read EXACT payload used during embedding
    # ------------------------------------------------------

    expected_payload = (
        EXPECTED_PAYLOAD_FILE.read_text(
            encoding="ascii"
        ).strip()
    )

    expected_bits = len(
        expected_payload
    )

    print(
        f"Expected Payload : "
        f"{expected_bits} bits"
    )

    # ------------------------------------------------------
    # Extract exact number of bits
    # ------------------------------------------------------

    recovered_packet = adaptive_extract(
        bands["LH"],
        edge_map,
        expected_bits
    )

    print(
        f"Recovered Bits   : "
        f"{len(recovered_packet)}"
    )

    # ------------------------------------------------------
    # Compare extracted packet with the EXACT packet
    # used by the embedding test.
    # ------------------------------------------------------

    if recovered_packet == expected_payload:

        print()
        print(
            "Packet Extraction : PASS"
        )

    else:

        print()
        print(
            "Packet Extraction : FAIL"
        )

        # Find first mismatch.

        limit = min(
            len(expected_payload),
            len(recovered_packet)
        )

        mismatch = None

        for index in range(limit):

            if (
                expected_payload[index]
                != recovered_packet[index]
            ):

                mismatch = index
                break

        if mismatch is None:
            mismatch = limit

        print(
            f"First Mismatch : {mismatch}"
        )

        print(
            "Expected:",
            expected_payload[
                max(0, mismatch - 16):
                mismatch + 16
            ]
        )

        print(
            "Recovered:",
            recovered_packet[
                max(0, mismatch - 16):
                mismatch + 16
            ]
        )

        raise RuntimeError(
            "Extracted packet does not match "
            "the packet used for embedding."
        )

    # ------------------------------------------------------
    # Parse payload packet
    # ------------------------------------------------------

    payload_length, encrypted_bits = (
        parse_payload_packet(
            recovered_packet
        )
    )

    print()
    print(
        f"Packet Payload Length : "
        f"{payload_length} bits"
    )

    print(
        f"Encrypted Bits        : "
        f"{len(encrypted_bits)} bits"
    )

    # ------------------------------------------------------
    # Convert encrypted binary → bytes
    # ------------------------------------------------------

    encrypted_data = binary_to_bytes(
        encrypted_bits
    )

    print(
        f"Encrypted Data        : "
        f"{len(encrypted_data)} bytes"
    )

    # ------------------------------------------------------
    # AES-256 CBC decryption
    # ------------------------------------------------------

    recovered_data = decrypt_file(
        encrypted_data,
        KEY_FILE
    )

    print(
        f"Recovered Data        : "
        f"{len(recovered_data)} bytes"
    )

    # ------------------------------------------------------
    # Read original secret
    # ------------------------------------------------------

    original_data = SECRET_FILE.read_bytes()

    print(
        f"Original Data         : "
        f"{len(original_data)} bytes"
    )

    # ------------------------------------------------------
    # Compare recovered secret
    # ------------------------------------------------------

    if recovered_data != original_data:

        print()
        print(
            "Original Data Match   : FAIL"
        )

        limit = min(
            len(recovered_data),
            len(original_data)
        )

        mismatch = None

        for index in range(limit):

            if (
                recovered_data[index]
                != original_data[index]
            ):

                mismatch = index
                break

        if mismatch is None:
            mismatch = limit

        print(
            f"First Data Mismatch : "
            f"{mismatch}"
        )

        raise RuntimeError(
            "Recovered data does not match "
            "the original secret."
        )

    # ------------------------------------------------------
    # Final validation
    # ------------------------------------------------------

    print()
    print(
        "AES Decryption        : PASS"
    )

    print(
        "Original Data Match   : PASS"
    )

    print()
    print("=" * 70)

    print(
        "REAL PAYLOAD EXTRACTION TEST PASSED!"
    )

    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()