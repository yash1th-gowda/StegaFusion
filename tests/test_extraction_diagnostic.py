"""
StegaFusion Extraction Diagnostic

Determines whether the extraction failure is caused by
edge-map changes between embedding and extraction.
"""

from pathlib import Path

import cv2
import numpy as np

from modules.steganography.edge_detector import generate_edge_map
from modules.steganography.adaptive_lsb import adaptive_extract
from modules.steganography.payload import (
    create_payload_packet,
)
from modules.crypto.aes_encrypt import encrypt_file
from modules.transform.wavelet_utils import apply_dwt


SECRET_FILE = Path(
    "input/secret_data/test_secret.txt"
)

KEY_FILE = Path(
    "input/keys/test_integration_key.bin"
)

ORIGINAL_FRAME = Path(
    "temp/frames/frame_00000.png"
)

STEGO_FRAME = Path(
    "temp/stego_frames/real_payload_frame.png"
)


def main():

    print("=" * 70)
    print("StegaFusion Extraction Diagnostic")
    print("=" * 70)

    # ------------------------------------------------------
    # Create expected payload
    # ------------------------------------------------------

    encrypted_data = encrypt_file(
        SECRET_FILE,
        KEY_FILE
    )

    payload = create_payload_packet(
        encrypted_data
    )

    total_bits = len(payload)

    # ------------------------------------------------------
    # Original frame
    # ------------------------------------------------------

    original = cv2.imread(
        str(ORIGINAL_FRAME)
    )

    if original is None:
        raise FileNotFoundError(
            ORIGINAL_FRAME
        )

    original_blue = original[:, :, 0]

    original_bands = apply_dwt(
        original_blue
    )

    original_edge = generate_edge_map(
        original_bands["LH"]
    )

    # ------------------------------------------------------
    # Stego frame
    # ------------------------------------------------------

    stego = cv2.imread(
        str(STEGO_FRAME)
    )

    if stego is None:
        raise FileNotFoundError(
            STEGO_FRAME
        )

    stego_blue = stego[:, :, 0]

    stego_bands = apply_dwt(
        stego_blue
    )

    stego_edge = generate_edge_map(
        stego_bands["LH"]
    )

    # ------------------------------------------------------
    # Compare edge maps
    # ------------------------------------------------------

    edge_difference = np.count_nonzero(
        original_edge != stego_edge
    )

    total_pixels = original_edge.size

    print()
    print(
        f"Total Edge Pixels      : "
        f"{np.count_nonzero(original_edge)}"
    )

    print(
        f"Total Edge Differences : "
        f"{edge_difference}"
    )

    print(
        f"Total Coefficients     : "
        f"{total_pixels}"
    )

    print(
        f"Expected Payload       : "
        f"{total_bits} bits"
    )

    # ------------------------------------------------------
    # Extract using STEGO edge map
    # ------------------------------------------------------

    extracted_stego_map = adaptive_extract(
        stego_bands["LH"],
        stego_edge,
        total_bits
    )

    # ------------------------------------------------------
    # Extract using ORIGINAL edge map
    # ------------------------------------------------------

    extracted_original_map = adaptive_extract(
        stego_bands["LH"],
        original_edge,
        total_bits
    )

    # ------------------------------------------------------
    # Compare
    # ------------------------------------------------------

    print()

    print(
        "Using STEGO edge map:"
    )

    print(
        f"Matches payload : "
        f"{extracted_stego_map == payload}"
    )

    print()

    print(
        "Using ORIGINAL edge map:"
    )

    print(
        f"Matches payload : "
        f"{extracted_original_map == payload}"
    )

    # ------------------------------------------------------
    # Find first mismatch
    # ------------------------------------------------------

    def first_mismatch(a, b):

        limit = min(
            len(a),
            len(b)
        )

        for i in range(limit):

            if a[i] != b[i]:
                return i

        if len(a) != len(b):
            return limit

        return None

    mismatch = first_mismatch(
        payload,
        extracted_stego_map
    )

    print()

    print(
        f"First Mismatch Index : "
        f"{mismatch}"
    )

    print()

    print("=" * 70)
    print("Diagnostic Completed")
    print("=" * 70)


if __name__ == "__main__":
    main()