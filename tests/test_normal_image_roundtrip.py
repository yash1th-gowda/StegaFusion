"""
------------------------------------------------------------
StegaFusion Normal Image DWT/LSB Round-Trip Test
------------------------------------------------------------
Tests the DWT + adaptive LSB pipeline using a synthetic
8-bit image with a normal 0-255 intensity distribution.

This isolates the steganography system from the current
cover video.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path

import cv2
import numpy as np

from modules.crypto.aes_encrypt import encrypt_file
from modules.steganography.payload import create_payload_packet
from modules.steganography.edge_detector import generate_edge_map
from modules.steganography.adaptive_lsb import (
    adaptive_embed,
    adaptive_extract,
)
from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)


SECRET_FILE = Path(
    "input/secret_data/test_secret.txt"
)

KEY_FILE = Path(
    "input/keys/test_integration_key.bin"
)


def first_mismatch(expected, recovered):

    limit = min(
        len(expected),
        len(recovered)
    )

    for i in range(limit):

        if expected[i] != recovered[i]:
            return i

    if len(expected) != len(recovered):
        return limit

    return None


def main():

    print("=" * 70)
    print("StegaFusion Normal Image DWT/LSB Round-Trip Test")
    print("=" * 70)

    # ------------------------------------------------------
    # Create normal synthetic 8-bit image
    # ------------------------------------------------------

    height = 1080
    width = 1920

    x = np.linspace(
        0,
        255,
        width,
        dtype=np.uint8
    )

    image = np.tile(
        x,
        (height, 1)
    )

    print()
    print(
        f"Image Shape : {image.shape}"
    )

    print(
        f"Image Dtype : {image.dtype}"
    )

    print(
        f"Image Range : {image.min()} -> {image.max()}"
    )

    # ------------------------------------------------------
    # Create encrypted payload
    # ------------------------------------------------------

    encrypted_data = encrypt_file(
        SECRET_FILE,
        KEY_FILE
    )

    payload = create_payload_packet(
        encrypted_data
    )

    print(
        f"Payload     : {len(payload)} bits"
    )

    # ------------------------------------------------------
    # DWT
    # ------------------------------------------------------

    bands = apply_dwt(
        image
    )

    # ------------------------------------------------------
    # Edge map
    # ------------------------------------------------------

    edge_map = generate_edge_map(
        bands["LH"]
    )

    print(
        f"Edge Pixels : "
        f"{np.count_nonzero(edge_map)}"
    )

    # ------------------------------------------------------
    # Capacity
    # ------------------------------------------------------

    capacity = (
        edge_map.size
        +
        np.count_nonzero(edge_map)
    )

    print(
        f"Capacity    : {capacity} bits"
    )

    if len(payload) > capacity:
        raise ValueError(
            "Payload exceeds image capacity."
        )

    # ------------------------------------------------------
    # Embed
    # ------------------------------------------------------

    stego_lh, embedded = adaptive_embed(
        bands["LH"],
        edge_map,
        payload
    )

    print(
        f"Embedded    : {embedded} bits"
    )

    # ------------------------------------------------------
    # Direct extraction
    # ------------------------------------------------------

    direct = adaptive_extract(
        stego_lh,
        edge_map,
        len(payload)
    )

    print()

    print(
        "Direct extraction:",
        "PASS" if direct == payload else "FAIL"
    )

    # ------------------------------------------------------
    # Inverse DWT
    # ------------------------------------------------------

    bands["LH"] = stego_lh

    reconstructed = apply_inverse_dwt(
        bands
    )

    # ------------------------------------------------------
    # Convert to uint8
    # ------------------------------------------------------

    reconstructed_uint8 = np.clip(
        np.rint(reconstructed),
        0,
        255
    ).astype(np.uint8)

    # ------------------------------------------------------
    # Forward DWT
    # ------------------------------------------------------

    recovered_bands = apply_dwt(
        reconstructed_uint8
    )

    recovered = adaptive_extract(
        recovered_bands["LH"],
        edge_map,
        len(payload)
    )

    mismatch = first_mismatch(
        payload,
        recovered
    )

    print()

    if mismatch is None:

        print(
            "uint8 DWT round trip : PASS"
        )

    else:

        print(
            "uint8 DWT round trip : FAIL"
        )

        print(
            f"First mismatch      : {mismatch}"
        )

    print()

    print(
        f"Reconstructed Range : "
        f"{reconstructed.min():.6f} -> "
        f"{reconstructed.max():.6f}"
    )

    print(
        f"Saved uint8 Range   : "
        f"{reconstructed_uint8.min()} -> "
        f"{reconstructed_uint8.max()}"
    )

    print()
    print("=" * 70)

    if mismatch is None:

        print(
            "NORMAL IMAGE ROUND-TRIP PASSED!"
        )

    else:

        print(
            "NORMAL IMAGE ROUND-TRIP FAILED!"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()