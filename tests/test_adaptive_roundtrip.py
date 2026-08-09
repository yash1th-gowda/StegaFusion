"""
------------------------------------------------------------
StegaFusion Adaptive DWT/LSB Round-Trip Test
------------------------------------------------------------
Validates both adaptive embedding modes:

    Smooth coefficient -> 1 bit
    Edge coefficient   -> 2 bits

Uses a deterministic image containing a localized
high-frequency region so that the Sobel detector produces
both edge and smooth regions in the LH DWT band.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

import numpy as np

from modules.crypto.aes_encrypt import encrypt_file
from modules.steganography.payload import create_payload_packet
from modules.steganography.edge_detector import generate_edge_map
from modules.steganography.adaptive_lsb import (
    adaptive_embed,
    adaptive_extract,
    calculate_capacity,
)
from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)


SECRET_FILE = "input/secret_data/test_secret.txt"
KEY_FILE = "input/keys/test_integration_key.bin"


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


# ==========================================================
# CREATE TEST IMAGE
# ==========================================================

def create_adaptive_test_image():

    height = 1080
    width = 1920

    image = np.full(
        (height, width),
        80,
        dtype=np.uint8
    )

    # ------------------------------------------------------
    # Create a localized high-frequency region.
    #
    # The rest of the image remains smooth so that we get
    # both edge and non-edge regions.
    # ------------------------------------------------------

    rng = np.random.default_rng(42)

    patch_height = 400
    patch_width = 600

    noise = rng.integers(
        0,
        256,
        size=(patch_height, patch_width),
        dtype=np.uint8
    )

    image[
        300:300 + patch_height,
        600:600 + patch_width
    ] = noise

    # ------------------------------------------------------
    # Add a few strong structured transitions around the
    # textured region.
    # ------------------------------------------------------

    image[
        280:300,
        580:1220
    ] = 255

    image[
        700:720,
        580:1220
    ] = 0

    image[
        280:720,
        580:600
    ] = 255

    image[
        280:720,
        1180:1200
    ] = 0

    return image


# ==========================================================
# MAIN TEST
# ==========================================================

def main():

    print("=" * 70)
    print("StegaFusion Adaptive DWT/LSB Round-Trip Test")
    print("=" * 70)

    # ------------------------------------------------------
    # Create test image
    # ------------------------------------------------------

    image = create_adaptive_test_image()

    print()
    print(
        f"Image Shape : {image.shape}"
    )

    print(
        f"Image Dtype : {image.dtype}"
    )

    print(
        f"Image Range : "
        f"{image.min()} -> {image.max()}"
    )

    # ------------------------------------------------------
    # Create payload
    # ------------------------------------------------------

    encrypted_data = encrypt_file(
        SECRET_FILE,
        KEY_FILE
    )

    payload = create_payload_packet(
        encrypted_data
    )

    print(
        f"Payload     : "
        f"{len(payload)} bits"
    )

    # ------------------------------------------------------
    # DWT
    # ------------------------------------------------------

    bands = apply_dwt(
        image
    )

    # ------------------------------------------------------
    # Generate edge map from LH
    # ------------------------------------------------------

    edge_map = generate_edge_map(
        bands["LH"]
    )

    edge_pixels = np.count_nonzero(
        edge_map
    )

    smooth_pixels = (
        edge_map.size -
        edge_pixels
    )

    print()
    print(
        f"Edge Pixels   : {edge_pixels}"
    )

    print(
        f"Smooth Pixels : {smooth_pixels}"
    )

    # ------------------------------------------------------
    # Capacity
    # ------------------------------------------------------

    capacity = calculate_capacity(
        edge_map
    )

    print(
        f"Capacity      : {capacity} bits"
    )

    if edge_pixels == 0:

        raise RuntimeError(
            "Zero edge pixels detected. "
            "Test image generation needs adjustment."
        )

    if smooth_pixels == 0:

        raise RuntimeError(
            "No smooth pixels detected. "
            "Both adaptive paths must be exercised."
        )

    if len(payload) > capacity:

        raise ValueError(
            "Payload exceeds adaptive capacity."
        )

    # ------------------------------------------------------
    # EMBEDDING
    # ------------------------------------------------------

    stego_lh, embedded = adaptive_embed(
        bands["LH"],
        edge_map,
        payload
    )

    print(
        f"Embedded      : {embedded} bits"
    )

    if embedded != len(payload):

        raise RuntimeError(
            "Not all payload bits were embedded."
        )

    # ------------------------------------------------------
    # Direct extraction
    # ------------------------------------------------------

    direct = adaptive_extract(
        stego_lh,
        edge_map,
        len(payload)
    )

    direct_mismatch = first_mismatch(
        payload,
        direct
    )

    print()

    if direct_mismatch is None:

        print(
            "Direct adaptive extraction : PASS"
        )

    else:

        print(
            "Direct adaptive extraction : FAIL"
        )

        print(
            f"First mismatch            : "
            f"{direct_mismatch}"
        )

        raise RuntimeError(
            "Adaptive embedding/extraction failed "
            "before reconstruction."
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

    # ------------------------------------------------------
    # Regenerate edge map
    # ------------------------------------------------------

    recovered_edge_map = generate_edge_map(
        recovered_bands["LH"]
    )

    recovered_edge_pixels = np.count_nonzero(
        recovered_edge_map
    )

    print()
    print(
        f"Recovered Edge Pixels : "
        f"{recovered_edge_pixels}"
    )

    # ------------------------------------------------------
    # Edge map comparison
    # ------------------------------------------------------

    edge_difference = np.count_nonzero(
        edge_map != recovered_edge_map
    )

    print(
        f"Edge Map Differences  : "
        f"{edge_difference}"
    )

    # ------------------------------------------------------
    # Extraction using regenerated edge map
    # ------------------------------------------------------

    recovered = adaptive_extract(
        recovered_bands["LH"],
        recovered_edge_map,
        len(payload)
    )

    mismatch = first_mismatch(
        payload,
        recovered
    )

    print()

    if mismatch is None:

        print(
            "Adaptive uint8 round trip : PASS"
        )

    else:

        print(
            "Adaptive uint8 round trip : FAIL"
        )

        print(
            f"First mismatch           : "
            f"{mismatch}"
        )

    # ------------------------------------------------------
    # Final result
    # ------------------------------------------------------

    print()
    print("=" * 70)

    if mismatch is None:

        print(
            "ADAPTIVE ROUND-TRIP TEST PASSED!"
        )

    else:

        print(
            "ADAPTIVE ROUND-TRIP TEST FAILED!"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()