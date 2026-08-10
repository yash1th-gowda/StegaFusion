"""
------------------------------------------------------------
StegaFusion Real PNG Round-Trip Diagnostic
------------------------------------------------------------

Determines whether the saved PNG changes the reconstructed
blue channel or its DWT coefficients.

This test does NOT modify production code.
------------------------------------------------------------
"""

from pathlib import Path

import cv2
import numpy as np

from modules.steganography.lsb_embed import (
    load_frame,
    save_frame,
)

from modules.steganography.edge_detector import (
    generate_edge_map,
)

from modules.steganography.adaptive_lsb import (
    adaptive_embed,
    adaptive_extract,
)

from modules.steganography.payload import (
    create_payload_packet,
)

from modules.crypto.aes_encrypt import (
    encrypt_file,
)

from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)


FRAME_FILE = Path(
    "temp/frames/frame_00000.png"
)

OUTPUT_FILE = Path(
    "temp/stego_frames/png_diagnostic.png"
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
    print("StegaFusion Real PNG Round-Trip Diagnostic")
    print("=" * 70)

    # ------------------------------------------------------
    # Load original frame
    # ------------------------------------------------------

    image = load_frame(
        FRAME_FILE
    )

    original_blue = image[:, :, 0].copy()

    print()
    print("Original Blue Channel")
    print(
        f"Shape : {original_blue.shape}"
    )
    print(
        f"Range : "
        f"{original_blue.min()} -> "
        f"{original_blue.max()}"
    )

    # ------------------------------------------------------
    # Create exact payload
    # ------------------------------------------------------

    encrypted = encrypt_file(
        SECRET_FILE,
        KEY_FILE
    )

    payload = create_payload_packet(
        encrypted
    )

    print(
        f"Payload : {len(payload)} bits"
    )

    # ------------------------------------------------------
    # DWT
    # ------------------------------------------------------

    bands = apply_dwt(
        original_blue
    )

    original_lh = bands["LH"].copy()

    # ------------------------------------------------------
    # Edge map
    # ------------------------------------------------------

    edge_map = generate_edge_map(
        original_lh
    )

    print(
        f"Edge Pixels : "
        f"{np.count_nonzero(edge_map)}"
    )

    # ------------------------------------------------------
    # Embed
    # ------------------------------------------------------

    stego_lh, embedded = adaptive_embed(
        original_lh,
        edge_map,
        payload
    )

    print(
        f"Embedded : {embedded}"
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
        "Direct Extraction : ",
        "PASS"
        if direct == payload
        else "FAIL"
    )

    # ------------------------------------------------------
    # Inverse DWT
    # ------------------------------------------------------

    stego_bands = bands.copy()

    stego_bands["LH"] = stego_lh

    reconstructed = apply_inverse_dwt(
        stego_bands
    )

    reconstructed_uint8 = np.clip(
        np.rint(reconstructed),
        0,
        255
    ).astype(np.uint8)

    # ------------------------------------------------------
    # Compare BEFORE PNG
    # ------------------------------------------------------

    pre_png_difference = (
        reconstructed_uint8.astype(np.int16)
        -
        original_blue.astype(np.int16)
    )

    print()
    print("Before PNG Save")
    print(
        f"Changed Pixels : "
        f"{np.count_nonzero(pre_png_difference)}"
    )
    print(
        f"Max Difference : "
        f"{np.max(np.abs(pre_png_difference))}"
    )

    # ------------------------------------------------------
    # Build complete stego image
    # ------------------------------------------------------

    stego_image = image.copy()

    stego_image[:, :, 0] = reconstructed_uint8

    # ------------------------------------------------------
    # Save PNG
    # ------------------------------------------------------

    save_frame(
        stego_image,
        OUTPUT_FILE
    )

    print()
    print(
        f"Saved : {OUTPUT_FILE}"
    )

    # ------------------------------------------------------
    # Read PNG again
    # ------------------------------------------------------

    loaded = load_frame(
        OUTPUT_FILE
    )

    loaded_blue = loaded[:, :, 0]

    # ------------------------------------------------------
    # Compare BLUE CHANNEL before/after PNG
    # ------------------------------------------------------

    png_blue_difference = (
        loaded_blue.astype(np.int16)
        -
        reconstructed_uint8.astype(np.int16)
    )

    print()
    print("PNG Blue Channel Comparison")
    print(
        f"Changed Pixels : "
        f"{np.count_nonzero(png_blue_difference)}"
    )
    print(
        f"Max Difference : "
        f"{np.max(np.abs(png_blue_difference))}"
    )
    print(
        f"Mean Difference : "
        f"{np.mean(np.abs(png_blue_difference)):.10f}"
    )

    # ------------------------------------------------------
    # DWT loaded PNG
    # ------------------------------------------------------

    loaded_bands = apply_dwt(
        loaded_blue
    )

    loaded_lh = loaded_bands["LH"]

    # ------------------------------------------------------
    # Compare stego LH before PNG vs after PNG
    # ------------------------------------------------------

    coefficient_difference = (
        loaded_lh -
        stego_lh
    )

    print()
    print("LH Coefficient Comparison")
    print(
        f"Changed Coefficients : "
        f"{np.count_nonzero(coefficient_difference)}"
    )
    print(
        f"Maximum Difference    : "
        f"{np.max(np.abs(coefficient_difference)):.10f}"
    )
    print(
        f"Mean Difference       : "
        f"{np.mean(np.abs(coefficient_difference)):.10f}"
    )

    # ------------------------------------------------------
    # Extract from loaded PNG
    # ------------------------------------------------------

    loaded_edge_map = generate_edge_map(
        loaded_lh
    )

    recovered = adaptive_extract(
        loaded_lh,
        loaded_edge_map,
        len(payload)
    )

    mismatch = first_mismatch(
        payload,
        recovered
    )

    print()
    print("Extraction After PNG")
    print(
        f"Recovered Bits : "
        f"{len(recovered)}"
    )

    print(
        f"First Mismatch : "
        f"{mismatch}"
    )

    if mismatch is None:

        print(
            "PNG Extraction : PASS"
        )

    else:

        print(
            "PNG Extraction : FAIL"
        )

        start = max(
            0,
            mismatch - 10
        )

        end = min(
            len(payload),
            mismatch + 20
        )

        print()
        print(
            "Expected:"
        )
        print(
            payload[start:end]
        )

        print(
            "Recovered:"
        )
        print(
            recovered[start:end]
        )

    # ------------------------------------------------------
    # Final
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("PNG ROUND-TRIP DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()