"""
StegaFusion Real Frame DWT Distortion Diagnostic
"""

from pathlib import Path

import numpy as np

from modules.steganography.lsb_embed import load_frame
from modules.steganography.edge_detector import generate_edge_map
from modules.steganography.adaptive_lsb import adaptive_embed
from modules.steganography.payload import create_payload_packet
from modules.crypto.aes_encrypt import encrypt_file
from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)


FRAME_FILE = Path(
    "temp/frames/frame_00000.png"
)

SECRET_FILE = Path(
    "input/secret_data/test_secret.txt"
)

KEY_FILE = Path(
    "input/keys/test_integration_key.bin"
)


def main():

    print("=" * 70)
    print("StegaFusion Real Frame DWT Distortion Diagnostic")
    print("=" * 70)

    # ------------------------------------------------------
    # Load frame
    # ------------------------------------------------------

    image = load_frame(
        FRAME_FILE
    )

    blue = image[:, :, 0]

    print()
    print("Original Blue Channel")
    print(
        f"Min        : {blue.min()}"
    )
    print(
        f"Max        : {blue.max()}"
    )
    print(
        f"Mean       : {blue.mean():.6f}"
    )
    print(
        f"Unique     : {len(np.unique(blue))}"
    )

    # ------------------------------------------------------
    # Create payload
    # ------------------------------------------------------

    encrypted = encrypt_file(
        SECRET_FILE,
        KEY_FILE
    )

    payload = create_payload_packet(
        encrypted
    )

    # ------------------------------------------------------
    # DWT
    # ------------------------------------------------------

    bands = apply_dwt(
        blue
    )

    edge_map = generate_edge_map(
        bands["LH"]
    )

    # ------------------------------------------------------
    # Embed
    # ------------------------------------------------------

    stego_lh, embedded = adaptive_embed(
        bands["LH"],
        edge_map,
        payload
    )

    print()
    print(
        f"Embedded Bits : {embedded}"
    )

    # ------------------------------------------------------
    # Coefficient difference
    # ------------------------------------------------------

    coefficient_difference = (
        stego_lh -
        bands["LH"]
    )

    print()
    print("LH Coefficient Changes")
    print(
        f"Min Difference  : "
        f"{coefficient_difference.min():.6f}"
    )
    print(
        f"Max Difference  : "
        f"{coefficient_difference.max():.6f}"
    )
    print(
        f"Mean Abs Change : "
        f"{np.mean(np.abs(coefficient_difference)):.6f}"
    )
    print(
        f"Changed Count   : "
        f"{np.count_nonzero(coefficient_difference)}"
    )

    # ------------------------------------------------------
    # Inverse DWT
    # ------------------------------------------------------

    bands["LH"] = stego_lh

    reconstructed = apply_inverse_dwt(
        bands
    )

    print()
    print("Reconstructed Float Image")
    print(
        f"Min    : {reconstructed.min():.6f}"
    )
    print(
        f"Max    : {reconstructed.max():.6f}"
    )
    print(
        f"Mean   : {reconstructed.mean():.6f}"
    )

    # ------------------------------------------------------
    # Count values outside original range
    # ------------------------------------------------------

    below_zero = np.count_nonzero(
        reconstructed < 0
    )

    above_six = np.count_nonzero(
        reconstructed > 6
    )

    print()
    print("Clipping Risk")
    print(
        f"Below 0 : {below_zero}"
    )
    print(
        f"Above 6 : {above_six}"
    )

    # ------------------------------------------------------
    # uint8 conversion
    # ------------------------------------------------------

    reconstructed_uint8 = np.clip(
        np.rint(reconstructed),
        0,
        255
    ).astype(np.uint8)

    # ------------------------------------------------------
    # Compare spatial image
    # ------------------------------------------------------

    spatial_difference = (
        reconstructed_uint8.astype(np.int16)
        -
        blue.astype(np.int16)
    )

    print()
    print("Spatial Difference")
    print(
        f"Min Difference : "
        f"{spatial_difference.min()}"
    )
    print(
        f"Max Difference : "
        f"{spatial_difference.max()}"
    )
    print(
        f"Mean Abs Diff  : "
        f"{np.mean(np.abs(spatial_difference)):.6f}"
    )
    print(
        f"Changed Pixels : "
        f"{np.count_nonzero(spatial_difference)}"
    )

    # ------------------------------------------------------
    # Forward DWT
    # ------------------------------------------------------

    recovered_bands = apply_dwt(
        reconstructed_uint8
    )

    dwt_difference = (
        recovered_bands["LH"]
        -
        stego_lh
    )

    print()
    print("DWT Coefficient Recovery")
    print(
        f"Max Difference : "
        f"{np.max(np.abs(dwt_difference)):.6f}"
    )
    print(
        f"Mean Difference : "
        f"{np.mean(np.abs(dwt_difference)):.6f}"
    )
    print(
        f"Changed Coefficients : "
        f"{np.count_nonzero(dwt_difference)}"
    )

    print()
    print("=" * 70)
    print("Diagnostic Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()