"""
------------------------------------------------------------
StegaFusion DWT Precision Diagnostic
------------------------------------------------------------
Determines whether payload corruption occurs because of
floating-point precision or uint8 image quantization
between inverse and forward DWT.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path

import numpy as np

from modules.crypto.aes_encrypt import encrypt_file
from modules.steganography.payload import create_payload_packet
from modules.steganography.edge_detector import generate_edge_map
from modules.steganography.adaptive_lsb import (
    adaptive_embed,
    adaptive_extract,
)
from modules.steganography.lsb_embed import load_frame
from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)


# ==========================================================
# PATHS
# ==========================================================

SECRET_FILE = Path(
    "input/secret_data/test_secret.txt"
)

KEY_FILE = Path(
    "input/keys/test_integration_key.bin"
)

FRAME_FILE = Path(
    "temp/frames/frame_00000.png"
)


# ==========================================================
# HELPER
# ==========================================================

def first_mismatch(expected: str, recovered: str):

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
# MAIN TEST
# ==========================================================

def main():

    print("=" * 70)
    print("StegaFusion DWT Precision Diagnostic")
    print("=" * 70)

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

    print()
    print(
        f"Payload Length : {len(payload)} bits"
    )

    # ------------------------------------------------------
    # Load frame
    # ------------------------------------------------------

    image = load_frame(
        FRAME_FILE
    )

    blue = image[:, :, 0]

    # ------------------------------------------------------
    # Original DWT
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

    print(
        f"Embedded Bits  : {embedded}"
    )

    if embedded != len(payload):
        raise RuntimeError(
            "Not all payload bits were embedded."
        )

    bands["LH"] = stego_lh

    # ------------------------------------------------------
    # Direct extraction
    # ------------------------------------------------------

    direct_recovered = adaptive_extract(
        stego_lh,
        edge_map,
        len(payload)
    )

    direct_mismatch = first_mismatch(
        payload,
        direct_recovered
    )

    print()

    if direct_mismatch is None:
        print(
            "Direct Coefficient Extraction : PASS"
        )
    else:
        print(
            "Direct Coefficient Extraction : FAIL"
        )
        print(
            f"First Mismatch              : "
            f"{direct_mismatch}"
        )

    # ------------------------------------------------------
    # Inverse DWT
    # ------------------------------------------------------

    reconstructed = apply_inverse_dwt(
        bands
    )

    # ------------------------------------------------------
    # Forward DWT WITHOUT uint8 conversion
    # ------------------------------------------------------

    bands_float = apply_dwt(
        reconstructed
    )

    recovered_float = adaptive_extract(
        bands_float["LH"],
        edge_map,
        len(payload)
    )

    mismatch_float = first_mismatch(
        payload,
        recovered_float
    )

    print()

    if mismatch_float is None:
        print(
            "Float DWT Round Trip          : PASS"
        )
    else:
        print(
            "Float DWT Round Trip          : FAIL"
        )
        print(
            f"First Mismatch              : "
            f"{mismatch_float}"
        )

    # ------------------------------------------------------
    # Convert to uint8
    # ------------------------------------------------------

    reconstructed_uint8 = np.clip(
        reconstructed,
        0,
        255
    ).astype(np.uint8)

    # ------------------------------------------------------
    # Forward DWT AFTER uint8 conversion
    # ------------------------------------------------------

    bands_uint8 = apply_dwt(
        reconstructed_uint8
    )

    recovered_uint8 = adaptive_extract(
        bands_uint8["LH"],
        edge_map,
        len(payload)
    )

    mismatch_uint8 = first_mismatch(
        payload,
        recovered_uint8
    )

    print()

    if mismatch_uint8 is None:
        print(
            "uint8 DWT Round Trip          : PASS"
        )
    else:
        print(
            "uint8 DWT Round Trip          : FAIL"
        )
        print(
            f"First Mismatch              : "
            f"{mismatch_uint8}"
        )

    # ------------------------------------------------------
    # Precision statistics
    # ------------------------------------------------------

    mean_error = np.mean(
        np.abs(
            reconstructed.astype(np.float64)
            -
            reconstructed_uint8.astype(np.float64)
        )
    )

    max_error = np.max(
        np.abs(
            reconstructed.astype(np.float64)
            -
            reconstructed_uint8.astype(np.float64)
        )
    )

    print()
    print(
        "Precision Statistics"
    )

    print(
        f"Original Range      : "
        f"{blue.min():.6f} -> {blue.max():.6f}"
    )

    print(
        f"Reconstructed Range : "
        f"{reconstructed.min():.6f} -> "
        f"{reconstructed.max():.6f}"
    )

    print(
        f"Mean Float/uint8 Error : "
        f"{mean_error:.8f}"
    )

    print(
        f"Maximum Float/uint8 Error : "
        f"{max_error:.8f}"
    )

    print()
    print("=" * 70)
    print("Diagnostic Completed")
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()