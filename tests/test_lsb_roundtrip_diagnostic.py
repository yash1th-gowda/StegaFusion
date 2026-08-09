"""
------------------------------------------------------------
StegaFusion LSB Round-Trip Diagnostic
------------------------------------------------------------
Determines whether payload corruption occurs:

1. Immediately after LSB embedding
2. After inverse DWT
3. After saving and reloading the PNG
4. After performing DWT again

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
from modules.steganography.lsb_embed import load_frame, save_frame
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

STEGO_FRAME = Path(
    "temp/stego_frames/lsb_diagnostic.png"
)

PAYLOAD_FILE = Path(
    "temp/test_payload.txt"
)


# ==========================================================
# HELPER
# ==========================================================

def compare_payload(
    expected: str,
    recovered: str,
    label: str
) -> None:

    mismatch = None

    limit = min(
        len(expected),
        len(recovered)
    )

    for i in range(limit):

        if expected[i] != recovered[i]:
            mismatch = i
            break

    if mismatch is None and len(expected) == len(recovered):

        print(
            f"{label:<35}: PASS"
        )

    else:

        print(
            f"{label:<35}: FAIL"
        )

        print(
            f"  Expected Length : {len(expected)}"
        )

        print(
            f"  Recovered Length: {len(recovered)}"
        )

        print(
            f"  First Mismatch  : {mismatch}"
        )


# ==========================================================
# MAIN TEST
# ==========================================================

def main():

    print("=" * 70)
    print("StegaFusion LSB Round-Trip Diagnostic")
    print("=" * 70)

    # ------------------------------------------------------
    # Create EXACT payload
    # ------------------------------------------------------

    encrypted_data = encrypt_file(
        SECRET_FILE,
        KEY_FILE
    )

    payload = create_payload_packet(
        encrypted_data
    )

    PAYLOAD_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    PAYLOAD_FILE.write_text(
        payload,
        encoding="ascii"
    )

    print()
    print(
        f"Payload Length : {len(payload)} bits"
    )

    # ------------------------------------------------------
    # Load original frame
    # ------------------------------------------------------

    image = load_frame(
        FRAME_FILE
    )

    blue = image[:, :, 0]

    # ------------------------------------------------------
    # ORIGINAL DWT
    # ------------------------------------------------------

    bands = apply_dwt(
        blue
    )

    original_lh = bands["LH"].copy()

    # ------------------------------------------------------
    # ORIGINAL EDGE MAP
    # ------------------------------------------------------

    edge_map = generate_edge_map(
        original_lh
    )

    print(
        f"Edge Pixels    : "
        f"{np.count_nonzero(edge_map)}"
    )

    # ------------------------------------------------------
    # EMBED
    # ------------------------------------------------------

    stego_lh, embedded_bits = adaptive_embed(
        original_lh,
        edge_map,
        payload
    )

    print(
        f"Embedded Bits  : "
        f"{embedded_bits}"
    )

    # ------------------------------------------------------
    # TEST 1
    #
    # Extract immediately from the modified coefficients.
    # NO inverse DWT.
    # ------------------------------------------------------

    immediate_extraction = adaptive_extract(
        stego_lh,
        edge_map,
        len(payload)
    )

    compare_payload(
        payload,
        immediate_extraction,
        "1. Direct coefficient extraction"
    )

    # ------------------------------------------------------
    # Replace LH for inverse DWT
    # ------------------------------------------------------

    bands["LH"] = stego_lh

    # ------------------------------------------------------
    # INVERSE DWT
    # ------------------------------------------------------

    reconstructed_blue = apply_inverse_dwt(
        bands
    )

    reconstructed_blue = np.clip(
        reconstructed_blue,
        0,
        255
    ).astype(np.uint8)

    # ------------------------------------------------------
    # Build stego frame
    # ------------------------------------------------------

    stego_frame = image.copy()

    stego_frame[:, :, 0] = reconstructed_blue

    STEGO_FRAME.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_frame(
        stego_frame,
        STEGO_FRAME
    )

    # ------------------------------------------------------
    # TEST 2
    #
    # Read the reconstructed PNG and perform DWT again.
    # ------------------------------------------------------

    reloaded = cv2.imread(
        str(STEGO_FRAME)
    )

    if reloaded is None:

        raise FileNotFoundError(
            f"Could not reload {STEGO_FRAME}"
        )

    reloaded_blue = reloaded[:, :, 0]

    reloaded_bands = apply_dwt(
        reloaded_blue
    )

    reloaded_lh = reloaded_bands["LH"]

    # ------------------------------------------------------
    # Generate extraction edge map
    # ------------------------------------------------------

    reloaded_edge = generate_edge_map(
        reloaded_lh
    )

    # ------------------------------------------------------
    # TEST 2A
    #
    # Extract using ORIGINAL edge map.
    # ------------------------------------------------------

    extracted_original_edge = adaptive_extract(
        reloaded_lh,
        edge_map,
        len(payload)
    )

    compare_payload(
        payload,
        extracted_original_edge,
        "2. After PNG/DWT using original edge"
    )

    # ------------------------------------------------------
    # TEST 2B
    #
    # Extract using regenerated edge map.
    # ------------------------------------------------------

    extracted_reloaded_edge = adaptive_extract(
        reloaded_lh,
        reloaded_edge,
        len(payload)
    )

    compare_payload(
        payload,
        extracted_reloaded_edge,
        "3. After PNG/DWT using new edge"
    )

    # ------------------------------------------------------
    # Compare coefficients
    # ------------------------------------------------------

    coefficient_difference = np.abs(
        stego_lh -
        reloaded_lh
    )

    print()

    print(
        "Coefficient Difference Statistics"
    )

    print(
        f"Maximum Difference : "
        f"{np.max(coefficient_difference):.8f}"
    )

    print(
        f"Mean Difference    : "
        f"{np.mean(coefficient_difference):.8f}"
    )

    print(
        f"Changed Coefficients: "
        f"{np.count_nonzero(coefficient_difference > 1e-6)}"
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