"""
------------------------------------------------------------
StegaFusion Real Frame QIM Strength Diagnostic
------------------------------------------------------------
Tests several quantization strengths against the actual
StegaFusion cover frame.

No production configuration is changed permanently.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path

import numpy as np

from modules.steganography.lsb_embed import load_frame
from modules.steganography.edge_detector import generate_edge_map
from modules.steganography.adaptive_lsb import (
    adaptive_embed,
    adaptive_extract,
)
from modules.steganography.payload import create_payload_packet
from modules.crypto.aes_encrypt import encrypt_file
from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)
import modules.steganography.lsb_utils as lsb_utils


FRAME_FILE = Path(
    "temp/frames/frame_00000.png"
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


def run_test(
    blue_channel,
    payload,
    edge_map,
    step
):

    # ------------------------------------------------------
    # Change only the runtime quantization strength.
    # Nothing is permanently modified.
    # ------------------------------------------------------

    lsb_utils.QUANTIZATION_STEP = step

    # ------------------------------------------------------
    # Fresh DWT for every test
    # ------------------------------------------------------

    bands = apply_dwt(
        blue_channel
    )

    original_lh = bands["LH"].copy()

    # ------------------------------------------------------
    # Embed
    # ------------------------------------------------------

    stego_lh, embedded = adaptive_embed(
        bands["LH"],
        edge_map,
        payload
    )

    if embedded != len(payload):

        return {
            "step": step,
            "embedded": embedded,
            "pass": False,
            "mismatch": None,
        }

    # ------------------------------------------------------
    # Direct extraction
    # ------------------------------------------------------

    direct = adaptive_extract(
        stego_lh,
        edge_map,
        len(payload)
    )

    direct_pass = (
        direct == payload
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
    # Use the ORIGINAL edge map first.
    #
    # This isolates coefficient robustness from edge-map
    # classification changes.
    # ------------------------------------------------------

    recovered = adaptive_extract(
        recovered_bands["LH"],
        edge_map,
        len(payload)
    )

    mismatch = first_mismatch(
        payload,
        recovered
    )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    coefficient_difference = (
        stego_lh -
        original_lh
    )

    dwt_recovery_difference = (
        recovered_bands["LH"] -
        stego_lh
    )

    negative_count = np.count_nonzero(
        reconstructed < 0
    )

    return {
        "step": step,
        "embedded": embedded,
        "direct_pass": direct_pass,
        "pass": mismatch is None,
        "mismatch": mismatch,
        "negative": negative_count,
        "max_embed_change": float(
            np.max(
                np.abs(
                    coefficient_difference
                )
            )
        ),
        "mean_embed_change": float(
            np.mean(
                np.abs(
                    coefficient_difference
                )
            )
        ),
        "max_dwt_error": float(
            np.max(
                np.abs(
                    dwt_recovery_difference
                )
            )
        ),
        "mean_dwt_error": float(
            np.mean(
                np.abs(
                    dwt_recovery_difference
                )
            )
        ),
    }


def main():

    print("=" * 70)
    print("StegaFusion Real Frame QIM Strength Diagnostic")
    print("=" * 70)

    # ------------------------------------------------------
    # Load real frame
    # ------------------------------------------------------

    image = load_frame(
        FRAME_FILE
    )

    blue_channel = image[:, :, 0]

    print()
    print("Real Blue Channel")
    print(
        f"Range  : "
        f"{blue_channel.min()} -> "
        f"{blue_channel.max()}"
    )
    print(
        f"Mean   : "
        f"{blue_channel.mean():.6f}"
    )

    # ------------------------------------------------------
    # Payload
    # ------------------------------------------------------

    encrypted = encrypt_file(
        SECRET_FILE,
        KEY_FILE
    )

    payload = create_payload_packet(
        encrypted
    )

    print(
        f"Payload: {len(payload)} bits"
    )

    # ------------------------------------------------------
    # IMPORTANT:
    # Edge map is generated from the ORIGINAL LH band.
    # ------------------------------------------------------

    original_bands = apply_dwt(
        blue_channel
    )

    edge_map = generate_edge_map(
        original_bands["LH"]
    )

    edge_pixels = np.count_nonzero(
        edge_map
    )

    print(
        f"Edge Pixels: {edge_pixels}"
    )

    print()
    print(
        "Testing quantization strengths..."
    )
    print("-" * 70)

    # ------------------------------------------------------
    # Test several strengths
    # ------------------------------------------------------

    steps = [
        1,
        2,
        3,
        4,
    ]

    results = []

    for step in steps:

        result = run_test(
            blue_channel,
            payload,
            edge_map,
            step
        )

        results.append(
            result
        )

        print()
        print(
            f"STEP = {step}"
        )

        print(
            f"Embedded Bits     : "
            f"{result['embedded']}"
        )

        print(
            f"Direct Extraction : "
            f"{'PASS' if result.get('direct_pass') else 'FAIL'}"
        )

        print(
            f"Round Trip        : "
            f"{'PASS' if result['pass'] else 'FAIL'}"
        )

        print(
            f"First Mismatch    : "
            f"{result['mismatch']}"
        )

        print(
            f"Negative Pixels   : "
            f"{result['negative']}"
        )

        print(
            f"Max Embed Change  : "
            f"{result['max_embed_change']:.6f}"
        )

        print(
            f"Mean Embed Change : "
            f"{result['mean_embed_change']:.6f}"
        )

        print(
            f"Max DWT Error     : "
            f"{result['max_dwt_error']:.6f}"
        )

        print(
            f"Mean DWT Error    : "
            f"{result['mean_dwt_error']:.6f}"
        )

    # ------------------------------------------------------
    # Restore current project setting
    # ------------------------------------------------------

    lsb_utils.QUANTIZATION_STEP = 4

    print()
    print("=" * 70)
    print("QIM Strength Diagnostic Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()