"""
StegaFusion Real Frame Comparison Diagnostic

Compares:
    1. real_payload_frame.png
    2. png_diagnostic.png

using the exact same extraction procedure.
"""

from pathlib import Path

import numpy as np

from modules.steganography.lsb_embed import load_frame
from modules.steganography.edge_detector import generate_edge_map
from modules.steganography.adaptive_lsb import adaptive_extract
from modules.transform.wavelet_utils import apply_dwt


REAL_FRAME = Path(
    "temp/stego_frames/real_payload_frame.png"
)

DIAGNOSTIC_FRAME = Path(
    "temp/stego_frames/png_diagnostic.png"
)

PAYLOAD_FILE = Path(
    "temp/test_payload.txt"
)


def extract_frame(path, expected_bits):

    image = load_frame(path)

    blue = image[:, :, 0]

    bands = apply_dwt(blue)

    edge_map = generate_edge_map(
        bands["LH"]
    )

    recovered = adaptive_extract(
        bands["LH"],
        edge_map,
        expected_bits
    )

    return image, blue, bands["LH"], edge_map, recovered


def first_mismatch(a, b):

    limit = min(len(a), len(b))

    for i in range(limit):
        if a[i] != b[i]:
            return i

    if len(a) != len(b):
        return limit

    return None


def main():

    print("=" * 70)
    print("StegaFusion Real Frame Comparison Diagnostic")
    print("=" * 70)

    if not REAL_FRAME.exists():
        raise FileNotFoundError(REAL_FRAME)

    if not DIAGNOSTIC_FRAME.exists():
        raise FileNotFoundError(DIAGNOSTIC_FRAME)

    if not PAYLOAD_FILE.exists():
        raise FileNotFoundError(PAYLOAD_FILE)

    expected = PAYLOAD_FILE.read_text(
        encoding="ascii"
    ).strip()

    expected_bits = len(expected)

    print()
    print(
        f"Expected Payload : {expected_bits} bits"
    )

    # ------------------------------------------------------
    # Extract REAL frame
    # ------------------------------------------------------

    (
        real_image,
        real_blue,
        real_lh,
        real_edge,
        real_payload,
    ) = extract_frame(
        REAL_FRAME,
        expected_bits
    )

    # ------------------------------------------------------
    # Extract diagnostic frame
    # ------------------------------------------------------

    (
        diag_image,
        diag_blue,
        diag_lh,
        diag_edge,
        diag_payload,
    ) = extract_frame(
        DIAGNOSTIC_FRAME,
        expected_bits
    )

    # ------------------------------------------------------
    # Basic image comparison
    # ------------------------------------------------------

    image_difference = (
        real_image.astype(np.int16)
        -
        diag_image.astype(np.int16)
    )

    print()
    print("IMAGE COMPARISON")
    print(
        f"Changed Pixels : "
        f"{np.count_nonzero(image_difference)}"
    )
    print(
        f"Maximum Difference : "
        f"{np.max(np.abs(image_difference))}"
    )
    print(
        f"Mean Absolute Difference : "
        f"{np.mean(np.abs(image_difference)):.10f}"
    )

    # ------------------------------------------------------
    # Blue channel comparison
    # ------------------------------------------------------

    blue_difference = (
        real_blue.astype(np.int16)
        -
        diag_blue.astype(np.int16)
    )

    print()
    print("BLUE CHANNEL COMPARISON")
    print(
        f"Changed Pixels : "
        f"{np.count_nonzero(blue_difference)}"
    )
    print(
        f"Maximum Difference : "
        f"{np.max(np.abs(blue_difference))}"
    )
    print(
        f"Mean Absolute Difference : "
        f"{np.mean(np.abs(blue_difference)):.10f}"
    )

    # ------------------------------------------------------
    # LH comparison
    # ------------------------------------------------------

    lh_difference = (
        real_lh -
        diag_lh
    )

    print()
    print("LH COMPARISON")
    print(
        f"Changed Coefficients : "
        f"{np.count_nonzero(lh_difference)}"
    )
    print(
        f"Maximum Difference : "
        f"{np.max(np.abs(lh_difference)):.10f}"
    )
    print(
        f"Mean Absolute Difference : "
        f"{np.mean(np.abs(lh_difference)):.10f}"
    )

    # ------------------------------------------------------
    # Edge map comparison
    # ------------------------------------------------------

    print()
    print("EDGE MAP")
    print(
        f"Real Edge Pixels : "
        f"{np.count_nonzero(real_edge)}"
    )
    print(
        f"Diagnostic Edge Pixels : "
        f"{np.count_nonzero(diag_edge)}"
    )

    # ------------------------------------------------------
    # Real payload comparison
    # ------------------------------------------------------

    real_mismatch = first_mismatch(
        expected,
        real_payload
    )

    print()
    print("REAL FRAME EXTRACTION")
    print(
        f"Recovered Bits : "
        f"{len(real_payload)}"
    )
    print(
        f"First Mismatch : "
        f"{real_mismatch}"
    )

    # ------------------------------------------------------
    # Diagnostic payload comparison
    # ------------------------------------------------------

    diag_mismatch = first_mismatch(
        expected,
        diag_payload
    )

    print()
    print("DIAGNOSTIC FRAME EXTRACTION")
    print(
        f"Recovered Bits : "
        f"{len(diag_payload)}"
    )
    print(
        f"First Mismatch : "
        f"{diag_mismatch}"
    )

    # ------------------------------------------------------
    # Compare extracted payloads directly
    # ------------------------------------------------------

    payload_difference = first_mismatch(
        real_payload,
        diag_payload
    )

    print()
    print("EXTRACTED PAYLOAD COMPARISON")
    print(
        f"First Difference : "
        f"{payload_difference}"
    )

    # ------------------------------------------------------
    # Print header bits
    # ------------------------------------------------------

    print()
    print("EXPECTED HEADER")
    print(
        expected[:32]
    )

    print()
    print("REAL FRAME HEADER")
    print(
        real_payload[:32]
    )

    print()
    print("DIAGNOSTIC FRAME HEADER")
    print(
        diag_payload[:32]
    )

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()