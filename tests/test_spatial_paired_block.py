"""
StegaFusion Spatial Paired-Block Unit Test

Tests direct embedding and extraction using the
MP4-resistant spatial paired-block method.

No production code is modified by this test.
"""

from pathlib import Path

import cv2

from modules.steganography.spatial.paired_block import (
    embed_payload,
    extract_payload,
    calculate_capacity,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

FRAME_PATH = Path(
    "temp/frames/frame_00040.png"
)

BLOCK_SIZE = 32
GAP = 16
DELTA = 4

PAYLOAD = "101100111000111100001111"


# ==========================================================
# MAIN TEST
# ==========================================================

def main():

    print("=" * 70)
    print("StegaFusion Spatial Paired-Block Unit Test")
    print("=" * 70)

    print()

    # ------------------------------------------------------
    # Load frame
    # ------------------------------------------------------

    frame = cv2.imread(
        str(FRAME_PATH)
    )

    if frame is None:
        raise FileNotFoundError(
            f"Unable to load frame: {FRAME_PATH}"
        )

    print(
        f"Frame       : {FRAME_PATH.resolve()}"
    )

    print(
        f"Shape       : {frame.shape}"
    )

    print(
        f"Block Size  : {BLOCK_SIZE} x {BLOCK_SIZE}"
    )

    print(
        f"Gap         : {GAP}"
    )

    print(
        f"Delta       : {DELTA}"
    )

    # ------------------------------------------------------
    # Capacity
    # ------------------------------------------------------

    capacity = calculate_capacity(
        frame,
        block_size=BLOCK_SIZE,
        gap=GAP,
    )

    print(
        f"Capacity    : {capacity} bits"
    )

    print(
        f"Payload     : {PAYLOAD}"
    )

    print(
        f"Payload Bits: {len(PAYLOAD)}"
    )

    print()

    # ------------------------------------------------------
    # Embed
    # ------------------------------------------------------

    stego, embedded = embed_payload(
        frame,
        PAYLOAD,
        block_size=BLOCK_SIZE,
        delta=DELTA,
        gap=GAP,
    )

    print(
        f"Embedded Bits : {embedded}"
    )

    # ------------------------------------------------------
    # Direct extraction
    #
    # The spatial method compares the original cover
    # against the stego frame.
    # ------------------------------------------------------

    recovered = extract_payload(
        frame,
        stego,
        len(PAYLOAD),
        block_size=BLOCK_SIZE,
        gap=GAP,
    )

    print()

    print(
        f"Recovered     : {recovered}"
    )

    print(
        f"Expected      : {PAYLOAD}"
    )

    print()

    if recovered == PAYLOAD:

        print(
            "Direct Extraction : PASS"
        )

    else:

        correct = sum(
            a == b
            for a, b in zip(
                PAYLOAD,
                recovered,
            )
        )

        print(
            f"Direct Extraction : FAIL"
        )

        print(
            f"Correct Bits      : "
            f"{correct}/{len(PAYLOAD)}"
        )

        for index, (expected, actual) in enumerate(
            zip(PAYLOAD, recovered)
        ):

            if expected != actual:

                print(
                    f"First Mismatch    : {index}"
                )

                print(
                    f"Expected          : {expected}"
                )

                print(
                    f"Recovered         : {actual}"
                )

                break

    # ------------------------------------------------------
    # Save stego frame
    # ------------------------------------------------------

    output_dir = Path(
        "temp/spatial"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir /
        "paired_block_test.png"
    )

    cv2.imwrite(
        str(output_path),
        stego,
    )

    print()

    print(
        f"Saved Stego Frame : "
        f"{output_path.resolve()}"
    )

    print()

    print("=" * 70)

    if recovered == PAYLOAD:

        print(
            "RESULT: SPATIAL PAIRED-BLOCK "
            "UNIT TEST PASSED."
        )

    else:

        print(
            "RESULT: SPATIAL PAIRED-BLOCK "
            "UNIT TEST FAILED."
        )

    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()