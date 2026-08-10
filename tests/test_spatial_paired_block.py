"""
StegaFusion Spatial Paired-Block Unit Test

Tests:
    PNG/frame
        ↓
    spatial embedding
        ↓
    spatial extraction

No MP4 encoding is involved.
"""

from pathlib import Path

import cv2

from config.config import PathConfig
from modules.steganography.spatial.paired_block import (
    BLOCK_SIZE,
    DELTA,
    calculate_capacity,
    embed_payload,
    extract_payload,
)


PAYLOAD = "101100111000111100001111"


def main():

    print("=" * 70)
    print("StegaFusion Spatial Paired-Block Unit Test")
    print("=" * 70)

    frame_path = (
        PathConfig.FRAME_DIR /
        "frame_00040.png"
    )

    if not frame_path.exists():
        raise FileNotFoundError(
            f"Frame not found: {frame_path}"
        )

    frame = cv2.imread(
        str(frame_path)
    )

    if frame is None:
        raise RuntimeError(
            f"Unable to read frame: {frame_path}"
        )

    print()
    print(f"Frame       : {frame_path}")
    print(f"Shape       : {frame.shape}")
    print(f"Block Size  : {BLOCK_SIZE} x {BLOCK_SIZE}")
    print(f"Delta       : {DELTA}")

    capacity = calculate_capacity(
        frame
    )

    print(f"Capacity    : {capacity} bits")
    print(f"Payload     : {PAYLOAD}")
    print(f"Payload Bits: {len(PAYLOAD)}")

    if len(PAYLOAD) > capacity:
        raise RuntimeError(
            "Payload exceeds frame capacity."
        )

    # ------------------------------------------------------
    # EMBED
    # ------------------------------------------------------

    stego, embedded = embed_payload(
        frame,
        PAYLOAD,
    )

    print()
    print(f"Embedded Bits : {embedded}")

    # ------------------------------------------------------
    # DIRECT EXTRACTION
    # ------------------------------------------------------

    recovered = extract_payload(
        stego,
        len(PAYLOAD),
    )

    print()
    print(f"Recovered     : {recovered}")
    print(f"Expected      : {PAYLOAD}")

    if recovered != PAYLOAD:

        for index, (expected, actual) in enumerate(
            zip(PAYLOAD, recovered)
        ):
            if expected != actual:
                print()
                print(
                    f"First Mismatch : {index}"
                )
                print(
                    f"Expected       : {expected}"
                )
                print(
                    f"Recovered      : {actual}"
                )
                break

        raise RuntimeError(
            "Spatial embed/extract failed."
        )

    print()
    print("Direct Extraction : PASS")

    # ------------------------------------------------------
    # SAVE TEST FRAME
    # ------------------------------------------------------

    output = (
        PathConfig.TEMP_DIR /
        "spatial" /
        "paired_block_test.png"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output),
        stego,
    )

    print()
    print(f"Saved Stego Frame : {output}")
    print()
    print("=" * 70)
    print("RESULT: SPATIAL PAIRED-BLOCK UNIT TEST PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()