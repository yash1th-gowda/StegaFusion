"""
StegaFusion MP4 Spatial Blind Extraction Diagnostic

Tests whether the spatial paired-block payload can be recovered
from the decoded MP4 frame WITHOUT the original cover frame.
"""

from pathlib import Path

import cv2
import numpy as np

from modules.steganography.spatial.paired_block import (
    BLOCK_SIZE,
    DELTA,
    get_block_pairs,
    embed_payload,
)


FRAME_INDEX = 40
PAYLOAD = ("101100111000111100001111" * 23)[:544]
BLOCK_SIZE_TEST = 32
DELTA_TEST = 4


def block_mean(channel, bounds):
    y1, y2, x1, x2 = bounds
    return float(np.mean(channel[y1:y2, x1:x2]))


def extract_absolute(frame, positions):
    channel = frame[:, :, 0]
    bits = []

    for left, right in positions:
        left_mean = block_mean(channel, left)
        right_mean = block_mean(channel, right)

        bits.append(
            "0" if left_mean > right_mean else "1"
        )

    return "".join(bits)


def main():

    print("=" * 70)
    print("StegaFusion MP4 Spatial Blind Extraction Diagnostic")
    print("=" * 70)

    frame_path = Path(
        "temp/frames/frame_00040.png"
    )

    frame = cv2.imread(str(frame_path))

    if frame is None:
        raise FileNotFoundError(frame_path)

    print()
    print(f"Frame       : {FRAME_INDEX}")
    print(f"Shape       : {frame.shape}")
    print(f"Payload     : {PAYLOAD}")
    print(f"Payload Bits: {len(PAYLOAD)}")
    print(f"Block Size  : {BLOCK_SIZE_TEST}")
    print(f"Delta       : {DELTA_TEST}")

    positions = get_block_pairs(
        frame.shape[0],
        frame.shape[1],
        BLOCK_SIZE_TEST,
    )

    if len(positions) < len(PAYLOAD):
        raise RuntimeError(
            "Insufficient block capacity."
        )

    stego, embedded = embed_payload(
        frame,
        PAYLOAD,
        block_size=BLOCK_SIZE_TEST,
        delta=DELTA_TEST,
    )

    print()
    print(f"Embedded Bits : {embedded}")

    output_dir = Path(
        "output/video_spatial_blind"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stego_path = (
        output_dir / "stego.png"
    )

    cv2.imwrite(
        str(stego_path),
        stego,
    )

    # ------------------------------------------------------
    # CREATE MP4V
    # ------------------------------------------------------

    video_path = (
        output_dir / "blind_test.mp4"
    )

    height, width = frame.shape[:2]

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        24.0,
        (width, height),
    )

    for _ in range(10):
        writer.write(stego)

    writer.release()

    print()
    print(f"MP4 : {video_path}")

    # ------------------------------------------------------
    # DECODE MP4
    # ------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    decoded = None

    while True:
        success, image = cap.read()

        if not success:
            break

        decoded = image

    cap.release()

    if decoded is None:
        raise RuntimeError(
            "Unable to decode MP4."
        )

    print(
        f"Decoded Shape : {decoded.shape}"
    )

    # ------------------------------------------------------
    # BLIND EXTRACTION
    #
    # IMPORTANT:
    # We deliberately DO NOT use `frame` here.
    # ------------------------------------------------------

    recovered = extract_absolute(
        decoded,
        positions[:len(PAYLOAD)],
    )

    correct = sum(
        a == b
        for a, b in zip(
            PAYLOAD,
            recovered,
        )
    )

    accuracy = (
        correct /
        len(PAYLOAD)
        * 100
    )

    mismatch = None

    for index, (expected, actual) in enumerate(
        zip(PAYLOAD, recovered)
    ):
        if expected != actual:
            mismatch = index
            break

    print()
    print(
        f"Recovered Payload : {recovered}"
    )

    print(
        f"Expected Payload  : {PAYLOAD}"
    )

    print()
    print(
        f"Correct Bits : "
        f"{correct}/{len(PAYLOAD)}"
    )

    print(
        f"Accuracy     : "
        f"{accuracy:.2f}%"
    )

    print(
        f"First Mismatch : {mismatch}"
    )

    print()

    if recovered == PAYLOAD:
        print(
            "RESULT: BLIND EXTRACTION PASS"
        )
    else:
        print(
            "RESULT: BLIND EXTRACTION FAIL"
        )


if __name__ == "__main__":
    main()