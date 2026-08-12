"""
StegaFusion Multi-Frame Spatial Payload Capacity Diagnostic

Determines how much payload can safely be embedded into each
video frame using the validated spatial paired-block method.

No production code is modified.
"""

from pathlib import Path

import cv2

from modules.steganography.spatial.paired_block import (
    calculate_capacity,
    embed_payload,
    extract_payload,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

FRAME_DIR = Path(
    "temp/frames"
)

OUTPUT_DIR = Path(
    "output/video_spatial_multiframe_capacity"
)

BLOCK_SIZE = 32
GAP = 16
DELTA = 4

FRAME_INDICES = [
    0,
    1,
    5,
    10,
    20,
    40,
    60,
    80,
    100,
    150,
    196,
]

PAYLOAD_SIZES = [
    544,
    576,
    608,
    640,
    672,
    704,
    736,
    768,
]


# ==========================================================
# PAYLOAD GENERATOR
# ==========================================================

def make_payload(bit_count: int) -> str:
    """
    Generate deterministic test payload.
    """

    pattern = (
        "101100111000111100001111"
    )

    repeats = (
        bit_count // len(pattern)
    ) + 1

    return (
        pattern * repeats
    )[:bit_count]


# ==========================================================
# MP4 ENCODING
# ==========================================================

def encode_frame(
    frame,
    output_path: Path,
):
    """
    Encode one stego frame through MP4V.
    """

    height, width = frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        24.0,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Unable to create {output_path}"
        )

    writer.write(frame)
    writer.release()


# ==========================================================
# MP4 DECODING
# ==========================================================

def decode_frame(
    video_path: Path,
):
    """
    Decode the first frame from an MP4.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open {video_path}"
        )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            f"Unable to decode {video_path}"
        )

    return frame


# ==========================================================
# BIT COMPARISON
# ==========================================================

def compare_bits(
    expected: str,
    recovered: str,
):
    correct = sum(
        a == b
        for a, b in zip(
            expected,
            recovered,
        )
    )

    mismatch = None

    for index, (
        expected_bit,
        recovered_bit,
    ) in enumerate(
        zip(expected, recovered)
    ):

        if expected_bit != recovered_bit:
            mismatch = index
            break

    return correct, mismatch


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "StegaFusion Multi-Frame Spatial "
        "Payload Capacity Diagnostic"
    )
    print("=" * 70)

    print()

    print(
        f"Block Size : {BLOCK_SIZE}"
    )

    print(
        f"Gap        : {GAP}"
    )

    print(
        f"Delta      : {DELTA}"
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # ------------------------------------------------------
    # TEST EACH PAYLOAD SIZE
    # ------------------------------------------------------

    for payload_size in PAYLOAD_SIZES:

        print("=" * 70)
        print(
            f"TESTING PAYLOAD SIZE : "
            f"{payload_size} BITS"
        )
        print("=" * 70)

        payload = make_payload(
            payload_size
        )

        size_passes = 0
        size_failures = 0

        worst_accuracy = 100.0
        worst_mismatch = None

        for frame_index in FRAME_INDICES:

            frame_path = (
                FRAME_DIR /
                f"frame_{frame_index:05d}.png"
            )

            frame = cv2.imread(
                str(frame_path)
            )

            if frame is None:

                print(
                    f"Frame {frame_index}: "
                    f"SKIP - not found"
                )

                continue

            capacity = calculate_capacity(
                frame,
                block_size=BLOCK_SIZE,
                gap=GAP,
            )

            if payload_size > capacity:

                print(
                    f"Frame {frame_index}: "
                    f"SKIP - capacity {capacity}"
                )

                continue

            # --------------------------------------------------
            # EMBED
            # --------------------------------------------------

            stego, embedded = embed_payload(
                frame,
                payload,
                block_size=BLOCK_SIZE,
                delta=DELTA,
                gap=GAP,
            )

            # --------------------------------------------------
            # DIRECT EXTRACTION
            # --------------------------------------------------

            direct = extract_payload(
                frame,
                stego,
                payload_size,
                block_size=BLOCK_SIZE,
                gap=GAP,
            )

            direct_correct, _ = compare_bits(
                payload,
                direct,
            )

            # --------------------------------------------------
            # MP4V
            # --------------------------------------------------

            output_path = (
                OUTPUT_DIR /
                f"payload_{payload_size}_"
                f"frame_{frame_index:05d}.mp4"
            )

            encode_frame(
                stego,
                output_path,
            )

            decoded = decode_frame(
                output_path
            )

            # --------------------------------------------------
            # VIDEO EXTRACTION
            # --------------------------------------------------

            recovered = extract_payload(
                frame,
                decoded,
                payload_size,
                block_size=BLOCK_SIZE,
                gap=GAP,
            )

            correct, mismatch = compare_bits(
                payload,
                recovered,
            )

            accuracy = (
                correct /
                payload_size
            ) * 100.0

            worst_accuracy = min(
                worst_accuracy,
                accuracy,
            )

            if mismatch is not None:

                worst_mismatch = mismatch

            if correct == payload_size:

                size_passes += 1
                result = "PASS"

            else:

                size_failures += 1
                result = "FAIL"

            print(
                f"Frame {frame_index:3d}: "
                f"Direct "
                f"{direct_correct:3d}/{payload_size:<3d} "
                f"Video "
                f"{correct:3d}/{payload_size:<3d} "
                f"{accuracy:7.2f}% "
                f"Mismatch={str(mismatch):>4} "
                f"{result}"
            )

        print()

        print(
            f"Payload {payload_size}: "
            f"{size_passes} PASS / "
            f"{size_failures} FAIL"
        )

        print(
            f"Worst Accuracy : "
            f"{worst_accuracy:.2f}%"
        )

        print()

        results.append(
            (
                payload_size,
                size_passes,
                size_failures,
                worst_accuracy,
                worst_mismatch,
            )
        )

    # ------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------

    print("=" * 70)
    print("CAPACITY SUMMARY")
    print("=" * 70)

    print()

    print(
        "PAYLOAD   PASS   FAIL   "
        "WORST_ACCURACY   RESULT"
    )

    safe_sizes = []

    for (
        payload_size,
        passes,
        failures,
        worst_accuracy,
        worst_mismatch,
    ) in results:

        result = (
            "PASS"
            if failures == 0
            else "FAIL"
        )

        if failures == 0:
            safe_sizes.append(
                payload_size
            )

        print(
            f"{payload_size:7d}   "
            f"{passes:4d}   "
            f"{failures:4d}   "
            f"{worst_accuracy:13.2f}%   "
            f"{result}"
        )

    print()

    if safe_sizes:

        print(
            f"Largest tested fully reliable "
            f"payload : {max(safe_sizes)} bits"
        )

    else:

        print(
            "No tested payload size was "
            "fully reliable."
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "No production embedding code "
        "was modified."
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()