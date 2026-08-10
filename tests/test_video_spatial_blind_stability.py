"""
StegaFusion MP4 Spatial Blind Extraction Stability Diagnostic

Tests the 544-bit spatial payload across multiple video frames.
No production embedding code is modified.
"""

from pathlib import Path

import cv2

from modules.steganography.spatial.paired_block import (
    embed_payload,
    extract_payload,
)


PAYLOAD = ("101100111000111100001111" * 23)[:544]

BLOCK_SIZE = 32
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

FRAME_DIR = Path("temp/frames")
OUTPUT_DIR = Path("output/video_spatial_blind/stability")


def compare_bits(expected, recovered):
    correct = sum(
        a == b
        for a, b in zip(expected, recovered)
    )

    mismatch = None

    for i, (a, b) in enumerate(
        zip(expected, recovered)
    ):
        if a != b:
            mismatch = i
            break

    return correct, mismatch


def make_video(frame_path, output_path):
    frame = cv2.imread(str(frame_path))

    if frame is None:
        raise FileNotFoundError(frame_path)

    height, width = frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        24.0,
        (width, height),
    )

    writer.write(frame)
    writer.release()


def decode_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

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


def main():

    print("=" * 70)
    print("StegaFusion MP4 Spatial Blind Extraction Stability Diagnostic")
    print("=" * 70)

    print()
    print(f"Payload Bits : {len(PAYLOAD)}")
    print(f"Block Size   : {BLOCK_SIZE}")
    print(f"Delta        : {DELTA}")
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for frame_index in FRAME_INDICES:

        print(f"Testing frame {frame_index}...")

        frame_path = (
            FRAME_DIR /
            f"frame_{frame_index:05d}.png"
        )

        frame = cv2.imread(
            str(frame_path)
        )

        if frame is None:
            print(
                f"SKIP - frame not found: {frame_path}"
            )
            continue

        stego, embedded = embed_payload(
            frame,
            PAYLOAD,
            block_size=BLOCK_SIZE,
            delta=DELTA,
        )

        direct = extract_payload(
            frame,
            stego,
            len(PAYLOAD),
            block_size=BLOCK_SIZE,
        )

        direct_correct, direct_mismatch = compare_bits(
            PAYLOAD,
            direct,
        )

        output_video = (
            OUTPUT_DIR /
            f"frame_{frame_index:05d}.mp4"
        )

        make_video(
            frame_path=frame_path,
            output_path=output_video,
        )

        # Replace the video source frame with the
        # stego frame so the diagnostic actually
        # tests the embedded image through MP4.
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            24.0,
            (
                stego.shape[1],
                stego.shape[0],
            ),
        )

        writer.write(stego)
        writer.release()

        decoded = decode_video(
            output_video
        )

        recovered = extract_payload(
            frame,
            decoded,
            len(PAYLOAD),
            block_size=BLOCK_SIZE,
        )

        correct, mismatch = compare_bits(
            PAYLOAD,
            recovered,
        )

        accuracy = (
            correct / len(PAYLOAD)
        ) * 100.0

        result = (
            "PASS"
            if correct == len(PAYLOAD)
            else "FAIL"
        )

        print(
            f"Capacity    : "
            f"{len(PAYLOAD)}"
        )

        print(
            f"Direct      : "
            f"{direct_correct}/{len(PAYLOAD)} "
            f"({direct_correct / len(PAYLOAD) * 100:.2f}%)"
        )

        print(
            f"Video       : "
            f"{correct}/{len(PAYLOAD)} "
            f"({accuracy:.2f}%)"
        )

        print(
            f"Mismatch    : {mismatch}"
        )

        print(
            f"Result      : {result}"
        )

        print()

        results.append(
            (
                frame_index,
                correct,
                accuracy,
                mismatch,
            )
        )

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print()
    print(
        "FRAME   CORRECT   ACCURACY   MISMATCH   RESULT"
    )

    passed = 0

    for (
        frame_index,
        correct,
        accuracy,
        mismatch,
    ) in results:

        result = (
            "PASS"
            if correct == len(PAYLOAD)
            else "FAIL"
        )

        if result == "PASS":
            passed += 1

        print(
            f"{frame_index:5d}   "
            f"{correct:7d}   "
            f"{accuracy:8.2f}%   "
            f"{str(mismatch):8s}   "
            f"{result}"
        )

    print()

    print(
        f"Frames tested : {len(results)}"
    )

    print(
        f"Frames passed : {passed}"
    )

    print(
        f"Frames failed : "
        f"{len(results) - passed}"
    )

    print()

    if results and passed == len(results):

        print(
            "RESULT: 544-BIT BLIND SPATIAL "
            "EXTRACTION PASSED ALL TESTED FRAMES."
        )

    else:

        print(
            "RESULT: BLIND SPATIAL EXTRACTION "
            "IS NOT YET FULLY STABLE."
        )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "No production embedding code was modified."
    )


if __name__ == "__main__":
    main()