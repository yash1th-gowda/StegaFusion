"""
StegaFusion Spatial Blind Extraction Margin Diagnostic

Inspects the exact bits that failed after MP4 encoding.
No production code is modified.
"""

from pathlib import Path

import cv2

from modules.steganography.spatial.paired_block import (
    embed_payload,
    get_block_pairs,
)


PAYLOAD = ("101100111000111100001111" * 23)[:544]

BLOCK_SIZE = 32
DELTA = 4
GAP = 16

TESTS = {
    40: 182,
    60: 157,
    150: 58,
}

FRAME_DIR = Path("temp/frames")
OUTPUT_DIR = Path("output/video_spatial_blind/margin")


def block_mean(channel, bounds):
    y1, y2, x1, x2 = bounds
    return float(channel[y1:y2, x1:x2].mean())


def make_mp4(frame, path):
    height, width = frame.shape[:2]

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        24.0,
        (width, height),
    )

    writer.write(frame)
    writer.release()


def decode_mp4(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Cannot decode {path}")

    return frame


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("StegaFusion Spatial Blind Extraction Margin Diagnostic")
    print("=" * 70)

    print()
    print(f"Payload Bits : {len(PAYLOAD)}")
    print(f"Block Size   : {BLOCK_SIZE}")
    print(f"Delta        : {DELTA}")
    print(f"Gap          : {GAP}")
    print()

    for frame_index, bit_index in TESTS.items():

        print("=" * 70)
        print(
            f"FRAME {frame_index} / BIT {bit_index}"
        )
        print("=" * 70)

        frame_path = (
            FRAME_DIR /
            f"frame_{frame_index:05d}.png"
        )

        original = cv2.imread(
            str(frame_path)
        )

        if original is None:
            print(
                f"ERROR: missing {frame_path}"
            )
            continue

        stego, embedded = embed_payload(
            original,
            PAYLOAD,
            block_size=BLOCK_SIZE,
            delta=DELTA,
        )

        pairs = get_block_pairs(
            original.shape[0],
            original.shape[1],
            BLOCK_SIZE,
        )

        pair = pairs[bit_index]

        original_channel = original[:, :, 0]
        stego_channel = stego[:, :, 0]

        original_left = block_mean(
            original_channel,
            pair[0],
        )

        original_right = block_mean(
            original_channel,
            pair[1],
        )

        stego_left = block_mean(
            stego_channel,
            pair[0],
        )

        stego_right = block_mean(
            stego_channel,
            pair[1],
        )

        original_difference = (
            original_left - original_right
        )

        embedded_difference = (
            stego_left - stego_right
        )

        output = (
            OUTPUT_DIR /
            f"frame_{frame_index:05d}.mp4"
        )

        make_mp4(
            stego,
            output,
        )

        decoded = decode_mp4(output)

        decoded_channel = decoded[:, :, 0]

        decoded_left = block_mean(
            decoded_channel,
            pair[0],
        )

        decoded_right = block_mean(
            decoded_channel,
            pair[1],
        )

        decoded_difference = (
            decoded_left - decoded_right
        )

        expected_bit = PAYLOAD[bit_index]

        recovered_bit = (
            "0"
            if (
                decoded_difference >= 0
            )
            else "1"
        )

        embedded_margin = abs(
            embedded_difference
        )

        decoded_margin = abs(
            decoded_difference
        )

        print()
        print(
            f"Expected Bit       : {expected_bit}"
        )
        print(
            f"Recovered Bit      : {recovered_bit}"
        )

        print()
        print(
            f"Original Left      : "
            f"{original_left:.6f}"
        )
        print(
            f"Original Right     : "
            f"{original_right:.6f}"
        )
        print(
            f"Original Diff      : "
            f"{original_difference:.6f}"
        )

        print()
        print(
            f"Embedded Left      : "
            f"{stego_left:.6f}"
        )
        print(
            f"Embedded Right     : "
            f"{stego_right:.6f}"
        )
        print(
            f"Embedded Diff      : "
            f"{embedded_difference:.6f}"
        )

        print()
        print(
            f"Decoded Left       : "
            f"{decoded_left:.6f}"
        )
        print(
            f"Decoded Right      : "
            f"{decoded_right:.6f}"
        )
        print(
            f"Decoded Diff       : "
            f"{decoded_difference:.6f}"
        )

        print()
        print(
            f"Embedded Margin    : "
            f"{embedded_margin:.6f}"
        )
        print(
            f"Decoded Margin     : "
            f"{decoded_margin:.6f}"
        )

        print()
        print(
            f"RESULT             : "
            f"{'PASS' if expected_bit == recovered_bit else 'FAIL'}"
        )

        print()

    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print()
    print(
        "No production embedding code was modified."
    )


if __name__ == "__main__":
    main()