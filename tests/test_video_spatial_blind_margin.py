"""
StegaFusion Spatial Blind Extraction Margin Diagnostic

Inspects the exact bits that failed after MP4 encoding.

This diagnostic reproduces the SAME differential extraction
logic used by modules.steganography.spatial.paired_block.

No production code is modified.
"""

from pathlib import Path

import cv2

from modules.steganography.spatial.paired_block import (
    embed_payload,
    get_block_pairs,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

PAYLOAD = ("101100111000111100001111" * 23)[:544]

BLOCK_SIZE = 32
DELTA = 4
GAP = 16

# Exact failures observed in the blind stability test.
TESTS = {
    40: 182,
    60: 157,
    150: 58,
}

FRAME_DIR = Path("temp/frames")
OUTPUT_DIR = Path(
    "output/video_spatial_blind/margin"
)


# ==========================================================
# BLOCK MEAN
# ==========================================================

def block_mean(channel, bounds):
    y1, y2, x1, x2 = bounds

    return float(
        channel[y1:y2, x1:x2].mean()
    )


# ==========================================================
# MP4 ENCODE
# ==========================================================

def make_mp4(frame, path):
    height, width = frame.shape[:2]

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        24.0,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Unable to create video: {path}"
        )

    writer.write(frame)
    writer.release()


# ==========================================================
# MP4 DECODE
# ==========================================================

def decode_mp4(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {path}"
        )

    ok, frame = cap.read()

    cap.release()

    if not ok:
        raise RuntimeError(
            f"Cannot decode {path}"
        )

    return frame


# ==========================================================
# MAIN
# ==========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        "StegaFusion Spatial Blind Extraction "
        "Margin Diagnostic"
    )
    print("=" * 70)

    print()
    print(
        f"Payload Bits : {len(PAYLOAD)}"
    )
    print(
        f"Block Size   : {BLOCK_SIZE}"
    )
    print(
        f"Delta        : {DELTA}"
    )
    print(
        f"Gap          : {GAP}"
    )
    print()

    for frame_index, bit_index in TESTS.items():

        print("=" * 70)
        print(
            f"FRAME {frame_index} / BIT {bit_index}"
        )
        print("=" * 70)

        frame_path = (
            FRAME_DIR
            / f"frame_{frame_index:05d}.png"
        )

        original = cv2.imread(
            str(frame_path)
        )

        if original is None:
            print(
                f"ERROR: missing {frame_path}"
            )
            continue

        # --------------------------------------------------
        # EMBED
        # --------------------------------------------------

        stego, embedded = embed_payload(
            original,
            PAYLOAD,
            block_size=BLOCK_SIZE,
            delta=DELTA,
            gap=GAP,
        )

        # --------------------------------------------------
        # GET EXACT PAIR
        # --------------------------------------------------

        pairs = get_block_pairs(
            original.shape[0],
            original.shape[1],
            BLOCK_SIZE,
            GAP,
        )

        if bit_index >= len(pairs):
            print(
                f"ERROR: bit index {bit_index} "
                f"exceeds available pairs "
                f"({len(pairs)})"
            )
            continue

        pair = pairs[bit_index]

        # --------------------------------------------------
        # CHANNELS
        # --------------------------------------------------

        original_channel = (
            original[:, :, 0]
        )

        stego_channel = (
            stego[:, :, 0]
        )

        # --------------------------------------------------
        # ORIGINAL MEANS
        # --------------------------------------------------

        original_left = block_mean(
            original_channel,
            pair[0],
        )

        original_right = block_mean(
            original_channel,
            pair[1],
        )

        original_difference = (
            original_left
            - original_right
        )

        # --------------------------------------------------
        # EMBEDDED MEANS
        # --------------------------------------------------

        stego_left = block_mean(
            stego_channel,
            pair[0],
        )

        stego_right = block_mean(
            stego_channel,
            pair[1],
        )

        embedded_difference = (
            stego_left
            - stego_right
        )

        # --------------------------------------------------
        # EMBEDDED CHANGES
        #
        # This is what the REAL extractor measures.
        # --------------------------------------------------

        embedded_left_change = (
            stego_left
            - original_left
        )

        embedded_right_change = (
            stego_right
            - original_right
        )

        embedded_change_margin = abs(
            embedded_left_change
            - embedded_right_change
        )

        # --------------------------------------------------
        # WRITE STEGO FRAME TO MP4
        # --------------------------------------------------

        output = (
            OUTPUT_DIR
            / f"frame_{frame_index:05d}.mp4"
        )

        make_mp4(
            stego,
            output,
        )

        # --------------------------------------------------
        # DECODE MP4
        # --------------------------------------------------

        decoded = decode_mp4(
            output
        )

        decoded_channel = (
            decoded[:, :, 0]
        )

        # --------------------------------------------------
        # DECODED MEANS
        # --------------------------------------------------

        decoded_left = block_mean(
            decoded_channel,
            pair[0],
        )

        decoded_right = block_mean(
            decoded_channel,
            pair[1],
        )

        decoded_difference = (
            decoded_left
            - decoded_right
        )

        # --------------------------------------------------
        # DECODED CHANGES
        #
        # IMPORTANT:
        # This exactly matches the production
        # extractor's logic:
        #
        # left_change  = decoded_left  - original_left
        # right_change = decoded_right - original_right
        # --------------------------------------------------

        decoded_left_change = (
            decoded_left
            - original_left
        )

        decoded_right_change = (
            decoded_right
            - original_right
        )

        decoded_change_difference = (
            decoded_left_change
            - decoded_right_change
        )

        decoded_change_margin = abs(
            decoded_change_difference
        )

        # --------------------------------------------------
        # EXPECTED / RECOVERED BIT
        #
        # Same decision rule as _extract_bit():
        #
        # if left_change > right_change:
        #     return "0"
        #
        # else:
        #     return "1"
        # --------------------------------------------------

        expected_bit = PAYLOAD[
            bit_index
        ]

        recovered_bit = (
            "0"
            if decoded_left_change
            > decoded_right_change
            else "1"
        )

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        result = (
            "PASS"
            if expected_bit == recovered_bit
            else "FAIL"
        )

        # --------------------------------------------------
        # OUTPUT
        # --------------------------------------------------

        print()

        print(
            f"Expected Bit       : "
            f"{expected_bit}"
        )

        print(
            f"Recovered Bit      : "
            f"{recovered_bit}"
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
            f"Embedded Left Change : "
            f"{embedded_left_change:.6f}"
        )

        print(
            f"Embedded Right Change: "
            f"{embedded_right_change:.6f}"
        )

        print(
            f"Embedded Change Margin: "
            f"{embedded_change_margin:.6f}"
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
            f"Decoded Left Change  : "
            f"{decoded_left_change:.6f}"
        )

        print(
            f"Decoded Right Change : "
            f"{decoded_right_change:.6f}"
        )

        print(
            f"Decoded Change Margin: "
            f"{decoded_change_margin:.6f}"
        )

        print()

        print(
            f"RESULT             : "
            f"{result}"
        )

        print()

    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print()

    print(
        "No production embedding code was modified."
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()