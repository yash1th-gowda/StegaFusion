"""
StegaFusion MP4 Spatial Payload Diagnostic

Research-only experiment.

Encodes a multi-bit payload using paired spatial blocks
and tests recovery after MP4V compression.

Bit 0:
    Left block is modified.

Bit 1:
    Right block is modified.

No production embedding code is modified.
"""

import cv2
import numpy as np

from config.config import PathConfig


# ==========================================================
# CONFIGURATION
# ==========================================================

FRAME_INDEX = 40

PAYLOAD = "101100111000111100001111"

BLOCK_SIZE = 32

DELTA = 8

GAP = 16

NUM_FRAMES = 10

OUTPUT_DIR = (
    PathConfig.OUTPUT_DIR
    / "video_spatial_payload"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD FRAME
# ==========================================================

def load_frame():

    video_path = (
        PathConfig.COVER_VIDEO_DIR
        / "sample.mp4"
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Unable to open video: {video_path}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        FRAME_INDEX
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            f"Unable to read frame {FRAME_INDEX}"
        )

    return frame


# ==========================================================
# BLOCK LOCATIONS
# ==========================================================

def generate_positions(
    height,
    width,
    count,
):

    positions = []

    pair_width = (
        BLOCK_SIZE * 2
        + GAP
    )

    pair_height = BLOCK_SIZE

    margin = GAP

    y = margin

    while (
        y + BLOCK_SIZE
        <= height - margin
    ):

        x = margin

        while (
            x
            + BLOCK_SIZE * 2
            + GAP
            <= width - margin
        ):

            left = (
                y,
                y + BLOCK_SIZE,
                x,
                x + BLOCK_SIZE,
            )

            right_x0 = (
                x
                + BLOCK_SIZE
                + GAP
            )

            right = (
                y,
                y + BLOCK_SIZE,
                right_x0,
                right_x0 + BLOCK_SIZE,
            )

            positions.append(
                (left, right)
            )

            if len(positions) >= count:
                return positions

            x += (
                pair_width
                + GAP
            )

        y += (
            pair_height
            + GAP
        )

    return positions


# ==========================================================
# VALIDATE POSITIONS
# ==========================================================

def validate_positions(
    positions,
    payload_length,
    height,
    width,
):

    if len(positions) < payload_length:

        raise RuntimeError(
            f"Only {len(positions)} valid block pairs "
            f"available for {payload_length} payload bits."
        )

    for index, (left, right) in enumerate(
        positions[:payload_length]
    ):

        for name, block in (
            ("left", left),
            ("right", right),
        ):

            y0, y1, x0, x1 = block

            if not (
                0 <= y0 < y1 <= height
                and
                0 <= x0 < x1 <= width
            ):

                raise RuntimeError(
                    f"Invalid {name} block "
                    f"at payload index {index}: "
                    f"{block}"
                )


# ==========================================================
# EMBED PAYLOAD
# ==========================================================

def embed_payload(
    blue,
    payload,
    positions,
):

    modified = blue.copy()

    for bit, pair in zip(
        payload,
        positions,
    ):

        left, right = pair

        if bit == "0":

            y0, y1, x0, x1 = left

        else:

            y0, y1, x0, x1 = right

        region = (
            modified[
                y0:y1,
                x0:x1
            ].astype(
                np.int16
            )
        )

        modified[
            y0:y1,
            x0:x1
        ] = np.clip(
            region + DELTA,
            0,
            255,
        ).astype(
            np.uint8
        )

    return modified


# ==========================================================
# WRITE VIDEO
# ==========================================================

def write_video(
    frames,
    output_path,
    fps=23.976,
):

    height, width = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Unable to create MP4."
        )

    for frame in frames:
        writer.write(frame)

    writer.release()


# ==========================================================
# READ VIDEO
# ==========================================================

def read_frame(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open MP4."
        )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            "Unable to decode MP4."
        )

    return frame


# ==========================================================
# EXTRACT PAYLOAD
# ==========================================================

def extract_payload(
    original_blue,
    decoded_blue,
    positions,
):

    recovered = []

    scores = []

    for index, (left, right) in enumerate(
        positions
    ):

        ly0, ly1, lx0, lx1 = left
        ry0, ry1, rx0, rx1 = right

        original_left = (
            original_blue[
                ly0:ly1,
                lx0:lx1
            ].astype(
                np.float32
            )
        )

        original_right = (
            original_blue[
                ry0:ry1,
                rx0:rx1
            ].astype(
                np.float32
            )
        )

        decoded_left = (
            decoded_blue[
                ly0:ly1,
                lx0:lx1
            ].astype(
                np.float32
            )
        )

        decoded_right = (
            decoded_blue[
                ry0:ry1,
                rx0:rx1
            ].astype(
                np.float32
            )
        )

        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------

        if (
            original_left.size == 0
            or original_right.size == 0
            or decoded_left.size == 0
            or decoded_right.size == 0
        ):

            raise RuntimeError(
                f"Empty spatial block at payload "
                f"index {index}."
            )

        left_score = float(
            np.mean(
                decoded_left
                - original_left
            )
        )

        right_score = float(
            np.mean(
                decoded_right
                - original_right
            )
        )

        # --------------------------------------------------
        # NaN / Inf protection
        # --------------------------------------------------

        if not (
            np.isfinite(left_score)
            and np.isfinite(right_score)
        ):

            raise RuntimeError(
                "Invalid spatial block produced "
                f"NaN/Inf at payload index {index}. "
                f"Left={left_score}, "
                f"Right={right_score}"
            )

        if left_score > right_score:

            recovered.append("0")

        else:

            recovered.append("1")

        scores.append(
            (
                left_score,
                right_score,
            )
        )

    return "".join(recovered), scores


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 78)
    print(
        "StegaFusion MP4 Spatial Payload Diagnostic"
    )
    print("=" * 78)

    source = load_frame()

    original_blue = (
        source[:, :, 0]
    )

    height, width = (
        original_blue.shape
    )

    # ------------------------------------------------------
    # Generate valid block positions
    # ------------------------------------------------------

    positions = generate_positions(
        height,
        width,
        len(PAYLOAD),
    )

    validate_positions(
        positions,
        len(PAYLOAD),
        height,
        width,
    )

    # Only use positions required by payload.
    positions = positions[
        :len(PAYLOAD)
    ]

    print()
    print(
        f"Frame       : {FRAME_INDEX}"
    )

    print(
        f"Blue Shape  : "
        f"{original_blue.shape}"
    )

    print(
        f"Payload     : {PAYLOAD}"
    )

    print(
        f"Payload Bits: {len(PAYLOAD)}"
    )

    print(
        f"Block Size  : "
        f"{BLOCK_SIZE} x {BLOCK_SIZE}"
    )

    print(
        f"Delta       : {DELTA}"
    )

    print(
        f"Valid Pairs : {len(positions)}"
    )

    print()

    # ------------------------------------------------------
    # EMBEDDING
    # ------------------------------------------------------

    stego_blue = embed_payload(
        original_blue,
        PAYLOAD,
        positions,
    )

    stego = source.copy()

    stego[:, :, 0] = stego_blue

    # ------------------------------------------------------
    # DIRECT EXTRACTION
    # ------------------------------------------------------

    direct_payload, _ = (
        extract_payload(
            original_blue,
            stego_blue,
            positions,
        )
    )

    print(
        f"Direct Payload : "
        f"{direct_payload}"
    )

    print(
        "Direct Result  : "
        +
        (
            "PASS"
            if direct_payload == PAYLOAD
            else "FAIL"
        )
    )

    # ------------------------------------------------------
    # MP4 ENCODING
    # ------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        /
        "spatial_payload.mp4"
    )

    frames = [
        stego.copy()
        for _ in range(NUM_FRAMES)
    ]

    write_video(
        frames,
        output_path,
    )

    print()
    print(
        f"MP4 : {output_path}"
    )

    decoded = read_frame(
        output_path
    )

    decoded_blue = (
        decoded[:, :, 0]
    )

    # ------------------------------------------------------
    # EXTRACTION AFTER MP4
    # ------------------------------------------------------

    recovered, scores = (
        extract_payload(
            original_blue,
            decoded_blue,
            positions,
        )
    )

    print()
    print(
        f"Recovered Payload : "
        f"{recovered}"
    )

    print(
        f"Expected Payload  : "
        f"{PAYLOAD}"
    )

    matches = sum(
        a == b
        for a, b in zip(
            PAYLOAD,
            recovered,
        )
    )

    accuracy = (
        matches
        /
        len(PAYLOAD)
        *
        100
    )

    print()
    print(
        f"Correct Bits : "
        f"{matches}/{len(PAYLOAD)}"
    )

    print(
        f"Accuracy     : "
        f"{accuracy:.2f}%"
    )

    mismatch = next(
        (
            i
            for i, (a, b)
            in enumerate(
                zip(
                    PAYLOAD,
                    recovered,
                )
            )
            if a != b
        ),
        None,
    )

    print(
        f"First Mismatch : "
        f"{mismatch}"
    )

    # ------------------------------------------------------
    # BIT SCORES
    # ------------------------------------------------------

    print()
    print(
        "BIT-BY-BIT SCORES"
    )

    print(
        "INDEX  EXPECTED  "
        "RECOVERED  "
        "LEFT      RIGHT"
    )

    print("-" * 60)

    for index, (
        expected,
        actual,
    ) in enumerate(
        zip(
            PAYLOAD,
            recovered,
        )
    ):

        left_score, right_score = (
            scores[index]
        )

        print(
            f"{index:5d}  "
            f"{expected:^8s}  "
            f"{actual:^9s}  "
            f"{left_score:8.3f}  "
            f"{right_score:8.3f}"
        )

    print()
    print("=" * 78)

    if accuracy == 100:

        print(
            "RESULT: MULTI-BIT SPATIAL PAYLOAD "
            "SURVIVED MP4V."
        )

    elif accuracy >= 90:

        print(
            "RESULT: SPATIAL PAYLOAD IS "
            "HIGHLY PROMISING."
        )

    elif accuracy >= 75:

        print(
            "RESULT: SPATIAL PAYLOAD IS "
            "PARTIALLY ROBUST."
        )

    else:

        print(
            "RESULT: SPATIAL PAYLOAD REQUIRES "
            "FURTHER WORK."
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "No production embedding code was modified."
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()