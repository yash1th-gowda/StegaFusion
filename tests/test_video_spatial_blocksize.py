"""
StegaFusion MP4 Spatial Block-Size Diagnostic

Tests whether smaller paired spatial blocks can carry
the real 544-bit StegaFusion payload through MP4V.

Experimental only.
No production embedding code is modified.
"""

import cv2
import numpy as np

from config.config import PathConfig


# ==========================================================
# CONFIGURATION
# ==========================================================

FRAME_INDEX = 40
DELTA = 8
GAP = 8
NUM_FRAMES = 10

PAYLOAD_BITS = 544

BLOCK_SIZES = [
    32,
    24,
    16,
]


OUTPUT_DIR = (
    PathConfig.OUTPUT_DIR
    / "video_spatial_blocksize"
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
        raise RuntimeError(
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
# POSITIONS
# ==========================================================

def generate_positions(
    height,
    width,
    block_size,
    gap,
):

    positions = []

    y = gap

    while y + block_size <= height - gap:

        x = gap

        while (
            x
            + block_size
            + gap
            + block_size
            <= width - gap
        ):

            left = (
                y,
                y + block_size,
                x,
                x + block_size,
            )

            right_x = (
                x
                + block_size
                + gap
            )

            right = (
                y,
                y + block_size,
                right_x,
                right_x + block_size,
            )

            positions.append(
                (left, right)
            )

            x += (
                block_size
                * 2
                + gap
            )

        y += (
            block_size
            + gap
        )

    return positions


# ==========================================================
# PAYLOAD
# ==========================================================

def create_payload(length):

    rng = np.random.default_rng(
        20260810
    )

    return "".join(
        rng.choice(
            ["0", "1"],
            size=length,
        )
    )


# ==========================================================
# EMBED
# ==========================================================

def embed_payload(
    blue,
    payload,
    positions,
):

    result = blue.copy()

    for bit, pair in zip(
        payload,
        positions,
    ):

        left, right = pair

        if bit == "0":
            target = left
        else:
            target = right

        y0, y1, x0, x1 = target

        region = (
            result[
                y0:y1,
                x0:x1
            ].astype(
                np.int16
            )
        )

        result[
            y0:y1,
            x0:x1
        ] = np.clip(
            region + DELTA,
            0,
            255,
        ).astype(
            np.uint8
        )

    return result


# ==========================================================
# EXTRACT
# ==========================================================

def extract_payload(
    original_blue,
    decoded_blue,
    payload,
    positions,
):

    recovered = []

    for pair in positions[
        :len(payload)
    ]:

        left, right = pair

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

        recovered.append(
            "0"
            if left_score > right_score
            else "1"
        )

    return "".join(recovered)


# ==========================================================
# VIDEO
# ==========================================================

def write_video(
    frame,
    output_path,
):

    height, width = frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        23.976,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Unable to create MP4."
        )

    for _ in range(NUM_FRAMES):
        writer.write(frame)

    writer.release()


def read_frame(
    output_path,
):

    cap = cv2.VideoCapture(
        str(output_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open generated MP4."
        )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            "Unable to decode generated MP4."
        )

    return frame


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 78)
    print(
        "StegaFusion MP4 Spatial Block-Size Diagnostic"
    )
    print("=" * 78)

    source = load_frame()

    original_blue = source[:, :, 0]

    height, width = original_blue.shape

    payload = create_payload(
        PAYLOAD_BITS
    )

    print()
    print(
        f"Frame       : {FRAME_INDEX}"
    )

    print(
        f"Blue Shape  : "
        f"{original_blue.shape}"
    )

    print(
        f"Payload     : "
        f"{PAYLOAD_BITS} bits"
    )

    print(
        f"Delta       : {DELTA}"
    )

    print()

    results = []

    for block_size in BLOCK_SIZES:

        print("=" * 78)
        print(
            f"BLOCK SIZE : "
            f"{block_size} x {block_size}"
        )
        print("=" * 78)

        positions = generate_positions(
            height,
            width,
            block_size,
            GAP,
        )

        capacity = len(positions)

        print(
            f"Available Pairs : {capacity}"
        )

        if capacity < PAYLOAD_BITS:

            print(
                f"RESULT : SKIP"
            )

            print(
                f"Reason : "
                f"{capacity} < "
                f"{PAYLOAD_BITS}"
            )

            results.append(
                (
                    block_size,
                    capacity,
                    None,
                )
            )

            print()
            continue

        stego_blue = embed_payload(
            original_blue,
            payload,
            positions,
        )

        stego = source.copy()
        stego[:, :, 0] = stego_blue

        output_path = (
            OUTPUT_DIR
            /
            f"block_{block_size}.mp4"
        )

        print(
            f"MP4 : {output_path}"
        )

        write_video(
            stego,
            output_path,
        )

        decoded = read_frame(
            output_path
        )

        decoded_blue = decoded[:, :, 0]

        recovered = extract_payload(
            original_blue,
            decoded_blue,
            payload,
            positions,
        )

        correct = sum(
            a == b
            for a, b in zip(
                payload,
                recovered,
            )
        )

        accuracy = (
            correct
            /
            PAYLOAD_BITS
            *
            100
        )

        mismatch = next(
            (
                i
                for i, (a, b)
                in enumerate(
                    zip(
                        payload,
                        recovered,
                    )
                )
                if a != b
            ),
            None,
        )

        spatial_diff = np.abs(
            stego_blue.astype(
                np.int16
            )
            -
            original_blue.astype(
                np.int16
            )
        )

        print()
        print(
            f"Correct Bits : "
            f"{correct}/{PAYLOAD_BITS}"
        )

        print(
            f"Accuracy     : "
            f"{accuracy:.2f}%"
        )

        print(
            f"First Mismatch : "
            f"{mismatch}"
        )

        print(
            f"Changed Pixels : "
            f"{np.count_nonzero(spatial_diff)}"
        )

        print(
            f"Maximum Spatial Difference : "
            f"{np.max(spatial_diff)}"
        )

        result = (
            "PASS"
            if accuracy == 100
            else "FAIL"
        )

        print(
            f"RESULT : {result}"
        )

        results.append(
            (
                block_size,
                capacity,
                accuracy,
            )
        )

        print()

    # ======================================================
    # SUMMARY
    # ======================================================

    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    print()

    print(
        "BLOCK       CAPACITY       ACCURACY       RESULT"
    )

    for (
        block_size,
        capacity,
        accuracy,
    ) in results:

        if accuracy is None:

            print(
                f"{block_size:5d}       "
                f"{capacity:8d}       "
                f"{'N/A':>8}       "
                f"SKIP"
            )

        else:

            result = (
                "PASS"
                if accuracy == 100
                else "FAIL"
            )

            print(
                f"{block_size:5d}       "
                f"{capacity:8d}       "
                f"{accuracy:7.2f}%       "
                f"{result}"
            )

    print()

    passing = [
        (
            block_size,
            capacity,
        )
        for (
            block_size,
            capacity,
            accuracy,
        ) in results
        if accuracy == 100
    ]

    if passing:

        print(
            "544-BIT MP4V CAPABLE CONFIGURATIONS:"
        )

        for block_size, capacity in passing:

            print(
                f"  {block_size}x{block_size}"
                f" -> {capacity} pairs"
            )

    else:

        print(
            "No tested block size carried "
            "the full 544-bit payload."
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