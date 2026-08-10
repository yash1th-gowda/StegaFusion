"""
StegaFusion MP4 Spatial Payload Capacity Diagnostic

Determines how many bits can survive MP4V compression
using the paired spatial-block method.

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

BLOCK_SIZE = 32
DELTA = 8
GAP = 16

NUM_FRAMES = 10

PAYLOAD_SIZES = [
    384,
    400,
    410,
    418,
]


OUTPUT_DIR = (
    PathConfig.OUTPUT_DIR
    / "video_spatial_capacity"
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
# GENERATE POSITIONS
# ==========================================================

def generate_positions(
    height,
    width,
):

    positions = []

    y = GAP

    while (
        y + BLOCK_SIZE
        <= height - GAP
    ):

        x = GAP

        while (
            x
            + BLOCK_SIZE * 2
            + GAP
            <= width - GAP
        ):

            left = (
                y,
                y + BLOCK_SIZE,
                x,
                x + BLOCK_SIZE,
            )

            right_x = (
                x
                + BLOCK_SIZE
                + GAP
            )

            right = (
                y,
                y + BLOCK_SIZE,
                right_x,
                right_x + BLOCK_SIZE,
            )

            positions.append(
                (left, right)
            )

            x += (
                BLOCK_SIZE * 2
                + GAP * 2
            )

        y += (
            BLOCK_SIZE
            + GAP
        )

    return positions


# ==========================================================
# CREATE PAYLOAD
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

            y0, y1, x0, x1 = left

        else:

            y0, y1, x0, x1 = right

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
    payload_length,
    positions,
):

    recovered = []

    for left, right in positions[
        :payload_length
    ]:

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

        if left_score > right_score:
            recovered.append("0")
        else:
            recovered.append("1")

    return "".join(recovered)


# ==========================================================
# WRITE MP4
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


# ==========================================================
# READ MP4
# ==========================================================

def read_first_frame(
    video_path,
):

    cap = cv2.VideoCapture(
        str(video_path)
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
        "StegaFusion MP4 Spatial Payload Capacity Diagnostic"
    )
    print("=" * 78)

    source = load_frame()

    original_blue = source[:, :, 0]

    height, width = (
        original_blue.shape
    )

    positions = generate_positions(
        height,
        width,
    )

    capacity = len(positions)

    print()
    print(
        f"Frame       : {FRAME_INDEX}"
    )

    print(
        f"Blue Shape  : "
        f"{original_blue.shape}"
    )

    print(
        f"Block Size  : "
        f"{BLOCK_SIZE} x {BLOCK_SIZE}"
    )

    print(
        f"Delta       : {DELTA}"
    )

    print(
        f"Available Block Pairs : {capacity}"
    )

    print()

    results = []

    for payload_size in PAYLOAD_SIZES:

        print(
            f"Testing payload : "
            f"{payload_size} bits..."
        )

        if payload_size > capacity:

            print(
                "  SKIP - insufficient "
                "block capacity"
            )

            results.append(
                (
                    payload_size,
                    None,
                    None,
                )
            )

            continue

        payload = create_payload(
            payload_size
        )

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
            f"payload_{payload_size}.mp4"
        )

        write_video(
            stego,
            output_path,
        )

        decoded = read_first_frame(
            output_path
        )

        decoded_blue = decoded[:, :, 0]

        recovered = extract_payload(
            original_blue,
            decoded_blue,
            payload_size,
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
            payload_size
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

        result = (
            "PASS"
            if accuracy == 100
            else "FAIL"
        )

        print(
            f"  Correct   : "
            f"{correct}/{payload_size}"
        )

        print(
            f"  Accuracy  : "
            f"{accuracy:.2f}%"
        )

        print(
            f"  Mismatch  : "
            f"{mismatch}"
        )

        print(
            f"  Result    : "
            f"{result}"
        )

        print()

        results.append(
            (
                payload_size,
                accuracy,
                result,
            )
        )

    # ======================================================
    # SUMMARY
    # ======================================================

    print()
    print("=" * 78)
    print(
        "CAPACITY SUMMARY"
    )
    print("=" * 78)

    print()

    print(
        "PAYLOAD     ACCURACY     RESULT"
    )

    print(
        "-------     --------     ------"
    )

    for (
        payload_size,
        accuracy,
        result,
    ) in results:

        if accuracy is None:

            print(
                f"{payload_size:7d}     "
                f"{'N/A':>8}     "
                f"SKIP"
            )

        else:

            print(
                f"{payload_size:7d}     "
                f"{accuracy:7.2f}%     "
                f"{result}"
            )

    # ======================================================
    # FIND LARGEST PERFECT PAYLOAD
    # ======================================================

    perfect = [
        size
        for size, accuracy, result
        in results
        if accuracy == 100
    ]

    print()

    if perfect:

        largest = max(perfect)

        print(
            f"Largest 100% payload : "
            f"{largest} bits"
        )

    else:

        print(
            "No tested payload achieved "
            "100% recovery."
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