"""
StegaFusion MP4 Spatial Frame Stability Diagnostic

Tests the 544-bit spatial paired-block method across
multiple frames of the source video.

Experimental only.
No production embedding code is modified.
"""

import cv2
import numpy as np

from config.config import PathConfig


# ==========================================================
# CONFIGURATION
# ==========================================================

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

BLOCK_SIZE = 32
GAP = 8
DELTA = 8

PAYLOAD_BITS = 544
NUM_FRAMES = 10


OUTPUT_DIR = (
    PathConfig.OUTPUT_DIR
    / "video_spatial_frame_stability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD FRAME
# ==========================================================

def load_frame(frame_index):

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
        frame_index
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            f"Unable to read frame {frame_index}"
        )

    return frame


# ==========================================================
# POSITIONS
# ==========================================================

def generate_positions(height, width):

    positions = []

    y = GAP

    while y + BLOCK_SIZE <= height - GAP:

        x = GAP

        while (
            x
            + BLOCK_SIZE
            + GAP
            + BLOCK_SIZE
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
                + GAP
            )

        y += (
            BLOCK_SIZE
            + GAP
        )

    return positions


# ==========================================================
# PAYLOAD
# ==========================================================

def create_payload():

    rng = np.random.default_rng(
        20260810
    )

    return "".join(
        rng.choice(
            ["0", "1"],
            size=PAYLOAD_BITS,
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

        target = (
            left
            if bit == "0"
            else right
        )

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

    left_scores = []
    right_scores = []

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

        left_scores.append(
            left_score
        )

        right_scores.append(
            right_score
        )

        recovered.append(
            "0"
            if left_score > right_score
            else "1"
        )

    return (
        "".join(recovered),
        left_scores,
        right_scores,
    )


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

def read_frame(output_path):

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
        "StegaFusion MP4 Spatial Frame Stability Diagnostic"
    )
    print("=" * 78)

    payload = create_payload()

    print()
    print(
        f"Payload     : {PAYLOAD_BITS} bits"
    )

    print(
        f"Block Size  : "
        f"{BLOCK_SIZE} x {BLOCK_SIZE}"
    )

    print(
        f"Delta       : {DELTA}"
    )

    print()

    results = []

    for frame_index in FRAME_INDICES:

        print(
            f"Testing frame {frame_index}..."
        )

        source = load_frame(
            frame_index
        )

        original_blue = source[:, :, 0]

        height, width = (
            original_blue.shape
        )

        positions = generate_positions(
            height,
            width,
        )

        capacity = len(positions)

        if capacity < PAYLOAD_BITS:

            print(
                f"  SKIP: capacity "
                f"{capacity} < "
                f"{PAYLOAD_BITS}"
            )

            results.append(
                (
                    frame_index,
                    capacity,
                    None,
                    None,
                    None,
                )
            )

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
            f"frame_{frame_index:03d}.mp4"
        )

        write_video(
            stego,
            output_path,
        )

        decoded = read_frame(
            output_path
        )

        decoded_blue = decoded[:, :, 0]

        (
            recovered,
            left_scores,
            right_scores,
        ) = extract_payload(
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

        stego_diff = np.abs(
            stego_blue.astype(
                np.int16
            )
            -
            original_blue.astype(
                np.int16
            )
        )

        changed_pixels = int(
            np.count_nonzero(
                stego_diff
            )
        )

        max_spatial = int(
            np.max(stego_diff)
        )

        mean_spatial = float(
            np.mean(stego_diff)
        )

        margin = np.abs(
            np.asarray(left_scores)
            -
            np.asarray(right_scores)
        )

        min_margin = float(
            np.min(margin)
        )

        mean_margin = float(
            np.mean(margin)
        )

        result = (
            "PASS"
            if accuracy == 100
            else "FAIL"
        )

        print(
            f"  Capacity    : {capacity}"
        )

        print(
            f"  Correct     : "
            f"{correct}/{PAYLOAD_BITS}"
        )

        print(
            f"  Accuracy    : "
            f"{accuracy:.2f}%"
        )

        print(
            f"  Mismatch    : "
            f"{mismatch}"
        )

        print(
            f"  Min Margin  : "
            f"{min_margin:.4f}"
        )

        print(
            f"  Mean Margin : "
            f"{mean_margin:.4f}"
        )

        print(
            f"  Result      : "
            f"{result}"
        )

        print()

        results.append(
            (
                frame_index,
                capacity,
                accuracy,
                min_margin,
                mean_margin,
            )
        )

    # ======================================================
    # SUMMARY
    # ======================================================

    print("=" * 78)
    print(
        "FRAME STABILITY SUMMARY"
    )
    print("=" * 78)

    print()

    print(
        "FRAME   CAPACITY   ACCURACY   "
        "MIN_MARGIN   MEAN_MARGIN   RESULT"
    )

    for result in results:

        (
            frame_index,
            capacity,
            accuracy,
            min_margin,
            mean_margin,
        ) = result

        if accuracy is None:

            print(
                f"{frame_index:5d}   "
                f"{capacity:8d}   "
                f"{'N/A':>8}   "
                f"{'N/A':>10}   "
                f"{'N/A':>11}   "
                f"SKIP"
            )

        else:

            status = (
                "PASS"
                if accuracy == 100
                else "FAIL"
            )

            print(
                f"{frame_index:5d}   "
                f"{capacity:8d}   "
                f"{accuracy:7.2f}%   "
                f"{min_margin:10.4f}   "
                f"{mean_margin:11.4f}   "
                f"{status}"
            )

    print()

    passed = [
        r
        for r in results
        if r[2] == 100
    ]

    print(
        f"Frames tested : "
        f"{len(results)}"
    )

    print(
        f"Frames passed : "
        f"{len(passed)}"
    )

    print(
        f"Frames failed : "
        f"{len(results) - len(passed)}"
    )

    print()

    if (
        len(passed)
        == len(results)
        and len(results) > 0
    ):

        print(
            "RESULT: 544-BIT PAYLOAD "
            "SURVIVED ON ALL TESTED FRAMES."
        )

    else:

        print(
            "RESULT: FRAME-DEPENDENT "
            "FAILURES DETECTED."
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