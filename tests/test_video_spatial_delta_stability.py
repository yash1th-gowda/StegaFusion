"""
StegaFusion MP4 Spatial Delta Stability Diagnostic

Tests the spatial paired-block method across multiple frames
and multiple embedding deltas.

Experimental only.
No production embedding code is modified.
"""

import cv2
import numpy as np

from config.config import PathConfig


FRAME_INDICES = [
    0, 1, 5, 10, 20, 40,
    60, 80, 100, 150, 196,
]

BLOCK_SIZE = 32
GAP = 8

DELTAS = [
    2, 4, 6, 8, 10, 12, 16
]

PAYLOAD_BITS = 544
NUM_FRAMES = 10

OUTPUT_DIR = (
    PathConfig.OUTPUT_DIR
    / "video_spatial_delta_stability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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


def embed_payload(
    blue,
    payload,
    positions,
    delta,
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
            region + delta,
            0,
            255,
        ).astype(
            np.uint8
        )

    return result


def extract_payload(
    original_blue,
    decoded_blue,
    payload,
    positions,
):

    recovered = []

    margins = []

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

        if left_score > right_score:
            recovered.append("0")
        else:
            recovered.append("1")

        margins.append(
            abs(
                left_score
                - right_score
            )
        )

    return (
        "".join(recovered),
        margins,
    )


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


def main():

    print("=" * 78)
    print(
        "StegaFusion MP4 Spatial Delta Stability Diagnostic"
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
        f"Frames      : {len(FRAME_INDICES)}"
    )

    print()

    all_results = []

    for delta in DELTAS:

        print("-" * 78)
        print(
            f"TESTING DELTA : {delta}"
        )
        print("-" * 78)

        delta_results = []

        for frame_index in FRAME_INDICES:

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
                    f"Frame {frame_index:3d}: "
                    f"SKIP capacity={capacity}"
                )

                continue

            stego_blue = embed_payload(
                original_blue,
                payload,
                positions,
                delta,
            )

            stego = source.copy()

            stego[:, :, 0] = stego_blue

            output_path = (
                OUTPUT_DIR
                /
                f"delta_{delta}"
                /
                f"frame_{frame_index:03d}.mp4"
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
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
                margins,
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

            min_margin = float(
                np.min(margins)
            )

            mean_margin = float(
                np.mean(margins)
            )

            passed = (
                correct
                == PAYLOAD_BITS
            )

            delta_results.append(
                (
                    frame_index,
                    accuracy,
                    min_margin,
                    mean_margin,
                    passed,
                )
            )

            print(
                f"Frame {frame_index:3d}: "
                f"{correct:3d}/{PAYLOAD_BITS} "
                f"{accuracy:7.2f}% "
                f"MinMargin={min_margin:7.3f} "
                f"MeanMargin={mean_margin:7.3f} "
                f"{'PASS' if passed else 'FAIL'}"
            )

        passed_count = sum(
            r[4]
            for r in delta_results
        )

        failed_count = (
            len(delta_results)
            - passed_count
        )

        if delta_results:

            worst_accuracy = min(
                r[1]
                for r in delta_results
            )

            worst_margin = min(
                r[2]
                for r in delta_results
            )

            mean_margin = np.mean(
                [
                    r[3]
                    for r in delta_results
                ]
            )

        else:

            worst_accuracy = 0
            worst_margin = 0
            mean_margin = 0

        print()

        print(
            f"Delta {delta}: "
            f"{passed_count} PASS / "
            f"{failed_count} FAIL"
        )

        print(
            f"Worst Accuracy : "
            f"{worst_accuracy:.2f}%"
        )

        print(
            f"Worst Margin   : "
            f"{worst_margin:.4f}"
        )

        print(
            f"Mean Margin    : "
            f"{mean_margin:.4f}"
        )

        all_results.append(
            (
                delta,
                passed_count,
                failed_count,
                worst_accuracy,
                worst_margin,
                mean_margin,
            )
        )

        print()

    print("=" * 78)
    print(
        "DELTA ROBUSTNESS SUMMARY"
    )
    print("=" * 78)

    print()

    print(
        "DELTA   PASS   FAIL   "
        "WORST_ACC   WORST_MARGIN   MEAN_MARGIN"
    )

    for result in all_results:

        (
            delta,
            passed_count,
            failed_count,
            worst_accuracy,
            worst_margin,
            mean_margin,
        ) = result

        print(
            f"{delta:5d}   "
            f"{passed_count:4d}   "
            f"{failed_count:4d}   "
            f"{worst_accuracy:9.2f}%   "
            f"{worst_margin:12.4f}   "
            f"{mean_margin:11.4f}"
        )

    print()

    reliable = [
        r
        for r in all_results
        if r[2] == 0
    ]

    if reliable:

        best = min(
            reliable,
            key=lambda r: r[0]
        )

        print(
            f"LOWEST FULLY RELIABLE DELTA : "
            f"{best[0]}"
        )

        print(
            f"Worst Accuracy : "
            f"{best[3]:.2f}%"
        )

        print(
            f"Worst Margin   : "
            f"{best[4]:.4f}"
        )

    else:

        print(
            "No tested Delta achieved "
            "100% recovery on every frame."
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