"""
StegaFusion MP4 Spatial Paired-Block Bit Diagnostic

Research-only experiment.

Tests whether one bit can be represented by choosing between
two spatial blocks and survive MP4V compression.

Bit 0:
    Left block  is brightened
    Right block unchanged

Bit 1:
    Right block is brightened
    Left block unchanged

No production embedding code is modified.
"""

from pathlib import Path

import cv2
import numpy as np

from config.config import PathConfig


# ==========================================================
# CONFIGURATION
# ==========================================================

FRAME_INDEX = 40

BLOCK_SIZES = [
    32,
    48,
    64,
    96,
]

DELTAS = [
    4,
    6,
    8,
    10,
    12,
    16,
]

NUM_FRAMES = 10

OUTPUT_DIR = (
    PathConfig.OUTPUT_DIR
    / "video_spatial_bit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD SOURCE FRAME
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
            f"Unable to create video: {output_path}"
        )

    for frame in frames:
        writer.write(frame)

    writer.release()


# ==========================================================
# READ FIRST FRAME
# ==========================================================

def read_video_frame(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open: {video_path}"
        )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            "Unable to decode MP4."
        )

    return frame


# ==========================================================
# CREATE PAIRED BLOCK SIGNAL
# ==========================================================

def create_bit_signal(
    blue,
    block_size,
    delta,
    bit,
):

    modified = blue.copy()

    height, width = blue.shape

    # Keep the pair centered.
    gap = block_size // 2

    total_width = (
        block_size * 2
        + gap
    )

    x_start = (
        width // 2
        - total_width // 2
    )

    y_start = (
        height // 2
        - block_size // 2
    )

    left_x0 = x_start
    left_x1 = (
        left_x0
        + block_size
    )

    right_x0 = (
        left_x1
        + gap
    )

    right_x1 = (
        right_x0
        + block_size
    )

    y0 = y_start
    y1 = (
        y0
        + block_size
    )

    if bit == 0:

        modified[
            y0:y1,
            left_x0:left_x1
        ] = np.clip(
            modified[
                y0:y1,
                left_x0:left_x1
            ].astype(np.int16)
            + delta,
            0,
            255,
        ).astype(np.uint8)

    else:

        modified[
            y0:y1,
            right_x0:right_x1
        ] = np.clip(
            modified[
                y0:y1,
                right_x0:right_x1
            ].astype(np.int16)
            + delta,
            0,
            255,
        ).astype(np.uint8)

    return modified, (
        y0,
        y1,
        left_x0,
        left_x1,
        right_x0,
        right_x1,
    )


# ==========================================================
# MEASURE BIT
# ==========================================================

def measure_bit(
    original_blue,
    decoded_blue,
    coordinates,
):

    (
        y0,
        y1,
        left_x0,
        left_x1,
        right_x0,
        right_x1,
    ) = coordinates

    original_left = (
        original_blue[
            y0:y1,
            left_x0:left_x1
        ].astype(np.float32)
    )

    original_right = (
        original_blue[
            y0:y1,
            right_x0:right_x1
        ].astype(np.float32)
    )

    decoded_left = (
        decoded_blue[
            y0:y1,
            left_x0:left_x1
        ].astype(np.float32)
    )

    decoded_right = (
        decoded_blue[
            y0:y1,
            right_x0:right_x1
        ].astype(np.float32)
    )

    left_delta = (
        decoded_left
        - original_left
    )

    right_delta = (
        decoded_right
        - original_right
    )

    left_score = float(
        np.mean(left_delta)
    )

    right_score = float(
        np.mean(right_delta)
    )

    # Bit 0 = left carries the signal.
    # Bit 1 = right carries the signal.
    recovered_bit = (
        0
        if left_score > right_score
        else 1
    )

    return (
        recovered_bit,
        left_score,
        right_score,
    )


# ==========================================================
# TEST ONE BIT
# ==========================================================

def test_bit(
    source,
    original_blue,
    block_size,
    delta,
    bit,
):

    modified_blue, coordinates = (
        create_bit_signal(
            original_blue,
            block_size,
            delta,
            bit,
        )
    )

    stego = source.copy()

    stego[:, :, 0] = (
        modified_blue
    )

    output_path = (
        OUTPUT_DIR
        /
        (
            f"size_{block_size}"
            f"_delta_{delta}"
            f"_bit_{bit}.mp4"
        )
    )

    frames = [
        stego.copy()
        for _ in range(NUM_FRAMES)
    ]

    write_video(
        frames,
        output_path,
    )

    decoded = read_video_frame(
        output_path
    )

    decoded_blue = (
        decoded[:, :, 0]
    )

    recovered_bit, left_score, right_score = (
        measure_bit(
            original_blue,
            decoded_blue,
            coordinates,
        )
    )

    return (
        recovered_bit,
        left_score,
        right_score,
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 78)
    print(
        "StegaFusion MP4 Spatial Paired-Block Bit Diagnostic"
    )
    print("=" * 78)

    source = load_frame()

    original_blue = (
        source[:, :, 0]
    )

    print()
    print(
        f"Frame       : {FRAME_INDEX}"
    )

    print(
        f"Blue Shape  : {original_blue.shape}"
    )

    print(
        f"Blue Range  : "
        f"{original_blue.min()} "
        f"-> "
        f"{original_blue.max()}"
    )

    print()

    print(
        "SIZE  DELTA   BIT0  BIT1  "
        "ACCURACY   "
        "B0_LEFT   B0_RIGHT   "
        "B1_LEFT   B1_RIGHT"
    )

    print("-" * 78)

    results = []

    for block_size in BLOCK_SIZES:

        for delta in DELTAS:

            (
                recovered_0,
                b0_left,
                b0_right,
            ) = test_bit(
                source,
                original_blue,
                block_size,
                delta,
                0,
            )

            (
                recovered_1,
                b1_left,
                b1_right,
            ) = test_bit(
                source,
                original_blue,
                block_size,
                delta,
                1,
            )

            correct = (
                int(recovered_0 == 0)
                +
                int(recovered_1 == 1)
            )

            accuracy = (
                correct
                / 2
                * 100
            )

            results.append(
                (
                    block_size,
                    delta,
                    accuracy,
                    recovered_0,
                    recovered_1,
                )
            )

            print(
                f"{block_size:4d}  "
                f"{delta:5d}   "
                f"{recovered_0:^4d}  "
                f"{recovered_1:^4d}  "
                f"{accuracy:7.1f}%   "
                f"{b0_left:8.3f}   "
                f"{b0_right:9.3f}   "
                f"{b1_left:8.3f}   "
                f"{b1_right:9.3f}"
            )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    best = max(
        results,
        key=lambda item: (
            item[2],
            -item[1],
            -item[0],
        ),
    )

    (
        best_size,
        best_delta,
        best_accuracy,
        best_bit0,
        best_bit1,
    ) = best

    print()
    print(
        f"Best Block Size : {best_size} x {best_size}"
    )

    print(
        f"Best Delta      : {best_delta}"
    )

    print(
        f"Bit Accuracy    : {best_accuracy:.1f}%"
    )

    print(
        f"Bit 0 Recovered : {best_bit0}"
    )

    print(
        f"Bit 1 Recovered : {best_bit1}"
    )

    print()

    if best_accuracy >= 100:

        print(
            "RESULT: A perfect paired-block bit "
            "configuration was observed."
        )

    elif best_accuracy >= 50:

        print(
            "RESULT: Spatial bit recovery is "
            "partially promising."
        )

    else:

        print(
            "RESULT: Paired-block spatial encoding "
            "is not yet reliable."
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