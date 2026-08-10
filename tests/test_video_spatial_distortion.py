"""
StegaFusion MP4 Spatial Distortion Diagnostic

Compares visual distortion for Delta 4, 6 and 8
using the proven 32x32 paired-block configuration.

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

DELTAS = [4, 6, 8]

PAYLOAD_BITS = 544
NUM_FRAMES = 10

OUTPUT_DIR = (
    PathConfig.OUTPUT_DIR
    / "video_spatial_distortion"
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
            f"Unable to open {video_path}"
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
            "Unable to decode MP4."
        )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            "Unable to decode frame."
        )

    return frame


def calculate_psnr(
    original,
    modified,
):

    original = (
        original.astype(
            np.float64
        )
    )

    modified = (
        modified.astype(
            np.float64
        )
    )

    mse = np.mean(
        (
            original
            - modified
        ) ** 2
    )

    if mse == 0:
        return float("inf")

    return float(
        10
        * np.log10(
            (255.0 ** 2)
            / mse
        )
    )


def main():

    print("=" * 78)
    print(
        "StegaFusion MP4 Spatial Distortion Diagnostic"
    )
    print("=" * 78)

    payload = create_payload()

    print()
    print(
        f"Payload    : {PAYLOAD_BITS} bits"
    )

    print(
        f"Block Size : "
        f"{BLOCK_SIZE} x {BLOCK_SIZE}"
    )

    print(
        f"Deltas     : {DELTAS}"
    )

    print()

    results = []

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

            original_blue = (
                source[:, :, 0]
            )

            positions = generate_positions(
                *original_blue.shape
            )

            stego_blue = embed_payload(
                original_blue,
                payload,
                positions,
                delta,
            )

            stego = source.copy()

            stego[:, :, 0] = stego_blue

            spatial_diff = np.abs(
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
                    spatial_diff
                )
            )

            max_difference = int(
                np.max(
                    spatial_diff
                )
            )

            mean_difference = float(
                np.mean(
                    spatial_diff
                )
            )

            psnr = calculate_psnr(
                original_blue,
                stego_blue,
            )

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

            decoded_blue = (
                decoded[:, :, 0]
            )

            decoded_difference = np.abs(
                decoded_blue.astype(
                    np.int16
                )
                -
                original_blue.astype(
                    np.int16
                )
            )

            decoded_changed = int(
                np.count_nonzero(
                    decoded_difference
                )
            )

            decoded_max = int(
                np.max(
                    decoded_difference
                )
            )

            decoded_mean = float(
                np.mean(
                    decoded_difference
                )
            )

            decoded_psnr = calculate_psnr(
                original_blue,
                decoded_blue,
            )

            print(
                f"Frame {frame_index:3d}: "
                f"Changed={changed_pixels:8d} "
                f"Max={max_difference:2d} "
                f"Mean={mean_difference:.6f} "
                f"PSNR={psnr:.2f} dB "
                f"DecodedMean={decoded_mean:.6f} "
                f"DecodedPSNR={decoded_psnr:.2f} dB"
            )

            delta_results.append(
                (
                    frame_index,
                    changed_pixels,
                    max_difference,
                    mean_difference,
                    psnr,
                    decoded_changed,
                    decoded_max,
                    decoded_mean,
                    decoded_psnr,
                )
            )

        results.append(
            (
                delta,
                delta_results,
            )
        )

        print()

    print("=" * 78)
    print(
        "DISTORTION SUMMARY"
    )
    print("=" * 78)

    print()

    print(
        "DELTA   MEAN_DIFF   MAX_DIFF   "
        "MEAN_PSNR   DECODED_MEAN   DECODED_PSNR"
    )

    for delta, delta_results in results:

        mean_diff = np.mean(
            [
                r[3]
                for r in delta_results
            ]
        )

        max_diff = max(
            r[2]
            for r in delta_results
        )

        mean_psnr = np.mean(
            [
                r[4]
                for r in delta_results
            ]
        )

        decoded_mean = np.mean(
            [
                r[7]
                for r in delta_results
            ]
        )

        decoded_psnr = np.mean(
            [
                r[8]
                for r in delta_results
            ]
        )

        print(
            f"{delta:5d}   "
            f"{mean_diff:9.6f}   "
            f"{max_diff:8d}   "
            f"{mean_psnr:9.2f}   "
            f"{decoded_mean:12.6f}   "
            f"{decoded_psnr:12.2f}"
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