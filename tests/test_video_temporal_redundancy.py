"""
StegaFusion Temporal Redundancy Diagnostic

Tests whether repeating the same embedded bit across
multiple video frames can survive MP4 compression.

Diagnostic only.
Does NOT modify production embedding code.
"""

from pathlib import Path

import cv2
import numpy as np

from config.config import PathConfig

from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)

from modules.steganography.lsb_utils import (
    embed_one_bit,
    extract_one_bit,
)


# ==========================================================
# SETTINGS
# ==========================================================

FRAME_COUNT = 197

PAYLOAD_BITS = (
    "101100111000111100001111"
)

# Number of video frames used for one logical bit.
REPETITIONS = (
    1,
    3,
    5,
    9,
)

TEST_BITS = len(
    PAYLOAD_BITS
)

FPS = 23.976023976023978

OUTPUT_DIR = (
    PathConfig.OUTPUT_DIR /
    "video_temporal_redundancy"
)


# ==========================================================
# SAFE RECONSTRUCTION
# ==========================================================

def reconstruct_uint8(
    bands
):

    reconstructed = apply_inverse_dwt(
        bands
    )

    return np.round(
        reconstructed
    ).clip(
        0,
        255
    ).astype(np.uint8)


# ==========================================================
# LOAD SOURCE FRAMES
# ==========================================================

def load_source_frames():

    frames = []

    for index in range(
        FRAME_COUNT
    ):

        path = (
            PathConfig.FRAME_DIR /
            f"frame_{index:05d}.png"
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        frame = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR
        )

        if frame is None:

            raise RuntimeError(
                f"Unable to read {path}"
            )

        frames.append(
            frame
        )

    return frames


# ==========================================================
# EMBED ONE BIT
# ==========================================================

def embed_bit_in_frame(
    frame,
    bit
):

    blue = frame[:, :, 0]

    bands = apply_dwt(
        blue
    )

    # ------------------------------------------------------
    # Use a known stable-ish LH location from our diagnostic.
    #
    # This is NOT production logic.
    # ------------------------------------------------------

    row = 28
    col = 555

    original = float(
        bands["LH"][
            row,
            col
        ]
    )

    embedded = embed_one_bit(
        int(
            round(
                original
            )
        ),
        bit
    )

    modified = {
        key: value.copy()
        for key, value in bands.items()
    }

    modified["LH"][
        row,
        col
    ] = float(
        embedded
    )

    reconstructed = reconstruct_uint8(
        modified
    )

    output = frame.copy()

    output[:, :, 0] = reconstructed

    return output


# ==========================================================
# CREATE VIDEO
# ==========================================================

def create_video(
    source_frames,
    repetitions
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR /
        f"repetition_{repetitions}.mp4"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (
            source_frames[0].shape[1],
            source_frames[0].shape[0],
        )
    )

    if not writer.isOpened():

        raise RuntimeError(
            f"Unable to create {output_path}"
        )

    frame_index = 0

    # ------------------------------------------------------
    # Each logical bit is repeated N times.
    # ------------------------------------------------------

    for bit in PAYLOAD_BITS:

        for _ in range(
            repetitions
        ):

            if frame_index >= len(
                source_frames
            ):
                break

            stego = embed_bit_in_frame(
                source_frames[
                    frame_index
                ],
                bit
            )

            writer.write(
                stego
            )

            frame_index += 1

    # ------------------------------------------------------
    # Fill remaining frames unchanged.
    # ------------------------------------------------------

    while frame_index < len(
        source_frames
    ):

        writer.write(
            source_frames[
                frame_index
            ]
        )

        frame_index += 1

    writer.release()

    return output_path


# ==========================================================
# READ VIDEO
# ==========================================================

def read_video(
    path
):

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Unable to open {path}"
        )

    frames = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        frames.append(
            frame
        )

    cap.release()

    return frames


# ==========================================================
# EXTRACT BIT
# ==========================================================

def extract_bit_from_frame(
    frame
):

    blue = frame[:, :, 0]

    bands = apply_dwt(
        blue
    )

    value = int(
        round(
            bands["LH"][
                28,
                555
            ]
        )
    )

    return extract_one_bit(
        value
    )


# ==========================================================
# MAJORITY VOTE
# ==========================================================

def majority_vote(
    bits
):

    ones = bits.count(
        "1"
    )

    zeros = bits.count(
        "0"
    )

    if ones >= zeros:

        return "1"

    return "0"


# ==========================================================
# TEST REPETITION
# ==========================================================

def test_repetition(
    source_frames,
    repetitions
):

    print()
    print("=" * 70)

    print(
        f"Testing repetition = "
        f"{repetitions}"
    )

    print("=" * 70)

    video_path = create_video(
        source_frames,
        repetitions
    )

    print(
        f"Video : {video_path}"
    )

    decoded_frames = read_video(
        video_path
    )

    print(
        f"Decoded Frames : "
        f"{len(decoded_frames)}"
    )

    recovered_bits = []

    frame_index = 0

    for bit_index in range(
        TEST_BITS
    ):

        observations = []

        for _ in range(
            repetitions
        ):

            if frame_index >= len(
                decoded_frames
            ):
                break

            recovered = (
                extract_bit_from_frame(
                    decoded_frames[
                        frame_index
                    ]
                )
            )

            observations.append(
                recovered
            )

            frame_index += 1

        if not observations:

            break

        voted = majority_vote(
            observations
        )

        recovered_bits.append(
            voted
        )

    recovered = "".join(
        recovered_bits
    )

    # ------------------------------------------------------
    # Compare
    # ------------------------------------------------------

    mismatch = None

    limit = min(
        len(PAYLOAD_BITS),
        len(recovered)
    )

    for index in range(
        limit
    ):

        if (
            PAYLOAD_BITS[index]
            != recovered[index]
        ):

            mismatch = index
            break

    if mismatch is None:

        if len(recovered) != len(
            PAYLOAD_BITS
        ):

            mismatch = limit

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    if recovered == PAYLOAD_BITS:

        result = "PASS"

    else:

        result = "FAIL"

    print()
    print(
        f"Expected Bits  : "
        f"{len(PAYLOAD_BITS)}"
    )

    print(
        f"Recovered Bits : "
        f"{len(recovered)}"
    )

    print(
        f"Result         : "
        f"{result}"
    )

    print(
        f"First Mismatch : "
        f"{mismatch}"
    )

    if mismatch is not None:

        print(
            "Expected:",
            PAYLOAD_BITS[
                max(
                    0,
                    mismatch - 8
                ):
                mismatch + 8
            ]
        )

        print(
            "Recovered:",
            recovered[
                max(
                    0,
                    mismatch - 8
                ):
                mismatch + 8
            ]
        )

    return result


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "StegaFusion Temporal Redundancy Diagnostic"
    )
    print("=" * 70)

    print()
    print(
        f"Payload Bits : "
        f"{TEST_BITS}"
    )

    print(
        f"Available Frames : "
        f"{FRAME_COUNT}"
    )

    source_frames = (
        load_source_frames()
    )

    for repetitions in REPETITIONS:

        required = (
            TEST_BITS *
            repetitions
        )

        if required > FRAME_COUNT:

            print()
            print(
                f"Skipping repetition "
                f"{repetitions}: "
                f"requires {required} frames."
            )

            continue

        test_repetition(
            source_frames,
            repetitions
        )

    print()
    print("=" * 70)

    print(
        "Temporal Redundancy "
        "Diagnostic Complete"
    )

    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()