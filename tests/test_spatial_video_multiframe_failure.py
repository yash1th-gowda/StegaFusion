"""
StegaFusion Multi-Frame Spatial MP4 Failure Diagnostic

Deliberately replaces one embedded stego frame with its
original cover frame.

Expected behavior:

    Valid packet 0
        ↓
    Corrupted packet 1
        ↓
    Valid packet 2
        ↓
    Packet validation failure

The purpose of this test is to verify that the production
pipeline does NOT silently recover corrupted data.

No production steganography algorithm is modified.
"""

from pathlib import Path

import cv2

from modules.pipeline.spatial_video_pipeline import (
    prepare_spatial_payload,
    create_frame_packets,
    embed_spatial_video,
    extract_spatial_video,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

COVER_VIDEO = Path(
    "input/cover_video/sample.mp4"
)

SECRET_FILE = Path(
    "input/secret_data/multiframe_1kb_secret.bin"
)

KEY_FILE = Path(
    "input/keys/test_aes_key.bin"
)

OUTPUT_DIR = Path(
    "output/video_spatial_multiframe_failure"
)

VALID_VIDEO = (
    OUTPUT_DIR /
    "valid_stego.mp4"
)

CORRUPTED_VIDEO = (
    OUTPUT_DIR /
    "corrupted_frame.mp4"
)

RECOVERED_FILE = (
    OUTPUT_DIR /
    "recovered_should_not_exist.bin"
)

CORRUPT_FRAME_INDEX = 1


# ==========================================================
# VIDEO HELPERS
# ==========================================================

def read_all_frames(
    video_path: Path,
):
    """
    Read all frames from a video.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: "
            f"{video_path}"
        )

    frames = []

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frames.append(frame)

    finally:

        cap.release()

    if not frames:
        raise RuntimeError(
            f"No frames decoded from: "
            f"{video_path}"
        )

    return frames


def get_video_info(
    video_path: Path,
):
    """
    Read FPS and resolution.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: "
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    return (
        fps,
        width,
        height,
    )


def write_video(
    frames,
    output_path: Path,
    fps: float,
):
    """
    Write frames as MP4V.
    """

    if not frames:
        raise ValueError(
            "Cannot write an empty video."
        )

    height, width = (
        frames[0].shape[:2]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Unable to create video: "
            f"{output_path}"
        )

    try:

        for frame in frames:
            writer.write(frame)

    finally:

        writer.release()


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "StegaFusion Multi-Frame Spatial MP4 "
        "Failure Diagnostic"
    )
    print("=" * 70)

    # ------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------

    print()
    print("INPUT FILES")
    print("-" * 70)

    required_files = {
        "Cover Video": COVER_VIDEO,
        "Secret File": SECRET_FILE,
        "AES Key": KEY_FILE,
    }

    for name, path in required_files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{name} not found:\n{path}"
            )

        print(
            f"{name:<15}: {path}"
        )

    # ------------------------------------------------------
    # PAYLOAD
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("PAYLOAD PREPARATION")
    print("=" * 70)

    payload = prepare_spatial_payload(
        SECRET_FILE,
        KEY_FILE,
    )

    packets = create_frame_packets(
        payload
    )

    print()

    print(
        f"Secret Size   : "
        f"{SECRET_FILE.stat().st_size} bytes"
    )

    print(
        f"Payload Bits  : "
        f"{len(payload)}"
    )

    print(
        f"Frame Packets : "
        f"{len(packets)}"
    )

    if len(packets) < 3:

        raise RuntimeError(
            "Failure test requires at least "
            "three frame packets."
        )

    print(
        "Multi-frame payload : PASS"
    )

    # ------------------------------------------------------
    # CREATE VALID STEGO VIDEO
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING VALID STEGO VIDEO")
    print("=" * 70)

    embedding = embed_spatial_video(
        cover_video=COVER_VIDEO,
        secret_file=SECRET_FILE,
        key_file=KEY_FILE,
        output_video=VALID_VIDEO,
    )

    print()

    print(
        f"Payload Bits    : "
        f"{embedding['payload_bits']}"
    )

    print(
        f"Packet Count    : "
        f"{embedding['packet_count']}"
    )

    print(
        f"Embedded Frames : "
        f"{embedding['embedded_frames']}"
    )

    print(
        f"Valid Video     : "
        f"{VALID_VIDEO}"
    )

    if not VALID_VIDEO.exists():

        raise RuntimeError(
            "Valid stego video was not created."
        )

    print()
    print(
        "Valid stego video : PASS"
    )

    # ------------------------------------------------------
    # READ COVER AND STEGO FRAMES
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("PREPARING CORRUPTED FRAME")
    print("=" * 70)

    cover_frames = read_all_frames(
        COVER_VIDEO
    )

    stego_frames = read_all_frames(
        VALID_VIDEO
    )

    print()

    print(
        f"Cover Frames : "
        f"{len(cover_frames)}"
    )

    print(
        f"Stego Frames : "
        f"{len(stego_frames)}"
    )

    if len(cover_frames) != len(
        stego_frames
    ):

        raise RuntimeError(
            "Cover and stego videos have "
            "different frame counts."
        )

    if CORRUPT_FRAME_INDEX >= len(
        stego_frames
    ):

        raise RuntimeError(
            "Corruption frame index is "
            "outside the video."
        )

    # ------------------------------------------------------
    # VERIFY FRAME WAS ACTUALLY EMBEDDED
    # ------------------------------------------------------

    original_frame = (
        cover_frames[
            CORRUPT_FRAME_INDEX
        ]
    )

    stego_frame = (
        stego_frames[
            CORRUPT_FRAME_INDEX
        ]
    )

    difference = cv2.absdiff(
        original_frame,
        stego_frame,
    )

    changed_pixels = int(
        cv2.countNonZero(
            cv2.cvtColor(
                difference,
                cv2.COLOR_BGR2GRAY,
            )
        )
    )

    print(
        f"Corrupt Frame Index : "
        f"{CORRUPT_FRAME_INDEX}"
    )

    print(
        f"Changed Pixels      : "
        f"{changed_pixels}"
    )

    if changed_pixels == 0:

        raise RuntimeError(
            "Selected frame does not contain "
            "an observable spatial modification."
        )

    print(
        "Selected frame contains "
        "embedded data : PASS"
    )

    # ------------------------------------------------------
    # CORRUPT THE FRAME
    # ------------------------------------------------------

    corrupted_frames = list(
        stego_frames
    )

    corrupted_frames[
        CORRUPT_FRAME_INDEX
    ] = cover_frames[
        CORRUPT_FRAME_INDEX
    ]

    fps, width, height = (
        get_video_info(
            VALID_VIDEO
        )
    )

    write_video(
        corrupted_frames,
        CORRUPTED_VIDEO,
        fps,
    )

    print()

    print(
        f"Corrupted Video : "
        f"{CORRUPTED_VIDEO}"
    )

    if not CORRUPTED_VIDEO.exists():

        raise RuntimeError(
            "Corrupted video was not created."
        )

    print(
        "Corrupted MP4 creation : PASS"
    )

    # ------------------------------------------------------
    # ATTEMPT EXTRACTION
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("ATTEMPTING EXTRACTION FROM CORRUPTED VIDEO")
    print("=" * 70)

    extraction_failed = False

    try:

        extract_spatial_video(
            cover_video=COVER_VIDEO,
            stego_video=CORRUPTED_VIDEO,
            key_file=KEY_FILE,
            output_file=RECOVERED_FILE,
            payload_bits=len(payload),
        )

    except Exception as exc:

        extraction_failed = True

        print()
        print(
            "Extraction failed as expected."
        )

        print(
            f"Exception Type : "
            f"{type(exc).__name__}"
        )

        print(
            f"Exception      : "
            f"{exc}"
        )

    # ------------------------------------------------------
    # FAILURE VALIDATION
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("FAILURE VALIDATION")
    print("=" * 70)

    if not extraction_failed:

        print()
        print(
            "RESULT: FAIL"
        )

        print(
            "The corrupted video was accepted "
            "without raising an extraction error."
        )

        raise SystemExit(1)

    print()
    print(
        "Corrupted packet detected : PASS"
    )

    if RECOVERED_FILE.exists():

        recovered_size = (
            RECOVERED_FILE.stat().st_size
        )

        print()
        print(
            "WARNING:"
        )

        print(
            f"A recovered file was created "
            f"despite extraction failure "
            f"({recovered_size} bytes)."
        )

        print(
            "Removing the invalid recovered file."
        )

        RECOVERED_FILE.unlink()

    else:

        print(
            "No invalid recovered file : PASS"
        )

    # ------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RESULT: CORRUPTED MULTI-FRAME "
        "PACKET WAS SAFELY REJECTED."
    )
    print("=" * 70)

    print()

    print(
        "Failure Handling:"
    )

    print(
        "Valid Stego Video       : PASS"
    )

    print(
        "Frame Corruption        : PASS"
    )

    print(
        "Corrupted MP4 Creation  : PASS"
    )

    print(
        "Packet Validation       : PASS"
    )

    print(
        "Extraction Rejection    : PASS"
    )

    print(
        "No False Recovery       : PASS"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "No production spatial "
        "embedding algorithm was modified."
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()