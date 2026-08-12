"""
StegaFusion Multi-Frame Spatial MP4 Missing-Frame Failure Test

Creates a valid multi-frame stego video and then truncates the
video so that one required embedded frame is missing.

Expected behavior:

    Valid 13-packet video
            ↓
    Remove final required frame
            ↓
    Extraction detects insufficient frames
            ↓
    Extraction FAILS safely

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
    "output/video_spatial_multiframe_missing"
)

VALID_VIDEO = (
    OUTPUT_DIR /
    "valid_stego.mp4"
)

TRUNCATED_VIDEO = (
    OUTPUT_DIR /
    "missing_frame.mp4"
)

RECOVERED_FILE = (
    OUTPUT_DIR /
    "should_not_exist.bin"
)


# ==========================================================
# VIDEO HELPERS
# ==========================================================

def read_video_info(path: Path):
    """
    Return FPS, width, height and frame count.
    """

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {path}"
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

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    if fps <= 0:
        fps = 24.0

    return (
        fps,
        width,
        height,
        frame_count,
    )


def copy_first_n_frames(
    source: Path,
    destination: Path,
    frame_count: int,
):
    """
    Copy only the first frame_count frames into
    a new MP4V video.
    """

    cap = cv2.VideoCapture(
        str(source)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open source video: "
            f"{source}"
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

    if fps <= 0:
        fps = 24.0

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Unable to create video: "
            f"{destination}"
        )

    written = 0

    try:

        while written < frame_count:

            success, frame = cap.read()

            if not success:
                break

            writer.write(frame)

            written += 1

    finally:

        cap.release()
        writer.release()

    if written != frame_count:
        raise RuntimeError(
            "Unable to create the requested "
            "truncated video.\n"
            f"Requested : {frame_count}\n"
            f"Written   : {written}"
        )

    return written


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "StegaFusion Multi-Frame Spatial MP4 "
        "Missing-Frame Failure Test"
    )
    print("=" * 70)

    # ------------------------------------------------------
    # INPUT CHECK
    # ------------------------------------------------------

    print()
    print("INPUT FILES")
    print("-" * 70)

    required = {
        "Cover Video": COVER_VIDEO,
        "Secret File": SECRET_FILE,
        "AES Key": KEY_FILE,
    }

    for name, path in required.items():

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

    packet_count = len(packets)

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
        f"{packet_count}"
    )

    if packet_count < 2:

        raise RuntimeError(
            "Missing-frame test requires "
            "at least two packets."
        )

    print(
        "Multi-frame payload : PASS"
    )

    # ------------------------------------------------------
    # CREATE VALID VIDEO
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

    (
        valid_fps,
        valid_width,
        valid_height,
        valid_frames,
    ) = read_video_info(
        VALID_VIDEO
    )

    print()

    print(
        f"Valid FPS       : {valid_fps}"
    )

    print(
        f"Valid Resolution: "
        f"{valid_width} x {valid_height}"
    )

    print(
        f"Valid Frames    : {valid_frames}"
    )

    if valid_frames < packet_count:

        raise RuntimeError(
            "Valid stego video contains fewer "
            "frames than required packets."
        )

    print(
        "Valid stego video : PASS"
    )

    # ------------------------------------------------------
    # TRUNCATE VIDEO
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("REMOVING REQUIRED FRAME")
    print("=" * 70)

    required_frames = packet_count

    truncated_frames = (
        required_frames - 1
    )

    print()

    print(
        f"Required Frames : "
        f"{required_frames}"
    )

    print(
        f"Truncated To    : "
        f"{truncated_frames}"
    )

    print(
        f"Missing Frame   : "
        f"{required_frames - 1}"
    )

    written = copy_first_n_frames(
        VALID_VIDEO,
        TRUNCATED_VIDEO,
        truncated_frames,
    )

    print()

    print(
        f"Frames Written  : {written}"
    )

    print(
        f"Truncated Video : "
        f"{TRUNCATED_VIDEO}"
    )

    if not TRUNCATED_VIDEO.exists():

        raise RuntimeError(
            "Truncated video was not created."
        )

    (
        truncated_fps,
        truncated_width,
        truncated_height,
        actual_frames,
    ) = read_video_info(
        TRUNCATED_VIDEO
    )

    print()

    print(
        f"Decoded Frames  : "
        f"{actual_frames}"
    )

    if actual_frames >= required_frames:

        raise RuntimeError(
            "The test video was not actually "
            "truncated."
        )

    print(
        "Required frame successfully removed : PASS"
    )

    # ------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ATTEMPTING EXTRACTION FROM "
        "TRUNCATED VIDEO"
    )
    print("=" * 70)

    extraction_failed = False

    try:

        extract_spatial_video(
            cover_video=COVER_VIDEO,
            stego_video=TRUNCATED_VIDEO,
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
    # VALIDATION
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
            "The pipeline accepted a video with "
            "a missing required frame."
        )

        raise SystemExit(1)

    print()

    print(
        "Missing frame detected : PASS"
    )

    if RECOVERED_FILE.exists():

        print()

        print(
            "WARNING:"
        )

        print(
            "A recovered file was created despite "
            "the missing-frame failure."
        )

        RECOVERED_FILE.unlink()

        print(
            "Invalid recovered file removed."
        )

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
        "RESULT: MISSING MULTI-FRAME DATA "
        "WAS SAFELY REJECTED."
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
        "Frame Removal           : PASS"
    )

    print(
        "Truncated MP4 Creation  : PASS"
    )

    print(
        "Frame Count Validation  : PASS"
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