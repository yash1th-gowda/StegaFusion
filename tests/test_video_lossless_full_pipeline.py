"""
StegaFusion Full Lossless Video Pipeline Test

Tests the complete pipeline:

    Cover Video
        ↓
    Frame Extraction
        ↓
    AES-256 Encryption
        ↓
    Payload Packet
        ↓
    DWT + Adaptive QIM
        ↓
    Stego Frames
        ↓
    FFV1 Lossless Video
        ↓
    Decoded Stego Frames
        ↓
    DWT Extraction
        ↓
    Payload Packet
        ↓
    AES-256 Decryption
        ↓
    Original Secret

This test intentionally uses FFV1 because the current mp4v
pipeline is lossy and has already been shown to corrupt the
embedded DWT coefficients.
"""

from pathlib import Path

import cv2
import numpy as np

from config.config import PathConfig

from modules.video.frame_extract import extract_frames
from modules.steganography.lsb_embed import load_frame, save_frame
from modules.steganography.edge_detector import generate_edge_map
from modules.steganography.adaptive_lsb import (
    adaptive_embed,
    adaptive_extract,
    calculate_capacity,
)
from modules.steganography.payload import (
    create_payload_packet,
    parse_payload_packet,
    binary_to_bytes,
)
from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)
from modules.crypto.aes_encrypt import encrypt_file
from modules.crypto.aes_decrypt import decrypt_file


# ==========================================================
# PATHS
# ==========================================================

SECRET_FILE = Path(
    "input/secret_data/test_secret.txt"
)

KEY_FILE = Path(
    "input/keys/test_integration_key.bin"
)

COVER_VIDEO = (
    PathConfig.COVER_VIDEO_DIR /
    "sample.mp4"
)

SOURCE_FRAME_DIR = PathConfig.FRAME_DIR

STEGO_FRAME_DIR = (
    PathConfig.TEMP_DIR /
    "video_lossless_full" /
    "stego_frames"
)

DECODED_FRAME_DIR = (
    PathConfig.TEMP_DIR /
    "video_lossless_full" /
    "decoded_frames"
)

OUTPUT_VIDEO = (
    PathConfig.OUTPUT_DIR /
    "video_lossless_full_pipeline.avi"
)


# ==========================================================
# HELPERS
# ==========================================================

def write_ffv1_video(
    frame_folder: Path,
    output_video: Path,
    fps: float,
) -> int:
    """
    Write PNG frames into an FFV1 lossless AVI video.
    """

    frames = sorted(
        frame_folder.glob("*.png")
    )

    if not frames:
        raise FileNotFoundError(
            f"No frames found in {frame_folder}"
        )

    first = cv2.imread(
        str(frames[0]),
        cv2.IMREAD_COLOR
    )

    if first is None:
        raise RuntimeError(
            f"Unable to read first frame: {frames[0]}"
        )

    height, width = first.shape[:2]

    output_video.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"FFV1"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Unable to open FFV1 video writer."
        )

    count = 0

    for frame_path in frames:

        frame = cv2.imread(
            str(frame_path),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            raise RuntimeError(
                f"Unable to read frame: {frame_path}"
            )

        writer.write(frame)

        count += 1

    writer.release()

    return count


def decode_video(
    video_path: Path,
    output_folder: Path,
) -> int:
    """
    Decode every video frame and save it as PNG.
    """

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_path = (
            output_folder /
            f"frame_{count:05d}.png"
        )

        cv2.imwrite(
            str(frame_path),
            frame
        )

        count += 1

    cap.release()

    return count


def first_mismatch(
    expected: str,
    recovered: str,
):
    """
    Return the first bit mismatch.
    """

    limit = min(
        len(expected),
        len(recovered)
    )

    for index in range(limit):

        if expected[index] != recovered[index]:
            return index

    if len(expected) != len(recovered):
        return limit

    return None


# ==========================================================
# MAIN
# ==========================================================

def main():

    import shutil

    # Clean previous test artifacts
    if SOURCE_FRAME_DIR.exists():
        shutil.rmtree(SOURCE_FRAME_DIR)

    if STEGO_FRAME_DIR.exists():
        shutil.rmtree(STEGO_FRAME_DIR)

    if DECODED_FRAME_DIR.exists():
        shutil.rmtree(DECODED_FRAME_DIR)

    if OUTPUT_VIDEO.exists():
        OUTPUT_VIDEO.unlink()

    SOURCE_FRAME_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    STEGO_FRAME_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    DECODED_FRAME_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print(
        "StegaFusion Full Lossless Video Pipeline Test"
    )
    print("=" * 70)

    # ------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------

    if not COVER_VIDEO.exists():
        raise FileNotFoundError(
            f"Cover video not found: {COVER_VIDEO}"
        )

    if not SECRET_FILE.exists():
        raise FileNotFoundError(
            f"Secret file not found: {SECRET_FILE}"
        )

    if not KEY_FILE.exists():
        raise FileNotFoundError(
            f"AES key not found: {KEY_FILE}"
        )

    # ------------------------------------------------------
    # Extract source video frames
    # ------------------------------------------------------

    print()
    print("Extracting source video frames...")

    SOURCE_FRAME_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    source_count, fps, resolution = extract_frames(
        COVER_VIDEO
    )

    print()
    print(
        f"Source Video       : {COVER_VIDEO}"
    )

    print(
        f"Source Frames      : {source_count}"
    )

    print(
        f"FPS                : {fps}"
    )

    print(
        f"Resolution         : "
        f"{resolution[0]} x {resolution[1]}"
    )

    if source_count == 0:
        raise RuntimeError(
            "No source frames were extracted."
        )

    # ------------------------------------------------------
    # Encrypt secret
    # ------------------------------------------------------

    encrypted_data = encrypt_file(
        SECRET_FILE,
        KEY_FILE
    )

    payload_bits = create_payload_packet(
        encrypted_data
    )

    print()
    print(
        f"Encrypted Data     : "
        f"{len(encrypted_data)} bytes"
    )

    print(
        f"Payload            : "
        f"{len(payload_bits)} bits"
    )

    # ------------------------------------------------------
    # Prepare stego frame directory
    # ------------------------------------------------------

    STEGO_FRAME_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Copy/process all frames
    #
    # Only frame 00000 carries the payload.
    # Remaining frames are copied unchanged.
    # ------------------------------------------------------

    print()
    print("Preparing stego frames...")

    source_frames = sorted(
        SOURCE_FRAME_DIR.glob("*.png")
    )

    if len(source_frames) != source_count:
        raise RuntimeError(
            "Extracted frame count does not match "
            "the source frame list."
        )

    embedded_bits = 0

    original_blue = None
    embedded_blue = None

    for index, source_path in enumerate(
        source_frames
    ):

        image = load_frame(
            source_path
        )

        if index != 0:

            output_path = (
                STEGO_FRAME_DIR /
                source_path.name
            )

            save_frame(
                image,
                output_path
            )

            continue

        # --------------------------------------------------
        # Embed payload into first frame
        # --------------------------------------------------

        blue_channel = image[:, :, 0]

        original_blue = blue_channel.copy()

        bands = apply_dwt(
            blue_channel
        )

        edge_map = generate_edge_map(
            bands["LH"]
        )

        capacity = calculate_capacity(
            edge_map
        )

        print()
        print(
            "Embedding Frame"
        )

        print(
            f"Shape              : "
            f"{blue_channel.shape}"
        )

        print(
            f"Blue Range         : "
            f"{blue_channel.min()} -> "
            f"{blue_channel.max()}"
        )

        print(
            f"Edge Pixels        : "
            f"{np.count_nonzero(edge_map)}"
        )

        print(
            f"Capacity           : "
            f"{capacity} bits"
        )

        if len(payload_bits) > capacity:
            raise RuntimeError(
                "Payload does not fit in frame."
            )

        stego_lh, embedded_bits = adaptive_embed(
            bands["LH"],
            edge_map,
            payload_bits
        )

        if embedded_bits != len(payload_bits):
            raise RuntimeError(
                f"Only {embedded_bits} of "
                f"{len(payload_bits)} bits embedded."
            )

        bands["LH"] = stego_lh

                # ------------------------------------------------------
        # Diagnostic 1: extraction directly from embedded LH
        # ------------------------------------------------------

        direct_lh = adaptive_extract(
            stego_lh,
            edge_map,
            len(payload_bits)
        )

        lh_mismatch = first_mismatch(
            payload_bits,
            direct_lh
        )

        print()
        print(
            "Extraction from Embedded LH : "
            f"{'PASS' if lh_mismatch is None else 'FAIL'}"
        )

        if lh_mismatch is not None:
            print(
                f"LH First Mismatch          : "
                f"{lh_mismatch}"
            )

            raise RuntimeError(
                "Payload is already corrupted immediately "
                "after adaptive_embed()."
            )

        reconstructed_blue = apply_inverse_dwt(
            bands
        )

        reconstructed_blue = np.round(
            reconstructed_blue
        ).clip(
            0,
            255
        ).astype(np.uint8)

        embedded_blue = (
            reconstructed_blue.copy()
        )

                # ------------------------------------------------------
        # Diagnostic 2: DWT → inverse DWT → DWT
        # ------------------------------------------------------

        roundtrip_bands = apply_dwt(
            reconstructed_blue
        )

        roundtrip_edge_map = generate_edge_map(
            roundtrip_bands["LH"]
        )

        roundtrip_recovered = adaptive_extract(
            roundtrip_bands["LH"],
            roundtrip_edge_map,
            len(payload_bits)
        )

        roundtrip_mismatch = first_mismatch(
            payload_bits,
            roundtrip_recovered
        )

        print()
        print(
            "Extraction After DWT Round-Trip : "
            f"{'PASS' if roundtrip_mismatch is None else 'FAIL'}"
        )

        if roundtrip_mismatch is not None:
            print(
                f"DWT First Mismatch             : "
                f"{roundtrip_mismatch}"
            )


        stego_image = image.copy()

        stego_image[:, :, 0] = (
            reconstructed_blue
        )

        output_path = (
            STEGO_FRAME_DIR /
            source_path.name
        )

        save_frame(
            stego_image,
            output_path
        )

    print()
    print(
        f"Embedded Bits      : "
        f"{embedded_bits}"
    )

    print(
        f"Stego Frames       : "
        f"{len(source_frames)}"
    )

    # ------------------------------------------------------
    # Direct extraction before video encoding
    # ------------------------------------------------------

    stego_first = load_frame(
        STEGO_FRAME_DIR /
        "frame_00000.png"
    )

    stego_blue = stego_first[:, :, 0]

    stego_bands = apply_dwt(
        stego_blue
    )

    stego_edge_map = generate_edge_map(
        stego_bands["LH"]
    )

    direct_recovered = adaptive_extract(
        stego_bands["LH"],
        stego_edge_map,
        len(payload_bits)
    )

    mismatch = first_mismatch(
        payload_bits,
        direct_recovered
    )

    print()
    print(
        f"Direct Extraction   : "
        f"{'PASS' if mismatch is None else 'FAIL'}"
    )

    if mismatch is not None:
        print(
            f"First Mismatch      : {mismatch}"
        )

        raise RuntimeError(
            "Payload failed before video encoding."
        )

    # ------------------------------------------------------
    # Spatial distortion
    # ------------------------------------------------------

    spatial_difference = (
        np.abs(
            embedded_blue.astype(np.int16)
            -
            original_blue.astype(np.int16)
        )
    )

    print()
    print(
        "Before Video Encoding"
    )

    print(
        f"Changed Pixels      : "
        f"{np.count_nonzero(spatial_difference)}"
    )

    print(
        f"Maximum Difference  : "
        f"{spatial_difference.max()}"
    )

    print(
        f"Mean Difference     : "
        f"{spatial_difference.mean():.10f}"
    )

    # ------------------------------------------------------
    # Write FFV1 video
    # ------------------------------------------------------

    print()
    print(
        "Writing FFV1 lossless video..."
    )

    written_frames = write_ffv1_video(
        STEGO_FRAME_DIR,
        OUTPUT_VIDEO,
        fps
    )

    print()
    print(
        f"Frames Written      : "
        f"{written_frames}"
    )

    print(
        f"Codec               : FFV1"
    )

    print(
        f"Output Video        : "
        f"{OUTPUT_VIDEO}"
    )

    # ------------------------------------------------------
    # Decode video
    # ------------------------------------------------------

    decoded_count = decode_video(
        OUTPUT_VIDEO,
        DECODED_FRAME_DIR
    )

    print()
    print(
        f"Decoded Frames      : "
        f"{decoded_count}"
    )

    if decoded_count != source_count:
        raise RuntimeError(
            "Decoded frame count does not match "
            "source frame count."
        )

    # ------------------------------------------------------
    # Compare embedded first frame against decoded frame
    # ------------------------------------------------------

    decoded_first = load_frame(
        DECODED_FRAME_DIR /
        "frame_00000.png"
    )

    decoded_blue = decoded_first[:, :, 0]

    video_difference = (
        np.abs(
            decoded_blue.astype(np.int16)
            -
            embedded_blue.astype(np.int16)
        )
    )

    print()
    print(
        "Lossless Video Distortion"
    )

    print(
        f"Changed Pixels      : "
        f"{np.count_nonzero(video_difference)}"
    )

    print(
        f"Maximum Difference  : "
        f"{video_difference.max()}"
    )

    print(
        f"Mean Difference     : "
        f"{video_difference.mean():.10f}"
    )

    if np.any(video_difference != 0):

        raise RuntimeError(
            "FFV1 video introduced unexpected "
            "pixel distortion."
        )

    # ------------------------------------------------------
    # Extract payload after FFV1
    # ------------------------------------------------------

    decoded_bands = apply_dwt(
        decoded_blue
    )

    decoded_edge_map = generate_edge_map(
        decoded_bands["LH"]
    )

    recovered_packet = adaptive_extract(
        decoded_bands["LH"],
        decoded_edge_map,
        len(payload_bits)
    )

    mismatch = first_mismatch(
        payload_bits,
        recovered_packet
    )

    print()
    print(
        f"Recovered Bits      : "
        f"{len(recovered_packet)}"
    )

    print(
        f"First Mismatch      : "
        f"{mismatch}"
    )

    if mismatch is not None:

        raise RuntimeError(
            "Payload did not survive FFV1 "
            "video round-trip."
        )

    print(
        "Packet Extraction   : PASS"
    )

    # ------------------------------------------------------
    # Parse payload
    # ------------------------------------------------------

    payload_length, encrypted_bits = (
        parse_payload_packet(
            recovered_packet
        )
    )

    encrypted_data_recovered = (
        binary_to_bytes(
            encrypted_bits
        )
    )

    print()
    print(
        f"Packet Payload      : "
        f"{payload_length} bits"
    )

    print(
        f"Encrypted Data      : "
        f"{len(encrypted_data_recovered)} bytes"
    )

    # ------------------------------------------------------
    # AES decryption
    # ------------------------------------------------------

    recovered_data = decrypt_file(
        encrypted_data_recovered,
        KEY_FILE
    )

    original_data = (
        SECRET_FILE.read_bytes()
    )

    print()
    print(
        f"Recovered Data      : "
        f"{len(recovered_data)} bytes"
    )

    print(
        f"Original Data       : "
        f"{len(original_data)} bytes"
    )

    if recovered_data != original_data:

        raise RuntimeError(
            "Recovered secret does not match "
            "the original secret."
        )

    print(
        "AES Decryption      : PASS"
    )

    print(
        "Original Match      : PASS"
    )

    # ------------------------------------------------------
    # Final result
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FULL LOSSLESS VIDEO PIPELINE PASSED!"
    )
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
