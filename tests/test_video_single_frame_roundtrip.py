"""
StegaFusion Video Single-Frame Round-Trip Test

Tests whether a payload embedded into one video frame
survives MP4 video encoding and decoding.

Pipeline:

    Secret File
        ↓
    AES Encryption
        ↓
    Payload Packet
        ↓
    Frame 00000
        ↓
    DWT
        ↓
    Adaptive LSB Embedding
        ↓
    Stego Frame
        ↓
    MP4 Encoding
        ↓
    MP4 Decoding
        ↓
    DWT
        ↓
    Adaptive Extraction
        ↓
    Payload
        ↓
    AES Decryption
        ↓
    Original Secret
"""

from pathlib import Path
import shutil

import cv2
import numpy as np

from config.config import PathConfig

from modules.crypto.aes_encrypt import encrypt_file
from modules.crypto.aes_decrypt import decrypt_file

from modules.steganography.payload import (
    create_payload_packet,
    parse_payload_packet,
    binary_to_bytes,
)

from modules.steganography.lsb_embed import (
    load_frame,
    save_frame,
)

from modules.steganography.edge_detector import (
    generate_edge_map,
)

from modules.steganography.adaptive_lsb import (
    adaptive_embed,
    adaptive_extract,
    calculate_capacity,
)

from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)

from modules.video.frame_reconstruct import (
    reconstruct_video,
)


# ==========================================================
# PATHS
# ==========================================================

SECRET_FILE = Path(
    "input/secret_data/test_secret.txt"
)

KEY_FILE = Path(
    "input/keys/test_integration_key.bin"
)

SOURCE_FRAME = Path(
    "temp/frames/frame_00000.png"
)

ROUNDTRIP_DIR = Path(
    "temp/video_roundtrip_frames"
)

STEGO_FRAME = (
    ROUNDTRIP_DIR /
    "frame_00000.png"
)

ROUNDTRIP_VIDEO = (
    PathConfig.OUTPUT_DIR /
    "video_single_frame_roundtrip.mp4"
)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("StegaFusion Video Single-Frame Round-Trip Test")
    print("=" * 70)

    # ------------------------------------------------------
    # Validate required files
    # ------------------------------------------------------

    required_files = [
        SECRET_FILE,
        KEY_FILE,
        SOURCE_FRAME,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # ------------------------------------------------------
    # Encrypt secret
    # ------------------------------------------------------

    encrypted_data = encrypt_file(
        SECRET_FILE,
        KEY_FILE
    )

    # ------------------------------------------------------
    # Create exact payload packet
    # ------------------------------------------------------

    payload_bits = create_payload_packet(
        encrypted_data
    )

    print()
    print(
        f"Encrypted Data : "
        f"{len(encrypted_data)} bytes"
    )

    print(
        f"Payload Bits   : "
        f"{len(payload_bits)} bits"
    )

    # ------------------------------------------------------
    # Load source frame
    # ------------------------------------------------------

    image = load_frame(
        SOURCE_FRAME
    )

    blue_channel = image[:, :, 0]

    print()
    print("Source Frame")

    print(
        f"Shape : "
        f"{blue_channel.shape}"
    )

    print(
        f"Dtype : "
        f"{blue_channel.dtype}"
    )

    print(
        f"Range : "
        f"{blue_channel.min()} -> "
        f"{blue_channel.max()}"
    )

    # ------------------------------------------------------
    # DWT
    # ------------------------------------------------------

    bands = apply_dwt(
        blue_channel
    )

    # ------------------------------------------------------
    # Edge detection
    # ------------------------------------------------------

    edge_map = generate_edge_map(
        bands["LH"]
    )

    edge_pixels = np.count_nonzero(
        edge_map
    )

    capacity = calculate_capacity(
        edge_map
    )

    print()
    print(
        f"Edge Pixels : "
        f"{edge_pixels}"
    )

    print(
        f"Capacity    : "
        f"{capacity} bits"
    )

    if len(payload_bits) > capacity:

        raise RuntimeError(
            "Payload does not fit in frame."
        )

    # ------------------------------------------------------
    # Embed payload
    # ------------------------------------------------------

    stego_lh, embedded_bits = adaptive_embed(
        bands["LH"],
        edge_map,
        payload_bits
    )

    if embedded_bits != len(payload_bits):

        raise RuntimeError(
            f"Embedded only {embedded_bits} "
            f"of {len(payload_bits)} bits."
        )

    # ------------------------------------------------------
    # Verify direct extraction before video encoding
    # ------------------------------------------------------

    direct_bits = adaptive_extract(
        stego_lh,
        edge_map,
        len(payload_bits)
    )

    if direct_bits != payload_bits:

        raise RuntimeError(
            "Direct extraction failed before "
            "video encoding."
        )

    print()
    print(
        "Direct Extraction : PASS"
    )

    # ------------------------------------------------------
    # Replace LH band
    # ------------------------------------------------------

    bands["LH"] = stego_lh

    # ------------------------------------------------------
    # Inverse DWT
    #
    # IMPORTANT:
    # Round before uint8 conversion.
    # ------------------------------------------------------

    reconstructed_blue = apply_inverse_dwt(
        bands
    )

    reconstructed_blue = np.clip(
        np.rint(reconstructed_blue),
        0,
        255
    ).astype(np.uint8)

    # ------------------------------------------------------
    # Create stego frame
    # ------------------------------------------------------

    stego_frame = image.copy()

    stego_frame[:, :, 0] = reconstructed_blue

    # ------------------------------------------------------
    # Prepare round-trip frame directory
    #
    # We need all 197 frames so that the existing video
    # reconstruction module can rebuild a complete video.
    # Only frame_00000 is modified.
    # ------------------------------------------------------

    if ROUNDTRIP_DIR.exists():

        shutil.rmtree(
            ROUNDTRIP_DIR
        )

    ROUNDTRIP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Copy all original extracted frames
    # ------------------------------------------------------

    source_frames = sorted(
        PathConfig.FRAME_DIR.glob("*.png")
    )

    if not source_frames:

        raise FileNotFoundError(
            "No extracted frames found."
        )

    for frame_path in source_frames:

        destination = (
            ROUNDTRIP_DIR /
            frame_path.name
        )

        shutil.copy2(
            frame_path,
            destination
        )

    # ------------------------------------------------------
    # Replace frame 00000 with stego version
    # ------------------------------------------------------

    save_frame(
        stego_frame,
        STEGO_FRAME
    )

    print()
    print(
        f"Frames Prepared : "
        f"{len(source_frames)}"
    )

    print(
        f"Stego Frame     : "
        f"{STEGO_FRAME}"
    )

    # ------------------------------------------------------
    # Reconstruct MP4
    # ------------------------------------------------------

    source_video = (
        PathConfig.COVER_VIDEO_DIR /
        "sample.mp4"
    )

    video = cv2.VideoCapture(
        str(source_video)
    )

    if not video.isOpened():

        raise RuntimeError(
            "Unable to open source video."
        )

    source_fps = video.get(
        cv2.CAP_PROP_FPS
    )

    video.release()

    ROUNDTRIP_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    written_frames = reconstruct_video(
        ROUNDTRIP_DIR,
        ROUNDTRIP_VIDEO,
        fps=source_fps
    )

    print()
    print(
        f"Video Frames Written : "
        f"{written_frames}"
    )

    print(
        f"Round-Trip Video     : "
        f"{ROUNDTRIP_VIDEO}"
    )

    # ------------------------------------------------------
    # Open encoded MP4
    # ------------------------------------------------------

    cap = cv2.VideoCapture(
        str(ROUNDTRIP_VIDEO)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Unable to open reconstructed MP4."
        )

    # ------------------------------------------------------
    # Read first decoded frame
    # ------------------------------------------------------

    success, decoded_frame = cap.read()

    cap.release()

    if not success:

        raise RuntimeError(
            "Unable to decode frame 00000 "
            "from reconstructed MP4."
        )

    print()
    print(
        "Decoded MP4 Frame"
    )

    print(
        f"Shape : "
        f"{decoded_frame.shape}"
    )

    print(
        f"Dtype : "
        f"{decoded_frame.dtype}"
    )

    print(
        f"Range : "
        f"{decoded_frame.min()} -> "
        f"{decoded_frame.max()}"
    )

    # ------------------------------------------------------
    # Compare original stego frame with decoded frame
    # ------------------------------------------------------

    decoded_blue = decoded_frame[:, :, 0]

    blue_difference = (
        decoded_blue.astype(np.int16)
        -
        stego_frame[:, :, 0].astype(np.int16)
    )

    changed_pixels = np.count_nonzero(
        blue_difference
    )

    max_difference = np.max(
        np.abs(blue_difference)
    )

    mean_difference = np.mean(
        np.abs(blue_difference)
    )

    print()
    print(
        "MP4 Blue Channel Distortion"
    )

    print(
        f"Changed Pixels : "
        f"{changed_pixels}"
    )

    print(
        f"Maximum Difference : "
        f"{max_difference}"
    )

    print(
        f"Mean Absolute Difference : "
        f"{mean_difference:.8f}"
    )

    # ------------------------------------------------------
    # DWT extraction from decoded MP4 frame
    # ------------------------------------------------------

    decoded_bands = apply_dwt(
        decoded_blue
    )

    decoded_edge_map = generate_edge_map(
        decoded_bands["LH"]
    )

    recovered_bits = adaptive_extract(
        decoded_bands["LH"],
        decoded_edge_map,
        len(payload_bits)
    )

    print()
    print(
        f"Recovered Bits : "
        f"{len(recovered_bits)}"
    )

    # ------------------------------------------------------
    # Compare payload
    # ------------------------------------------------------

    if recovered_bits != payload_bits:

        mismatch = None

        limit = min(
            len(payload_bits),
            len(recovered_bits)
        )

        for index in range(limit):

            if (
                payload_bits[index]
                != recovered_bits[index]
            ):

                mismatch = index
                break

        if mismatch is None:
            mismatch = limit

        print()
        print(
            "Video Payload Extraction : FAIL"
        )

        print(
            f"First Mismatch : "
            f"{mismatch}"
        )

        print(
            "Expected:",
            payload_bits[
                max(0, mismatch - 16):
                mismatch + 16
            ]
        )

        print(
            "Recovered:",
            recovered_bits[
                max(0, mismatch - 16):
                mismatch + 16
            ]
        )

        raise RuntimeError(
            "Payload did not survive MP4 "
            "encoding/decoding."
        )

    print()
    print(
        "Video Payload Extraction : PASS"
    )

    # ------------------------------------------------------
    # Parse packet
    # ------------------------------------------------------

    payload_length, encrypted_bits = (
        parse_payload_packet(
            recovered_bits
        )
    )

    print()
    print(
        f"Packet Payload Length : "
        f"{payload_length} bits"
    )

    print(
        f"Encrypted Bits        : "
        f"{len(encrypted_bits)} bits"
    )

    # ------------------------------------------------------
    # Convert encrypted bits to bytes
    # ------------------------------------------------------

    recovered_encrypted_data = binary_to_bytes(
        encrypted_bits
    )

    print(
        f"Encrypted Data        : "
        f"{len(recovered_encrypted_data)} bytes"
    )

    # ------------------------------------------------------
    # AES decryption
    # ------------------------------------------------------

    recovered_data = decrypt_file(
        recovered_encrypted_data,
        KEY_FILE
    )

    original_data = (
        SECRET_FILE.read_bytes()
    )

    print(
        f"Recovered Data        : "
        f"{len(recovered_data)} bytes"
    )

    print(
        f"Original Data         : "
        f"{len(original_data)} bytes"
    )

    # ------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------

    if recovered_data != original_data:

        raise RuntimeError(
            "Recovered secret does not match "
            "the original secret."
        )

    print()
    print(
        "AES Decryption : PASS"
    )

    print(
        "Original Data Match : PASS"
    )

    print()
    print("=" * 70)
    print(
        "VIDEO SINGLE-FRAME ROUND-TRIP PASSED!"
    )
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()