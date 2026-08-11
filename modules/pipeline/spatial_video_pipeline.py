"""
StegaFusion Spatial MP4 Video Pipeline

End-to-end spatial-domain steganography pipeline.

Workflow
--------
Embedding:

    Secret File
        ↓
    AES-256 Encryption
        ↓
    Payload Packet
        ↓
    Spatial Paired-Block Embedding
        ↓
    MP4V Video

Extraction:

    Original Cover Video + Stego MP4
        ↓
    Spatial Paired-Block Extraction
        ↓
    Payload Packet
        ↓
    AES-256 Decryption
        ↓
    Recovered Secret File

Important
---------
The current spatial extraction algorithm requires the original
cover frame as a reference. Therefore this first production
pipeline does NOT claim true cover-independent blind extraction.

The validated spatial configuration is:

    Block Size : 32
    Gap        : 16
    Delta      : 4
    Channel    : Blue
    Codec      : MP4V
"""

from pathlib import Path

import cv2

from modules.crypto.aes_decrypt import decrypt_file
from modules.crypto.aes_encrypt import encrypt_file
from modules.steganography.payload import (
    binary_to_bytes,
    bytes_to_binary,
    create_payload_packet,
    parse_payload_packet,
    write_file,
)
from modules.steganography.spatial.paired_block import (
    BLOCK_SIZE,
    DELTA,
    GAP,
    calculate_capacity,
    embed_payload,
    extract_payload,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

FPS_DEFAULT = 24.0


# ==========================================================
# PAYLOAD PREPARATION
# ==========================================================

def prepare_spatial_payload(
    secret_file: Path,
    key_file: Path,
) -> str:
    """
    Encrypt the secret file and create the binary payload packet.

    Returns
    -------
    str
        Complete binary payload packet.
    """

    encrypted_data = encrypt_file(
        Path(secret_file),
        Path(key_file),
    )

    return create_payload_packet(
        encrypted_data
    )


# ==========================================================
# VIDEO INFORMATION
# ==========================================================

def get_video_info(
    video_path: Path,
):
    """
    Read basic video information.

    Returns
    -------
    tuple
        (fps, width, height, frame_count)
    """

    video_path = Path(video_path)

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Unable to open video: {video_path}"
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
        fps = FPS_DEFAULT

    return (
        fps,
        width,
        height,
        frame_count,
    )


# ==========================================================
# FRAME CAPACITY
# ==========================================================

def get_spatial_capacity(
    frame,
    block_size: int = BLOCK_SIZE,
    gap: int = GAP,
) -> int:
    """
    Return spatial payload capacity in bits.
    """

    return calculate_capacity(
        frame,
        block_size=block_size,
        gap=gap,
    )


# ==========================================================
# EMBED ONE FRAME
# ==========================================================

def embed_spatial_frame(
    frame,
    payload: str,
    block_size: int = BLOCK_SIZE,
    delta: int = DELTA,
    gap: int = GAP,
):
    """
    Embed a complete payload into one video frame.

    Returns
    -------
    tuple
        (stego_frame, embedded_bits)
    """

    capacity = get_spatial_capacity(
        frame,
        block_size=block_size,
        gap=gap,
    )

    if len(payload) > capacity:
        raise ValueError(
            "Spatial payload exceeds frame capacity.\n"
            f"Payload : {len(payload)} bits\n"
            f"Capacity: {capacity} bits"
        )

    return embed_payload(
        frame,
        payload,
        block_size=block_size,
        delta=delta,
        gap=gap,
    )


# ==========================================================
# WRITE SPATIAL MP4
# ==========================================================

def embed_spatial_video(
    cover_video: Path,
    secret_file: Path,
    key_file: Path,
    output_video: Path,
    block_size: int = BLOCK_SIZE,
    delta: int = DELTA,
    gap: int = GAP,
):
    """
    Create an MP4V stego video.

    The complete payload is embedded into the first frame.
    All remaining frames are copied unchanged.

    Returns
    -------
    dict
        Embedding information.
    """

    cover_video = Path(
        cover_video
    )

    secret_file = Path(
        secret_file
    )

    key_file = Path(
        key_file
    )

    output_video = Path(
        output_video
    )

    if not cover_video.exists():
        raise FileNotFoundError(
            f"Cover video not found: {cover_video}"
        )

    if not secret_file.exists():
        raise FileNotFoundError(
            f"Secret file not found: {secret_file}"
        )

    if not key_file.exists():
        raise FileNotFoundError(
            f"AES key not found: {key_file}"
        )

    payload = prepare_spatial_payload(
        secret_file,
        key_file,
    )

    payload_bits = len(payload)

    cap = cv2.VideoCapture(
        str(cover_video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open cover video: "
            f"{cover_video}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = FPS_DEFAULT

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

    output_video.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Unable to create output video: "
            f"{output_video}"
        )

    embedded_bits = 0

    frame_index = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            if frame_index == 0:

                capacity = get_spatial_capacity(
                    frame,
                    block_size=block_size,
                    gap=gap,
                )

                if payload_bits > capacity:
                    raise ValueError(
                        "Payload is too large for "
                        "the first video frame.\n"
                        f"Payload : {payload_bits} bits\n"
                        f"Capacity: {capacity} bits"
                    )

                stego_frame, embedded_bits = (
                    embed_spatial_frame(
                        frame,
                        payload,
                        block_size=block_size,
                        delta=delta,
                        gap=gap,
                    )
                )

                writer.write(
                    stego_frame
                )

            else:

                writer.write(
                    frame
                )

            frame_index += 1

    finally:

        cap.release()
        writer.release()

    return {
        "payload_bits": payload_bits,
        "embedded_bits": embedded_bits,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frame_count,
        "output_video": output_video,
    }


# ==========================================================
# DECODE FIRST FRAME
# ==========================================================

def _read_first_frame(
    video_path: Path,
):
    """
    Read the first frame from a video.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: "
            f"{video_path}"
        )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            f"Unable to decode first frame: "
            f"{video_path}"
        )

    return frame


# ==========================================================
# EXTRACT SPATIAL PAYLOAD
# ==========================================================

def extract_spatial_payload(
    original_frame,
    received_frame,
    bit_count: int,
    block_size: int = BLOCK_SIZE,
    gap: int = GAP,
):
    """
    Extract a binary payload from a received frame.

    The original cover frame is required because the
    spatial paired-block extractor measures changes
    relative to the original frame.
    """

    return extract_payload(
        original_frame,
        received_frame,
        bit_count,
        block_size=block_size,
        gap=gap,
    )


# ==========================================================
# RECOVER SECRET FILE
# ==========================================================

def recover_secret_file(
    original_frame,
    received_frame,
    key_file: Path,
    bit_count: int,
    output_file: Path,
    block_size: int = BLOCK_SIZE,
    gap: int = GAP,
):
    """
    Extract, parse and decrypt the embedded payload.

    Returns
    -------
    bytes
        Recovered plaintext data.
    """

    packet = extract_spatial_payload(
        original_frame,
        received_frame,
        bit_count,
        block_size=block_size,
        gap=gap,
    )

    payload_length, encrypted_bits = (
        parse_payload_packet(
            packet
        )
    )

    if payload_length % 8 != 0:
        raise ValueError(
            "Recovered payload length is "
            "not byte-aligned."
        )

    encrypted_data = binary_to_bytes(
        encrypted_bits
    )

    recovered_data = decrypt_file(
        encrypted_data,
        Path(key_file),
    )

    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_file(
        output_file,
        recovered_data,
    )

    return recovered_data


# ==========================================================
# EXTRACT SPATIAL VIDEO
# ==========================================================

def extract_spatial_video(
    cover_video: Path,
    stego_video: Path,
    key_file: Path,
    output_file: Path,
    payload_bits: int,
    block_size: int = BLOCK_SIZE,
    gap: int = GAP,
):
    """
    Recover the secret file from an MP4V stego video.

    The original cover video is required as the reference.
    """

    cover_video = Path(
        cover_video
    )

    stego_video = Path(
        stego_video
    )

    key_file = Path(
        key_file
    )

    output_file = Path(
        output_file
    )

    original_frame = _read_first_frame(
        cover_video
    )

    received_frame = _read_first_frame(
        stego_video
    )

    recovered_data = recover_secret_file(
        original_frame,
        received_frame,
        key_file,
        payload_bits,
        output_file,
        block_size=block_size,
        gap=gap,
    )

    return {
        "payload_bits": payload_bits,
        "recovered_bytes": len(
            recovered_data
        ),
        "output_file": output_file,
    }


# ==========================================================
# MODULE TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Spatial MP4 Video Pipeline")
    print("=" * 70)

    print()
    print("Module loaded successfully.")
    print()
    print("Configuration:")
    print(
        f"Block Size : {BLOCK_SIZE}"
    )
    print(
        f"Gap        : {GAP}"
    )
    print(
        f"Delta      : {DELTA}"
    )
    print()
    print(
        "Spatial MP4 pipeline ready."
    )