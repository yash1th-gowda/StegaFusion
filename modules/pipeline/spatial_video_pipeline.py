"""
StegaFusion Spatial Multi-Frame MP4 Video Pipeline

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
    Multi-Frame Payload Chunking
        ↓
    Frame Packet Creation
        ↓
    Spatial Paired-Block Embedding
        ↓
    MP4V Video

Extraction:

    Original Cover Video + Stego MP4
        ↓
    Spatial Paired-Block Extraction
        ↓
    Frame Packet Parsing
        ↓
    Chunk Reassembly
        ↓
    Payload Packet Parsing
        ↓
    AES-256 Decryption
        ↓
    Recovered Secret File

Important
---------
The current spatial extraction algorithm requires the original
cover video as a reference.

This pipeline therefore does NOT claim true cover-independent
blind extraction.

Validated spatial configuration:

    Block Size : 32
    Gap        : 16
    Delta      : 4
    Channel    : Blue
    Codec      : MP4V

Multi-frame configuration:

    Maximum Frame Packet : 768 bits
    Header               : 64 bits
    Maximum Chunk Data   : 704 bits
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

from modules.steganography.multiframe_payload import (
    split_payload,
    combine_chunks,
)

from modules.steganography.multiframe_packet import (
    MAX_FRAME_BITS,
    MAX_CHUNK_DATA_BITS,
    create_packet,
    parse_packet,
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
    Encrypt the secret file and create the normal
    StegaFusion payload packet.

    Returns
    -------
    str
        Complete binary payload packet.
    """

    secret_file = Path(secret_file)
    key_file = Path(key_file)

    encrypted_data = encrypt_file(
        secret_file,
        key_file,
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
    Embed a binary packet into one video frame.

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
# CREATE FRAME PACKETS
# ==========================================================

def create_frame_packets(
    payload: str,
):
    """
    Split the complete payload into multi-frame chunks
    and create one packet for each chunk.

    Returns
    -------
    list[str]
        Frame packets.
    """

    chunks = split_payload(
        payload,
        MAX_CHUNK_DATA_BITS,
    )

    if not chunks:
        raise ValueError(
            "Payload cannot be empty."
        )

    total_chunks = len(chunks)

    packets = []

    for index, chunk in enumerate(chunks):

        packet = create_packet(
            chunk_data=chunk,
            total_chunks=total_chunks,
            chunk_index=index,
        )

        packets.append(packet)

    return packets


# ==========================================================
# PARSE FRAME PACKETS
# ==========================================================

def parse_frame_packets(
    packets,
):
    """
    Parse frame packets and reconstruct the original
    binary payload.

    Packets may arrive in any order because each packet
    contains its chunk index.

    Returns
    -------
    str
        Reconstructed binary payload.
    """

    if not packets:
        raise ValueError(
            "No frame packets supplied."
        )

    parsed_packets = [
        parse_packet(packet)
        for packet in packets
    ]

    total_chunks = parsed_packets[0].total_chunks

    if total_chunks != len(parsed_packets):
        raise ValueError(
            "Incorrect number of frame packets.\n"
            f"Expected : {total_chunks}\n"
            f"Received : {len(parsed_packets)}"
        )

    chunks = [None] * total_chunks

    for parsed in parsed_packets:

        if parsed.total_chunks != total_chunks:
            raise ValueError(
                "Frame packet total_chunks mismatch."
            )

        if chunks[parsed.chunk_index] is not None:
            raise ValueError(
                f"Duplicate chunk index: "
                f"{parsed.chunk_index}"
            )

        chunks[
            parsed.chunk_index
        ] = parsed.chunk_data

    if any(
        chunk is None
        for chunk in chunks
    ):
        raise ValueError(
            "Missing frame packet."
        )

    return combine_chunks(
        chunks
    )


# ==========================================================
# WRITE MULTI-FRAME SPATIAL MP4
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
    Create a multi-frame MP4V stego video.

    The encrypted payload is split into frame packets.
    One packet is embedded into each consecutive frame.

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
            f"Cover video not found: "
            f"{cover_video}"
        )

    if not secret_file.exists():
        raise FileNotFoundError(
            f"Secret file not found: "
            f"{secret_file}"
        )

    if not key_file.exists():
        raise FileNotFoundError(
            f"AES key not found: "
            f"{key_file}"
        )

    # ------------------------------------------------------
    # PREPARE REAL ENCRYPTED PAYLOAD
    # ------------------------------------------------------

    payload = prepare_spatial_payload(
        secret_file,
        key_file,
    )

    payload_bits = len(payload)

    # ------------------------------------------------------
    # CREATE FRAME PACKETS
    # ------------------------------------------------------

    packets = create_frame_packets(
        payload
    )

    packet_count = len(packets)

    # ------------------------------------------------------
    # OPEN COVER VIDEO
    # ------------------------------------------------------

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

    if packet_count > frame_count:
        cap.release()

        raise ValueError(
            "Video does not contain enough frames "
            "for the complete payload.\n"
            f"Required frames : {packet_count}\n"
            f"Available frames: {frame_count}"
        )

    # ------------------------------------------------------
    # OUTPUT VIDEO
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # EMBEDDING
    # ------------------------------------------------------

    embedded_bits = 0
    embedded_frames = 0
    frame_index = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            if frame_index < packet_count:

                packet = packets[
                    frame_index
                ]

                capacity = get_spatial_capacity(
                    frame,
                    block_size=block_size,
                    gap=gap,
                )

                if len(packet) > capacity:

                    raise ValueError(
                        "Frame packet exceeds "
                        "spatial capacity.\n"
                        f"Frame    : {frame_index}\n"
                        f"Packet   : {len(packet)} bits\n"
                        f"Capacity : {capacity} bits"
                    )

                stego_frame, embedded = (
                    embed_spatial_frame(
                        frame,
                        packet,
                        block_size=block_size,
                        delta=delta,
                        gap=gap,
                    )
                )

                writer.write(
                    stego_frame
                )

                embedded_bits += embedded
                embedded_frames += 1

            else:

                writer.write(
                    frame
                )

            frame_index += 1

    finally:

        cap.release()
        writer.release()

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    expected_bits = sum(
        len(packet)
        for packet in packets
    )

    if embedded_bits != expected_bits:

        raise RuntimeError(
            "Not all frame packet bits were embedded.\n"
            f"Expected: {expected_bits}\n"
            f"Embedded: {embedded_bits}"
        )

    return {
        "payload_bits": payload_bits,
        "embedded_bits": embedded_bits,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frame_count,
        "packet_count": packet_count,
        "packet_sizes": [
            len(packet)
            for packet in packets
        ],
        "embedded_frames": embedded_frames,
        "output_video": output_video,
    }


# ==========================================================
# EXTRACT ALL FRAMES
# ==========================================================

def _read_video_frames(
    video_path: Path,
):
    """
    Read every frame from a video.
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

            frames.append(
                frame
            )

    finally:

        cap.release()

    if not frames:
        raise RuntimeError(
            f"No frames decoded from: "
            f"{video_path}"
        )

    return frames


# ==========================================================
# EXTRACT ONE SPATIAL PACKET
# ==========================================================

def extract_spatial_payload(
    original_frame,
    received_frame,
    bit_count: int,
    block_size: int = BLOCK_SIZE,
    gap: int = GAP,
):
    """
    Extract a binary payload from one received frame.

    The original cover frame is required because the
    paired-block extractor measures spatial changes
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
# EXTRACT FRAME PACKET
# ==========================================================

def extract_frame_packet(
    original_frame,
    received_frame,
    block_size: int = BLOCK_SIZE,
    gap: int = GAP,
):
    """
    Extract the maximum safe frame packet.

    The 64-bit header tells us the actual chunk length.
    Therefore extracting MAX_FRAME_BITS is safe even for
    the final shorter packet.
    """

    capacity = get_spatial_capacity(
        original_frame,
        block_size=block_size,
        gap=gap,
    )

    if capacity < MAX_FRAME_BITS:

        raise ValueError(
            "Frame capacity is too small for "
            "the configured frame packet.\n"
            f"Capacity: {capacity}\n"
            f"Required: {MAX_FRAME_BITS}"
        )

    return extract_spatial_payload(
        original_frame,
        received_frame,
        MAX_FRAME_BITS,
        block_size=block_size,
        gap=gap,
    )


# ==========================================================
# RECOVER SECRET FILE
# ==========================================================

def recover_secret_file(
    original_frames,
    received_frames,
    key_file: Path,
    output_file: Path,
    block_size: int = BLOCK_SIZE,
    gap: int = GAP,
):
    """
    Extract all multi-frame packets, reconstruct the
    payload packet and decrypt the original secret file.

    Returns
    -------
    bytes
        Recovered plaintext data.
    """

    if len(
        original_frames
    ) != len(
        received_frames
    ):
        raise ValueError(
            "Original and received videos "
            "have different frame counts.\n"
            f"Original : {len(original_frames)}\n"
            f"Received : {len(received_frames)}"
        )

    # ------------------------------------------------------
    # FIRST PACKET
    # ------------------------------------------------------

    first_packet_bits = extract_frame_packet(
        original_frames[0],
        received_frames[0],
        block_size=block_size,
        gap=gap,
    )

    first_parsed = parse_packet(
        first_packet_bits
    )

    total_chunks = (
        first_parsed.total_chunks
    )

    if total_chunks <= 0:
        raise ValueError(
            "Invalid total chunk count."
        )

    if total_chunks > len(
        original_frames
    ):
        raise ValueError(
            "Stego video does not contain "
            "enough frames for all chunks.\n"
            f"Required: {total_chunks}\n"
            f"Available: {len(original_frames)}"
        )

    packets = [
        first_packet_bits
    ]

    # ------------------------------------------------------
    # REMAINING PACKETS
    # ------------------------------------------------------

    for frame_index in range(
        1,
        total_chunks,
    ):

        packet_bits = extract_frame_packet(
            original_frames[frame_index],
            received_frames[frame_index],
            block_size=block_size,
            gap=gap,
        )

        packets.append(
            packet_bits
        )

    # ------------------------------------------------------
    # RECONSTRUCT PAYLOAD
    # ------------------------------------------------------

    payload = parse_frame_packets(
        packets
    )

    # ------------------------------------------------------
    # PARSE ORIGINAL AES PAYLOAD
    # ------------------------------------------------------

    payload_length, encrypted_bits = (
        parse_payload_packet(
            payload
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

    # ------------------------------------------------------
    # AES DECRYPTION
    # ------------------------------------------------------

    recovered_data = decrypt_file(
        encrypted_data,
        Path(key_file),
    )

    # ------------------------------------------------------
    # WRITE RECOVERED FILE
    # ------------------------------------------------------

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
    payload_bits: int = None,
    block_size: int = BLOCK_SIZE,
    gap: int = GAP,
):
    """
    Recover the secret file from a multi-frame MP4V
    stego video.

    The original cover video is required as the
    extraction reference.

    payload_bits is retained for API compatibility
    with the previous single-frame pipeline.
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

    if not cover_video.exists():
        raise FileNotFoundError(
            f"Cover video not found: "
            f"{cover_video}"
        )

    if not stego_video.exists():
        raise FileNotFoundError(
            f"Stego video not found: "
            f"{stego_video}"
        )

    if not key_file.exists():
        raise FileNotFoundError(
            f"AES key not found: "
            f"{key_file}"
        )

    original_frames = _read_video_frames(
        cover_video
    )

    received_frames = _read_video_frames(
        stego_video
    )

    recovered_data = recover_secret_file(
        original_frames,
        received_frames,
        key_file,
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
    print(
        "StegaFusion Spatial Multi-Frame MP4 Video Pipeline"
    )
    print("=" * 70)

    print()
    print(
        "Module loaded successfully."
    )

    print()
    print(
        "Configuration:"
    )

    print(
        f"Block Size          : {BLOCK_SIZE}"
    )

    print(
        f"Gap                 : {GAP}"
    )

    print(
        f"Delta               : {DELTA}"
    )

    print(
        f"Maximum Frame Bits  : {MAX_FRAME_BITS}"
    )

    print(
        f"Maximum Chunk Bits  : {MAX_CHUNK_DATA_BITS}"
    )

    print()
    print(
        "Spatial multi-frame MP4 pipeline ready."
    )