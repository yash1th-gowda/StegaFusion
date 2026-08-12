"""
StegaFusion Multi-Frame Spatial MP4 Video Diagnostic

Tests the complete multi-frame flow using ONE MP4V video:

    Payload
        ↓
    Chunking
        ↓
    Frame packets
        ↓
    Spatial embedding
        ↓
    ONE MP4V video
        ↓
    Decode all frames
        ↓
    Spatial extraction
        ↓
    Packet parsing
        ↓
    Chunk reassembly
        ↓
    Original payload

Diagnostic only.
No production steganography algorithm is modified.
"""

from pathlib import Path

import cv2

from modules.steganography.multiframe_payload import (
    combine_chunks,
    split_payload,
)

from modules.steganography.multiframe_packet import (
    MAX_CHUNK_DATA_BITS,
    create_packet,
    parse_packet,
)

from modules.steganography.spatial.paired_block import (
    calculate_capacity,
    embed_payload,
    extract_payload,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

FRAME_DIR = Path("temp/frames")

OUTPUT_DIR = Path(
    "output/video_spatial_multiframe"
)

OUTPUT_VIDEO = (
    OUTPUT_DIR /
    "multiframe_stego.mp4"
)

FRAME_INDICES = [
    0,
    1,
    2,
]

BLOCK_SIZE = 32
GAP = 16
DELTA = 4

PAYLOAD_BITS = 2000

FPS = 23.976023976023978


# ==========================================================
# PAYLOAD
# ==========================================================

def make_payload(bit_count: int) -> str:

    pattern = (
        "101100111000111100001111"
    )

    repeats = (
        bit_count // len(pattern)
    ) + 1

    return (
        pattern * repeats
    )[:bit_count]


# ==========================================================
# BIT COMPARISON
# ==========================================================

def compare_bits(
    expected: str,
    recovered: str,
):
    correct = sum(
        a == b
        for a, b in zip(
            expected,
            recovered,
        )
    )

    mismatch = None

    for index, (
        expected_bit,
        recovered_bit,
    ) in enumerate(
        zip(
            expected,
            recovered,
        )
    ):

        if expected_bit != recovered_bit:
            mismatch = index
            break

    if (
        mismatch is None
        and
        len(expected) != len(recovered)
    ):

        mismatch = min(
            len(expected),
            len(recovered),
        )

    return correct, mismatch


# ==========================================================
# WRITE MULTI-FRAME MP4
# ==========================================================

def write_video(
    frames,
    output_path: Path,
):

    if not frames:
        raise ValueError(
            "No frames supplied."
        )

    height, width = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        FPS,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():

        raise RuntimeError(
            f"Unable to create video: "
            f"{output_path}"
        )

    for frame in frames:
        writer.write(frame)

    writer.release()


# ==========================================================
# READ ALL VIDEO FRAMES
# ==========================================================

def read_video(
    video_path: Path,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Unable to open video: "
            f"{video_path}"
        )

    frames = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        frames.append(frame)

    cap.release()

    if not frames:

        raise RuntimeError(
            "Video contained no decodable frames."
        )

    return frames


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 78)
    print(
        "StegaFusion Multi-Frame Spatial MP4 "
        "Video Diagnostic"
    )
    print("=" * 78)

    print()

    print(
        f"Original Payload : "
        f"{PAYLOAD_BITS} bits"
    )

    print(
        f"Chunk Data Limit : "
        f"{MAX_CHUNK_DATA_BITS} bits"
    )

    print(
        f"Block Size       : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"Gap              : "
        f"{GAP}"
    )

    print(
        f"Delta            : "
        f"{DELTA}"
    )

    print()

    # ------------------------------------------------------
    # PAYLOAD
    # ------------------------------------------------------

    payload = make_payload(
        PAYLOAD_BITS
    )

    chunks = split_payload(
        payload,
        MAX_CHUNK_DATA_BITS,
    )

    print(
        "PAYLOAD CHUNKING"
    )

    print(
        "-" * 78
    )

    print(
        f"Payload Bits : "
        f"{len(payload)}"
    )

    print(
        f"Chunk Count  : "
        f"{len(chunks)}"
    )

    for index, chunk in enumerate(
        chunks
    ):

        print(
            f"Chunk {index:2d} : "
            f"{len(chunk)} bits"
        )

    print()

    if len(chunks) != len(
        FRAME_INDICES
    ):

        raise RuntimeError(
            "This diagnostic expects exactly "
            f"{len(FRAME_INDICES)} chunks."
        )

    # ------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # PREPARE STEGO FRAMES
    # ------------------------------------------------------

    stego_frames = []

    packet_lengths = {}

    print(
        "PREPARING STEGO FRAMES"
    )

    print(
        "-" * 78
    )

    for chunk_index, chunk in enumerate(
        chunks
    ):

        frame_index = (
            FRAME_INDICES[
                chunk_index
            ]
        )

        frame_path = (
            FRAME_DIR
            / f"frame_{frame_index:05d}.png"
        )

        print()

        print(
            f"Frame {frame_index} "
            f"→ Chunk {chunk_index}"
        )

        original = cv2.imread(
            str(frame_path)
        )

        if original is None:

            raise FileNotFoundError(
                f"Unable to load frame: "
                f"{frame_path}"
            )

        capacity = calculate_capacity(
            original,
            block_size=BLOCK_SIZE,
            gap=GAP,
        )

        print(
            f"Capacity : "
            f"{capacity} bits"
        )

        # --------------------------------------------------
        # CREATE PACKET
        # --------------------------------------------------

        packet = create_packet(
            chunk_data=chunk,
            total_chunks=len(chunks),
            chunk_index=chunk_index,
        )

        packet_lengths[
            chunk_index
        ] = len(packet)

        print(
            f"Packet   : "
            f"{len(packet)} bits"
        )

        if len(packet) > capacity:

            raise RuntimeError(
                f"Packet for frame "
                f"{frame_index} exceeds capacity."
            )

        # --------------------------------------------------
        # EMBED
        # --------------------------------------------------

        stego, embedded = embed_payload(
            original,
            packet,
            block_size=BLOCK_SIZE,
            delta=DELTA,
            gap=GAP,
        )

        print(
            f"Embedded : "
            f"{embedded} bits"
        )

        if embedded != len(packet):

            raise RuntimeError(
                "Embedded bit count does not "
                "match packet length."
            )

        # --------------------------------------------------
        # DIRECT EXTRACTION
        # --------------------------------------------------

        direct = extract_payload(
            original,
            stego,
            len(packet),
            block_size=BLOCK_SIZE,
            gap=GAP,
        )

        if direct != packet:

            correct, mismatch = compare_bits(
                packet,
                direct,
            )

            print(
                f"Direct Extraction : "
                f"FAIL "
                f"{correct}/{len(packet)}"
            )

            print(
                f"Mismatch          : "
                f"{mismatch}"
            )

            raise RuntimeError(
                f"Direct extraction failed "
                f"for frame {frame_index}."
            )

        print(
            "Direct Extraction : PASS"
        )

        stego_frames.append(
            stego
        )

    # ------------------------------------------------------
    # WRITE ONE MP4
    # ------------------------------------------------------

    print()
    print(
        "WRITING SINGLE MULTI-FRAME MP4"
    )

    print(
        "-" * 78
    )

    write_video(
        stego_frames,
        OUTPUT_VIDEO,
    )

    print(
        f"Output Video : "
        f"{OUTPUT_VIDEO}"
    )

    # ------------------------------------------------------
    # DECODE VIDEO
    # ------------------------------------------------------

    decoded_frames = read_video(
        OUTPUT_VIDEO
    )

    print(
        f"Decoded Frames : "
        f"{len(decoded_frames)}"
    )

    if len(decoded_frames) != len(
        stego_frames
    ):

        raise RuntimeError(
            "Decoded frame count does not "
            "match the number of stego frames."
        )

    # ------------------------------------------------------
    # EXTRACT PACKETS
    # ------------------------------------------------------

    recovered_chunks = {}

    print()
    print(
        "EXTRACTING FRAME PACKETS"
    )

    print(
        "-" * 78
    )

    for position, frame_index in enumerate(
        FRAME_INDICES
    ):

        original_path = (
            FRAME_DIR
            / f"frame_{frame_index:05d}.png"
        )

        original = cv2.imread(
            str(original_path)
        )

        if original is None:

            raise FileNotFoundError(
                original_path
            )

        decoded = decoded_frames[
            position
        ]

        expected_packet_length = (
            packet_lengths[position]
        )

        recovered_packet = extract_payload(
            original,
            decoded,
            expected_packet_length,
            block_size=BLOCK_SIZE,
            gap=GAP,
        )

        # --------------------------------------------------
        # PARSE PACKET
        # --------------------------------------------------

        parsed = parse_packet(
            recovered_packet
        )

        correct, mismatch = compare_bits(
            recovered_packet,
            create_packet(
                parsed.chunk_data,
                parsed.total_chunks,
                parsed.chunk_index,
            ),
        )

        print()

        print(
            f"Frame {frame_index}"
        )

        print(
            f"Packet Bits : "
            f"{len(recovered_packet)}"
        )

        print(
            f"Accuracy    : "
            f"{correct / len(recovered_packet) * 100:.2f}%"
        )

        print(
            f"Mismatch    : "
            f"{mismatch}"
        )

        print(
            f"Chunk Index : "
            f"{parsed.chunk_index}"
        )

        print(
            f"Chunk Size  : "
            f"{parsed.chunk_length}"
        )

        if mismatch is not None:

            raise RuntimeError(
                f"Packet corruption detected "
                f"on frame {frame_index}."
            )

        recovered_chunks[
            parsed.chunk_index
        ] = parsed.chunk_data

        print(
            "Result      : PASS"
        )

    # ------------------------------------------------------
    # VALIDATE CHUNKS
    # ------------------------------------------------------

    print()
    print(
        "CHUNK VALIDATION"
    )

    print(
        "-" * 78
    )

    if len(
        recovered_chunks
    ) != len(chunks):

        raise RuntimeError(
            "Not all expected chunks "
            "were recovered."
        )

    for index, original_chunk in enumerate(
        chunks
    ):

        recovered_chunk = (
            recovered_chunks[index]
        )

        if recovered_chunk != original_chunk:

            raise RuntimeError(
                f"Chunk {index} does not "
                "match the original."
            )

        print(
            f"Chunk {index:2d} : PASS"
        )

    # ------------------------------------------------------
    # REASSEMBLY
    # ------------------------------------------------------

    ordered_chunks = [
        recovered_chunks[index]
        for index in range(
            len(chunks)
        )
    ]

    reconstructed = combine_chunks(
        ordered_chunks
    )

    correct, mismatch = compare_bits(
        payload,
        reconstructed,
    )

    print()
    print(
        "REASSEMBLY"
    )

    print(
        "-" * 78
    )

    print(
        f"Original Payload : "
        f"{len(payload)} bits"
    )

    print(
        f"Recovered Payload: "
        f"{len(reconstructed)} bits"
    )

    print(
        f"Correct Bits     : "
        f"{correct}/{len(payload)}"
    )

    print(
        f"Accuracy         : "
        f"{correct / len(payload) * 100:.2f}%"
    )

    print(
        f"First Mismatch   : "
        f"{mismatch}"
    )

    if reconstructed != payload:

        print(
            "Payload Match    : FAIL"
        )

        raise RuntimeError(
            "Reconstructed payload does not "
            "match the original payload."
        )

    print(
        "Payload Match    : PASS"
    )

    # ------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "FINAL RESULT"
    )
    print("=" * 78)

    print()

    print(
        "Payload Chunking       : PASS"
    )

    print(
        "Frame Packet Creation  : PASS"
    )

    print(
        "Spatial Embedding      : PASS"
    )

    print(
        "Single MP4V Encoding   : PASS"
    )

    print(
        "All Frames Decoded     : PASS"
    )

    print(
        "Packet Extraction      : PASS"
    )

    print(
        "Packet Parsing         : PASS"
    )

    print(
        "Chunk Validation       : PASS"
    )

    print(
        "Chunk Reassembly       : PASS"
    )

    print(
        "Original Payload Match : PASS"
    )

    print()

    print(
        "RESULT: MULTI-FRAME SPATIAL "
        "MP4 VIDEO PIPELINE PASSED."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No production steganography "
        "algorithm was modified."
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()