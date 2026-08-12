"""
StegaFusion Multi-Frame Spatial MP4 Integration Diagnostic

Pipeline tested:

    Payload
        ↓
    Split into chunks
        ↓
    Frame packets
        ↓
    Spatial paired-block embedding
        ↓
    MP4V encoding
        ↓
    MP4V decoding
        ↓
    Spatial extraction
        ↓
    Frame packet parsing
        ↓
    Chunk reassembly
        ↓
    Original payload

This is a diagnostic only.
No production steganography algorithm is modified.
"""

from pathlib import Path

import cv2

from modules.steganography.multiframe_payload import (
    DEFAULT_CHUNK_BITS,
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

FRAME_INDICES = [
    0,
    1,
    2,
]

BLOCK_SIZE = 32
GAP = 16
DELTA = 4

PAYLOAD_BITS = 2000


# ==========================================================
# PAYLOAD
# ==========================================================

def make_payload(
    bit_count: int,
) -> str:
    """
    Generate a deterministic test payload.
    """

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
    """
    Compare two binary payloads.
    """

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

    if len(expected) != len(recovered):

        if mismatch is None:

            mismatch = min(
                len(expected),
                len(recovered),
            )

    return correct, mismatch


# ==========================================================
# WRITE SINGLE-FRAME MP4
# ==========================================================

def write_mp4(
    frame,
    output_path: Path,
):
    """
    Encode one stego frame as an MP4V video.
    """

    height, width = frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        24.0,
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

    writer.write(frame)

    writer.release()


# ==========================================================
# READ SINGLE-FRAME MP4
# ==========================================================

def read_mp4(
    video_path: Path,
):
    """
    Decode the first frame from an MP4 video.
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
            f"Unable to decode video: "
            f"{video_path}"
        )

    return frame


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 78)
    print(
        "StegaFusion Multi-Frame Spatial MP4 "
        "Integration Diagnostic"
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
    # CREATE PAYLOAD
    # ------------------------------------------------------

    payload = make_payload(
        PAYLOAD_BITS
    )

    print(
        f"Payload Generated: "
        f"{len(payload)} bits"
    )

    print()

    # ------------------------------------------------------
    # SPLIT PAYLOAD
    # ------------------------------------------------------

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
        f"Chunk Count : "
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

    # ------------------------------------------------------
    # CHECK FRAME AVAILABILITY
    # ------------------------------------------------------

    if len(chunks) > len(
        FRAME_INDICES
    ):

        raise RuntimeError(
            "Not enough test frames "
            "for the payload chunks."
        )

    # ------------------------------------------------------
    # PREPARE OUTPUT
    # ------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # EMBED EACH CHUNK
    # ------------------------------------------------------

    recovered_chunks = {}

    print(
        "FRAME PROCESSING"
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

        print(
            f"Source : "
            f"{frame_path}"
        )

        # --------------------------------------------------
        # LOAD FRAME
        # --------------------------------------------------

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
        # CREATE FRAME PACKET
        # --------------------------------------------------

        packet = create_packet(
            chunk_data=chunk,
            total_chunks=len(chunks),
            chunk_index=chunk_index,
        )

        print(
            f"Packet   : "
            f"{len(packet)} bits"
        )

        if len(packet) > capacity:

            raise RuntimeError(
                f"Frame {frame_index} "
                f"cannot hold packet."
            )

        # --------------------------------------------------
        # SPATIAL EMBEDDING
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
                "Not all packet bits "
                "were embedded."
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
                f"Direct Extraction : FAIL "
                f"({correct}/{len(packet)})"
            )

            print(
                f"First Mismatch    : "
                f"{mismatch}"
            )

            raise RuntimeError(
                "Direct spatial packet "
                "extraction failed."
            )

        print(
            "Direct Extraction : PASS"
        )

        # --------------------------------------------------
        # WRITE MP4
        # --------------------------------------------------

        output_video = (
            OUTPUT_DIR
            / f"frame_{frame_index:05d}.mp4"
        )

        write_mp4(
            stego,
            output_video,
        )

        print(
            f"MP4      : "
            f"{output_video}"
        )

        # --------------------------------------------------
        # DECODE MP4
        # --------------------------------------------------

        decoded = read_mp4(
            output_video
        )

        # --------------------------------------------------
        # EXTRACT PACKET AFTER MP4
        # --------------------------------------------------

        recovered_packet = extract_payload(
            original,
            decoded,
            len(packet),
            block_size=BLOCK_SIZE,
            gap=GAP,
        )

        correct, mismatch = compare_bits(
            packet,
            recovered_packet,
        )

        print(
            f"Video Extraction : "
            f"{correct}/{len(packet)} "
            f"({correct / len(packet) * 100:.2f}%)"
        )

        print(
            f"Mismatch         : "
            f"{mismatch}"
        )

        if recovered_packet != packet:

            print(
                "Result           : FAIL"
            )

            raise RuntimeError(
                f"MP4 packet extraction "
                f"failed on frame "
                f"{frame_index}."
            )

        print(
            "Result           : PASS"
        )

        # --------------------------------------------------
        # PARSE PACKET
        # --------------------------------------------------

        parsed = parse_packet(
            recovered_packet
        )

        print(
            f"Parsed Chunk     : "
            f"{parsed.chunk_index}"
        )

        print(
            f"Parsed Length    : "
            f"{parsed.chunk_length}"
        )

        if (
            parsed.total_chunks
            != len(chunks)
        ):

            raise RuntimeError(
                "Total chunk metadata "
                "does not match."
            )

        if (
            parsed.chunk_index
            != chunk_index
        ):

            raise RuntimeError(
                "Chunk index metadata "
                "does not match."
            )

        if (
            parsed.chunk_data
            != chunk
        ):

            raise RuntimeError(
                "Recovered chunk does "
                "not match original chunk."
            )

        recovered_chunks[
            parsed.chunk_index
        ] = parsed.chunk_data

    # ------------------------------------------------------
    # REASSEMBLE
    # ------------------------------------------------------

    print()
    print(
        "REASSEMBLY"
    )

    print(
        "-" * 78
    )

    if len(
        recovered_chunks
    ) != len(chunks):

        raise RuntimeError(
            "Not all chunks were recovered."
        )

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

    print(
        f"Original Payload   : "
        f"{len(payload)} bits"
    )

    print(
        f"Reconstructed      : "
        f"{len(reconstructed)} bits"
    )

    print(
        f"Correct Bits       : "
        f"{correct}/{len(payload)}"
    )

    print(
        f"Accuracy           : "
        f"{correct / len(payload) * 100:.2f}%"
    )

    print(
        f"First Mismatch     : "
        f"{mismatch}"
    )

    print()

    if reconstructed == payload:

        print(
            "Payload Reassembly : PASS"
        )

    else:

        print(
            "Payload Reassembly : FAIL"
        )

        raise RuntimeError(
            "Reconstructed payload "
            "does not match original."
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
        "Payload Chunking        : PASS"
    )

    print(
        "Frame Packet Creation   : PASS"
    )

    print(
        "Spatial Embedding       : PASS"
    )

    print(
        "Direct Extraction       : PASS"
    )

    print(
        "MP4V Encoding           : PASS"
    )

    print(
        "MP4V Extraction         : PASS"
    )

    print(
        "Packet Parsing          : PASS"
    )

    print(
        "Chunk Reassembly        : PASS"
    )

    print(
        "Original Payload Match  : PASS"
    )

    print()

    print(
        "RESULT: MULTI-FRAME SPATIAL "
        "MP4 PIPELINE PASSED."
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