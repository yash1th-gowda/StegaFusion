"""
StegaFusion Real Multi-Frame Spatial MP4 End-to-End Test

Uses a real secret file and real AES encryption to verify that
the production spatial video pipeline can:

    Secret File
        ↓
    AES Encryption
        ↓
    Payload Packet
        ↓
    Multi-Frame Chunking
        ↓
    Frame Packets
        ↓
    Spatial Embedding
        ↓
    ONE MP4V VIDEO
        ↓
    Multi-Frame Extraction
        ↓
    Chunk Reassembly
        ↓
    Payload Parsing
        ↓
    AES Decryption
        ↓
    Original Secret File

This test does not modify production steganography code.
"""

from pathlib import Path

from modules.pipeline.spatial_video_pipeline import (
    prepare_spatial_payload,
    create_frame_packets,
    embed_spatial_video,
    extract_spatial_video,
)

from modules.steganography.multiframe_packet import (
    MAX_CHUNK_DATA_BITS,
    MAX_FRAME_BITS,
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
    "output/video_spatial_multiframe_real_1kb"
)

OUTPUT_VIDEO = (
    OUTPUT_DIR /
    "multiframe_1kb_stego.mp4"
)

RECOVERED_FILE = (
    OUTPUT_DIR /
    "recovered_multiframe_1kb_secret.bin"
)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "StegaFusion Real Multi-Frame Spatial MP4 "
        "End-to-End Test"
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
    # PAYLOAD PREPARATION
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("REAL PAYLOAD PREPARATION")
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
        f"Secret Size       : "
        f"{SECRET_FILE.stat().st_size} bytes"
    )

    print(
        f"Payload Bits      : "
        f"{len(payload)}"
    )

    print(
        f"Maximum Chunk     : "
        f"{MAX_CHUNK_DATA_BITS} bits"
    )

    print(
        f"Maximum Packet    : "
        f"{MAX_FRAME_BITS} bits"
    )

    print(
        f"Frame Packets     : "
        f"{len(packets)}"
    )

    print()

    for index, packet in enumerate(
        packets
    ):

        print(
            f"Packet {index:2d}        : "
            f"{len(packet)} bits"
        )

    # ------------------------------------------------------
    # CRITICAL MULTI-FRAME CHECK
    # ------------------------------------------------------

    if len(packets) < 2:

        raise RuntimeError(
            "The test payload did not produce "
            "multiple frame packets."
        )

    print()

    print(
        "Real payload spans multiple frames : PASS"
    )

    # ------------------------------------------------------
    # EMBEDDING
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("MULTI-FRAME SPATIAL EMBEDDING")
    print("=" * 70)

    embedding = embed_spatial_video(
        cover_video=COVER_VIDEO,
        secret_file=SECRET_FILE,
        key_file=KEY_FILE,
        output_video=OUTPUT_VIDEO,
    )

    print()

    print(
        f"Payload Bits     : "
        f"{embedding['payload_bits']}"
    )

    print(
        f"Embedded Bits    : "
        f"{embedding['embedded_bits']}"
    )

    print(
        f"Video Frames     : "
        f"{embedding['frames']}"
    )

    print(
        f"Embedded Frames  : "
        f"{embedding['embedded_frames']}"
    )

    print(
        f"Packet Count     : "
        f"{embedding['packet_count']}"
    )

    print(
        f"Packet Sizes     : "
        f"{embedding['packet_sizes']}"
    )

    print(
        f"Output Video     : "
        f"{embedding['output_video']}"
    )

    print()

    if embedding["packet_count"] < 2:

        raise RuntimeError(
            "Production pipeline did not create "
            "multiple frame packets."
        )

    if (
        embedding["embedded_frames"]
        != embedding["packet_count"]
    ):

        raise RuntimeError(
            "Not all frame packets were embedded."
        )

    if not OUTPUT_VIDEO.exists():

        raise RuntimeError(
            "Stego video was not created."
        )

    print(
        "Multi-Frame Spatial Embedding : PASS"
    )

    print(
        "MP4V Encoding : PASS"
    )

    # ------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("MULTI-FRAME SPATIAL EXTRACTION")
    print("=" * 70)

    extraction = extract_spatial_video(
        cover_video=COVER_VIDEO,
        stego_video=OUTPUT_VIDEO,
        key_file=KEY_FILE,
        output_file=RECOVERED_FILE,
        payload_bits=len(payload),
    )

    print()

    print(
        f"Recovered Bytes : "
        f"{extraction['recovered_bytes']}"
    )

    print(
        f"Recovered File  : "
        f"{extraction['output_file']}"
    )

    print()

    if not RECOVERED_FILE.exists():

        raise RuntimeError(
            "Recovered secret file was not created."
        )

    print(
        "Multi-Frame Spatial Extraction : PASS"
    )

    # ------------------------------------------------------
    # BYTE-LEVEL VERIFICATION
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("RECOVERED FILE VERIFICATION")
    print("=" * 70)

    original_data = (
        SECRET_FILE.read_bytes()
    )

    recovered_data = (
        RECOVERED_FILE.read_bytes()
    )

    print()

    print(
        f"Original Size  : "
        f"{len(original_data)} bytes"
    )

    print(
        f"Recovered Size : "
        f"{len(recovered_data)} bytes"
    )

    if original_data != recovered_data:

        print()
        print(
            "Original Match : FAIL"
        )

        raise SystemExit(1)

    print()

    print(
        "Original Match : PASS"
    )

    # ------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RESULT: REAL MULTI-FRAME SPATIAL MP4 "
        "PIPELINE PASSED."
    )
    print("=" * 70)

    print()

    print(
        "Pipeline:"
    )

    print(
        "AES Encryption           : PASS"
    )

    print(
        "Payload Packet           : PASS"
    )

    print(
        "Multi-Frame Chunking     : PASS"
    )

    print(
        "Frame Packet Creation    : PASS"
    )

    print(
        "Spatial Embedding        : PASS"
    )

    print(
        "MP4V Encoding            : PASS"
    )

    print(
        "Multi-Frame Extraction   : PASS"
    )

    print(
        "Packet Parsing           : PASS"
    )

    print(
        "Chunk Reassembly         : PASS"
    )

    print(
        "AES Decryption           : PASS"
    )

    print(
        "Original File Match      : PASS"
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