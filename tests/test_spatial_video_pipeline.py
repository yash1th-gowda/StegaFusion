"""
StegaFusion Spatial MP4 End-to-End Pipeline Test

Workflow:

    Secret File
        ↓
    AES Encryption
        ↓
    Payload Packet
        ↓
    Spatial Paired-Block Embedding
        ↓
    MP4V Encoding
        ↓
    MP4V Decoding
        ↓
    Spatial Extraction
        ↓
    Payload Parsing
        ↓
    AES Decryption
        ↓
    Recovered Secret File

This test does not modify production steganography code.
"""

from pathlib import Path

from modules.pipeline.spatial_video_pipeline import (
    prepare_spatial_payload,
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
    "input/secret_data/test_secret.txt"
)

KEY_FILE = Path(
    "input/keys/test_aes_key.bin"
)

OUTPUT_DIR = Path(
    "output/video_spatial_pipeline"
)

OUTPUT_VIDEO = (
    OUTPUT_DIR /
    "spatial_stego.mp4"
)

RECOVERED_FILE = (
    OUTPUT_DIR /
    "recovered_test_secret.txt"
)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("StegaFusion Spatial MP4 End-to-End Pipeline Test")
    print("=" * 70)

    print()

    # ------------------------------------------------------
    # CHECK INPUTS
    # ------------------------------------------------------

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

    print()

    # ------------------------------------------------------
    # PREPARE PAYLOAD
    # ------------------------------------------------------

    print("=" * 70)
    print("PAYLOAD PREPARATION")
    print("=" * 70)

    payload = prepare_spatial_payload(
        SECRET_FILE,
        KEY_FILE,
    )

    print()
    print(
        f"Secret Size  : "
        f"{SECRET_FILE.stat().st_size} bytes"
    )

    print(
        f"Payload Bits : "
        f"{len(payload)}"
    )

    print()
    print(
        "AES + Payload Packet : PASS"
    )

    # ------------------------------------------------------
    # EMBEDDING
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("SPATIAL VIDEO EMBEDDING")
    print("=" * 70)

    embedding = embed_spatial_video(
        cover_video=COVER_VIDEO,
        secret_file=SECRET_FILE,
        key_file=KEY_FILE,
        output_video=OUTPUT_VIDEO,
    )

    print()

    print(
        f"Payload Bits  : "
        f"{embedding['payload_bits']}"
    )

    print(
        f"Embedded Bits : "
        f"{embedding['embedded_bits']}"
    )

    print(
        f"Resolution    : "
        f"{embedding['width']} x "
        f"{embedding['height']}"
    )

    print(
        f"FPS           : "
        f"{embedding['fps']}"
    )

    print(
        f"Frames        : "
        f"{embedding['frames']}"
    )

    print(
        f"Output Video  : "
        f"{embedding['output_video']}"
    )

    print()

    if (
        embedding["embedded_bits"]
        != embedding["payload_bits"]
    ):

        raise RuntimeError(
            "Embedding did not process "
            "the complete payload."
        )

    if not OUTPUT_VIDEO.exists():

        raise RuntimeError(
            "Stego video was not created."
        )

    print(
        "Spatial Embedding : PASS"
    )

    # ------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("SPATIAL MP4 EXTRACTION")
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
        f"Payload Bits     : "
        f"{extraction['payload_bits']}"
    )

    print(
        f"Recovered Bytes  : "
        f"{extraction['recovered_bytes']}"
    )

    print(
        f"Recovered File   : "
        f"{extraction['output_file']}"
    )

    print()

    if not RECOVERED_FILE.exists():

        raise RuntimeError(
            "Recovered secret file was not created."
        )

    print(
        "MP4 Spatial Extraction : PASS"
    )

    # ------------------------------------------------------
    # FILE COMPARISON
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

    print()

    if original_data != recovered_data:

        print(
            "Original Match : FAIL"
        )

        print()
        print(
            "The recovered file does not "
            "match the original secret file."
        )

        raise SystemExit(1)

    print(
        "Original Match : PASS"
    )

    # ------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RESULT: SPATIAL MP4 END-TO-END "
        "PIPELINE PASSED."
    )
    print("=" * 70)

    print()

    print(
        "Pipeline:"
    )

    print(
        "AES Encryption        : PASS"
    )

    print(
        "Payload Packet        : PASS"
    )

    print(
        "Spatial Embedding     : PASS"
    )

    print(
        "MP4V Encoding         : PASS"
    )

    print(
        "Spatial Extraction    : PASS"
    )

    print(
        "AES Decryption        : PASS"
    )

    print(
        "Original File Match   : PASS"
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