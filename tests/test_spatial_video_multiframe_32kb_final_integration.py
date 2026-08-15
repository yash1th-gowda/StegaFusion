"""
StegaFusion 32 KB FINAL PRODUCTION API INTEGRATION TEST

Validates:

    32 KB Secret
        ↓
    AES-256 Encryption
        ↓
    Multi-frame Packetization
        ↓
    Spatial Embedding
        ↓
    MP4V Encoding
        ↓
    MP4V Decoding
        ↓
    Spatial Extraction
        ↓
    Packet Reassembly
        ↓
    AES-256 Decryption
        ↓
    Byte-for-byte validation
        ↓
    SHA-256 validation

No production steganography algorithm is modified.
"""

from pathlib import Path
import hashlib

from modules.pipeline.spatial_video_pipeline import (
    embed_spatial_video,
    extract_spatial_video,
)


# ==========================================================
# INPUT / OUTPUT PATHS
# ==========================================================

COVER_VIDEO = Path(
    "input/cover_video/sample_long.mp4"
)

SECRET_FILE = Path(
    "input/secret_data/scalability_test/secret_32kb.bin"
)

KEY_FILE = Path(
    "input/keys/test_aes_key.bin"
)

OUTPUT_DIR = Path(
    "output/video_spatial_multiframe_32kb_final_integration"
)

OUTPUT_VIDEO = (
    OUTPUT_DIR / "stego_32kb.mp4"
)

RECOVERED_DIR = (
    OUTPUT_DIR / "recovered"
)

RECOVERED_FILE = (
    RECOVERED_DIR / "recovered_32kb.bin"
)


# ==========================================================
# SHA-256
# ==========================================================

def sha256_file(
    path: Path,
) -> str:
    """
    Return SHA-256 hash of a file.
    """

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ==========================================================
# INPUT VALIDATION
# ==========================================================

def validate_inputs():
    """
    Validate required input files.
    """

    print()
    print("## INPUT VALIDATION")

    for path in (
        COVER_VIDEO,
        SECRET_FILE,
        KEY_FILE,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Required input does not exist: "
                f"{path}"
            )

        print(
            f"{path} : EXISTS"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)

    print(
        "StegaFusion 32 KB FINAL "
        "PRODUCTION API INTEGRATION TEST"
    )

    print("=" * 70)

    # ------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # ------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RECOVERED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------

    validate_inputs()

    original_size = (
        SECRET_FILE.stat().st_size
    )

    original_hash = sha256_file(
        SECRET_FILE
    )

    print()
    print(
        f"Original Secret Size : "
        f"{original_size} bytes"
    )

    print(
        f"Original SHA-256     : "
        f"{original_hash}"
    )

    if original_size != 32768:

        raise RuntimeError(
            "Expected a 32768-byte secret file."
        )

    # ======================================================
    # PRODUCTION EMBEDDING
    # ======================================================

    print()
    print("## PRODUCTION EMBEDDING")

    result = embed_spatial_video(
        cover_video=COVER_VIDEO,
        secret_file=SECRET_FILE,
        key_file=KEY_FILE,
        output_video=OUTPUT_VIDEO,
        delta=5,
    )

    print(
        "Embedding : PASS"
    )

    print(
        f"Payload Bits      : "
        f"{result['payload_bits']}"
    )

    print(
        f"Embedded Bits     : "
        f"{result['embedded_bits']}"
    )

    print(
        f"Packet Count      : "
        f"{result['packet_count']}"
    )

    print(
        f"Embedded Frames   : "
        f"{result['embedded_frames']}"
    )

    print(
        f"Video Frames      : "
        f"{result['frames']}"
    )

    print(
        f"Output Video      : "
        f"{result['output_video']}"
    )

    print(
        f"Stego Video       : "
        f"{'EXISTS' if OUTPUT_VIDEO.exists() else 'MISSING'}"
    )

    if not OUTPUT_VIDEO.exists():

        raise RuntimeError(
            "Production embedding did not create "
            "the expected stego video."
        )

    # ======================================================
    # PRODUCTION EXTRACTION
    # ======================================================

    print()
    print("## PRODUCTION EXTRACTION")

    extraction = extract_spatial_video(
        cover_video=COVER_VIDEO,
        stego_video=OUTPUT_VIDEO,
        key_file=KEY_FILE,
        output_file=RECOVERED_FILE,
        delta=5,
    )

    print(
        "Extraction : PASS"
    )

    print(
        f"Recovered File : "
        f"{RECOVERED_FILE}"
    )

    print(
        f"Recovered File Exists : "
        f"{'YES' if RECOVERED_FILE.exists() else 'NO'}"
    )

    if not RECOVERED_FILE.exists():

        raise RuntimeError(
            "Production extraction did not "
            "create the recovered file."
        )

    # ======================================================
    # BYTE-FOR-BYTE VALIDATION
    # ======================================================

    recovered_size = (
        RECOVERED_FILE.stat().st_size
    )

    print()
    print("## BYTE-FOR-BYTE VALIDATION")

    print(
        f"Original Size  : "
        f"{original_size} bytes"
    )

    print(
        f"Recovered Size : "
        f"{recovered_size} bytes"
    )

    original_bytes = (
        SECRET_FILE.read_bytes()
    )

    recovered_bytes = (
        RECOVERED_FILE.read_bytes()
    )

    byte_match = (
        original_bytes ==
        recovered_bytes
    )

    print(
        f"Byte-for-byte Match : "
        f"{'TRUE' if byte_match else 'FALSE'}"
    )

    # ======================================================
    # SHA-256 VALIDATION
    # ======================================================

    recovered_hash = sha256_file(
        RECOVERED_FILE
    )

    hash_match = (
        original_hash ==
        recovered_hash
    )

    print()
    print("## SHA-256 VALIDATION")

    print(
        f"Original SHA-256  : "
        f"{original_hash}"
    )

    print(
        f"Recovered SHA-256 : "
        f"{recovered_hash}"
    )

    print(
        f"SHA-256 Match     : "
        f"{'TRUE' if hash_match else 'FALSE'}"
    )

    # ======================================================
    # FINAL RESULT
    # ======================================================

    if (
        recovered_size == original_size
        and byte_match
        and hash_match
    ):

        print()
        print("=" * 70)

        print(
            "RESULT: 32 KB COMPLETE "
            "PRODUCTION INTEGRATION PASS."
        )

        print("=" * 70)

        print()
        print(
            "The original 32 KB secret survived:"
        )

        print(
            "AES-256 encryption"
        )

        print(
            "→ multi-frame packetization"
        )

        print(
            "→ spatial embedding"
        )

        print(
            "→ MP4V encoding"
        )

        print(
            "→ MP4V decoding"
        )

        print(
            "→ spatial extraction"
        )

        print(
            "→ packet reassembly"
        )

        print(
            "→ AES-256 decryption"
        )

        print(
            "→ byte-for-byte recovery"
        )

        print()
        print(
            "No production steganography "
            "algorithm was modified."
        )

    else:

        print()
        print("=" * 70)

        print(
            "RESULT: 32 KB INTEGRATION FAIL."
        )

        print("=" * 70)

        raise RuntimeError(
            "32 KB byte/hash validation failed."
        )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
