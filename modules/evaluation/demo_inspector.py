"""
StegaFusion Demo Inspector

Stages:

    Stage 1
        Secret
            ↓
        AES-256-CBC
            ↓
        IV + Ciphertext
            ↓
        StegaFusion Payload

    Stage 2
        Payload
            ↓
        Multi-frame packetization
            ↓
        Frame packet mapping

This module only INSPECTS the existing production pipeline.
It does not modify the production steganography algorithm.
"""

from pathlib import Path
import hashlib

from modules.crypto.aes_encrypt import encrypt_file

from modules.steganography.payload import (
    create_payload_packet,
)

from modules.steganography.multiframe_packet import (
    MAX_FRAME_BITS,
    MAX_CHUNK_DATA_BITS,
    create_packet,
    parse_packet,
)

from modules.pipeline.spatial_video_pipeline import (
    create_frame_packets,
)


# ==========================================================
# HASH
# ==========================================================

def sha256_bytes(data: bytes) -> str:
    """Return SHA-256 hash of byte data."""

    return hashlib.sha256(
        data
    ).hexdigest()


# ==========================================================
# STAGE 1
# ==========================================================

def inspect_encryption_stage(
    secret_file: Path,
    key_file: Path,
) -> dict:
    """
    Inspect the real StegaFusion encryption/payload stage.
    """

    secret_file = Path(
        secret_file
    )

    key_file = Path(
        key_file
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
    # ORIGINAL SECRET
    # ------------------------------------------------------

    plaintext = (
        secret_file.read_bytes()
    )

    original_size = len(
        plaintext
    )

    original_hash = sha256_bytes(
        plaintext
    )

    # ------------------------------------------------------
    # REAL PRODUCTION ENCRYPTION
    # ------------------------------------------------------

    encrypted_data = encrypt_file(
        secret_file,
        key_file,
    )

    encrypted_size = len(
        encrypted_data
    )

    iv_size = 16

    if encrypted_size < iv_size:

        raise RuntimeError(
            "Encrypted output is smaller "
            "than the expected IV size."
        )

    iv = encrypted_data[
        :iv_size
    ]

    ciphertext = encrypted_data[
        iv_size:
    ]

    # ------------------------------------------------------
    # PAYLOAD
    # ------------------------------------------------------

    payload_packet = (
        create_payload_packet(
            encrypted_data
        )
    )

    payload_bits = len(
        payload_packet
    )

    encrypted_bits = (
        encrypted_size * 8
    )

    header_bits = 32

    return {
        "secret_file": str(
            secret_file
        ),

        "secret_size": original_size,

        "secret_sha256": original_hash,

        "encrypted_size": encrypted_size,

        "encrypted_bits": encrypted_bits,

        "iv_size": iv_size,

        "iv_preview": iv.hex(
            " "
        ),

        "ciphertext_size": len(
            ciphertext
        ),

        "encrypted_preview": (
            encrypted_data[
                :16
            ].hex(" ")
        ),

        "payload_header_bits": (
            header_bits
        ),

        "payload_bits": payload_bits,

        "payload_packet": (
            payload_packet
        ),

        "payload_preview": (
            payload_packet[
                :64
            ]
        ),
    }


# ==========================================================
# STAGE 2
# ==========================================================

def inspect_packetization_stage(
    payload_packet: str,
) -> dict:
    """
    Inspect real multi-frame packetization.
    """

    if not payload_packet:

        raise ValueError(
            "Payload packet cannot be empty."
        )

    packets = create_frame_packets(
        payload_packet
    )

    if not packets:

        raise RuntimeError(
            "Production packetization "
            "returned no packets."
        )

    packet_count = len(
        packets
    )

    parsed_packets = [
        parse_packet(packet)
        for packet in packets
    ]

    # ------------------------------------------------------
    # VALIDATE PACKET METADATA
    # ------------------------------------------------------

    expected_total = packet_count

    for expected_index, parsed in enumerate(
        parsed_packets
    ):

        if parsed.total_chunks != expected_total:

            raise RuntimeError(
                "Packet total_chunks mismatch."
            )

        if parsed.chunk_index != expected_index:

            raise RuntimeError(
                "Packet ordering mismatch."
            )

    # ------------------------------------------------------
    # PACKET SIZES
    # ------------------------------------------------------

    packet_sizes = [
        len(packet)
        for packet in packets
    ]

    maximum_packet_bits = max(
        packet_sizes
    )

    minimum_packet_bits = min(
        packet_sizes
    )

    average_packet_bits = (
        sum(packet_sizes)
        / packet_count
    )

    # ------------------------------------------------------
    # FRAME MAPPING
    # ------------------------------------------------------

    frame_mapping = []

    for index, packet in enumerate(
        packets
    ):

        frame_mapping.append(
            {
                "packet_number": index + 1,
                "packet_index": index,
                "frame_index": index,
                "packet_bits": len(packet),
                "chunk_index": parsed_packets[
                    index
                ].chunk_index,
                "total_chunks": parsed_packets[
                    index
                ].total_chunks,
            }
        )

    # ------------------------------------------------------
    # REASSEMBLY VALIDATION
    # ------------------------------------------------------

    chunks = [
        parsed.chunk_data
        for parsed in parsed_packets
    ]

    reconstructed = "".join(
        chunks
    )

    reconstruction_match = (
        reconstructed
        == payload_packet
    )

    if not reconstruction_match:

        raise RuntimeError(
            "Packet reassembly does not "
            "match the original payload."
        )

    return {
        "payload_bits": len(
            payload_packet
        ),

        "max_frame_bits": (
            MAX_FRAME_BITS
        ),

        "max_chunk_data_bits": (
            MAX_CHUNK_DATA_BITS
        ),

        "packet_count": packet_count,

        "minimum_packet_bits": (
            minimum_packet_bits
        ),

        "maximum_packet_bits": (
            maximum_packet_bits
        ),

        "average_packet_bits": (
            average_packet_bits
        ),

        "frame_mapping": frame_mapping,

        "reconstruction_match": (
            reconstruction_match
        ),
    }


# ==========================================================
# DISPLAY — STAGE 1
# ==========================================================

def print_encryption_stage(
    report: dict,
) -> None:

    print()
    print("=" * 70)

    print(
        "StegaFusion DEMO INSPECTOR"
    )

    print(
        "STAGE 1 — ENCRYPTION + PAYLOAD"
    )

    print("=" * 70)

    print()
    print("## ORIGINAL SECRET")

    print(
        f"File          : "
        f"{report['secret_file']}"
    )

    print(
        f"Size          : "
        f"{report['secret_size']} bytes"
    )

    print(
        f"SHA-256       : "
        f"{report['secret_sha256']}"
    )

    print()
    print("## AES-256 ENCRYPTION")

    print(
        "Algorithm     : AES-256-CBC"
    )

    print(
        f"IV Size       : "
        f"{report['iv_size']} bytes"
    )

    print(
        f"Ciphertext    : "
        f"{report['ciphertext_size']} bytes"
    )

    print(
        f"Encrypted Data: "
        f"{report['encrypted_size']} bytes"
    )

    print(
        f"Encrypted Bits: "
        f"{report['encrypted_bits']}"
    )

    print(
        f"Encrypted Preview: "
        f"{report['encrypted_preview']}"
    )

    print()
    print(
        "## STEGAFUSION PAYLOAD PACKET"
    )

    print(
        f"Header        : "
        f"{report['payload_header_bits']} bits"
    )

    print(
        f"Payload Bits  : "
        f"{report['payload_bits']}"
    )

    print(
        f"Payload Preview: "
        f"{report['payload_preview']}"
    )

    print()
    print(
        "Stage 1 Status: PASS"
    )


# ==========================================================
# DISPLAY — STAGE 2
# ==========================================================

def print_packetization_stage(
    report: dict,
) -> None:

    print()
    print("=" * 70)

    print(
        "STAGE 2 — MULTI-FRAME PACKETIZATION"
    )

    print("=" * 70)

    print()
    print("## PAYLOAD")

    print(
        f"Payload Bits       : "
        f"{report['payload_bits']}"
    )

    print(
        f"Maximum Frame Bits : "
        f"{report['max_frame_bits']}"
    )

    print(
        f"Maximum Data Bits  : "
        f"{report['max_chunk_data_bits']}"
    )

    print()
    print("## PACKETIZATION")

    print(
        f"Packet Count       : "
        f"{report['packet_count']}"
    )

    print(
        f"Minimum Packet Bits: "
        f"{report['minimum_packet_bits']}"
    )

    print(
        f"Maximum Packet Bits: "
        f"{report['maximum_packet_bits']}"
    )

    print(
        f"Average Packet Bits: "
        f"{report['average_packet_bits']:.2f}"
    )

    print()
    print("## PACKET → FRAME MAPPING")

    mappings = report[
        "frame_mapping"
    ]

    # First three
    for mapping in mappings[:3]:

        print(
            f"Packet "
            f"{mapping['packet_number']:03d}"
            f" → Frame "
            f"{mapping['frame_index']:03d}"
            f" | "
            f"{mapping['packet_bits']} bits"
        )

    print(
        "..."
    )

    # Middle packet
    middle = mappings[
        len(mappings) // 2
    ]

    print(
        f"Packet "
        f"{middle['packet_number']:03d}"
        f" → Frame "
        f"{middle['frame_index']:03d}"
        f" | "
        f"{middle['packet_bits']} bits"
    )

    print(
        "..."
    )

    # Last three
    for mapping in mappings[-3:]:

        print(
            f"Packet "
            f"{mapping['packet_number']:03d}"
            f" → Frame "
            f"{mapping['frame_index']:03d}"
            f" | "
            f"{mapping['packet_bits']} bits"
        )

    print()
    print(
        "Packet Reassembly : "
        f"{'PASS' if report['reconstruction_match'] else 'FAIL'}"
    )

    print()
    print(
        "Stage 2 Status: PASS"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    secret_file = Path(
        "input/secret_data/scalability_test/"
        "secret_32kb.bin"
    )

    key_file = Path(
        "input/keys/test_aes_key.bin"
    )

    # ------------------------------------------------------
    # STAGE 1
    # ------------------------------------------------------

    stage1 = (
        inspect_encryption_stage(
            secret_file=secret_file,
            key_file=key_file,
        )
    )

    print_encryption_stage(
        stage1
    )

    # ------------------------------------------------------
    # STAGE 2
    # ------------------------------------------------------

    stage2 = (
        inspect_packetization_stage(
            payload_packet=stage1[
                "payload_packet"
            ],
        )
    )

    print_packetization_stage(
        stage2
    )

    # ------------------------------------------------------
    # FINAL
    # ------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "RESULT: STAGES 1–2 INSPECTION PASS."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()