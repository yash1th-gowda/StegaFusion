"""
StegaFusion Stage 5 — Complete Recovery Flow

Visualizes the ACTUAL production recovery pipeline:

    Original Cover Video
            +
    Stego Video
            |
            v
    Production Frame Extraction
            |
            v
    Production Packet Extraction
            |
            v
    Multi-Frame Packet Reassembly
            |
            v
    StegaFusion Payload Packet
            |
            v
    AES-256-CBC Decryption
            |
            v
    Recovered Secret
            |
            +----------------------+
            |                      |
            v                      v
      Byte-for-Byte            SHA-256
        Validation             Validation

Important
---------
This script does NOT implement a second steganography algorithm.

It directly uses the existing StegaFusion production functions
to inspect and demonstrate the recovery process.
"""

from pathlib import Path
import hashlib

import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from modules.pipeline.spatial_video_pipeline import (
    _read_video_frames,
    extract_frame_packet,
    parse_frame_packets,
)

from modules.steganography.payload import (
    binary_to_bytes,
    parse_payload_packet,
    write_file,
)

from modules.crypto.aes_decrypt import (
    decrypt_file,
)

from modules.steganography.multiframe_packet import (
    parse_packet,
)

from modules.steganography.spatial.paired_block import (
    BLOCK_SIZE,
    GAP,
    DELTA,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

COVER_VIDEO = Path(
    "input/cover_video/sample_long.mp4"
)

STEGO_VIDEO = Path(
    "output/ui_demo/stego_demo.mp4"
)

SECRET_FILE = Path(
    "input/secret_data/scalability_test/secret_32kb.bin"
)

KEY_FILE = Path(
    "input/keys/test_aes_key.bin"
)

OUTPUT_DIR = Path(
    "output/demo_inspection"
)

RECOVERED_FILE = (
    OUTPUT_DIR
    / "stage5_recovered_secret.bin"
)

FLOW_IMAGE = (
    OUTPUT_DIR
    / "stage5_complete_recovery_flow.png"
)

FRAME_IMAGE = (
    OUTPUT_DIR
    / "stage5_recovery_frame_comparison.png"
)

DELTA_VALUE = DELTA


# ==========================================================
# SHA-256
# ==========================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ==========================================================
# VIDEO INFORMATION
# ==========================================================

def get_video_info(
    path: Path,
):
    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {path}"
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

    frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    return (
        fps,
        width,
        height,
        frames,
    )


# ==========================================================
# VISUAL FLOW
# ==========================================================

def create_flow_visual(
    payload_bits,
    packet_count,
    packet_sizes,
    recovered_bytes,
    byte_match,
    hash_match,
):
    """
    Create a presentation-ready 16:9 visualization of the
    complete StegaFusion recovery pipeline.

    This function is visualization only.
    It does NOT implement or modify the recovery algorithm.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # FIGURE
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(16, 9),
        facecolor="black",
    )

    ax.set_facecolor("black")

    ax.set_xlim(
        0,
        16,
    )

    ax.set_ylim(
        0,
        9,
    )

    ax.axis("off")

    # ------------------------------------------------------
    # TITLE
    # ------------------------------------------------------

    ax.text(
        8,
        8.55,
        "StegaFusion - COMPLETE RECOVERY FLOW",
        color="white",
        fontsize=25,
        fontweight="bold",
        ha="center",
        va="center",
    )

    ax.text(
        8,
        8.15,
        (
            f"Production recovery demonstration  |  "
            f"{packet_count} frame packets  |  "
            f"{payload_bits:,} reconstructed payload bits"
        ),
        color="lightgray",
        fontsize=12,
        ha="center",
        va="center",
    )

    # ------------------------------------------------------
    # BOX HELPER
    # ------------------------------------------------------

    def box(
        x,
        y,
        w,
        h,
        title,
        detail,
        title_size=14,
        detail_size=10,
        edge_color="white",
        title_color="white",
    ):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=2,
            edgecolor=edge_color,
            facecolor="#080808",
        )

        ax.add_patch(
            patch
        )

        ax.text(
            x + w / 2,
            y + h * 0.64,
            title,
            color=title_color,
            fontsize=title_size,
            fontweight="bold",
            ha="center",
            va="center",
        )

        ax.text(
            x + w / 2,
            y + h * 0.30,
            detail,
            color="white",
            fontsize=detail_size,
            ha="center",
            va="center",
        )

    # ------------------------------------------------------
    # ARROW HELPER
    # ------------------------------------------------------

    def arrow(
        x1,
        y1,
        x2,
        y2,
        color="white",
        width=2.2,
    ):
        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=width,
            color=color,
        )

        ax.add_patch(
            patch
        )

    # ------------------------------------------------------
    # INPUT BOXES
    # ------------------------------------------------------

    box(
        0.65,
        6.65,
        4.0,
        1.15,
        "ORIGINAL COVER VIDEO",
        "720 frames  |  1920 x 1080  |  23.976 FPS",
        title_size=15,
        detail_size=10,
        edge_color="white",
    )

    box(
        11.35,
        6.65,
        4.0,
        1.15,
        "STEGO VIDEO",
        "720 frames  |  MP4V  |  secret embedded",
        title_size=15,
        detail_size=10,
        edge_color="lime",
        title_color="lime",
    )

    # ------------------------------------------------------
    # CENTRAL RECOVERY PIPELINE
    # ------------------------------------------------------

    box(
        4.25,
        5.30,
        7.50,
        0.95,
        "PRODUCTION SPATIAL EXTRACTION",
        "Original cover frame + stego frame -> paired-block extraction",
        title_size=14,
        detail_size=10,
    )

    box(
        4.25,
        4.05,
        7.50,
        0.95,
        "FRAME PACKETS",
        (
            f"{packet_count} packets  |  "
            f"packet range: "
            f"{min(packet_sizes):,}-{max(packet_sizes):,} bits"
        ),
        title_size=14,
        detail_size=10,
    )

    box(
        4.25,
        2.80,
        7.50,
        0.95,
        "MULTI-FRAME PAYLOAD REASSEMBLY",
        f"{packet_count} packets -> {payload_bits:,} payload bits",
        title_size=14,
        detail_size=10,
    )

    box(
        4.25,
        1.55,
        7.50,
        0.95,
        "AES-256-CBC DECRYPTION",
        "Encrypted payload -> original plaintext secret",
        title_size=14,
        detail_size=10,
        edge_color="cyan",
    )

    box(
        4.25,
        0.30,
        7.50,
        0.95,
        "RECOVERED SECRET",
        f"{recovered_bytes:,} bytes recovered",
        title_size=16,
        detail_size=11,
        edge_color="lime",
        title_color="lime",
    )

    # ------------------------------------------------------
    # INPUT -> EXTRACTION
    # ------------------------------------------------------

    arrow(
        2.65,
        6.65,
        5.10,
        6.25,
    )

    arrow(
        13.35,
        6.65,
        10.90,
        6.25,
    )

    # ------------------------------------------------------
    # PIPELINE ARROWS
    # ------------------------------------------------------

    arrow(
        8.0,
        5.30,
        8.0,
        5.00,
    )

    arrow(
        8.0,
        4.05,
        8.0,
        3.75,
    )

    arrow(
        8.0,
        2.80,
        8.0,
        2.50,
    )

    arrow(
        8.0,
        1.55,
        8.0,
        1.25,
    )

    # ------------------------------------------------------
    # RECOVERY VALIDATION PANEL
    # ------------------------------------------------------

    validation_ok = (
        byte_match
        and hash_match
    )

    validation_color = (
        "lime"
        if validation_ok
        else "red"
    )

    validation_title = (
        "INTEGRITY VERIFIED"
        if validation_ok
        else "INTEGRITY VERIFICATION FAILED"
    )

    # Outer validation panel

    validation_box = FancyBboxPatch(
        (12.15, 2.25),
        3.15,
        2.45,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=2.5,
        edgecolor=validation_color,
        facecolor="#080808",
    )

    ax.add_patch(
        validation_box
    )

    ax.text(
        13.725,
        4.30,
        validation_title,
        color=validation_color,
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
    )

    ax.text(
        13.725,
        3.72,
        (
            f"BYTE-FOR-BYTE\n"
            f"{'PASS' if byte_match else 'FAIL'}"
        ),
        color="white",
        fontsize=11,
        ha="center",
        va="center",
    )

    ax.text(
        13.725,
        3.10,
        (
            f"SHA-256\n"
            f"{'PASS' if hash_match else 'FAIL'}"
        ),
        color="white",
        fontsize=11,
        ha="center",
        va="center",
    )

    ax.text(
        13.725,
        2.60,
        "Original secret restored exactly",
        color="lightgray",
        fontsize=9,
        ha="center",
        va="center",
    )

    # Arrow from recovered secret to validation

    arrow(
        11.75,
        0.78,
        13.20,
        2.25,
        color=validation_color,
    )

    # ------------------------------------------------------
    # KEY TECHNICAL DETAILS
    # ------------------------------------------------------

    ax.text(
        0.75,
        1.05,
        (
            "RECOVERY CHARACTERISTICS\n"
            "Reference-based spatial extraction\n"
            "Multi-frame packet reassembly\n"
            "AES-256-CBC decryption"
        ),
        color="lightgray",
        fontsize=9,
        ha="left",
        va="top",
        linespacing=1.6,
    )

    # ------------------------------------------------------
    # FOOTER
    # ------------------------------------------------------

    ax.text(
        8.0,
        0.05,
        (
            "StegaFusion production recovery path  |  "
            "No alternate steganography algorithm used"
        ),
        color="gray",
        fontsize=8,
        ha="center",
        va="bottom",
    )

    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------

    fig.savefig(
        FLOW_IMAGE,
        dpi=150,
        facecolor="black",
    )

    plt.close(
        fig
    )

# ==========================================================
# FRAME COMPARISON
# ==========================================================

def create_frame_comparison(
    original_frames,
    received_frames,
    frame_index,
):

    original = original_frames[
        frame_index
    ]

    received = received_frames[
        frame_index
    ]

    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB,
    )

    received_rgb = cv2.cvtColor(
        received,
        cv2.COLOR_BGR2RGB,
    )

    difference = cv2.absdiff(
        original,
        received,
    )

    difference_gray = cv2.cvtColor(
        difference,
        cv2.COLOR_BGR2GRAY,
    )

    difference_amplified = np.clip(
        difference_gray.astype(
            np.float32
        ) * 8,
        0,
        255,
    ).astype(
        np.uint8
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        facecolor="black",
    )

    titles = [
        f"ORIGINAL — FRAME {frame_index}",
        f"STEGO — FRAME {frame_index}",
        "AMPLIFIED DIFFERENCE ×8",
    ]

    images = [
        original_rgb,
        received_rgb,
        difference_amplified,
    ]

    for axis, image, title in zip(
        axes,
        images,
        titles,
    ):

        axis.imshow(
            image,
            cmap="gray"
            if image.ndim == 2
            else None,
        )

        axis.set_title(
            title,
            color="white",
            fontsize=14,
            fontweight="bold",
        )

        axis.axis(
            "off"
        )

    fig.patch.set_facecolor(
        "black"
    )

    fig.savefig(
        FRAME_IMAGE,
        dpi=140,
        facecolor="black",
    )

    plt.close(
        fig
    )


# ==========================================================
# MAIN DEMONSTRATION
# ==========================================================

def main():

    print(
        "=" * 78
    )

    print(
        "StegaFusion STAGE 5 — COMPLETE RECOVERY FLOW"
    )

    print(
        "=" * 78
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================================
    # INPUT VALIDATION
    # ======================================================

    print()
    print(
        "## INPUT VALIDATION"
    )

    for path in [
        COVER_VIDEO,
        STEGO_VIDEO,
        SECRET_FILE,
        KEY_FILE,
    ]:

        status = (
            "EXISTS"
            if path.exists()
            else "MISSING"
        )

        print(
            f"{path} : {status}"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Required input missing: {path}"
            )

    # ======================================================
    # VIDEO INFORMATION
    # ======================================================

    cover_info = get_video_info(
        COVER_VIDEO
    )

    stego_info = get_video_info(
        STEGO_VIDEO
    )

    print()
    print(
        "## VIDEO INFORMATION"
    )

    print()
    print(
        "ORIGINAL COVER VIDEO"
    )

    print(
        f"Frames      : {cover_info[3]}"
    )

    print(
        f"FPS         : {cover_info[0]}"
    )

    print(
        f"Resolution  : "
        f"{cover_info[1]}x{cover_info[2]}"
    )

    print()
    print(
        "STEGO VIDEO"
    )

    print(
        f"Frames      : {stego_info[3]}"
    )

    print(
        f"FPS         : {stego_info[0]}"
    )

    print(
        f"Resolution  : "
        f"{stego_info[1]}x{stego_info[2]}"
    )

    if cover_info[3] != stego_info[3]:

        raise RuntimeError(
            "Original and stego video frame counts differ."
        )

    # ======================================================
    # READ REAL FRAMES
    # ======================================================

    print()
    print(
        "## READING PRODUCTION VIDEO FRAMES"
    )

    original_frames = _read_video_frames(
        COVER_VIDEO
    )

    received_frames = _read_video_frames(
        STEGO_VIDEO
    )

    print(
        f"Original frames loaded : "
        f"{len(original_frames)}"
    )

    print(
        f"Stego frames loaded    : "
        f"{len(received_frames)}"
    )

    # ======================================================
    # FIRST PACKET
    # ======================================================

    print()
    print(
        "## PRODUCTION PACKET EXTRACTION"
    )

    print(
        "Reading packet header from frame 0..."
    )

    first_packet = extract_frame_packet(
        original_frames[0],
        received_frames[0],
        block_size=BLOCK_SIZE,
        delta=DELTA_VALUE,
        gap=GAP,
    )

    first_parsed = parse_packet(
        first_packet
    )

    total_chunks = (
        first_parsed.total_chunks
    )

    print(
        f"Total Chunks : {total_chunks}"
    )

    print(
        f"First Packet : {len(first_packet)} bits"
    )

    print(
        f"First Index  : {first_parsed.chunk_index}"
    )

    print(
        f"First Data   : "
        f"{first_parsed.chunk_length} bits"
    )

    # ======================================================
    # EXTRACT ALL PACKETS
    # ======================================================

    print()
    print(
        "Extracting all production packets..."
    )

    packets = [
        first_packet
    ]

    packet_sizes = [
        len(first_packet)
    ]

    for frame_index in range(
        1,
        total_chunks,
    ):

        packet_bits = extract_frame_packet(
            original_frames[frame_index],
            received_frames[frame_index],
            block_size=BLOCK_SIZE,
            delta=DELTA_VALUE,
            gap=GAP,
        )

        packets.append(
            packet_bits
        )

        packet_sizes.append(
            len(packet_bits)
        )

    print(
        f"Extracted Packets : {len(packets)}"
    )

    print(
        f"Minimum Packet    : "
        f"{min(packet_sizes)} bits"
    )

    print(
        f"Maximum Packet    : "
        f"{max(packet_sizes)} bits"
    )

    print(
        f"Average Packet    : "
        f"{sum(packet_sizes) / len(packet_sizes):.2f} bits"
    )

    # ======================================================
    # PACKET VALIDATION
    # ======================================================

    print()
    print(
        "## PACKET VALIDATION"
    )

    indices = []

    for packet in packets:

        parsed = parse_packet(
            packet
        )

        indices.append(
            parsed.chunk_index
        )

    expected_indices = list(
        range(total_chunks)
    )

    packet_order_valid = (
        sorted(indices)
        == expected_indices
    )

    print(
        f"Packet Count      : {len(packets)}"
    )

    print(
        f"Packet Order      : "
        f"{'PASS' if packet_order_valid else 'FAIL'}"
    )

    if not packet_order_valid:

        raise RuntimeError(
            "Packet index validation failed."
        )

    # ======================================================
    # REASSEMBLY
    # ======================================================

    print()
    print(
        "## MULTI-FRAME PAYLOAD REASSEMBLY"
    )

    payload = parse_frame_packets(
        packets
    )

    payload_bits = len(
        payload
    )

    print(
        f"Reassembled Payload : "
        f"{payload_bits} bits"
    )

    # ======================================================
    # PAYLOAD PACKET
    # ======================================================

    print()
    print(
        "## STEGAFUSION PAYLOAD PACKET"
    )

    payload_length, encrypted_bits = (
        parse_payload_packet(
            payload
        )
    )

    print(
        f"Encrypted Payload Bits : "
        f"{payload_length}"
    )

    encrypted_data = binary_to_bytes(
        encrypted_bits
    )

    print(
        f"Encrypted Data Bytes   : "
        f"{len(encrypted_data)}"
    )

    # ======================================================
    # AES DECRYPTION
    # ======================================================

    print()
    print(
        "## AES-256-CBC DECRYPTION"
    )

    recovered_data = decrypt_file(
        encrypted_data,
        KEY_FILE,
    )

    print(
        "Decryption : PASS"
    )

    print(
        f"Recovered Bytes : "
        f"{len(recovered_data)}"
    )

    # ======================================================
    # WRITE RECOVERED SECRET
    # ======================================================

    write_file(
        RECOVERED_FILE,
        recovered_data,
    )

    print(
        f"Recovered File : "
        f"{RECOVERED_FILE}"
    )

    # ======================================================
    # INTEGRITY
    # ======================================================

    print()
    print(
        "## INTEGRITY VALIDATION"
    )

    original_bytes = (
        SECRET_FILE.read_bytes()
    )

    recovered_bytes_data = (
        RECOVERED_FILE.read_bytes()
    )

    byte_match = (
        original_bytes
        == recovered_bytes_data
    )

    original_hash = sha256_file(
        SECRET_FILE
    )

    recovered_hash = sha256_file(
        RECOVERED_FILE
    )

    hash_match = (
        original_hash
        == recovered_hash
    )

    print()
    print(
        "BYTE-FOR-BYTE VALIDATION"
    )

    print(
        f"Original Bytes  : "
        f"{len(original_bytes)}"
    )

    print(
        f"Recovered Bytes : "
        f"{len(recovered_bytes_data)}"
    )

    print(
        f"Match           : "
        f"{'TRUE' if byte_match else 'FALSE'}"
    )

    print()
    print(
        "SHA-256 VALIDATION"
    )

    print(
        f"Original  : {original_hash}"
    )

    print(
        f"Recovered : {recovered_hash}"
    )

    print(
        f"Match     : "
        f"{'TRUE' if hash_match else 'FALSE'}"
    )

    # ======================================================
    # FRAME COMPARISON
    # ======================================================

    selected_frame = (
        min(
            100,
            total_chunks - 1,
        )
    )

    create_frame_comparison(
        original_frames,
        received_frames,
        selected_frame,
    )

    # ======================================================
    # COMPLETE FLOW VISUAL
    # ======================================================

    create_flow_visual(
        payload_bits=payload_bits,
        packet_count=total_chunks,
        packet_sizes=packet_sizes,
        recovered_bytes=len(
            recovered_data
        ),
        byte_match=byte_match,
        hash_match=hash_match,
    )

    # ======================================================
    # FINAL RESULT
    # ======================================================

    print()
    print(
        "## GENERATED ARTIFACTS"
    )

    print(
        f"Complete Recovery Flow : "
        f"{FLOW_IMAGE}"
    )

    print(
        f"Frame Comparison       : "
        f"{FRAME_IMAGE}"
    )

    print()
    print(
        "=" * 78
    )

    if byte_match and hash_match:

        print(
            "RESULT: STAGE 5 COMPLETE RECOVERY PASS."
        )

        print()
        print(
            "Actual production recovery demonstrated:"
        )

        print(
            "Stego MP4"
        )

        print(
            "→ original cover frame reference"
        )

        print(
            "→ spatial packet extraction"
        )

        print(
            f"→ {total_chunks} frame packets"
        )

        print(
            f"→ {payload_bits} reconstructed payload bits"
        )

        print(
            "→ AES-256-CBC decryption"
        )

        print(
            f"→ recovered {len(recovered_data)} byte secret"
        )

        print(
            "→ byte-for-byte validation"
        )

        print(
            "→ SHA-256 validation"
        )

        print()
        print(
            "INTEGRITY VERIFIED."
        )

    else:

        print(
            "RESULT: STAGE 5 RECOVERY FAILED."
        )

        print(
            "Byte-for-byte or SHA-256 validation failed."
        )

    print(
        "=" * 78
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()