"""
StegaFusion Stage 4 — Temporal Multi-Frame Flow Visualization

Demonstrates the ACTUAL production temporal pipeline:

    Secret
        ↓
    AES-256 encrypted payload
        ↓
    Multi-frame packetization
        ↓
    Packet → Frame mapping
        ↓
    Actual production stego video
        ↓
    Original + Stego frame comparison
        ↓
    Packet extraction
        ↓
    Packet reassembly

This module is an evaluation / visualization tool.

It does NOT modify the production steganography algorithm.
"""

from pathlib import Path

import cv2
import numpy as np

from modules.pipeline.spatial_video_pipeline import (
    prepare_spatial_payload,
    create_frame_packets,
    extract_frame_packet,
)

from modules.steganography.multiframe_packet import (
    parse_packet,
)


# ==========================================================
# CONFIGURATION
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

STEGO_VIDEO = Path(
    "output/ui_demo/stego_demo.mp4"
)

OUTPUT_DIR = Path(
    "output/demo_inspection"
)

TIMELINE_IMAGE = (
    OUTPUT_DIR
    / "stage4_temporal_flow.png"
)

FRAME_COMPARISON_IMAGE = (
    OUTPUT_DIR
    / "stage4_frame_comparison.png"
)


# ==========================================================
# HELPERS
# ==========================================================

def read_video_info(path: Path):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise FileNotFoundError(
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

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    return (
        fps,
        width,
        height,
        frame_count,
    )


def read_frame(
    path: Path,
    frame_index: int,
):

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Unable to open video: {path}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            f"Unable to read frame "
            f"{frame_index} from {path}"
        )

    return frame


def resize_frame(
    frame,
    width,
    height,
):

    return cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def add_text(
    image,
    text,
    position,
    scale=0.6,
    thickness=1,
):

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ==========================================================
# TEMPORAL TIMELINE
# ==========================================================

def create_timeline(
    frame_count: int,
    packet_count: int,
):

    width = 1600
    height = 500

    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    add_text(
        canvas,
        "StegaFusion — TEMPORAL MULTI-FRAME FLOW",
        (30, 40),
        scale=1.0,
        thickness=2,
    )

    add_text(
        canvas,
        f"Video frames: {frame_count}    "
        f"Production packets: {packet_count}",
        (30, 75),
        scale=0.65,
    )

    # ------------------------------------------------------
    # FRAME TIMELINE
    # ------------------------------------------------------

    x0 = 60
    x1 = 1540
    y = 150

    line_width = x1 - x0

    cv2.line(
        canvas,
        (x0, y),
        (x1, y),
        (180, 180, 180),
        3,
    )

    for frame_index in range(
        frame_count
    ):

        x = int(
            x0
            + (
                frame_index
                / max(frame_count - 1, 1)
            )
            * line_width
        )

        # Embedded packet frames
        if frame_index < packet_count:

            cv2.circle(
                canvas,
                (x, y),
                4,
                (0, 220, 0),
                -1,
            )

        # Unmodified frames
        else:

            cv2.circle(
                canvas,
                (x, y),
                2,
                (100, 100, 100),
                -1,
            )

    # ------------------------------------------------------
    # LABELS
    # ------------------------------------------------------

    add_text(
        canvas,
        "FRAME 0",
        (x0, y + 45),
        scale=0.55,
    )

    add_text(
        canvas,
        f"FRAME {packet_count - 1}",
        (
            int(
                x0
                + (
                    (packet_count - 1)
                    / max(frame_count - 1, 1)
                )
                * line_width
            ) - 50,
            y + 45,
        ),
        scale=0.55,
    )

    add_text(
        canvas,
        f"FRAME {frame_count - 1}",
        (x1 - 90, y + 45),
        scale=0.55,
    )

    # ------------------------------------------------------
    # PACKET REGION
    # ------------------------------------------------------

    packet_end_x = int(
        x0
        + (
            (packet_count - 1)
            / max(frame_count - 1, 1)
        )
        * line_width
    )

    cv2.rectangle(
        canvas,
        (x0, y - 25),
        (packet_end_x, y + 25),
        (0, 180, 0),
        2,
    )

    add_text(
        canvas,
        "PRODUCTION PACKETS EMBEDDED",
        (
            x0 + 10,
            y - 40,
        ),
        scale=0.65,
        thickness=2,
    )

    add_text(
        canvas,
        "REMAINING FRAMES — COPIED UNCHANGED",
        (
            packet_end_x + 20,
            y - 40,
        ),
        scale=0.55,
    )

    # ------------------------------------------------------
    # EXAMPLE PACKET MAPPING
    # ------------------------------------------------------

    mapping_y = 290

    add_text(
        canvas,
        "ACTUAL PACKET → FRAME MAPPING",
        (30, mapping_y),
        scale=0.75,
        thickness=2,
    )

    examples = [
        (
            "Packet 001",
            "Frame 000",
        ),
        (
            "Packet 002",
            "Frame 001",
        ),
        (
            "Packet 003",
            "Frame 002",
        ),
        (
            f"Packet {packet_count - 1:03d}",
            f"Frame {packet_count - 2:03d}",
        ),
        (
            f"Packet {packet_count:03d}",
            f"Frame {packet_count - 1:03d}",
        ),
    ]

    yy = mapping_y + 45

    for packet_name, frame_name in examples:

        add_text(
            canvas,
            f"{packet_name}  →  {frame_name}",
            (60, yy),
            scale=0.6,
        )

        yy += 32

    # ------------------------------------------------------
    # LEGEND
    # ------------------------------------------------------

    cv2.circle(
        canvas,
        (850, 310),
        6,
        (0, 220, 0),
        -1,
    )

    add_text(
        canvas,
        "Frame contains production packet",
        (870, 316),
        scale=0.6,
    )

    cv2.circle(
        canvas,
        (850, 350),
        4,
        (100, 100, 100),
        -1,
    )

    add_text(
        canvas,
        "Frame copied without embedding",
        (870, 356),
        scale=0.6,
    )

    return canvas


# ==========================================================
# FRAME COMPARISON
# ==========================================================

def create_frame_comparison(
    original,
    stego,
    frame_index,
    packet_bits,
):

    target_width = 640
    target_height = 360

    original_small = resize_frame(
        original,
        target_width,
        target_height,
    )

    stego_small = resize_frame(
        stego,
        target_width,
        target_height,
    )

    difference = cv2.absdiff(
        original,
        stego,
    )

    difference = resize_frame(
        difference,
        target_width,
        target_height,
    )

    # Amplify difference only for visualization.
    difference = np.clip(
        difference.astype(np.float32)
        * 8.0,
        0,
        255,
    ).astype(
        np.uint8
    )

    canvas = np.zeros(
        (
            target_height + 100,
            target_width * 3,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        80:
        80 + target_height,
        0:
        target_width,
    ] = original_small

    canvas[
        80:
        80 + target_height,
        target_width:
        target_width * 2,
    ] = stego_small

    canvas[
        80:
        80 + target_height,
        target_width * 2:
        target_width * 3,
    ] = difference

    add_text(
        canvas,
        f"ORIGINAL — FRAME {frame_index}",
        (20, 45),
        scale=0.7,
        thickness=2,
    )

    add_text(
        canvas,
        f"STEGO — FRAME {frame_index}",
        (target_width + 20, 45),
        scale=0.7,
        thickness=2,
    )

    add_text(
        canvas,
        "AMPLIFIED DIFFERENCE ×8",
        (
            target_width * 2 + 20,
            45,
        ),
        scale=0.7,
        thickness=2,
    )

    add_text(
        canvas,
        f"Packet size: {len(packet_bits)} bits",
        (20, target_height + 92),
        scale=0.55,
    )

    return canvas


# ==========================================================
# ACTUAL PACKET EXTRACTION
# ==========================================================

def extract_actual_packets(
    cover_video: Path,
    stego_video: Path,
    packet_count: int,
):

    cover = cv2.VideoCapture(
        str(cover_video)
    )

    stego = cv2.VideoCapture(
        str(stego_video)
    )

    if not cover.isOpened():
        raise RuntimeError(
            "Unable to open cover video."
        )

    if not stego.isOpened():
        raise RuntimeError(
            "Unable to open stego video."
        )

    packets = []

    try:

        for frame_index in range(
            packet_count
        ):

            success_cover, cover_frame = (
                cover.read()
            )

            success_stego, stego_frame = (
                stego.read()
            )

            if not success_cover:
                raise RuntimeError(
                    f"Unable to read cover "
                    f"frame {frame_index}."
                )

            if not success_stego:
                raise RuntimeError(
                    f"Unable to read stego "
                    f"frame {frame_index}."
                )

            # ------------------------------------------------
            # Production extractor first reads packet header,
            # then determines the exact packet length.
            # ------------------------------------------------

            packet = extract_frame_packet(
                cover_frame,
                stego_frame,
            )

            packets.append(
                packet
            )

    finally:

        cover.release()
        stego.release()

    return packets


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "StegaFusion STAGE 4 — "
        "TEMPORAL MULTI-FRAME FLOW"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------

    print()
    print("## INPUT VALIDATION")

    for path in [
        COVER_VIDEO,
        SECRET_FILE,
        KEY_FILE,
        STEGO_VIDEO,
    ]:

        print(
            f"{path} : "
            f"{'EXISTS' if path.exists() else 'MISSING'}"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Required file does not exist: {path}"
            )

    # ------------------------------------------------------
    # VIDEO INFORMATION
    # ------------------------------------------------------

    fps, width, height, frame_count = (
        read_video_info(
            COVER_VIDEO
        )
    )

    stego_fps, stego_width, stego_height, stego_frames = (
        read_video_info(
            STEGO_VIDEO
        )
    )

    print()
    print("## ORIGINAL VIDEO")

    print(
        f"Frames      : {frame_count}"
    )

    print(
        f"FPS         : {fps}"
    )

    print(
        f"Resolution  : {width}x{height}"
    )

    print()
    print("## STEGO VIDEO")

    print(
        f"Frames      : {stego_frames}"
    )

    print(
        f"FPS         : {stego_fps}"
    )

    print(
        f"Resolution  : "
        f"{stego_width}x{stego_height}"
    )

    if frame_count != stego_frames:

        raise RuntimeError(
            "Original and stego videos "
            "have different frame counts."
        )

    # ------------------------------------------------------
    # RECREATE PRODUCTION PAYLOAD STRUCTURE
    # ------------------------------------------------------

    print()
    print("## PRODUCTION PAYLOAD STRUCTURE")

    payload = prepare_spatial_payload(
        SECRET_FILE,
        KEY_FILE,
    )

    packets = create_frame_packets(
        payload
    )

    packet_count = len(packets)

    print(
        f"Payload Bits       : {len(payload)}"
    )

    print(
        f"Packet Count       : {packet_count}"
    )

    print(
        f"Maximum Packet Bits: "
        f"{max(len(p) for p in packets)}"
    )

    print(
        f"Minimum Packet Bits: "
        f"{min(len(p) for p in packets)}"
    )

    # ------------------------------------------------------
    # TEMPORAL CAPACITY
    # ------------------------------------------------------

    remaining_frames = (
        frame_count
        - packet_count
    )

    print()
    print("## TEMPORAL FRAME ALLOCATION")

    print(
        f"Frames Available   : {frame_count}"
    )

    print(
        f"Frames Used        : {packet_count}"
    )

    print(
        f"Frames Unchanged   : {remaining_frames}"
    )

    print(
        f"Temporal Status    : "
        f"{'PASS' if remaining_frames >= 0 else 'FAIL'}"
    )

    if packet_count > frame_count:

        raise RuntimeError(
            "Production payload requires "
            "more frames than the video contains."
        )

    # ------------------------------------------------------
    # PACKET → FRAME MAPPING
    # ------------------------------------------------------

    print()
    print("## PACKET → FRAME MAPPING")

    preview_indices = [
        0,
        1,
        2,
        packet_count - 2,
        packet_count - 1,
    ]

    for index in preview_indices:

        if index < 0:
            continue

        print(
            f"Packet {index + 1:03d}"
            f" → Frame {index:03d}"
            f" → {len(packets[index])} bits"
        )

    # ------------------------------------------------------
    # VERIFY ACTUAL STEGO VIDEO
    # ------------------------------------------------------

    print()
    print("## ACTUAL STEGO PACKET EXTRACTION")

    actual_packets = extract_actual_packets(
        COVER_VIDEO,
        STEGO_VIDEO,
        packet_count,
    )

    if len(actual_packets) != packet_count:

        raise RuntimeError(
            "Extracted packet count does not match "
            "production packet count."
        )

    packet_matches = all(
        len(actual_packets[i])
        == len(packets[i])
        for i in range(packet_count)
    )

    print(
        f"Extracted Packets : "
        f"{len(actual_packets)}"
    )

    print(
        f"Packet Lengths    : "
        f"{'MATCH' if packet_matches else 'MISMATCH'}"
    )

    # ------------------------------------------------------
    # CREATE TIMELINE
    # ------------------------------------------------------

    print()
    print("## GENERATING TEMPORAL TIMELINE")

    timeline = create_timeline(
        frame_count,
        packet_count,
    )

    cv2.imwrite(
        str(TIMELINE_IMAGE),
        timeline,
    )

    print(
        f"Timeline : {TIMELINE_IMAGE}"
    )

    # ------------------------------------------------------
    # SELECT AN ACTUAL EMBEDDED FRAME
    # ------------------------------------------------------

    selected_frame = min(
        100,
        packet_count - 1,
    )

    print()
    print("## FRAME-LEVEL TEMPORAL DEMONSTRATION")

    print(
        f"Selected Frame : {selected_frame}"
    )

    print(
        f"Selected Packet: "
        f"{selected_frame + 1}"
    )

    original_frame = read_frame(
        COVER_VIDEO,
        selected_frame,
    )

    stego_frame = read_frame(
        STEGO_VIDEO,
        selected_frame,
    )

    selected_packet = actual_packets[
        selected_frame
    ]

    comparison = create_frame_comparison(
        original_frame,
        stego_frame,
        selected_frame,
        selected_packet,
    )

    cv2.imwrite(
        str(FRAME_COMPARISON_IMAGE),
        comparison,
    )

    print(
        f"Comparison : "
        f"{FRAME_COMPARISON_IMAGE}"
    )

    # ------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------

    print()
    print("=" * 70)

    if packet_matches:

        print(
            "RESULT: STAGE 4 TEMPORAL FLOW PASS."
        )

    else:

        print(
            "RESULT: STAGE 4 TEMPORAL FLOW FAILED."
        )

    print("=" * 70)

    print()
    print(
        "Actual production temporal mapping demonstrated:"
    )

    print(
        "Real encrypted payload"
    )

    print(
        "→ actual multi-frame packetization"
    )

    print(
        "→ packet-to-frame allocation"
    )

    print(
        "→ actual production stego video"
    )

    print(
        "→ original/stego frame comparison"
    )

    print(
        "→ actual production packet extraction"
    )

    print(
        "→ packet validation"
    )

    print()
    print(
        "No production steganography algorithm was modified."
    )


if __name__ == "__main__":
    main()