"""
StegaFusion Capacity Analysis

Provides a read-only capacity estimator for the production
multi-frame spatial video steganography pipeline.

This module does NOT modify the production embedding algorithm.

It reuses the same:
    - AES payload preparation
    - frame packetization
    - spatial capacity calculation

used by the production pipeline.
"""

from pathlib import Path
from dataclasses import dataclass
import cv2

from modules.pipeline.spatial_video_pipeline import (
    prepare_spatial_payload,
    create_frame_packets,
)

from modules.steganography.spatial.paired_block import (
    calculate_capacity,
    BLOCK_SIZE,
    GAP,
    DELTA,
)


@dataclass
class CapacityReport:
    """Capacity analysis result."""

    video: Path
    secret_file: Path

    frames: int
    fps: float
    width: int
    height: int

    secret_bytes: int
    payload_bits: int

    packet_bits: int
    packet_count: int

    spatial_capacity_bits: int

    required_frames: int
    available_frames: int
    remaining_frames: int

    block_size: int
    gap: int
    delta: int

    fits_video: bool


def analyze_capacity(
    video: Path,
    secret_file: Path,
    key_file: Path,
    block_size: int = BLOCK_SIZE,
    gap: int = GAP,
    delta: int = DELTA,
) -> CapacityReport:
    """
    Analyze whether a secret file can fit into a video.

    The calculation follows the same production payload
    preparation and packetization path used by
    embed_spatial_video().

    No video is modified.
    No payload is embedded.
    """

    video = Path(video)
    secret_file = Path(secret_file)
    key_file = Path(key_file)

    if not video.exists():
        raise FileNotFoundError(
            f"Cover video not found: {video}"
        )

    if not secret_file.exists():
        raise FileNotFoundError(
            f"Secret file not found: {secret_file}"
        )

    if not key_file.exists():
        raise FileNotFoundError(
            f"AES key not found: {key_file}"
        )

    if delta <= 0:
        raise ValueError(
            "delta must be positive."
        )

    # ------------------------------------------------------
    # READ VIDEO METADATA
    # ------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video}"
        )

    try:
        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
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

        success, first_frame = cap.read()

    finally:
        cap.release()

    if fps <= 0:
        fps = 0.0

    if frame_count <= 0:
        raise ValueError(
            "Video contains no readable frames."
        )

    if not success or first_frame is None:
        raise RuntimeError(
            "Unable to read the first video frame."
        )

    # ------------------------------------------------------
    # PREPARE THE REAL PRODUCTION PAYLOAD
    # ------------------------------------------------------

    payload = prepare_spatial_payload(
        secret_file,
        key_file,
    )

    payload_bits = len(payload)

    # ------------------------------------------------------
    # CREATE THE REAL PRODUCTION PACKETS
    # ------------------------------------------------------

    packets = create_frame_packets(
        payload
    )

    packet_count = len(packets)

    if packet_count == 0:
        raise ValueError(
            "Production packetization produced no packets."
        )

    # ------------------------------------------------------
    # DETERMINE PACKET SIZE
    # ------------------------------------------------------

    packet_bits = max(
        len(packet)
        for packet in packets
    )

    # ------------------------------------------------------
    # DETERMINE SPATIAL CAPACITY
    # ------------------------------------------------------

    spatial_capacity_bits = calculate_capacity(
        first_frame,
        block_size=block_size,
        gap=gap,
    )

    # ------------------------------------------------------
    # VALIDATE EVERY PACKET AGAINST CAPACITY
    # ------------------------------------------------------

    oversized_packets = [
        len(packet)
        for packet in packets
        if len(packet) > spatial_capacity_bits
    ]

    if oversized_packets:
        raise ValueError(
            "One or more production packets exceed "
            "the spatial capacity of the video frame.\n"
            f"Maximum packet : {max(oversized_packets)} bits\n"
            f"Capacity       : {spatial_capacity_bits} bits"
        )

    # ------------------------------------------------------
    # FRAME REQUIREMENT
    # ------------------------------------------------------

    required_frames = packet_count

    remaining_frames = (
        frame_count - required_frames
    )

    fits_video = (
        required_frames <= frame_count
    )

    # ------------------------------------------------------
    # RETURN REPORT
    # ------------------------------------------------------

    return CapacityReport(
        video=video,
        secret_file=secret_file,

        frames=frame_count,
        fps=fps,
        width=width,
        height=height,

        secret_bytes=secret_file.stat().st_size,
        payload_bits=payload_bits,

        packet_bits=packet_bits,
        packet_count=packet_count,

        spatial_capacity_bits=spatial_capacity_bits,

        required_frames=required_frames,
        available_frames=frame_count,
        remaining_frames=remaining_frames,

        block_size=block_size,
        gap=gap,
        delta=delta,

        fits_video=fits_video,
    )


def print_capacity_report(
    report: CapacityReport,
) -> None:
    """
    Print a human-readable capacity report.
    """

    print()
    print("=" * 70)
    print(
        "StegaFusion CAPACITY ANALYSIS"
    )
    print("=" * 70)

    print()
    print("## VIDEO")

    print(
        f"Video              : "
        f"{report.video}"
    )

    print(
        f"Frames             : "
        f"{report.frames}"
    )

    print(
        f"FPS                : "
        f"{report.fps}"
    )

    print(
        f"Resolution         : "
        f"{report.width}x{report.height}"
    )

    duration = (
        report.frames / report.fps
        if report.fps > 0
        else 0.0
    )

    print(
        f"Duration           : "
        f"{duration:.2f} seconds"
    )

    print()
    print("## SECRET")

    print(
        f"Secret File        : "
        f"{report.secret_file}"
    )

    print(
        f"Secret Size        : "
        f"{report.secret_bytes} bytes"
    )

    print(
        f"Encrypted Payload  : "
        f"{report.payload_bits} bits"
    )

    print()
    print("## PACKETIZATION")

    print(
        f"Packet Size        : "
        f"{report.packet_bits} bits"
    )

    print(
        f"Packet Count       : "
        f"{report.packet_count}"
    )

    print()
    print("## SPATIAL CAPACITY")

    print(
        f"Block Size         : "
        f"{report.block_size}"
    )

    print(
        f"Gap                : "
        f"{report.gap}"
    )

    print(
        f"Delta              : "
        f"{report.delta}"
    )

    print(
        f"Spatial Capacity   : "
        f"{report.spatial_capacity_bits} bits/frame"
    )

    print()
    print("## FRAME REQUIREMENT")

    print(
        f"Required Frames    : "
        f"{report.required_frames}"
    )

    print(
        f"Available Frames   : "
        f"{report.available_frames}"
    )

    print(
        f"Remaining Frames   : "
        f"{report.remaining_frames}"
    )

    print()
    print(
        f"Capacity Status    : "
        f"{'PASS' if report.fits_video else 'FAIL'}"
    )

    print("=" * 70)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Analyze StegaFusion "
            "multi-frame video capacity."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Cover video path.",
    )

    parser.add_argument(
        "--secret",
        required=True,
        help="Secret file path.",
    )

    parser.add_argument(
        "--key",
        required=True,
        help="AES key path.",
    )

    parser.add_argument(
        "--delta",
        type=int,
        default=DELTA,
        help=(
            "Embedding delta. "
            "Default: production configuration."
        ),
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=BLOCK_SIZE,
        help=(
            "Spatial block size."
        ),
    )

    parser.add_argument(
        "--gap",
        type=int,
        default=GAP,
        help=(
            "Gap between paired blocks."
        ),
    )

    args = parser.parse_args()

    report = analyze_capacity(
        video=Path(args.video),
        secret_file=Path(args.secret),
        key_file=Path(args.key),
        block_size=args.block_size,
        gap=args.gap,
        delta=args.delta,
    )

    print_capacity_report(
        report
    )

    if not report.fits_video:
        raise SystemExit(1)