"""
StegaFusion Multi-Frame Capacity Scalability Experiment

Capacity-only experiment.

This test does NOT perform embedding or MP4V encoding.

It uses the same production:
    AES payload preparation
    packetization
    spatial capacity calculation

used by the production pipeline.

Purpose:
    Determine how the required number of video frames scales
    with encrypted secret-file size.

No production steganography algorithm is modified.
"""

from pathlib import Path

from modules.evaluation.capacity import (
    analyze_capacity,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

COVER_VIDEO = Path(
    "input/cover_video/sample_long.mp4"
)

KEY_FILE = Path(
    "input/keys/test_aes_key.bin"
)

SECRET_FILES = [
    (
        "4 KB",
        Path(
            "input/secret_data/scalability_test/"
            "secret_4kb.bin"
        ),
    ),
    (
        "8 KB",
        Path(
            "input/secret_data/scalability_test/"
            "secret_8kb.bin"
        ),
    ),
    (
        "16 KB",
        Path(
            "input/secret_data/scalability_test/"
            "secret_16kb.bin"
        ),
    ),
    (
        "32 KB",
        Path(
            "input/secret_data/scalability_test/"
            "secret_32kb.bin"
        ),
    ),
]

DELTA = 5


# ==========================================================
# INPUT VALIDATION
# ==========================================================

def validate_inputs():

    print()
    print("## INPUT VALIDATION")

    if not COVER_VIDEO.exists():

        raise FileNotFoundError(
            f"Cover video not found: "
            f"{COVER_VIDEO}"
        )

    print(
        f"{COVER_VIDEO} : EXISTS"
    )

    if not KEY_FILE.exists():

        raise FileNotFoundError(
            f"AES key not found: "
            f"{KEY_FILE}"
        )

    print(
        f"{KEY_FILE} : EXISTS"
    )

    for label, secret_file in SECRET_FILES:

        if not secret_file.exists():

            raise FileNotFoundError(
                f"{label} secret file not found: "
                f"{secret_file}"
            )

        print(
            f"{secret_file} : EXISTS"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 78)

    print(
        "StegaFusion MULTI-FRAME CAPACITY "
        "SCALABILITY EXPERIMENT"
    )

    print("=" * 78)

    validate_inputs()

    print()
    print(
        f"Cover Video : {COVER_VIDEO}"
    )

    print(
        f"Delta       : {DELTA}"
    )

    print(
        "Block Size  : 32"
    )

    print(
        "Gap         : 16"
    )

    results = []

    # ======================================================
    # ANALYZE EACH SECRET SIZE
    # ======================================================

    for label, secret_file in SECRET_FILES:

        print()
        print("-" * 78)

        print(
            f"ANALYZING {label}"
        )

        print("-" * 78)

        report = analyze_capacity(
            video=COVER_VIDEO,
            secret_file=secret_file,
            key_file=KEY_FILE,
            delta=DELTA,
        )

        results.append(
            (
                label,
                report,
            )
        )

        print()

        print(
            f"Secret Size       : "
            f"{report.secret_bytes} bytes"
        )

        print(
            f"Payload Bits      : "
            f"{report.payload_bits}"
        )

        print(
            f"Packet Size       : "
            f"{report.packet_bits} bits"
        )

        print(
            f"Packet Count      : "
            f"{report.packet_count}"
        )

        print(
            f"Spatial Capacity  : "
            f"{report.spatial_capacity_bits} bits/frame"
        )

        print(
            f"Required Frames   : "
            f"{report.required_frames}"
        )

        print(
            f"Available Frames  : "
            f"{report.available_frames}"
        )

        print(
            f"Remaining Frames  : "
            f"{report.remaining_frames}"
        )

        print(
            f"Status            : "
            f"{'PASS' if report.fits_video else 'FAIL'}"
        )

    # ======================================================
    # SUMMARY TABLE
    # ======================================================

    print()
    print("=" * 78)

    print(
        "## SCALABILITY RESULTS"
    )

    print("=" * 78)

    print()

    print(
        f"{'SECRET':<10}"
        f"{'PAYLOAD BITS':>15}"
        f"{'PACKETS':>10}"
        f"{'REQ. FRAMES':>15}"
        f"{'AVAILABLE':>12}"
        f"{'REMAINING':>12}"
        f"{'STATUS':>10}"
    )

    print("-" * 84)

    for label, report in results:

        print(
            f"{label:<10}"
            f"{report.payload_bits:>15}"
            f"{report.packet_count:>10}"
            f"{report.required_frames:>15}"
            f"{report.available_frames:>12}"
            f"{report.remaining_frames:>12}"
            f"{'PASS' if report.fits_video else 'FAIL':>10}"
        )

    # ======================================================
    # EXPECTED PRODUCTION VALUES
    # ======================================================

    expected = {
        "4 KB": (
            33056,
            47,
            47,
        ),

        "8 KB": (
            65824,
            94,
            94,
        ),

        "16 KB": (
            131360,
            187,
            187,
        ),

        "32 KB": (
            262432,
            373,
            373,
        ),
    }

    print()
    print(
        "## KNOWN RESULT VALIDATION"
    )

    print("-" * 78)

    all_pass = True

    for label, report in results:

        expected_payload, expected_packets, expected_frames = (
            expected[label]
        )

        payload_ok = (
            report.payload_bits
            == expected_payload
        )

        packets_ok = (
            report.packet_count
            == expected_packets
        )

        frames_ok = (
            report.required_frames
            == expected_frames
        )

        capacity_ok = (
            report.fits_video
        )

        passed = (
            payload_ok
            and packets_ok
            and frames_ok
            and capacity_ok
        )

        all_pass = (
            all_pass
            and passed
        )

        print(
            f"{label:<8} : "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if not passed:

            print(
                f"  Expected payload : "
                f"{expected_payload}"
            )

            print(
                f"  Actual payload   : "
                f"{report.payload_bits}"
            )

            print(
                f"  Expected packets : "
                f"{expected_packets}"
            )

            print(
                f"  Actual packets   : "
                f"{report.packet_count}"
            )

            print(
                f"  Expected frames  : "
                f"{expected_frames}"
            )

            print(
                f"  Actual frames    : "
                f"{report.required_frames}"
            )

    # ======================================================
    # FINAL RESULT
    # ======================================================

    print()
    print("=" * 78)

    if all_pass:

        print(
            "RESULT: CAPACITY SCALABILITY "
            "EXPERIMENT PASS."
        )

    else:

        print(
            "RESULT: CAPACITY SCALABILITY "
            "EXPERIMENT FAIL."
        )

    print("=" * 78)

    if not all_pass:

        raise RuntimeError(
            "Capacity scalability results "
            "did not match the known production values."
        )

    print()
    print(
        "No production steganography "
        "algorithm was modified."
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()