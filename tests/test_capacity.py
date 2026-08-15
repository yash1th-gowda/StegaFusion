"""
StegaFusion Capacity Regression Test

Validates the production capacity estimator against
the known 32 KB long-video configuration.
"""

from pathlib import Path

from modules.evaluation.capacity import analyze_capacity


VIDEO = Path(
    "input/cover_video/sample_long.mp4"
)

SECRET = Path(
    "input/secret_data/scalability_test/secret_32kb.bin"
)

KEY = Path(
    "input/keys/test_aes_key.bin"
)


def main():

    print("=" * 70)
    print(
        "StegaFusion CAPACITY REGRESSION TEST"
    )
    print("=" * 70)

    report = analyze_capacity(
        video=VIDEO,
        secret_file=SECRET,
        key_file=KEY,
        delta=5,
    )

    print()
    print("## CAPACITY VALIDATION")

    print(
        f"Payload Bits       : "
        f"{report.payload_bits}"
    )

    print(
        f"Packet Count       : "
        f"{report.packet_count}"
    )

    print(
        f"Spatial Capacity   : "
        f"{report.spatial_capacity_bits}"
    )

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

    # ------------------------------------------------------
    # KNOWN PRODUCTION VALUES
    # ------------------------------------------------------

    assert report.payload_bits == 262432

    assert report.packet_count == 373

    assert report.packet_bits == 768

    assert report.spatial_capacity_bits == 792

    assert report.required_frames == 373

    assert report.available_frames == 720

    assert report.remaining_frames == 347

    assert report.fits_video is True

    print()
    print(
        "All capacity assertions : PASS"
    )

    print()
    print("=" * 70)

    print(
        "RESULT: CAPACITY REGRESSION PASS."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()