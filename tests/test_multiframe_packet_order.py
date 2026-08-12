"""
StegaFusion Multi-Frame Packet Ordering and Validation Test

Tests the packet layer independently from video/frame extraction.

The packet format is expected to support packets arriving in
arbitrary order because each packet contains its own chunk_index.

Expected:

    Normal order       -> PASS
    Reversed order     -> PASS
    Random order       -> PASS
    Duplicate index    -> REJECT
    Missing index      -> REJECT
    Wrong total count  -> REJECT

No production steganography algorithm is modified.
"""

from modules.steganography.multiframe_packet import (
    create_packet,
    parse_packet,
)

from modules.pipeline.spatial_video_pipeline import (
    create_frame_packets,
    parse_frame_packets,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

PAYLOAD_SIZE = 2000


# ==========================================================
# PAYLOAD GENERATION
# ==========================================================

def make_payload(length: int) -> str:
    """
    Create deterministic binary payload.
    """

    pattern = "101100111000111100001111"

    return (
        pattern
        * ((length // len(pattern)) + 1)
    )[:length]


# ==========================================================
# ASSERTION HELPERS
# ==========================================================

def require(
    condition: bool,
    message: str,
):
    """
    Raise an error when a test condition fails.
    """

    if not condition:
        raise RuntimeError(message)


# ==========================================================
# VALID ORDER TEST
# ==========================================================

def test_normal_order(
    payload: str,
    packets: list[str],
):
    """
    Verify normal packet order.
    """

    reconstructed = parse_frame_packets(
        packets
    )

    require(
        reconstructed == payload,
        "Normal packet reconstruction failed.",
    )

    print(
        "Normal order                  PASS"
    )


# ==========================================================
# REVERSED ORDER TEST
# ==========================================================

def test_reversed_order(
    payload: str,
    packets: list[str],
):
    """
    Packets should reconstruct correctly
    even when supplied in reverse order.
    """

    reordered = list(
        reversed(packets)
    )

    reconstructed = parse_frame_packets(
        reordered
    )

    require(
        reconstructed == payload,
        "Reversed packet reconstruction failed.",
    )

    print(
        "Reversed order                PASS"
    )


# ==========================================================
# RANDOM ORDER TEST
# ==========================================================

def test_random_order(
    payload: str,
    packets: list[str],
):
    """
    Test a deterministic non-sequential order.

    Every packet must still be present exactly once.
    """

    if len(packets) < 3:

        raise RuntimeError(
            "Random-order test requires at least "
            "three packets."
        )

    indexes = list(
        range(len(packets))
    )

    # Deterministic permutation.
    #
    # For three packets:
    #
    # Original:
    #   0 1 2
    #
    # Reordered:
    #   2 0 1
    #
    # Every packet is still present exactly once.

    reordered_indexes = (
        [indexes[-1]]
        + indexes[:-1]
    )

    reordered = [
        packets[index]
        for index in reordered_indexes
    ]

    # Safety check: the test itself must not
    # accidentally remove or duplicate packets.

    if len(reordered) != len(packets):

        raise RuntimeError(
            "Random ordering test produced "
            "the wrong number of packets."
        )

    if sorted(reordered_indexes) != indexes:

        raise RuntimeError(
            "Random ordering test did not "
            "contain every packet exactly once."
        )

    reconstructed = parse_frame_packets(
        reordered
    )

    require(
        reconstructed == payload,
        "Random packet reconstruction failed.",
    )

    print(
        "Deterministic random order     PASS"
    )


# ==========================================================
# DUPLICATE INDEX TEST
# ==========================================================

def test_duplicate_index(
    packets: list[str],
):
    """
    Duplicate chunk indexes must be rejected.
    """

    if len(packets) < 3:

        raise RuntimeError(
            "Duplicate test requires at least "
            "three packets."
        )

    malformed = list(
        packets
    )

    # Replace packet 2 with packet 1.
    #
    # Result:
    #
    # 0, 1, 1, 3, ...
    #
    malformed[2] = malformed[1]

    try:

        parse_frame_packets(
            malformed
        )

    except ValueError as exc:

        print(
            "Duplicate chunk index         PASS"
        )

        print(
            f"  Rejected with : {exc}"
        )

        return

    raise RuntimeError(
        "Duplicate chunk index was "
        "incorrectly accepted."
    )


# ==========================================================
# MISSING INDEX TEST
# ==========================================================

def test_missing_index(
    packets: list[str],
):
    """
    Missing chunk indexes must be rejected.

    We remove one packet entirely.
    """

    if len(packets) < 3:

        raise RuntimeError(
            "Missing-index test requires at least "
            "three packets."
        )

    malformed = (
        packets[:1]
        + packets[2:]
    )

    try:

        parse_frame_packets(
            malformed
        )

    except ValueError as exc:

        print(
            "Missing chunk index            PASS"
        )

        print(
            f"  Rejected with : {exc}"
        )

        return

    raise RuntimeError(
        "Missing chunk index was "
        "incorrectly accepted."
    )


# ==========================================================
# WRONG TOTAL CHUNKS TEST
# ==========================================================

def test_wrong_total_chunks(
    packets: list[str],
):
    """
    Packets reporting different total_chunks
    metadata must be rejected.
    """

    if len(packets) < 2:

        raise RuntimeError(
            "Wrong-total test requires at least "
            "two packets."
        )

    # Parse packet 0 so that its chunk data can
    # be reused with intentionally incorrect
    # total_chunks metadata.

    original = parse_packet(
        packets[0]
    )

    wrong_total = (
        original.total_chunks + 1
    )

    malformed_packet = create_packet(
        original.chunk_data,
        wrong_total,
        original.chunk_index,
    )

    malformed = list(
        packets
    )

    malformed[0] = malformed_packet

    try:

        parse_frame_packets(
            malformed
        )

    except ValueError as exc:

        print(
            "Wrong total_chunks             PASS"
        )

        print(
            f"  Rejected with : {exc}"
        )

        return

    raise RuntimeError(
        "Wrong total_chunks metadata "
        "was incorrectly accepted."
    )


# ==========================================================
# PACKET METADATA TEST
# ==========================================================

def test_packet_indexes(
    packets: list[str],
):
    """
    Verify every packet contains a unique and
    complete index range.
    """

    decoded = [
        parse_packet(packet)
        for packet in packets
    ]

    total_chunks = (
        decoded[0].total_chunks
    )

    indexes = sorted(
        item.chunk_index
        for item in decoded
    )

    expected = list(
        range(total_chunks)
    )

    require(
        indexes == expected,
        (
            "Packet indexes are not a complete "
            "0..N-1 sequence."
        ),
    )

    require(
        all(
            item.total_chunks == total_chunks
            for item in decoded
        ),
        "Packet total_chunks metadata is inconsistent.",
    )

    print(
        "Packet metadata validation      PASS"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)

    print(
        "StegaFusion Multi-Frame Packet "
        "Ordering and Validation Test"
    )

    print("=" * 70)

    print()

    payload = make_payload(
        PAYLOAD_SIZE
    )

    packets = create_frame_packets(
        payload
    )

    print(
        f"Original Payload : "
        f"{len(payload)} bits"
    )

    print(
        f"Packet Count     : "
        f"{len(packets)}"
    )

    print()

    print(
        "PACKET STRUCTURE"
    )

    print(
        "-" * 70
    )

    for index, packet in enumerate(
        packets
    ):

        decoded = parse_packet(
            packet
        )

        print(
            f"Packet {index:2d} : "
            f"Index={decoded.chunk_index:<2d} "
            f"Total={decoded.total_chunks:<2d} "
            f"Data={decoded.chunk_length:<3d} bits "
            f"Packet={len(packet):<3d} bits"
        )

    print()

    print(
        "=" * 70
    )

    print(
        "ORDERING TESTS"
    )

    print(
        "=" * 70
    )

    test_packet_indexes(
        packets
    )

    test_normal_order(
        payload,
        packets,
    )

    test_reversed_order(
        payload,
        packets,
    )

    test_random_order(
        payload,
        packets,
    )

    print()

    print(
        "=" * 70
    )

    print(
        "FAILURE VALIDATION"
    )

    print(
        "=" * 70
    )

    test_duplicate_index(
        packets
    )

    test_missing_index(
        packets
    )

    test_wrong_total_chunks(
        packets
    )

    print()

    print(
        "=" * 70
    )

    print(
        "RESULT: MULTI-FRAME PACKET "
        "ORDERING AND VALIDATION PASSED."
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Normal Order                  : PASS"
    )

    print(
        "Reversed Order                : PASS"
    )

    print(
        "Random Order                  : PASS"
    )

    print(
        "Duplicate Detection           : PASS"
    )

    print(
        "Missing Packet Detection      : PASS"
    )

    print(
        "Metadata Validation            : PASS"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This test validates the packet layer "
        "independently from spatial video "
        "frame alignment."
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