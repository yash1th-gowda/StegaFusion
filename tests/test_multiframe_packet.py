"""
StegaFusion Multi-Frame Packet Unit Tests

Tests:
    - packet creation
    - packet parsing
    - maximum chunk size
    - multiple chunk indexes
    - invalid magic
    - invalid version
    - invalid chunk index
    - truncated packets
    - invalid binary input

No production steganography code is modified.
"""

from modules.steganography.multiframe_packet import (
    HEADER_BITS,
    MAX_CHUNK_DATA_BITS,
    MAX_FRAME_BITS,
    create_packet,
    parse_packet,
)


# ==========================================================
# TEST HELPERS
# ==========================================================

def make_payload(bit_count: int) -> str:
    pattern = (
        "101100111000111100001111"
    )

    repeats = (
        bit_count // len(pattern)
    ) + 1

    return (
        pattern * repeats
    )[:bit_count]


# ==========================================================
# BASIC PACKET
# ==========================================================

def test_basic_packet():

    chunk = make_payload(500)

    packet = create_packet(
        chunk_data=chunk,
        total_chunks=4,
        chunk_index=2,
    )

    decoded = parse_packet(
        packet
    )

    assert decoded.version == 1
    assert decoded.total_chunks == 4
    assert decoded.chunk_index == 2
    assert decoded.chunk_length == 500
    assert decoded.chunk_data == chunk

    assert len(packet) == (
        HEADER_BITS + 500
    )


# ==========================================================
# MAXIMUM CHUNK
# ==========================================================

def test_maximum_chunk():

    chunk = make_payload(
        MAX_CHUNK_DATA_BITS
    )

    packet = create_packet(
        chunk_data=chunk,
        total_chunks=1,
        chunk_index=0,
    )

    assert len(packet) == (
        MAX_FRAME_BITS
    )

    decoded = parse_packet(
        packet
    )

    assert decoded.chunk_length == (
        MAX_CHUNK_DATA_BITS
    )

    assert decoded.chunk_data == chunk


# ==========================================================
# MULTIPLE CHUNK INDEXES
# ==========================================================

def test_multiple_indexes():

    for index in range(5):

        chunk = make_payload(
            200
        )

        packet = create_packet(
            chunk_data=chunk,
            total_chunks=5,
            chunk_index=index,
        )

        decoded = parse_packet(
            packet
        )

        assert decoded.total_chunks == 5
        assert decoded.chunk_index == index
        assert decoded.chunk_data == chunk


# ==========================================================
# SINGLE CHUNK
# ==========================================================

def test_single_chunk():

    chunk = make_payload(
        704
    )

    packet = create_packet(
        chunk_data=chunk,
        total_chunks=1,
        chunk_index=0,
    )

    decoded = parse_packet(
        packet
    )

    assert decoded.total_chunks == 1
    assert decoded.chunk_index == 0
    assert decoded.chunk_data == chunk


# ==========================================================
# INVALID MAGIC
# ==========================================================

def test_invalid_magic():

    chunk = make_payload(
        100
    )

    packet = create_packet(
        chunk_data=chunk,
        total_chunks=1,
        chunk_index=0,
    )

    corrupted = (
        "0000000000000000"
        + packet[16:]
    )

    try:

        parse_packet(
            corrupted
        )

    except ValueError as error:

        assert "magic" in str(
            error
        ).lower()

    else:

        raise AssertionError(
            "Invalid magic was accepted."
        )


# ==========================================================
# INVALID VERSION
# ==========================================================

def test_invalid_version():

    chunk = make_payload(
        100
    )

    packet = create_packet(
        chunk_data=chunk,
        total_chunks=1,
        chunk_index=0,
    )

    corrupted = (
        packet[:16]
        + "00000000"
        + packet[24:]
    )

    try:

        parse_packet(
            corrupted
        )

    except ValueError as error:

        assert "version" in str(
            error
        ).lower()

    else:

        raise AssertionError(
            "Invalid version was accepted."
        )


# ==========================================================
# INVALID CHUNK INDEX
# ==========================================================

def test_invalid_chunk_index():

    chunk = make_payload(
        100
    )

    try:

        create_packet(
            chunk_data=chunk,
            total_chunks=3,
            chunk_index=3,
        )

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Invalid chunk index was accepted."
        )


# ==========================================================
# INVALID TOTAL CHUNKS
# ==========================================================

def test_invalid_total_chunks():

    chunk = make_payload(
        100
    )

    try:

        create_packet(
            chunk_data=chunk,
            total_chunks=0,
            chunk_index=0,
        )

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Invalid total chunk count was accepted."
        )


# ==========================================================
# EMPTY CHUNK
# ==========================================================

def test_empty_chunk():

    try:

        create_packet(
            chunk_data="",
            total_chunks=1,
            chunk_index=0,
        )

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Empty chunk was accepted."
        )


# ==========================================================
# OVERSIZED CHUNK
# ==========================================================

def test_oversized_chunk():

    chunk = make_payload(
        MAX_CHUNK_DATA_BITS + 1
    )

    try:

        create_packet(
            chunk_data=chunk,
            total_chunks=1,
            chunk_index=0,
        )

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Oversized chunk was accepted."
        )


# ==========================================================
# INVALID BINARY CHUNK
# ==========================================================

def test_invalid_binary_chunk():

    try:

        create_packet(
            chunk_data="10102010",
            total_chunks=1,
            chunk_index=0,
        )

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Invalid binary chunk was accepted."
        )


# ==========================================================
# TRUNCATED PACKET
# ==========================================================

def test_truncated_packet():

    chunk = make_payload(
        200
    )

    packet = create_packet(
        chunk_data=chunk,
        total_chunks=1,
        chunk_index=0,
    )

    truncated = packet[:-20]

    try:

        parse_packet(
            truncated
        )

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Truncated packet was accepted."
        )


# ==========================================================
# INVALID BINARY PACKET
# ==========================================================

def test_invalid_binary_packet():

    try:

        parse_packet(
            "10101010XYZ"
        )

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Invalid binary packet was accepted."
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "StegaFusion Multi-Frame Packet Unit Test"
    )
    print("=" * 70)

    tests = [
        (
            "Basic packet",
            test_basic_packet,
        ),
        (
            "Maximum chunk",
            test_maximum_chunk,
        ),
        (
            "Multiple indexes",
            test_multiple_indexes,
        ),
        (
            "Single chunk",
            test_single_chunk,
        ),
        (
            "Invalid magic",
            test_invalid_magic,
        ),
        (
            "Invalid version",
            test_invalid_version,
        ),
        (
            "Invalid chunk index",
            test_invalid_chunk_index,
        ),
        (
            "Invalid total chunks",
            test_invalid_total_chunks,
        ),
        (
            "Empty chunk",
            test_empty_chunk,
        ),
        (
            "Oversized chunk",
            test_oversized_chunk,
        ),
        (
            "Invalid binary chunk",
            test_invalid_binary_chunk,
        ),
        (
            "Truncated packet",
            test_truncated_packet,
        ),
        (
            "Invalid binary packet",
            test_invalid_binary_packet,
        ),
    ]

    passed = 0
    failed = 0

    print()

    for name, test_function in tests:

        try:

            test_function()

            print(
                f"{name:<30} PASS"
            )

            passed += 1

        except Exception as error:

            print(
                f"{name:<30} FAIL"
            )

            print(
                f"  {error}"
            )

            failed += 1

    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print()

    print(
        f"Tests Passed : {passed}"
    )

    print(
        f"Tests Failed : {failed}"
    )

    print(
        f"Total Tests  : {len(tests)}"
    )

    print()

    if failed == 0:

        print(
            "RESULT: MULTI-FRAME PACKET "
            "UNIT TEST PASSED."
        )

    else:

        print(
            "RESULT: MULTI-FRAME PACKET "
            "UNIT TEST FAILED."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()