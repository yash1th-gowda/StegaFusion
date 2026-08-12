"""
StegaFusion Multi-Frame Payload Unit Tests

Tests:
    - payload splitting
    - payload reconstruction
    - exact chunk boundaries
    - chunk count
    - chunk sizes
    - invalid input handling

No production steganography code is modified.
"""

from modules.steganography.multiframe_payload import (
    DEFAULT_CHUNK_BITS,
    calculate_chunk_count,
    combine_chunks,
    get_chunk_sizes,
    split_payload,
)


# ==========================================================
# TEST HELPERS
# ==========================================================

def make_payload(bit_count: int) -> str:
    """
    Generate a deterministic binary payload.
    """

    pattern = (
        "101100111000111100001111"
    )

    repeats = (
        bit_count // len(pattern)
    ) + 1

    return (
        pattern * repeats
    )[:bit_count]


def assert_equal(
    actual,
    expected,
    message: str,
):
    if actual != expected:
        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected!r}\n"
            f"Actual  : {actual!r}"
        )


# ==========================================================
# TEST EMPTY PAYLOAD
# ==========================================================

def test_empty_payload():

    payload = ""

    chunks = split_payload(
        payload,
        DEFAULT_CHUNK_BITS,
    )

    assert_equal(
        chunks,
        [],
        "Empty payload should produce no chunks.",
    )

    reconstructed = combine_chunks(
        chunks
    )

    assert_equal(
        reconstructed,
        "",
        "Empty payload reconstruction failed.",
    )


# ==========================================================
# TEST SMALL PAYLOAD
# ==========================================================

def test_small_payload():

    payload = make_payload(
        100
    )

    chunks = split_payload(
        payload,
        DEFAULT_CHUNK_BITS,
    )

    assert_equal(
        len(chunks),
        1,
        "100-bit payload should produce one chunk.",
    )

    assert_equal(
        len(chunks[0]),
        100,
        "Small payload chunk has incorrect size.",
    )

    reconstructed = combine_chunks(
        chunks
    )

    assert_equal(
        reconstructed,
        payload,
        "Small payload reconstruction failed.",
    )


# ==========================================================
# TEST EXACT CHUNK SIZE
# ==========================================================

def test_exact_chunk():

    payload = make_payload(
        DEFAULT_CHUNK_BITS
    )

    chunks = split_payload(
        payload,
        DEFAULT_CHUNK_BITS,
    )

    assert_equal(
        len(chunks),
        1,
        "Exactly one chunk should be produced.",
    )

    assert_equal(
        len(chunks[0]),
        DEFAULT_CHUNK_BITS,
        "Exact chunk size is incorrect.",
    )

    reconstructed = combine_chunks(
        chunks
    )

    assert_equal(
        reconstructed,
        payload,
        "Exact chunk reconstruction failed.",
    )


# ==========================================================
# TEST ONE BIT OVER CHUNK SIZE
# ==========================================================

def test_one_bit_over_chunk():

    payload = make_payload(
        DEFAULT_CHUNK_BITS + 1
    )

    chunks = split_payload(
        payload,
        DEFAULT_CHUNK_BITS,
    )

    assert_equal(
        len(chunks),
        2,
        "769-bit payload should produce two chunks.",
    )

    assert_equal(
        len(chunks[0]),
        DEFAULT_CHUNK_BITS,
        "First chunk has incorrect size.",
    )

    assert_equal(
        len(chunks[1]),
        1,
        "Second chunk should contain one bit.",
    )

    reconstructed = combine_chunks(
        chunks
    )

    assert_equal(
        reconstructed,
        payload,
        "769-bit reconstruction failed.",
    )


# ==========================================================
# TEST MULTIPLE CHUNKS
# ==========================================================

def test_multiple_chunks():

    payload = make_payload(
        2000
    )

    chunks = split_payload(
        payload,
        DEFAULT_CHUNK_BITS,
    )

    expected_sizes = [
        768,
        768,
        464,
    ]

    actual_sizes = [
        len(chunk)
        for chunk in chunks
    ]

    assert_equal(
        actual_sizes,
        expected_sizes,
        "2000-bit chunk sizes are incorrect.",
    )

    reconstructed = combine_chunks(
        chunks
    )

    assert_equal(
        reconstructed,
        payload,
        "Multi-chunk reconstruction failed.",
    )


# ==========================================================
# TEST CHUNK COUNT
# ==========================================================

def test_chunk_count():

    test_cases = [
        (0, 0),
        (1, 1),
        (767, 1),
        (768, 1),
        (769, 2),
        (1536, 2),
        (1537, 3),
        (2000, 3),
    ]

    for payload_bits, expected_count in test_cases:

        actual_count = calculate_chunk_count(
            payload_bits,
            DEFAULT_CHUNK_BITS,
        )

        assert_equal(
            actual_count,
            expected_count,
            (
                f"Incorrect chunk count "
                f"for {payload_bits} bits."
            ),
        )


# ==========================================================
# TEST CHUNK SIZES
# ==========================================================

def test_chunk_sizes():

    cases = [
        (
            0,
            [],
        ),
        (
            1,
            [1],
        ),
        (
            768,
            [768],
        ),
        (
            769,
            [768, 1],
        ),
        (
            1536,
            [768, 768],
        ),
        (
            2000,
            [768, 768, 464],
        ),
    ]

    for payload_bits, expected_sizes in cases:

        actual_sizes = get_chunk_sizes(
            payload_bits,
            DEFAULT_CHUNK_BITS,
        )

        assert_equal(
            actual_sizes,
            expected_sizes,
            (
                f"Incorrect chunk sizes "
                f"for {payload_bits} bits."
            ),
        )


# ==========================================================
# TEST INVALID PAYLOAD
# ==========================================================

def test_invalid_payload():

    try:

        split_payload(
            "10102001",
            DEFAULT_CHUNK_BITS,
        )

    except ValueError:
        pass

    else:

        raise AssertionError(
            "Invalid binary payload "
            "was accepted."
        )


# ==========================================================
# TEST INVALID CHUNK SIZE
# ==========================================================

def test_invalid_chunk_size():

    invalid_sizes = [
        0,
        -1,
    ]

    for size in invalid_sizes:

        try:

            split_payload(
                "101010",
                size,
            )

        except ValueError:
            pass

        else:

            raise AssertionError(
                f"Invalid chunk size {size} "
                "was accepted."
            )


# ==========================================================
# TEST NON-INTEGER CHUNK SIZE
# ==========================================================

def test_non_integer_chunk_size():

    try:

        split_payload(
            "101010",
            768.0,
        )

    except TypeError:
        pass

    else:

        raise AssertionError(
            "Non-integer chunk size was accepted."
        )


# ==========================================================
# TEST INVALID CHUNK LIST
# ==========================================================

def test_invalid_chunks():

    invalid_cases = [
        [
            "101010",
            "102010",
        ],
        [
            "101010",
            123,
        ],
    ]

    for chunks in invalid_cases:

        try:

            combine_chunks(
                chunks
            )

        except (
            TypeError,
            ValueError,
        ):
            pass

        else:

            raise AssertionError(
                f"Invalid chunks were accepted: "
                f"{chunks!r}"
            )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(
        "StegaFusion Multi-Frame Payload Unit Test"
    )
    print("=" * 70)

    tests = [
        (
            "Empty payload",
            test_empty_payload,
        ),
        (
            "Small payload",
            test_small_payload,
        ),
        (
            "Exact chunk",
            test_exact_chunk,
        ),
        (
            "One bit over chunk",
            test_one_bit_over_chunk,
        ),
        (
            "Multiple chunks",
            test_multiple_chunks,
        ),
        (
            "Chunk count",
            test_chunk_count,
        ),
        (
            "Chunk sizes",
            test_chunk_sizes,
        ),
        (
            "Invalid payload",
            test_invalid_payload,
        ),
        (
            "Invalid chunk size",
            test_invalid_chunk_size,
        ),
        (
            "Non-integer chunk size",
            test_non_integer_chunk_size,
        ),
        (
            "Invalid chunks",
            test_invalid_chunks,
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
            "RESULT: MULTI-FRAME PAYLOAD "
            "UNIT TEST PASSED."
        )

    else:

        print(
            "RESULT: MULTI-FRAME PAYLOAD "
            "UNIT TEST FAILED."
        )

        raise SystemExit(1)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()