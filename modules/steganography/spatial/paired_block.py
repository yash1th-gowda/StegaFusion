"""
StegaFusion Spatial Paired-Block Steganography

MP4-resistant spatial-domain embedding.
"""

import numpy as np


# ==========================================================
# CONFIGURATION
# ==========================================================

BLOCK_SIZE = 32
GAP = 16
DELTA = 4
CHANNEL = 0


# ==========================================================
# BLOCK PAIRS
# ==========================================================

def get_block_pairs(
    height: int,
    width: int,
    block_size: int = BLOCK_SIZE,
    gap: int = GAP,
):
    """
    Generate non-overlapping horizontal paired blocks.

    Each pair is:

        [ Block A ][ GAP ][ Block B ]

    One payload bit is stored per pair.

    Pairs do not overlap each other.
    """

    pairs = []

    rows = height // block_size

    pair_width = (
        block_size
        + gap
        + block_size
    )

    pairs_per_row = width // pair_width

    for row in range(rows):

        y1 = row * block_size
        y2 = y1 + block_size

        for pair_index in range(pairs_per_row):

            x1 = pair_index * pair_width
            x2 = x1 + block_size

            right_x1 = x2 + gap
            right_x2 = right_x1 + block_size

            if right_x2 > width:
                continue

            pairs.append(
                (
                    (
                        y1,
                        y2,
                        x1,
                        x2,
                    ),
                    (
                        y1,
                        y2,
                        right_x1,
                        right_x2,
                    ),
                )
            )

    return pairs


# ==========================================================
# CAPACITY
# ==========================================================

def calculate_capacity(
    frame,
    block_size: int = BLOCK_SIZE,
    gap: int = GAP,
):
    """
    Return the number of payload bits that can be embedded.
    """

    height, width = frame.shape[:2]

    return len(
        get_block_pairs(
            height,
            width,
            block_size,
            gap,
        )
    )


# ==========================================================
# BLOCK MEAN
# ==========================================================

def _block_mean(
    channel,
    bounds,
):
    y1, y2, x1, x2 = bounds

    return float(
        np.mean(
            channel[y1:y2, x1:x2]
        )
    )


# ==========================================================
# EMBED ONE BIT
# ==========================================================

def _embed_bit(
    channel,
    pair,
    bit: int,
    delta: int = DELTA,
):
    """
    Embed one bit using a spatial block signal.

    Bit 0:
        LEFT block receives the signal.

    Bit 1:
        RIGHT block receives the signal.
    """

    left, right = pair

    if bit == 0:
        y1, y2, x1, x2 = left
    else:
        y1, y2, x1, x2 = right

    channel[
        y1:y2,
        x1:x2
    ] += delta


# ==========================================================
# EMBED PAYLOAD
# ==========================================================

def embed_payload(
    frame,
    payload: str,
    block_size: int = BLOCK_SIZE,
    delta: int = DELTA,
    gap: int = GAP,
):
    """
    Embed a binary payload into a frame.

    Returns
    -------
    stego_frame:
        Modified uint8 BGR frame.

    embedded:
        Number of embedded bits.
    """

    if not payload:
        return frame.copy(), 0

    if any(
        bit not in "01"
        for bit in payload
    ):
        raise ValueError(
            "Payload must contain only '0' and '1'."
        )

    if delta <= 0:
        raise ValueError(
            "delta must be positive."
        )

    capacity = calculate_capacity(
        frame,
        block_size,
        gap,
    )

    if len(payload) > capacity:
        raise ValueError(
            f"Payload requires {len(payload)} bits, "
            f"but frame capacity is {capacity} bits."
        )

    stego = frame.astype(
        np.float32
    ).copy()

    channel = stego[:, :, CHANNEL]

    pairs = get_block_pairs(
        frame.shape[0],
        frame.shape[1],
        block_size,
        gap,
    )

    for bit, pair in zip(
        payload,
        pairs,
    ):
        _embed_bit(
            channel,
            pair,
            int(bit),
            delta,
        )

    stego = np.rint(
        stego
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )

    return stego, len(payload)


# ==========================================================
# EXTRACT ONE BIT
# ==========================================================

def _extract_bit(
    original_channel,
    stego_channel,
    pair,
):
    """
    Extract one bit by comparing the spatial
    change in the two paired blocks.
    """

    left, right = pair

    original_left = _block_mean(
        original_channel,
        left,
    )

    original_right = _block_mean(
        original_channel,
        right,
    )

    stego_left = _block_mean(
        stego_channel,
        left,
    )

    stego_right = _block_mean(
        stego_channel,
        right,
    )

    left_change = (
        stego_left
        - original_left
    )

    right_change = (
        stego_right
        - original_right
    )

    if left_change > right_change:
        return "0"

    return "1"


# ==========================================================
# EXTRACT PAYLOAD
# ==========================================================

def extract_payload(
    original_frame,
    stego_frame,
    bit_count: int,
    block_size: int = BLOCK_SIZE,
    gap: int = GAP,
):
    """
    Extract a binary payload.

    Extraction requires the original cover frame
    because the MP4-resistant method measures
    spatial changes relative to the original.
    """

    if bit_count < 0:
        raise ValueError(
            "bit_count cannot be negative."
        )

    capacity = calculate_capacity(
        original_frame,
        block_size,
        gap,
    )

    if bit_count > capacity:
        raise ValueError(
            f"Requested {bit_count} bits, "
            f"but frame capacity is {capacity} bits."
        )

    original_channel = (
        original_frame[:, :, CHANNEL]
    )

    stego_channel = (
        stego_frame[:, :, CHANNEL]
    )

    pairs = get_block_pairs(
        original_frame.shape[0],
        original_frame.shape[1],
        block_size,
        gap,
    )

    bits = []

    for pair in pairs[:bit_count]:

        bits.append(
            _extract_bit(
                original_channel,
                stego_channel,
                pair,
            )
        )

    return "".join(bits)