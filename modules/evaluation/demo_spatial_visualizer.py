"""
StegaFusion Stage 3 — Spatial Microscope Visualization

Demonstrates the REAL production spatial embedding mechanism.

Pipeline:

    Real Secret
        ↓
    AES-256 Encryption
        ↓
    StegaFusion Payload
        ↓
    Multi-frame Packetization
        ↓
    Production Packet
        ↓
    Production Spatial Embedding
        ↓
    Selected Paired Blocks
        ↓
    Production Spatial Extraction
        ↓
    Bit Validation

This is an evaluation / visualization module only.

NO production steganography algorithm is modified.
"""

from pathlib import Path

import cv2
import numpy as np

from modules.pipeline.spatial_video_pipeline import (
    prepare_spatial_payload,
    create_frame_packets,
    embed_spatial_frame,
)

from modules.steganography.spatial.paired_block import (
    BLOCK_SIZE,
    GAP,
    DELTA,
    get_block_pairs,
    extract_payload,
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

OUTPUT_DIR = Path(
    "output/demo_inspection"
)

OUTPUT_MICROSCOPE = (
    OUTPUT_DIR
    / "stage3_spatial_microscope.png"
)

OUTPUT_FULL_FRAME = (
    OUTPUT_DIR
    / "stage3_full_frame_selected_pair.png"
)

OUTPUT_DIFFERENCE = (
    OUTPUT_DIR
    / "stage3_selected_pair_difference.png"
)

# Production demonstration delta
DELTA_VALUE = 5

# None = automatically find a visually useful pair.
# Set to an integer if you later want a specific pair.
SELECTED_PAIR_INDEX = None


# ==========================================================
# VIDEO FRAME
# ==========================================================

def read_first_frame(
    video_path: Path,
):
    """
    Read the first frame and basic video information.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise FileNotFoundError(
            f"Unable to open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
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

    success, frame = cap.read()

    cap.release()

    if not success:

        raise RuntimeError(
            "Unable to read first video frame."
        )

    return (
        frame,
        fps,
        total_frames,
        width,
        height,
    )


# ==========================================================
# BLOCK MEAN
# ==========================================================

def block_mean(
    frame,
    bounds,
):
    """
    Calculate the mean intensity of the
    production Blue channel inside one block.
    """

    y1, y2, x1, x2 = bounds

    blue_channel = frame[
        y1:y2,
        x1:x2,
        0
    ]

    return float(
        np.mean(
            blue_channel
        )
    )


# ==========================================================
# AUTOMATIC DEMONSTRATION PAIR SELECTION
# ==========================================================

def find_best_demo_pair(
    frame,
    pairs,
):
    """
    Automatically select a visually informative
    production block pair.

    Preference is given to blocks that are:

        - sufficiently bright
        - locally textured
        - not almost completely black
        - not completely uniform

    This changes ONLY which pair is visualized.

    It does NOT modify the production algorithm.
    """

    blue = frame[:, :, 0]

    best_index = 0
    best_score = -1.0

    for index, pair in enumerate(pairs):

        left, right = pair

        ly1, ly2, lx1, lx2 = left
        ry1, ry2, rx1, rx2 = right

        left_region = blue[
            ly1:ly2,
            lx1:lx2
        ]

        right_region = blue[
            ry1:ry2,
            rx1:rx2
        ]

        left_mean = float(
            np.mean(
                left_region
            )
        )

        right_mean = float(
            np.mean(
                right_region
            )
        )

        left_std = float(
            np.std(
                left_region
            )
        )

        right_std = float(
            np.std(
                right_region
            )
        )

        # Prefer regions with visible brightness.
        brightness_score = min(
            left_mean,
            right_mean,
        )

        # Prefer regions containing visual texture.
        texture_score = (
            left_std
            + right_std
        )

        # Avoid extremely dark blocks.
        if brightness_score < 15:

            continue

        score = (
            brightness_score
            + texture_score * 2.0
        )

        if score > best_score:

            best_score = score
            best_index = index

    return best_index


# ==========================================================
# UNION OF TWO BLOCK BOUNDS
# ==========================================================

def union_bounds(
    pair,
):
    """
    Return the bounding rectangle containing
    both blocks of a pair.
    """

    left, right = pair

    y1 = min(
        left[0],
        right[0],
    )

    y2 = max(
        left[1],
        right[1],
    )

    x1 = min(
        left[2],
        right[2],
    )

    x2 = max(
        left[3],
        right[3],
    )

    return (
        y1,
        y2,
        x1,
        x2,
    )


# ==========================================================
# DRAW SELECTED PAIR ON FULL FRAME
# ==========================================================

def draw_pair_on_frame(
    image,
    pair,
):
    """
    Draw a clean highlight around the selected
    pair on the original full frame.
    """

    output = image.copy()

    left, right = pair

    ly1, ly2, lx1, lx2 = left
    ry1, ry2, rx1, rx2 = right

    # LEFT = green
    cv2.rectangle(
        output,
        (lx1, ly1),
        (lx2 - 1, ly2 - 1),
        (0, 255, 0),
        3,
    )

    # RIGHT = blue
    cv2.rectangle(
        output,
        (rx1, ry1),
        (rx2 - 1, ry2 - 1),
        (255, 0, 0),
        3,
    )

    # Connecting line between pair
    left_center = (
        (lx1 + lx2) // 2,
        (ly1 + ly2) // 2,
    )

    right_center = (
        (rx1 + rx2) // 2,
        (ry1 + ry2) // 2,
    )

    cv2.line(
        output,
        left_center,
        right_center,
        (0, 255, 255),
        2,
    )

    # Label above the pair
    label_x = min(
        lx1,
        rx1,
    )

    label_y = max(
        35,
        min(
            ly1,
            ry1,
        ) - 10,
    )

    cv2.putText(
        output,
        "SELECTED PAIR",
        (
            label_x,
            label_y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


# ==========================================================
# CROP SELECTED PAIR
# ==========================================================

def crop_pair(
    image,
    pair,
    padding=80,
):
    """
    Crop a larger region around the selected pair.
    """

    y1, y2, x1, x2 = union_bounds(
        pair
    )

    y1 = max(
        0,
        y1 - padding,
    )

    x1 = max(
        0,
        x1 - padding,
    )

    y2 = min(
        image.shape[0],
        y2 + padding,
    )

    x2 = min(
        image.shape[1],
        x2 + padding,
    )

    return image[
        y1:y2,
        x1:x2,
    ].copy(), (
        y1,
        y2,
        x1,
        x2,
    )


# ==========================================================
# DRAW PAIR ON CROP
# ==========================================================

def draw_pair_on_crop(
    image,
    pair,
    crop_bounds,
):
    """
    Draw pair boundaries using coordinates
    relative to the cropped image.
    """

    output = image.copy()

    crop_y1, _, crop_x1, _ = crop_bounds

    left, right = pair

    ly1, ly2, lx1, lx2 = left
    ry1, ry2, rx1, rx2 = right

    # Convert global → crop coordinates
    left_rect = (
        lx1 - crop_x1,
        ly1 - crop_y1,
        lx2 - crop_x1,
        ly2 - crop_y1,
    )

    right_rect = (
        rx1 - crop_x1,
        ry1 - crop_y1,
        rx2 - crop_x1,
        ry2 - crop_y1,
    )

    cv2.rectangle(
        output,
        (
            left_rect[0],
            left_rect[1],
        ),
        (
            left_rect[2] - 1,
            left_rect[3] - 1,
        ),
        (0, 255, 0),
        5,
    )

    cv2.rectangle(
        output,
        (
            right_rect[0],
            right_rect[1],
        ),
        (
            right_rect[2] - 1,
            right_rect[3] - 1,
        ),
        (255, 0, 0),
        5,
    )

    # Labels outside / above blocks where possible
    cv2.putText(
        output,
        "LEFT",
        (
            left_rect[0],
            max(
                30,
                left_rect[1] - 10,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        "RIGHT",
        (
            right_rect[0],
            max(
                30,
                right_rect[1] - 10,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return output


# ==========================================================
# RESIZE IMAGE
# ==========================================================

def resize_for_display(
    image,
    width=700,
):
    """
    Resize while preserving aspect ratio.
    """

    if image.size == 0:

        raise ValueError(
            "Cannot resize an empty image."
        )

    scale = (
        width
        / image.shape[1]
    )

    height = max(
        1,
        int(
            image.shape[0]
            * scale
        ),
    )

    return cv2.resize(
        image,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )


# ==========================================================
# CREATE LABELED PANEL
# ==========================================================

def create_panel(
    image,
    title,
    width=700,
):
    """
    Create a presentation-friendly image panel.
    """

    display = resize_for_display(
        image,
        width,
    )

    header_height = 55

    panel = np.zeros(
        (
            display.shape[0]
            + header_height,
            display.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    panel[
        header_height:,
        :
    ] = display

    cv2.putText(
        panel,
        title,
        (
            18,
            36,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return panel


# ==========================================================
# CREATE MICROSCOPE ARTIFACT
# ==========================================================

def create_microscope(
    original_crop,
    stego_crop,
    difference_crop,
    selected_bit,
    extracted_bit,
    selected_index,
    left_change,
    right_change,
    packet_match,
    bit_match,
):
    """
    Create the final reviewer-friendly
    Stage 3 microscope visualization.
    """

    panel_width = 600

    original_panel = create_panel(
        original_crop,
        "ORIGINAL",
        panel_width,
    )

    stego_panel = create_panel(
        stego_crop,
        "STEGO",
        panel_width,
    )

    difference_panel = create_panel(
        difference_crop,
        "AMPLIFIED DIFFERENCE ×8",
        panel_width,
    )

    panel_height = max(
        original_panel.shape[0],
        stego_panel.shape[0],
        difference_panel.shape[0],
    )

    gap = 15

    header_height = 125
    info_height = 245

    total_width = (
        panel_width * 3
        + gap * 2
    )

    total_height = (
        header_height
        + panel_height
        + info_height
    )

    canvas = np.zeros(
        (
            total_height,
            total_width,
            3,
        ),
        dtype=np.uint8,
    )

    # ------------------------------------------------------
    # HEADER
    # ------------------------------------------------------

    cv2.putText(
        canvas,
        "StegaFusion — SPATIAL MICROSCOPE",
        (
            25,
            42,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        (
            f"Production packet bit #{selected_index}    "
            f"Original = {selected_bit}    "
            f"Extracted = {extracted_bit}"
        ),
        (
            25,
            80,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        (
            "GREEN = LEFT BLOCK     "
            "BLUE = RIGHT BLOCK"
        ),
        (
            25,
            108,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # ------------------------------------------------------
    # PANELS
    # ------------------------------------------------------

    panels = [
        original_panel,
        stego_panel,
        difference_panel,
    ]

    panel_y = header_height

    for index, panel in enumerate(
        panels
    ):

        x = (
            index
            * (
                panel_width
                + gap
            )
        )

        canvas[
            panel_y:
            panel_y + panel.shape[0],
            x:
            x + panel.shape[1],
        ] = panel

    # ------------------------------------------------------
    # INFORMATION SECTION
    # ------------------------------------------------------

    info_y = (
        header_height
        + panel_height
        + 30
    )

    cv2.putText(
        canvas,
        "HOW ONE PRODUCTION BIT IS REPRESENTED",
        (
            25,
            info_y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    lines = [
        f"Packet bit             : {selected_bit}",
        f"LEFT block change      : {left_change:+.4f}",
        f"RIGHT block change     : {right_change:+.4f}",
        f"Embedding delta        : +{DELTA_VALUE}",
        f"Extracted bit          : {extracted_bit}",
        f"Bit validation         : "
        f"{'TRUE' if bit_match else 'FALSE'}",
        f"Complete packet        : "
        f"{'MATCH' if packet_match else 'MISMATCH'}",
    ]

    for index, line in enumerate(
        lines
    ):

        good = (
            "TRUE" in line
            or "MATCH" in line
        )

        cv2.putText(
            canvas,
            line,
            (
                30,
                info_y
                + 35
                + index * 26,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                (0, 255, 0)
                if good
                else (230, 230, 230)
            ),
            1,
            cv2.LINE_AA,
        )

    return canvas


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 78)

    print(
        "StegaFusion STAGE 3 — "
        "SPATIAL MICROSCOPE VISUALIZATION"
    )

    print("=" * 78)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------

    print()
    print("## INPUT VALIDATION")

    for path in (
        COVER_VIDEO,
        SECRET_FILE,
        KEY_FILE,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Required input does not exist: {path}"
            )

        print(
            f"{path} : EXISTS"
        )

    # ------------------------------------------------------
    # READ VIDEO FRAME
    # ------------------------------------------------------

    (
        original_frame,
        fps,
        total_frames,
        width,
        height,
    ) = read_first_frame(
        COVER_VIDEO
    )

    print()
    print("## VIDEO FRAME")

    print(
        "Frame Index : 0"
    )

    print(
        f"Resolution  : "
        f"{width}x{height}"
    )

    print(
        f"FPS         : {fps}"
    )

    print(
        f"Total Frames: {total_frames}"
    )

    # ------------------------------------------------------
    # PREPARE REAL PRODUCTION PAYLOAD
    # ------------------------------------------------------

    print()
    print("## PRODUCTION PAYLOAD")

    payload = prepare_spatial_payload(
        secret_file=SECRET_FILE,
        key_file=KEY_FILE,
    )

    packets = create_frame_packets(
        payload
    )

    packet = packets[0]

    print(
        f"Complete Payload Bits : "
        f"{len(payload)}"
    )

    print(
        f"Packet Count          : "
        f"{len(packets)}"
    )

    print(
        f"Selected Packet Bits  : "
        f"{len(packet)}"
    )

    print(
        f"Selected Packet Preview: "
        f"{packet[:64]}"
    )

    # ------------------------------------------------------
    # SPATIAL CONFIGURATION
    # ------------------------------------------------------

    print()
    print("## SPATIAL CONFIGURATION")

    print(
        f"Block Size : "
        f"{BLOCK_SIZE}x{BLOCK_SIZE}"
    )

    print(
        f"Gap        : "
        f"{GAP} pixels"
    )

    print(
        f"Delta      : "
        f"{DELTA_VALUE}"
    )

    print(
        "Channel    : Blue"
    )

    # ------------------------------------------------------
    # GET ALL PRODUCTION BLOCK PAIRS
    # ------------------------------------------------------

    pairs = get_block_pairs(
        height,
        width,
        BLOCK_SIZE,
        GAP,
    )

    print()
    print("## PAIRED-BLOCK GEOMETRY")

    print(
        f"Total Block Pairs : "
        f"{len(pairs)}"
    )

    print(
        f"Spatial Capacity  : "
        f"{len(pairs)} bits/frame"
    )

    # ------------------------------------------------------
    # SELECT VISUALLY INFORMATIVE PAIR
    # ------------------------------------------------------

    if SELECTED_PAIR_INDEX is None:

        selected_index = (
            find_best_demo_pair(
                original_frame,
                pairs,
            )
        )

        selection_mode = (
            "AUTOMATIC — visually informative"
        )

    else:

        selected_index = min(
            SELECTED_PAIR_INDEX,
            len(packet) - 1,
        )

        selection_mode = "MANUAL"

    # The selected bit MUST be inside the actual
    # production packet.
    if selected_index >= len(packet):

        selected_index = (
            len(packet) - 1
        )

    selected_pair = pairs[
        selected_index
    ]

    selected_bit = packet[
        selected_index
    ]

    print(
        f"Selected Pair     : "
        f"{selected_index}"
    )

    print(
        f"Selected Bit      : "
        f"{selected_bit}"
    )

    print(
        f"Pair Selection    : "
        f"{selection_mode}"
    )

    # ------------------------------------------------------
    # PRODUCTION EMBEDDING
    # ------------------------------------------------------

    print()
    print("## PRODUCTION SPATIAL EMBEDDING")

    stego_frame, embedded_bits = (
        embed_spatial_frame(
            original_frame,
            packet,
            block_size=BLOCK_SIZE,
            delta=DELTA_VALUE,
            gap=GAP,
        )
    )

    print(
        "Embedding : PASS"
    )

    print(
        f"Embedded Bits : "
        f"{embedded_bits}"
    )

    # ------------------------------------------------------
    # PRODUCTION EXTRACTION
    # ------------------------------------------------------

    print()
    print("## PRODUCTION SPATIAL EXTRACTION")

    extracted_packet = extract_payload(
        original_frame,
        stego_frame,
        bit_count=len(packet),
        block_size=BLOCK_SIZE,
        gap=GAP,
    )

    packet_match = (
        extracted_packet
        == packet
    )

    extracted_bit = (
        extracted_packet[
            selected_index
        ]
    )

    bit_match = (
        selected_bit
        == extracted_bit
    )

    print(
        f"Extracted Bits : "
        f"{len(extracted_packet)}"
    )

    print(
        f"Packet Match   : "
        f"{'TRUE' if packet_match else 'FALSE'}"
    )

    print()
    print("## SELECTED BIT VALIDATION")

    print(
        f"Pair Index     : "
        f"{selected_index}"
    )

    print(
        f"Original Bit   : "
        f"{selected_bit}"
    )

    print(
        f"Extracted Bit  : "
        f"{extracted_bit}"
    )

    print(
        f"Bit Match      : "
        f"{'TRUE' if bit_match else 'FALSE'}"
    )

    if not packet_match:

        raise RuntimeError(
            "Production packet extraction failed."
        )

    if not bit_match:

        raise RuntimeError(
            "Selected bit extraction failed."
        )

    # ------------------------------------------------------
    # SIGNAL ANALYSIS
    # ------------------------------------------------------

    left_original = block_mean(
        original_frame,
        selected_pair[0],
    )

    right_original = block_mean(
        original_frame,
        selected_pair[1],
    )

    left_stego = block_mean(
        stego_frame,
        selected_pair[0],
    )

    right_stego = block_mean(
        stego_frame,
        selected_pair[1],
    )

    left_change = (
        left_stego
        - left_original
    )

    right_change = (
        right_stego
        - right_original
    )

    print()
    print("## SELECTED PAIR SIGNAL")

    print(
        f"Original LEFT mean  : "
        f"{left_original:.4f}"
    )

    print(
        f"Stego LEFT mean     : "
        f"{left_stego:.4f}"
    )

    print(
        f"LEFT change         : "
        f"{left_change:.4f}"
    )

    print(
        f"Original RIGHT mean : "
        f"{right_original:.4f}"
    )

    print(
        f"Stego RIGHT mean    : "
        f"{right_stego:.4f}"
    )

    print(
        f"RIGHT change        : "
        f"{right_change:.4f}"
    )

    # ------------------------------------------------------
    # FULL FRAME VISUALIZATION
    # ------------------------------------------------------

    full_frame = (
        draw_pair_on_frame(
            original_frame,
            selected_pair,
        )
    )

    cv2.imwrite(
        str(OUTPUT_FULL_FRAME),
        full_frame,
    )

    # ------------------------------------------------------
    # CROP ORIGINAL + STEGO
    # ------------------------------------------------------

    original_crop, crop_bounds = (
        crop_pair(
            original_frame,
            selected_pair,
            padding=80,
        )
    )

    stego_crop, _ = (
        crop_pair(
            stego_frame,
            selected_pair,
            padding=80,
        )
    )

    original_crop = (
        draw_pair_on_crop(
            original_crop,
            selected_pair,
            crop_bounds,
        )
    )

    stego_crop = (
        draw_pair_on_crop(
            stego_crop,
            selected_pair,
            crop_bounds,
        )
    )

    # ------------------------------------------------------
    # DIFFERENCE
    # ------------------------------------------------------

    difference = cv2.absdiff(
        original_frame,
        stego_frame,
    )

    amplified_difference = (
        cv2.convertScaleAbs(
            difference,
            alpha=8.0,
            beta=0,
        )
    )

    difference_crop, _ = (
        crop_pair(
            amplified_difference,
            selected_pair,
            padding=80,
        )
    )

    difference_crop = (
        draw_pair_on_crop(
            difference_crop,
            selected_pair,
            crop_bounds,
        )
    )

    # ------------------------------------------------------
    # MICROSCOPE IMAGE
    # ------------------------------------------------------

    microscope = create_microscope(
        original_crop=original_crop,
        stego_crop=stego_crop,
        difference_crop=difference_crop,
        selected_bit=selected_bit,
        extracted_bit=extracted_bit,
        selected_index=selected_index,
        left_change=left_change,
        right_change=right_change,
        packet_match=packet_match,
        bit_match=bit_match,
    )

    cv2.imwrite(
        str(OUTPUT_MICROSCOPE),
        microscope,
    )

    cv2.imwrite(
        str(OUTPUT_DIFFERENCE),
        difference_crop,
    )

    # ------------------------------------------------------
    # BLUE-CHANNEL PIXEL STATISTICS
    # ------------------------------------------------------

    blue_difference = difference[:, :, 0]

    changed_pixels = int(
        np.count_nonzero(
            blue_difference
        )
    )

    total_pixels = int(
        blue_difference.size
    )

    mean_absolute_change = float(
        np.mean(
            blue_difference
        )
    )

    maximum_change = int(
        np.max(
            blue_difference
        )
    )
    # ------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------

    print()
    print("## GENERATED ARTIFACTS")

    print(
        f"Spatial Microscope : "
        f"{OUTPUT_MICROSCOPE}"
    )

    print(
        f"Full Frame         : "
        f"{OUTPUT_FULL_FRAME}"
    )

    print(
        f"Difference         : "
        f"{OUTPUT_DIFFERENCE}"
    )

    print()
    print("## BLUE-CHANNEL PIXEL DIFFERENCE")

    print(
        f"Changed Pixel Values : "
        f"{changed_pixels}"
    )

    print(
        f"Total Pixel Values   : "
        f"{total_pixels}"
    )

    print(
        f"Mean Absolute Change : "
        f"{mean_absolute_change:.6f}"
    )

    print(
        f"Maximum Change       : "
        f"{maximum_change}"
    )

    # ------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------

    print()
    print("=" * 78)

    print(
        "RESULT: STAGE 3 SPATIAL "
        "MICROSCOPE PASS."
    )

    print("=" * 78)

    print()
    print(
        "Actual production demonstration:"
    )

    print(
        "Real secret"
    )

    print(
        "→ AES-256 encryption"
    )

    print(
        "→ multi-frame packetization"
    )

    print(
        "→ selected production packet"
    )

    print(
        "→ actual paired-block spatial embedding"
    )

    print(
        "→ production spatial extraction"
    )

    print(
        "→ bit validation"
    )

    print(
        "→ complete packet validation"
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