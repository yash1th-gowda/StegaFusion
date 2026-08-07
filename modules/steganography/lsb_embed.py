"""
------------------------------------------------------------
StegaFusion LSB Embed Pipeline
------------------------------------------------------------
Integrates DWT, Edge Detection and Adaptive LSB
to generate the first Stego Frame.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path

import cv2
import numpy as np

from config.config import PathConfig

from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)

from modules.steganography.edge_detector import (
    generate_edge_map,
)

from modules.steganography.adaptive_lsb import (
    adaptive_embed,
    calculate_capacity,
)


# ==========================================================
# LOAD FRAME
# ==========================================================

def load_frame(frame_path: Path):

    image = cv2.imread(str(frame_path))

    if image is None:
        raise FileNotFoundError(frame_path)

    return image


# ==========================================================
# SAVE FRAME
# ==========================================================

def save_frame(image, output_path: Path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        image
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Integration Test")
    print("=" * 70)

    frame_path = PathConfig.FRAME_DIR / "frame_00000.png"

    image = load_frame(frame_path)

    blue = image[:, :, 0]

    bands = apply_dwt(blue)

    edge_map = generate_edge_map(
        bands["LH"]
    )

    payload = "101100111000111100001111"

    capacity = calculate_capacity(edge_map)

    print(f"Embedding Capacity : {capacity} bits")

    stego_lh, embedded = adaptive_embed(
        bands["LH"],
        edge_map,
        payload
    )

    bands["LH"] = stego_lh

    reconstructed = apply_inverse_dwt(
        bands
    )

    reconstructed = np.clip(
        reconstructed,
        0,
        255
    ).astype(np.uint8)

    stego = image.copy()

    stego[:, :, 0] = reconstructed

    output = (
        PathConfig.TEMP_DIR /
        "stego_frames" /
        "stego_frame_00000.png"
    )

    save_frame(
        stego,
        output
    )

    print()

    print(
        f"Embedded Bits : {embedded}"
    )

    print(
        f"Saved : {output}"
    )

    print()

    print("First Stego Frame Generated Successfully!")