"""
------------------------------------------------------------
StegaFusion DWT Module
------------------------------------------------------------
Applies one-level Haar DWT to the Blue channel of a video frame.

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
    save_band
)


# ==========================================================
# LOAD FRAME
# ==========================================================

def load_frame(frame_path: Path) -> np.ndarray:
    """
    Load an image frame.

    Args:
        frame_path (Path): Path to frame image.

    Returns:
        numpy.ndarray
    """

    image = cv2.imread(str(frame_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to load frame: {frame_path}"
        )

    return image


# ==========================================================
# SPLIT RGB CHANNELS
# ==========================================================

def split_channels(image: np.ndarray):
    """
    Split image into Blue, Green and Red channels.
    """

    blue, green, red = cv2.split(image)

    return blue, green, red


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Haar DWT Test")
    print("=" * 70)

    # Load first extracted frame
    frame_path = PathConfig.FRAME_DIR / "frame_00000.png"

    image = load_frame(frame_path)

    blue, green, red = split_channels(image)

    # Apply Haar DWT
    bands = apply_dwt(blue)

    # Create output directory
    dwt_output = PathConfig.TEMP_DIR / "dwt"
    dwt_output.mkdir(parents=True, exist_ok=True)

    # Save all bands
    for band_name, band in bands.items():

        save_path = dwt_output / f"{band_name}.png"

        save_band(
            band,
            save_path
        )

        print(
            f"{band_name:<3} "
            f"Shape: {band.shape} "
            f"Saved: {save_path.name}"
        )

    print()
    print("Haar DWT Completed Successfully!")