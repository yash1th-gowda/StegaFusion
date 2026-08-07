"""
------------------------------------------------------------
StegaFusion DWT Module
------------------------------------------------------------
Reads a frame, separates RGB channels,
and prepares the Blue channel for Haar DWT.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path

import cv2
import numpy as np

from config.config import PathConfig


# ==========================================================
# LOAD IMAGE
# ==========================================================

def load_frame(frame_path: Path) -> np.ndarray:
    """
    Loads an image frame.

    Args:
        frame_path (Path)

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
    Splits image into Blue, Green and Red channels.

    Returns:
        tuple
    """

    blue, green, red = cv2.split(image)

    return blue, green, red


# ==========================================================
# IMAGE INFORMATION
# ==========================================================

def image_information(image: np.ndarray):

    height, width, channels = image.shape

    return {
        "Width": width,
        "Height": height,
        "Channels": channels,
        "Datatype": image.dtype
    }


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion DWT Preparation")
    print("=" * 70)

    frame = PathConfig.FRAME_DIR / "frame_00000.png"

    image = load_frame(frame)

    blue, green, red = split_channels(image)

    info = image_information(image)

    print()

    for key, value in info.items():
        print(f"{key:<12}: {value}")

    print()

    print(f"Blue Shape   : {blue.shape}")
    print(f"Green Shape  : {green.shape}")
    print(f"Red Shape    : {red.shape}")

    print()

    print("Frame Successfully Prepared For DWT")