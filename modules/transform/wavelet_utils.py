"""
------------------------------------------------------------
StegaFusion Wavelet Utilities
------------------------------------------------------------
Provides reusable functions for Haar DWT and visualization.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path

import cv2
import numpy as np
import pywt


# ==========================================================
# APPLY HAAR DWT
# ==========================================================

def apply_dwt(channel: np.ndarray):
    """
    Apply one-level Haar DWT.

    Args:
        channel: Single-channel image

    Returns:
        Dictionary containing LL, LH, HL and HH bands.
    """

    LL, (LH, HL, HH) = pywt.dwt2(channel, "haar")

    return {
        "LL": LL,
        "LH": LH,
        "HL": HL,
        "HH": HH
    }
# ==========================================================
# APPLY INVERSE DWT
# ==========================================================

def apply_inverse_dwt(bands: dict) -> np.ndarray:
    """
    Reconstruct image using inverse Haar DWT.

    Args:
        bands:
            Dictionary containing
            LL, LH, HL, HH.

    Returns:
        Reconstructed image.
    """

    reconstructed = pywt.idwt2(
        (
            bands["LL"],
            (
                bands["LH"],
                bands["HL"],
                bands["HH"]
            )
        ),
        "haar"
    )

    return reconstructed

# ==========================================================
# NORMALIZE BAND
# ==========================================================

def normalize_band(band: np.ndarray):

    normalized = cv2.normalize(
        band,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return normalized.astype(np.uint8)


# ==========================================================
# SAVE BAND
# ==========================================================

def save_band(band: np.ndarray, path: Path):

    cv2.imwrite(str(path), normalize_band(band))