"""
------------------------------------------------------------
StegaFusion Evaluation Metrics
------------------------------------------------------------
Computes objective image quality metrics.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


# ==========================================================
# MSE
# ==========================================================

def calculate_mse(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> float:
    """
    Mean Squared Error.
    """

    return np.mean(
        (original.astype(np.float64) -
         reconstructed.astype(np.float64)) ** 2
    )


# ==========================================================
# PSNR
# ==========================================================

def calculate_psnr(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Returns infinity when images are identical.
    """

    mse = calculate_mse(
        original,
        reconstructed
    )

    if mse == 0:
        return float("inf")

    return peak_signal_noise_ratio(
        original,
        reconstructed,
        data_range=255
    )


# ==========================================================
# SSIM
# ==========================================================

def calculate_ssim(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> float:
    """
    Structural Similarity Index.
    """

    return structural_similarity(
        original,
        reconstructed,
        data_range=255
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    image = np.random.randint(
        0,
        256,
        (512, 512),
        dtype=np.uint8
    )

    reconstructed = image.copy()

    mse = calculate_mse(
        image,
        reconstructed
    )

    psnr = calculate_psnr(
        image,
        reconstructed
    )

    ssim = calculate_ssim(
        image,
        reconstructed
    )

    print("=" * 60)
    print("StegaFusion Metrics Test")
    print("=" * 60)

    print(f"MSE  : {mse:.6f}")
    print(f"PSNR : {psnr:.6f}")
    print(f"SSIM : {ssim:.6f}")