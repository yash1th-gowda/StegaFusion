"""
------------------------------------------------------------
StegaFusion Edge Detection Module
------------------------------------------------------------
Detects edge regions using the Sobel operator.
The edge map will be used by the Adaptive LSB engine
to determine embedding capacity.

Author      : Yashwanth Gowda M
Version     : 1.0.0
Created     : August 2026
------------------------------------------------------------
"""

from pathlib import Path

import cv2
import numpy as np


# ==========================================================
# COMPUTE SOBEL GRADIENT
# ==========================================================

def compute_gradient(image: np.ndarray) -> np.ndarray:
    """
    Computes the gradient magnitude using Sobel filters.

    Args:
        image : Grayscale image

    Returns:
        Gradient magnitude image
    """

    grad_x = cv2.Sobel(
        image,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    grad_y = cv2.Sobel(
        image,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)

    return gradient


# ==========================================================
# CREATE EDGE MAP
# ==========================================================

def generate_edge_map(
    image: np.ndarray,
    threshold: float = 40.0
) -> np.ndarray:
    """
    Creates a binary edge map.

    Returns:
        Binary edge image
    """

    gradient = compute_gradient(image)

    edge_map = np.zeros_like(image, dtype=np.uint8)

    edge_map[gradient >= threshold] = 255

    return edge_map


# ==========================================================
# SAVE EDGE MAP
# ==========================================================

def save_edge_map(
    edge_map: np.ndarray,
    output_path: Path
):

    cv2.imwrite(
        str(output_path),
        edge_map
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Edge Detection Test")
    print("=" * 70)

    sample = Path("temp/dwt/LH.png")

    image = cv2.imread(
        str(sample),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            "LH.png not found."
        )

    edge_map = generate_edge_map(image)

    output = Path("temp/dwt/edge_map.png")

    save_edge_map(
        edge_map,
        output
    )

    print()

    print(f"Input Shape : {image.shape}")
    print(f"Edge Pixels : {np.count_nonzero(edge_map)}")

    print()
    print(f"Saved : {output}")

    print()
    print("Edge Detection Successful!")