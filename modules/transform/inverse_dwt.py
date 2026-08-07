"""
------------------------------------------------------------
StegaFusion Inverse DWT Module
------------------------------------------------------------
Reconstructs Blue channel from
LL, LH, HL and HH.

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


# ==========================================================
# LOAD IMAGE
# ==========================================================

def load_frame(path: Path):

    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(path)

    return image


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("StegaFusion Inverse DWT Test")
    print("=" * 70)

    frame = PathConfig.FRAME_DIR / "frame_00000.png"

    image = load_frame(frame)

    blue = image[:, :, 0]

    bands = apply_dwt(blue)

    reconstructed = apply_inverse_dwt(bands)

    reconstructed = np.clip(
        reconstructed,
        0,
        255
    ).astype(np.uint8)

    output = PathConfig.TEMP_DIR / "dwt" / "reconstructed_blue.png"

    cv2.imwrite(
        str(output),
        reconstructed
    )

    print()

    print(
        f"Original Shape      : {blue.shape}"
    )

    print(
        f"Recovered Shape     : {reconstructed.shape}"
    )

    print()

    print(
        f"Saved : {output}"
    )

    print()

    print("Inverse DWT Successful!")