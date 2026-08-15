"""
StegaFusion Video Quality Evaluation

Compares the original cover video against a decoded stego video
after the complete spatial embedding + MP4V encoding pipeline.

Metrics:
    - MSE
    - PSNR
    - SSIM

The evaluation is performed frame-by-frame on corresponding
decoded frames.

No production steganography algorithm is modified.
"""

from pathlib import Path
import math

import cv2
import numpy as np

from skimage.metrics import structural_similarity


# ==========================================================
# INPUTS
# ==========================================================

ORIGINAL_VIDEO = Path(
    "input/cover_video/sample_long.mp4"
)

STEGO_VIDEO = Path(
    "output/video_spatial_multiframe_32kb_longvideo_integration/"
    "stego_32kb_longvideo.mp4"
)


# ==========================================================
# CONFIGURATION
# ==========================================================

# Print a few individual frame measurements.
SAMPLE_FRAME_INTERVAL = 100


# ==========================================================
# VIDEO INFORMATION
# ==========================================================

def get_video_info(
    capture,
):
    """
    Return basic video metadata.
    """

    frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = (
        frames / fps
        if fps > 0
        else 0.0
    )

    return {
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
    }


# ==========================================================
# MSE
# ==========================================================

def calculate_mse(
    original,
    stego,
):
    """
    Calculate Mean Squared Error.
    """

    original_float = (
        original.astype(
            np.float64
        )
    )

    stego_float = (
        stego.astype(
            np.float64
        )
    )

    difference = (
        original_float
        - stego_float
    )

    return float(
        np.mean(
            difference ** 2
        )
    )


# ==========================================================
# PSNR
# ==========================================================

def calculate_psnr(
    mse,
):
    """
    Calculate PSNR from MSE.

    Returns infinity when the frames are identical.
    """

    if mse == 0:
        return float("inf")

    max_pixel = 255.0

    return float(
        10.0
        * math.log10(
            (max_pixel ** 2)
            / mse
        )
    )


# ==========================================================
# SSIM
# ==========================================================

def calculate_ssim(
    original,
    stego,
):
    """
    Calculate SSIM across RGB/BGR channels.
    """

    return float(
        structural_similarity(
            original,
            stego,
            data_range=255,
            channel_axis=2,
        )
    )


# ==========================================================
# INPUT VALIDATION
# ==========================================================

def validate_inputs():

    print()
    print(
        "## INPUT VALIDATION"
    )

    if not ORIGINAL_VIDEO.exists():

        raise FileNotFoundError(
            f"Original video not found: "
            f"{ORIGINAL_VIDEO}"
        )

    print(
        f"{ORIGINAL_VIDEO} : EXISTS"
    )

    if not STEGO_VIDEO.exists():

        raise FileNotFoundError(
            f"Stego video not found: "
            f"{STEGO_VIDEO}"
        )

    print(
        f"{STEGO_VIDEO} : EXISTS"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 78)

    print(
        "StegaFusion VIDEO QUALITY EVALUATION"
    )

    print("=" * 78)

    validate_inputs()

    # ------------------------------------------------------
    # OPEN VIDEOS
    # ------------------------------------------------------

    original_capture = (
        cv2.VideoCapture(
            str(ORIGINAL_VIDEO)
        )
    )

    stego_capture = (
        cv2.VideoCapture(
            str(STEGO_VIDEO)
        )
    )

    if not original_capture.isOpened():

        raise RuntimeError(
            "Could not open original video."
        )

    if not stego_capture.isOpened():

        original_capture.release()

        raise RuntimeError(
            "Could not open stego video."
        )

    original_info = get_video_info(
        original_capture
    )

    stego_info = get_video_info(
        stego_capture
    )

    # ------------------------------------------------------
    # VIDEO INFORMATION
    # ------------------------------------------------------

    print()
    print(
        "## ORIGINAL VIDEO"
    )

    print(
        f"Frames      : "
        f"{original_info['frames']}"
    )

    print(
        f"FPS         : "
        f"{original_info['fps']}"
    )

    print(
        f"Resolution  : "
        f"{original_info['width']}x"
        f"{original_info['height']}"
    )

    print(
        f"Duration    : "
        f"{original_info['duration']:.2f} seconds"
    )

    print()
    print(
        "## STEGO VIDEO"
    )

    print(
        f"Frames      : "
        f"{stego_info['frames']}"
    )

    print(
        f"FPS         : "
        f"{stego_info['fps']}"
    )

    print(
        f"Resolution  : "
        f"{stego_info['width']}x"
        f"{stego_info['height']}"
    )

    print(
        f"Duration    : "
        f"{stego_info['duration']:.2f} seconds"
    )

    # ------------------------------------------------------
    # COMPATIBILITY CHECK
    # ------------------------------------------------------

    if (
        original_info["frames"]
        != stego_info["frames"]
    ):

        original_capture.release()
        stego_capture.release()

        raise RuntimeError(
            "Frame count mismatch between "
            "original and stego videos."
        )

    if (
        original_info["width"]
        != stego_info["width"]
        or
        original_info["height"]
        != stego_info["height"]
    ):

        original_capture.release()
        stego_capture.release()

        raise RuntimeError(
            "Resolution mismatch between "
            "original and stego videos."
        )

    # ------------------------------------------------------
    # FRAME-BY-FRAME EVALUATION
    # ------------------------------------------------------

    print()
    print(
        "## FRAME-BY-FRAME QUALITY ANALYSIS"
    )

    mse_values = []
    psnr_values = []
    ssim_values = []

    frame_index = 0

    while True:

        original_ok, original_frame = (
            original_capture.read()
        )

        stego_ok, stego_frame = (
            stego_capture.read()
        )

        if (
            not original_ok
            and not stego_ok
        ):
            break

        if (
            not original_ok
            or not stego_ok
        ):

            original_capture.release()
            stego_capture.release()

            raise RuntimeError(
                "One video ended before "
                "the other."
            )

        if (
            original_frame.shape
            != stego_frame.shape
        ):

            original_capture.release()
            stego_capture.release()

            raise RuntimeError(
                f"Frame shape mismatch "
                f"at frame {frame_index}."
            )

        mse = calculate_mse(
            original_frame,
            stego_frame,
        )

        psnr = calculate_psnr(
            mse
        )

        ssim = calculate_ssim(
            original_frame,
            stego_frame,
        )

        mse_values.append(
            mse
        )

        psnr_values.append(
            psnr
        )

        ssim_values.append(
            ssim
        )

        if (
            frame_index == 0
            or
            frame_index % SAMPLE_FRAME_INTERVAL == 0
        ):

            psnr_text = (
                "inf"
                if math.isinf(psnr)
                else f"{psnr:.4f} dB"
            )

            print(
                f"Frame {frame_index:04d} | "
                f"MSE={mse:.6f} | "
                f"PSNR={psnr_text} | "
                f"SSIM={ssim:.6f}"
            )

        frame_index += 1

    original_capture.release()
    stego_capture.release()

    # ------------------------------------------------------
    # SAFETY CHECK
    # ------------------------------------------------------

    if not mse_values:

        raise RuntimeError(
            "No frames were evaluated."
        )

    # ------------------------------------------------------
    # AGGREGATE METRICS
    # ------------------------------------------------------

    finite_psnr = [
        value
        for value in psnr_values
        if math.isfinite(value)
    ]

    average_mse = float(
        np.mean(
            mse_values
        )
    )

    maximum_mse = float(
        np.max(
            mse_values
        )
    )

    average_ssim = float(
        np.mean(
            ssim_values
        )
    )

    minimum_ssim = float(
        np.min(
            ssim_values
        )
    )

    if finite_psnr:

        average_psnr = float(
            np.mean(
                finite_psnr
            )
        )

        minimum_psnr = float(
            np.min(
                finite_psnr
            )
        )

    else:

        average_psnr = float("inf")
        minimum_psnr = float("inf")

    # ------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------

    print()
    print("=" * 78)

    print(
        "## QUALITY RESULTS"
    )

    print("=" * 78)

    print()
    print(
        f"Frames Compared       : "
        f"{frame_index}"
    )

    print(
        f"Average MSE           : "
        f"{average_mse:.8f}"
    )

    print(
        f"Maximum MSE           : "
        f"{maximum_mse:.8f}"
    )

    if math.isinf(average_psnr):

        print(
            "Average PSNR          : "
            "inf"
        )

        print(
            "Minimum PSNR          : "
            "inf"
        )

    else:

        print(
            f"Average PSNR          : "
            f"{average_psnr:.4f} dB"
        )

        print(
            f"Minimum PSNR          : "
            f"{minimum_psnr:.4f} dB"
        )

    print(
        f"Average SSIM          : "
        f"{average_ssim:.8f}"
    )

    print(
        f"Minimum SSIM          : "
        f"{minimum_ssim:.8f}"
    )

    # ------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------

    print()
    print("=" * 78)

    print(
        "RESULT: VIDEO QUALITY "
        "EVALUATION COMPLETE."
    )

    print("=" * 78)

    print()
    print(
        "Comparison:"
    )

    print(
        "Original cover video"
    )

    print(
        "VS"
    )

    print(
        "Decoded MP4V stego video"
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