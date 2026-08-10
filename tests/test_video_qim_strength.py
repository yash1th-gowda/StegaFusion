"""
StegaFusion Video QIM Strength Diagnostic

Tests whether different quantization strengths survive
the complete MP4 encoding/decoding pipeline.

This is diagnostic only.
It does NOT modify production configuration.
"""

from pathlib import Path
import shutil
import cv2
import numpy as np

from config.config import PathConfig

from modules.crypto.aes_encrypt import encrypt_file
from modules.steganography.payload import create_payload_packet
from modules.steganography.lsb_embed import load_frame, save_frame
from modules.steganography.edge_detector import generate_edge_map
from modules.steganography.adaptive_lsb import (
    adaptive_embed,
    adaptive_extract,
)
from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)
from modules.video.frame_reconstruct import reconstruct_video
from modules.steganography import lsb_utils


SECRET_FILE = Path(
    "input/secret_data/test_secret.txt"
)

KEY_FILE = Path(
    "input/keys/test_integration_key.bin"
)

FRAME_DIR = Path(
    "temp/frames"
)

SOURCE_FRAME = (
    FRAME_DIR /
    "frame_00000.png"
)

TEST_ROOT = Path(
    "temp/video_qim_strength"
)

OUTPUT_DIR = Path(
    "output"
)


def find_mismatch(expected, recovered):

    limit = min(
        len(expected),
        len(recovered)
    )

    for i in range(limit):

        if expected[i] != recovered[i]:
            return i

    if len(expected) != len(recovered):
        return limit

    return None


def run_test(
    payload_bits,
    original_image,
    step,
    test_number,
    fps,
):

    print()
    print("=" * 70)
    print(f"STEP = {step}")
    print("=" * 70)

    # ------------------------------------------------------
    # Temporarily change QIM strength
    # ------------------------------------------------------

    old_step = lsb_utils.QUANTIZATION_STEP

    lsb_utils.QUANTIZATION_STEP = step

    try:

        blue = original_image[:, :, 0]

        bands = apply_dwt(
            blue
        )

        edge_map = generate_edge_map(
            bands["LH"]
        )

        # --------------------------------------------------
        # Embed
        # --------------------------------------------------

        stego_lh, embedded = adaptive_embed(
            bands["LH"],
            edge_map,
            payload_bits
        )

        if embedded != len(payload_bits):

            print(
                f"Embedded Bits : "
                f"{embedded}/{len(payload_bits)}"
            )

            return {
                "step": step,
                "direct": False,
                "video": False,
                "mismatch": None,
                "changed_pixels": None,
                "max_difference": None,
                "mean_difference": None,
            }

        # --------------------------------------------------
        # Direct extraction
        # --------------------------------------------------

        direct_bits = adaptive_extract(
            stego_lh,
            edge_map,
            len(payload_bits)
        )

        direct_pass = (
            direct_bits == payload_bits
        )

        # --------------------------------------------------
        # Reconstruct stego frame
        # --------------------------------------------------

        bands["LH"] = stego_lh

        reconstructed_blue = apply_inverse_dwt(
            bands
        )

        reconstructed_blue = np.clip(
            np.rint(reconstructed_blue),
            0,
            255
        ).astype(np.uint8)

        stego_frame = original_image.copy()

        stego_frame[:, :, 0] = (
            reconstructed_blue
        )

        # --------------------------------------------------
        # Spatial distortion before MP4
        # --------------------------------------------------

        spatial_difference = (
            stego_frame[:, :, 0].astype(np.int16)
            -
            original_image[:, :, 0].astype(np.int16)
        )

        changed_pixels = np.count_nonzero(
            spatial_difference
        )

        max_difference = np.max(
            np.abs(spatial_difference)
        )

        mean_difference = np.mean(
            np.abs(spatial_difference)
        )

        # --------------------------------------------------
        # Prepare temporary video frame directory
        # --------------------------------------------------

        test_dir = (
            TEST_ROOT /
            f"step_{step}"
        )

        if test_dir.exists():
            shutil.rmtree(test_dir)

        test_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Copy original frames
        source_frames = sorted(
            FRAME_DIR.glob("*.png")
        )

        for frame_path in source_frames:

            destination = (
                test_dir /
                frame_path.name
            )

            shutil.copy2(
                frame_path,
                destination
            )

        # Replace first frame
        save_frame(
            stego_frame,
            test_dir / "frame_00000.png"
        )

        # --------------------------------------------------
        # Build MP4
        # --------------------------------------------------

        output_video = (
            OUTPUT_DIR /
            f"video_qim_step_{step}.mp4"
        )

        output_video.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        reconstruct_video(
            test_dir,
            output_video,
            fps=fps
        )

        # --------------------------------------------------
        # Decode first MP4 frame
        # --------------------------------------------------

        cap = cv2.VideoCapture(
            str(output_video)
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Unable to open {output_video}"
            )

        success, decoded = cap.read()

        cap.release()

        if not success:

            raise RuntimeError(
                "Unable to decode first frame."
            )

        decoded_blue = decoded[:, :, 0]

        # --------------------------------------------------
        # Measure MP4 distortion
        # --------------------------------------------------

        mp4_difference = (
            decoded_blue.astype(np.int16)
            -
            stego_frame[:, :, 0].astype(np.int16)
        )

        mp4_changed_pixels = np.count_nonzero(
            mp4_difference
        )

        mp4_max_difference = np.max(
            np.abs(mp4_difference)
        )

        mp4_mean_difference = np.mean(
            np.abs(mp4_difference)
        )

        # --------------------------------------------------
        # Extract after MP4
        # --------------------------------------------------

        decoded_bands = apply_dwt(
            decoded_blue
        )

        decoded_edge_map = generate_edge_map(
            decoded_bands["LH"]
        )

        recovered_bits = adaptive_extract(
            decoded_bands["LH"],
            decoded_edge_map,
            len(payload_bits)
        )

        video_pass = (
            recovered_bits == payload_bits
        )

        mismatch = find_mismatch(
            payload_bits,
            recovered_bits
        )

        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        print(
            f"Embedded Bits       : "
            f"{embedded}"
        )

        print(
            f"Direct Extraction   : "
            f"{'PASS' if direct_pass else 'FAIL'}"
        )

        print(
            f"Video Extraction    : "
            f"{'PASS' if video_pass else 'FAIL'}"
        )

        print(
            f"First Mismatch      : "
            f"{mismatch}"
        )

        print()
        print(
            "Spatial Distortion Before MP4"
        )

        print(
            f"Changed Pixels      : "
            f"{changed_pixels}"
        )

        print(
            f"Maximum Difference  : "
            f"{max_difference}"
        )

        print(
            f"Mean Difference     : "
            f"{mean_difference:.8f}"
        )

        print()
        print(
            "MP4 Distortion"
        )

        print(
            f"Changed Pixels      : "
            f"{mp4_changed_pixels}"
        )

        print(
            f"Maximum Difference  : "
            f"{mp4_max_difference}"
        )

        print(
            f"Mean Difference     : "
            f"{mp4_mean_difference:.8f}"
        )

        return {
            "step": step,
            "direct": direct_pass,
            "video": video_pass,
            "mismatch": mismatch,
            "changed_pixels": changed_pixels,
            "max_difference": max_difference,
            "mean_difference": mean_difference,
        }

    finally:

        lsb_utils.QUANTIZATION_STEP = old_step


def main():

    print("=" * 70)
    print("StegaFusion Video QIM Strength Diagnostic")
    print("=" * 70)

    # ------------------------------------------------------
    # Validate files
    # ------------------------------------------------------

    for path in (
        SECRET_FILE,
        KEY_FILE,
        SOURCE_FRAME,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Missing: {path}"
            )

    # ------------------------------------------------------
    # Create exact payload
    # ------------------------------------------------------

    encrypted_data = encrypt_file(
        SECRET_FILE,
        KEY_FILE
    )

    payload_bits = create_payload_packet(
        encrypted_data
    )

    # ------------------------------------------------------
    # Load frame
    # ------------------------------------------------------

    image = load_frame(
        SOURCE_FRAME
    )

    # ------------------------------------------------------
    # Read FPS
    # ------------------------------------------------------

    source_video = (
        PathConfig.COVER_VIDEO_DIR /
        "sample.mp4"
    )

    cap = cv2.VideoCapture(
        str(source_video)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Unable to open source video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    cap.release()

    print()
    print(
        f"Payload : "
        f"{len(payload_bits)} bits"
    )

    print(
        f"FPS     : "
        f"{fps}"
    )

    print(
        f"Frame   : "
        f"{image.shape}"
    )

    # ------------------------------------------------------
    # Test strengths
    # ------------------------------------------------------

    steps = [
        1,
        2,
        3,
        4,
        6,
        8,
        12,
        16,
        24,
        32,
    ]

    results = []

    for test_number, step in enumerate(
        steps,
        start=1
    ):

        result = run_test(
            payload_bits,
            image,
            step,
            test_number,
            fps,
        )

        results.append(
            result
        )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"{'STEP':>6} "
        f"{'DIRECT':>10} "
        f"{'VIDEO':>10} "
        f"{'MISMATCH':>10} "
        f"{'MAX IMG':>10} "
        f"{'MEAN IMG':>12}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['step']:>6} "
            f"{'PASS' if result['direct'] else 'FAIL':>10} "
            f"{'PASS' if result['video'] else 'FAIL':>10} "
            f"{str(result['mismatch']):>10} "
            f"{str(result['max_difference']):>10} "
            f"{str(round(result['mean_difference'], 6)):>12}"
        )

    print()
    print("=" * 70)
    print("Diagnostic Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()