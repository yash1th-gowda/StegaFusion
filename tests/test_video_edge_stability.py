"""
StegaFusion Video Edge Stability Diagnostic

Determines whether MP4 compression changes the adaptive
edge map used for DWT-domain extraction.

This is diagnostic only.
It does not modify production code.
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
from modules.transform.wavelet_utils import apply_dwt
from modules.transform.wavelet_utils import apply_inverse_dwt
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

OUTPUT_DIR = Path(
    "output"
)

TEMP_DIR = Path(
    "temp/video_edge_stability"
)


def first_mismatch(expected, recovered):

    limit = min(
        len(expected),
        len(recovered)
    )

    for index in range(limit):

        if expected[index] != recovered[index]:
            return index

    if len(expected) != len(recovered):
        return limit

    return None


def main():

    print("=" * 70)
    print("StegaFusion Video Edge Stability Diagnostic")
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

    print()
    print(
        f"Payload : {len(payload_bits)} bits"
    )

    # ------------------------------------------------------
    # Load source frame
    # ------------------------------------------------------

    original_image = load_frame(
        SOURCE_FRAME
    )

    original_blue = (
        original_image[:, :, 0]
    )

    print(
        f"Frame   : {original_image.shape}"
    )

    print(
        f"Blue    : "
        f"{original_blue.min()} -> "
        f"{original_blue.max()}"
    )

    # ------------------------------------------------------
    # DWT
    # ------------------------------------------------------

    original_bands = apply_dwt(
        original_blue
    )

    original_lh = original_bands["LH"]

    # ------------------------------------------------------
    # ORIGINAL EDGE MAP
    # ------------------------------------------------------

    original_edge_map = generate_edge_map(
        original_lh
    )

    original_edge_pixels = np.count_nonzero(
        original_edge_map
    )

    print()
    print(
        f"Original Edge Pixels : "
        f"{original_edge_pixels}"
    )

    # ------------------------------------------------------
    # Use a known diagnostic step
    # ------------------------------------------------------

    old_step = lsb_utils.QUANTIZATION_STEP

    lsb_utils.QUANTIZATION_STEP = 4

    try:

        # --------------------------------------------------
        # Embed
        # --------------------------------------------------

        stego_lh, embedded = adaptive_embed(
            original_lh,
            original_edge_map,
            payload_bits
        )

        print(
            f"Embedded Bits       : "
            f"{embedded}"
        )

        # --------------------------------------------------
        # Direct extraction using ORIGINAL edge map
        # --------------------------------------------------

        direct_original_map = adaptive_extract(
            stego_lh,
            original_edge_map,
            len(payload_bits)
        )

        print()
        print(
            "Direct Extraction using Original Map :",
            "PASS"
            if direct_original_map == payload_bits
            else "FAIL"
        )

        # --------------------------------------------------
        # Reconstruct stego frame
        # --------------------------------------------------

        stego_bands = dict(
            original_bands
        )

        stego_bands["LH"] = stego_lh

        reconstructed_blue = (
            apply_inverse_dwt(
                stego_bands
            )
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
        # Prepare 197 frames
        # --------------------------------------------------

        if TEMP_DIR.exists():

            shutil.rmtree(
                TEMP_DIR
            )

        TEMP_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        source_frames = sorted(
            FRAME_DIR.glob("*.png")
        )

        print()
        print(
            f"Source Frames : "
            f"{len(source_frames)}"
        )

        for frame_path in source_frames:

            shutil.copy2(
                frame_path,
                TEMP_DIR / frame_path.name
            )

        save_frame(
            stego_frame,
            TEMP_DIR / "frame_00000.png"
        )

        # --------------------------------------------------
        # Get source FPS
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Encode MP4
        # --------------------------------------------------

        output_video = (
            OUTPUT_DIR /
            "video_edge_stability.mp4"
        )

        output_video.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        reconstruct_video(
            TEMP_DIR,
            output_video,
            fps=fps
        )

        print(
            f"MP4 : {output_video}"
        )

        # --------------------------------------------------
        # Decode first frame
        # --------------------------------------------------

        cap = cv2.VideoCapture(
            str(output_video)
        )

        if not cap.isOpened():

            raise RuntimeError(
                "Unable to open generated MP4."
            )

        success, decoded = cap.read()

        cap.release()

        if not success:

            raise RuntimeError(
                "Unable to decode generated MP4."
            )

        decoded_blue = (
            decoded[:, :, 0]
        )

        # --------------------------------------------------
        # Decode DWT
        # --------------------------------------------------

        decoded_bands = apply_dwt(
            decoded_blue
        )

        decoded_lh = decoded_bands["LH"]

        # --------------------------------------------------
        # REGENERATED EDGE MAP
        # --------------------------------------------------

        decoded_edge_map = generate_edge_map(
            decoded_lh
        )

        decoded_edge_pixels = np.count_nonzero(
            decoded_edge_map
        )

        print()
        print(
            f"Decoded Edge Pixels : "
            f"{decoded_edge_pixels}"
        )

        # --------------------------------------------------
        # Compare edge maps
        # --------------------------------------------------

        edge_difference = (
            original_edge_map
            != decoded_edge_map
        )

        changed_edges = np.count_nonzero(
            edge_difference
        )

        total_pixels = (
            original_edge_map.size
        )

        edge_difference_percent = (
            changed_edges /
            total_pixels *
            100
        )

        print()
        print(
            "EDGE MAP COMPARISON"
        )

        print(
            f"Different Pixels : "
            f"{changed_edges}"
        )

        print(
            f"Difference       : "
            f"{edge_difference_percent:.6f}%"
        )

        # --------------------------------------------------
        # Extraction using ORIGINAL edge map
        #
        # This is the critical experiment.
        # --------------------------------------------------

        recovered_original_map = (
            adaptive_extract(
                decoded_lh,
                original_edge_map,
                len(payload_bits)
            )
        )

        mismatch_original_map = (
            first_mismatch(
                payload_bits,
                recovered_original_map
            )
        )

        print()
        print(
            "EXTRACTION USING ORIGINAL EDGE MAP"
        )

        print(
            f"Recovered Bits : "
            f"{len(recovered_original_map)}"
        )

        print(
            f"First Mismatch : "
            f"{mismatch_original_map}"
        )

        if recovered_original_map == payload_bits:

            print(
                "Result : PASS"
            )

        else:

            print(
                "Result : FAIL"
            )

        # --------------------------------------------------
        # Extraction using regenerated map
        # --------------------------------------------------

        recovered_decoded_map = (
            adaptive_extract(
                decoded_lh,
                decoded_edge_map,
                len(payload_bits)
            )
        )

        mismatch_decoded_map = (
            first_mismatch(
                payload_bits,
                recovered_decoded_map
            )
        )

        print()
        print(
            "EXTRACTION USING REGENERATED EDGE MAP"
        )

        print(
            f"Recovered Bits : "
            f"{len(recovered_decoded_map)}"
        )

        print(
            f"First Mismatch : "
            f"{mismatch_decoded_map}"
        )

        if recovered_decoded_map == payload_bits:

            print(
                "Result : PASS"
            )

        else:

            print(
                "Result : FAIL"
            )

        # --------------------------------------------------
        # Compare DWT coefficients
        # --------------------------------------------------

        coefficient_difference = (
            decoded_lh -
            stego_lh
        )

        print()
        print(
            "DWT COEFFICIENT DISTORTION"
        )

        print(
            f"Maximum Difference : "
            f"{np.max(np.abs(coefficient_difference)):.6f}"
        )

        print(
            f"Mean Absolute Diff  : "
            f"{np.mean(np.abs(coefficient_difference)):.6f}"
        )

        print(
            f"Changed Coeffs     : "
            f"{np.count_nonzero(coefficient_difference)}"
        )

        # --------------------------------------------------
        # Final interpretation
        # --------------------------------------------------

        print()
        print("=" * 70)
        print("INTERPRETATION")
        print("=" * 70)

        if (
            recovered_original_map
            == payload_bits
        ):

            print(
                "The payload survives when the ORIGINAL "
                "edge map is used."
            )

            print()
            print(
                "Therefore the main problem is EDGE MAP "
                "INSTABILITY after MP4 compression."
            )

        else:

            print(
                "The payload does NOT survive even with "
                "the ORIGINAL edge map."
            )

            print()
            print(
                "Therefore MP4 compression is directly "
                "corrupting the embedded DWT coefficients."
            )

        print()
        print(
            "Do NOT change production QIM yet."
        )

        print("=" * 70)

    finally:

        lsb_utils.QUANTIZATION_STEP = old_step


if __name__ == "__main__":
    main()