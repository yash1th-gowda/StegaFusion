"""
StegaFusion Lossless Video Round-Trip Diagnostic

Tests whether the existing DWT/QIM payload survives a video
encode/decode path without lossy MP4 compression.

This is diagnostic only.
It does not modify production configuration.
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

TEMP_DIR = Path(
    "temp/lossless_video_frames"
)

OUTPUT_VIDEO = Path(
    "output/lossless_test.avi"
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
    print("StegaFusion Lossless Video Round-Trip Diagnostic")
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
        f"Encrypted Data : "
        f"{len(encrypted_data)} bytes"
    )

    print(
        f"Payload Bits   : "
        f"{len(payload_bits)} bits"
    )

    # ------------------------------------------------------
    # Load source frame
    # ------------------------------------------------------

    image = load_frame(
        SOURCE_FRAME
    )

    blue = image[:, :, 0]

    print()
    print("Source Frame")
    print(
        f"Shape : {image.shape}"
    )

    print(
        f"Dtype : {image.dtype}"
    )

    print(
        f"Blue Range : "
        f"{blue.min()} -> {blue.max()}"
    )

    # ------------------------------------------------------
    # DWT
    # ------------------------------------------------------

    bands = apply_dwt(
        blue
    )

    original_lh = bands["LH"].copy()

    # ------------------------------------------------------
    # Edge map
    # ------------------------------------------------------

    edge_map = generate_edge_map(
        original_lh
    )

    print()
    print(
        f"Edge Pixels : "
        f"{np.count_nonzero(edge_map)}"
    )

    # ------------------------------------------------------
    # Use the currently validated production setting
    # ------------------------------------------------------

    print(
        f"Quantization Step : "
        f"{lsb_utils.QUANTIZATION_STEP}"
    )

    # ------------------------------------------------------
    # Embed
    # ------------------------------------------------------

    stego_lh, embedded = adaptive_embed(
        original_lh,
        edge_map,
        payload_bits
    )

    if embedded != len(payload_bits):

        raise RuntimeError(
            f"Only {embedded} of "
            f"{len(payload_bits)} bits embedded."
        )

    # ------------------------------------------------------
    # Direct extraction
    # ------------------------------------------------------

    direct_bits = adaptive_extract(
        stego_lh,
        edge_map,
        len(payload_bits)
    )

    print()
    print(
        "Direct Extraction :",
        "PASS"
        if direct_bits == payload_bits
        else "FAIL"
    )

    # ------------------------------------------------------
    # Inverse DWT
    # ------------------------------------------------------

    bands["LH"] = stego_lh

    reconstructed_blue = apply_inverse_dwt(
        bands
    )

    reconstructed_blue = np.clip(
        np.rint(reconstructed_blue),
        0,
        255
    ).astype(np.uint8)

    stego_frame = image.copy()

    stego_frame[:, :, 0] = (
        reconstructed_blue
    )

    # ------------------------------------------------------
    # Spatial difference before video
    # ------------------------------------------------------

    spatial_difference = (
        stego_frame[:, :, 0].astype(np.int16)
        -
        image[:, :, 0].astype(np.int16)
    )

    print()
    print("Before Video Encoding")

    print(
        f"Changed Pixels : "
        f"{np.count_nonzero(spatial_difference)}"
    )

    print(
        f"Maximum Difference : "
        f"{np.max(np.abs(spatial_difference))}"
    )

    print(
        f"Mean Difference : "
        f"{np.mean(np.abs(spatial_difference)):.10f}"
    )

    # ------------------------------------------------------
    # Prepare frames
    # ------------------------------------------------------

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

    if not source_frames:

        raise FileNotFoundError(
            "No extracted frames found."
        )

    for frame_path in source_frames:

        shutil.copy2(
            frame_path,
            TEMP_DIR / frame_path.name
        )

    # Replace first frame with stego frame
    save_frame(
        stego_frame,
        TEMP_DIR / "frame_00000.png"
    )

    print()
    print(
        f"Frames Prepared : "
        f"{len(source_frames)}"
    )

    # ------------------------------------------------------
    # IMPORTANT
    #
    # OpenCV's standard VideoWriter does not provide a
    # guaranteed lossless codec through mp4v.
    #
    # We therefore first test FFV1 availability.
    # ------------------------------------------------------

    output_video = OUTPUT_VIDEO

    output_video.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    first_frame = cv2.imread(
        str(TEMP_DIR / "frame_00000.png")
    )

    height, width = first_frame.shape[:2]

    fps = 23.976023976023978

    fourcc = cv2.VideoWriter_fourcc(
        *"FFV1"
    )

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        print()
        print(
            "FFV1 codec is not available through "
            "this OpenCV installation."
        )

        print()
        print(
            "The diagnostic cannot continue."
        )

        print(
            "We need to use another lossless encoder."
        )

        return

    # ------------------------------------------------------
    # Write all frames
    # ------------------------------------------------------

    for frame_path in source_frames:

        frame = cv2.imread(
            str(TEMP_DIR / frame_path.name)
        )

        if frame is None:

            raise RuntimeError(
                f"Unable to read {frame_path}"
            )

        writer.write(
            frame
        )

    writer.release()

    print()
    print(
        f"Lossless Video : "
        f"{output_video}"
    )

    # ------------------------------------------------------
    # Decode first frame
    # ------------------------------------------------------

    cap = cv2.VideoCapture(
        str(output_video)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Unable to open lossless output video."
        )

    success, decoded = cap.read()

    cap.release()

    if not success:

        raise RuntimeError(
            "Unable to decode first frame."
        )

    decoded_blue = decoded[:, :, 0]

    print()
    print("Decoded Frame")

    print(
        f"Shape : {decoded.shape}"
    )

    print(
        f"Dtype : {decoded.dtype}"
    )

    print(
        f"Blue Range : "
        f"{decoded_blue.min()} -> "
        f"{decoded_blue.max()}"
    )

    # ------------------------------------------------------
    # Compare decoded frame against stego frame
    # ------------------------------------------------------

    video_difference = (
        decoded_blue.astype(np.int16)
        -
        stego_frame[:, :, 0].astype(np.int16)
    )

    print()
    print("Lossless Video Distortion")

    print(
        f"Changed Pixels : "
        f"{np.count_nonzero(video_difference)}"
    )

    print(
        f"Maximum Difference : "
        f"{np.max(np.abs(video_difference))}"
    )

    print(
        f"Mean Difference : "
        f"{np.mean(np.abs(video_difference)):.10f}"
    )

    # ------------------------------------------------------
    # DWT after video
    # ------------------------------------------------------

    decoded_bands = apply_dwt(
        decoded_blue
    )

    decoded_edge_map = generate_edge_map(
        decoded_bands["LH"]
    )

    # ------------------------------------------------------
    # Extract
    # ------------------------------------------------------

    recovered_bits = adaptive_extract(
        decoded_bands["LH"],
        decoded_edge_map,
        len(payload_bits)
    )

    mismatch = first_mismatch(
        payload_bits,
        recovered_bits
    )

    print()
    print(
        f"Recovered Bits : "
        f"{len(recovered_bits)}"
    )

    print(
        f"First Mismatch : "
        f"{mismatch}"
    )

    if recovered_bits == payload_bits:

        print()
        print(
            "LOSSLESS VIDEO EXTRACTION : PASS"
        )

        print()
        print("=" * 70)
        print(
            "LOSSLESS VIDEO ROUND-TRIP PASSED!"
        )
        print("=" * 70)

    else:

        print()
        print(
            "LOSSLESS VIDEO EXTRACTION : FAIL"
        )

        print()
        print(
            "The current embedding survives PNG "
            "but not this video codec."
        )

        print()
        print("=" * 70)
        print(
            "LOSSLESS VIDEO ROUND-TRIP FAILED"
        )
        print("=" * 70)


if __name__ == "__main__":
    main()