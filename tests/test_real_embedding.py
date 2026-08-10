"""
------------------------------------------------------------
StegaFusion Real Payload Embedding Test
------------------------------------------------------------

Tests the complete path from a real secret file through:

    Secret File
        ↓
    AES-256 Encryption
        ↓
    Payload Packet
        ↓
    DWT
        ↓
    Edge Detection
        ↓
    Adaptive LSB Embedding
        ↓
    Inverse DWT
        ↓
    Rounded uint8 conversion
        ↓
    PNG Stego Frame

The exact payload used for embedding is saved so that the
extraction diagnostics can compare against the actual
payload rather than re-encrypting the file with a new IV.

------------------------------------------------------------
"""

from pathlib import Path

import numpy as np

from modules.crypto.aes_encrypt import encrypt_file

from modules.steganography.payload import (
    create_payload_packet,
)

from modules.steganography.lsb_embed import (
    load_frame,
    save_frame,
)

from modules.steganography.edge_detector import (
    generate_edge_map,
)

from modules.steganography.adaptive_lsb import (
    adaptive_embed,
    calculate_capacity,
)

from modules.steganography.lsb_utils import (
    QUANTIZATION_STEP,
)

from modules.transform.wavelet_utils import (
    apply_dwt,
    apply_inverse_dwt,
)


# ==========================================================
# PATHS
# ==========================================================

SECRET_FILE = Path(
    "input/secret_data/test_secret.txt"
)

KEY_FILE = Path(
    "input/keys/test_integration_key.bin"
)

FRAME_FILE = Path(
    "temp/frames/frame_00000.png"
)

OUTPUT_FILE = Path(
    "temp/stego_frames/real_payload_frame.png"
)

EXPECTED_PAYLOAD_FILE = Path(
    "temp/test_payload.txt"
)


# ==========================================================
# MAIN TEST
# ==========================================================

def main():

    print("=" * 70)
    print("StegaFusion Real Payload Embedding Test")
    print("=" * 70)

    # ------------------------------------------------------
    # Validate input files
    # ------------------------------------------------------

    if not SECRET_FILE.exists():
        raise FileNotFoundError(
            f"Secret file not found: {SECRET_FILE}"
        )

    if not KEY_FILE.exists():
        raise FileNotFoundError(
            f"Key file not found: {KEY_FILE}"
        )

    if not FRAME_FILE.exists():
        raise FileNotFoundError(
            f"Frame not found: {FRAME_FILE}"
        )

    # ------------------------------------------------------
    # Encrypt secret file
    # ------------------------------------------------------

    encrypted_data = encrypt_file(
        SECRET_FILE,
        KEY_FILE
    )

    # ------------------------------------------------------
    # Create payload packet
    # ------------------------------------------------------

    payload_bits = create_payload_packet(
        encrypted_data
    )

    # ------------------------------------------------------
    # Save EXACT payload used for embedding
    # ------------------------------------------------------

    EXPECTED_PAYLOAD_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    EXPECTED_PAYLOAD_FILE.write_text(
        payload_bits,
        encoding="ascii"
    )

    # ------------------------------------------------------
    # Display payload information
    # ------------------------------------------------------

    print()

    print(
        f"Encrypted Data : "
        f"{len(encrypted_data)} bytes"
    )

    print(
        f"Payload Bits   : "
        f"{len(payload_bits)} bits"
    )

    print(
        f"Expected Payload Saved : "
        f"{EXPECTED_PAYLOAD_FILE}"
    )

    # ------------------------------------------------------
    # Load cover frame
    # ------------------------------------------------------

    image = load_frame(
        FRAME_FILE
    )

    # ------------------------------------------------------
    # Extract blue channel
    # ------------------------------------------------------

    blue_channel = image[:, :, 0]

    print()

    print("Blue Channel")

    print(
        f"Shape : {blue_channel.shape}"
    )

    print(
        f"Dtype : {blue_channel.dtype}"
    )

    print(
        f"Range : "
        f"{blue_channel.min()} -> "
        f"{blue_channel.max()}"
    )

    # ------------------------------------------------------
    # Apply DWT
    # ------------------------------------------------------

    bands = apply_dwt(
        blue_channel
    )

    # ------------------------------------------------------
    # Generate edge map
    # ------------------------------------------------------

    edge_map = generate_edge_map(
        bands["LH"]
    )

    # ------------------------------------------------------
    # Calculate capacity
    # ------------------------------------------------------

    capacity = calculate_capacity(
        edge_map
    )

    print()

    print(
        f"Capacity       : "
        f"{capacity} bits"
    )

    print(
        f"Quantization Step: "
        f"{QUANTIZATION_STEP}"
    )

    print(
        f"Edge Pixels     : "
        f"{np.count_nonzero(edge_map)}"
    )

    print(
        f"Smooth Pixels   : "
        f"{edge_map.size - np.count_nonzero(edge_map)}"
    )

    # ------------------------------------------------------
    # Capacity validation
    # ------------------------------------------------------

    if len(payload_bits) > capacity:

        raise ValueError(
            "Payload does not fit inside "
            "the selected frame."
        )

    # ------------------------------------------------------
    # Embed payload
    # ------------------------------------------------------

    stego_lh, embedded_bits = adaptive_embed(
        bands["LH"],
        edge_map,
        payload_bits
    )

    # ------------------------------------------------------
    # Verify embedding count
    # ------------------------------------------------------

    if embedded_bits != len(payload_bits):

        raise RuntimeError(
            f"Only {embedded_bits} of "
            f"{len(payload_bits)} bits were embedded."
        )

    # ------------------------------------------------------
    # Store modified LH band
    # ------------------------------------------------------

    bands["LH"] = stego_lh

    # ------------------------------------------------------
    # Inverse DWT
    # ------------------------------------------------------

    reconstructed_blue = apply_inverse_dwt(
        bands
    )

    # ------------------------------------------------------
    # IMPORTANT:
    #
    # Round before uint8 conversion.
    #
    # Direct astype(np.uint8) truncates floating-point
    # values and can alter the DWT coefficients after
    # reconstruction, especially for the low-range real
    # video frame.
    # ------------------------------------------------------

    reconstructed_blue = np.round(
        reconstructed_blue
    ).clip(
        0,
        255
    ).astype(np.uint8)

    # ------------------------------------------------------
    # Build stego frame
    # ------------------------------------------------------

    stego_frame = image.copy()

    stego_frame[:, :, 0] = reconstructed_blue

    # ------------------------------------------------------
    # Save stego frame
    # ------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_frame(
        stego_frame,
        OUTPUT_FILE
    )

    # ------------------------------------------------------
    # Display results
    # ------------------------------------------------------

    print()

    print(
        f"Embedded Bits   : "
        f"{embedded_bits}"
    )

    print(
        f"Output          : "
        f"{OUTPUT_FILE}"
    )

    print()

    print(
        "Real Payload Embedding Test PASSED!"
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()