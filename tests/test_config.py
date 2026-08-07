"""
------------------------------------------------------------
StegaFusion Configuration Test
------------------------------------------------------------
Tests all configuration settings.
------------------------------------------------------------
"""

from config.config import (
    ProjectConfig,
    PathConfig,
    AESConfig,
    VideoConfig,
    DWTConfig,
    EmbeddingConfig,
    DebugConfig,
)


def main():

    print("=" * 70)
    print("StegaFusion Configuration Test")
    print("=" * 70)

    print("\nPROJECT INFORMATION")
    print("-" * 70)
    print(f"Project Name      : {ProjectConfig.PROJECT_NAME}")
    print(f"Version           : {ProjectConfig.VERSION}")
    print(f"Author            : {ProjectConfig.AUTHOR}")
    print(f"Description       : {ProjectConfig.DESCRIPTION}")

    print("\nDIRECTORY PATHS")
    print("-" * 70)
    print(f"Project Root      : {PathConfig.PROJECT_ROOT}")
    print(f"Input Directory   : {PathConfig.INPUT_DIR}")
    print(f"Output Directory  : {PathConfig.OUTPUT_DIR}")
    print(f"Frame Directory   : {PathConfig.FRAME_DIR}")

    print("\nAES SETTINGS")
    print("-" * 70)
    print(f"Key Size          : {AESConfig.KEY_SIZE}")
    print(f"Mode              : {AESConfig.MODE}")
    print(f"Block Size        : {AESConfig.BLOCK_SIZE}")

    print("\nVIDEO SETTINGS")
    print("-" * 70)
    print(f"Codec             : {VideoConfig.VIDEO_CODEC}")
    print(f"Frame Format      : {VideoConfig.FRAME_FORMAT}")

    print("\nDWT SETTINGS")
    print("-" * 70)
    print(f"Wavelet           : {DWTConfig.WAVELET}")
    print(f"Level             : {DWTConfig.LEVEL}")

    print("\nEMBEDDING SETTINGS")
    print("-" * 70)
    print(f"LSB Bits          : {EmbeddingConfig.LSB_BITS}")
    print(f"Payload           : {EmbeddingConfig.MAX_PAYLOAD_PERCENTAGE}")

    print("\nDEBUG SETTINGS")
    print("-" * 70)
    print(f"Debug             : {DebugConfig.DEBUG}")
    print(f"Verbose           : {DebugConfig.VERBOSE}")

    print("\n")
    print("=" * 70)
    print("CONFIGURATION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()