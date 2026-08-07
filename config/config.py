"""
------------------------------------------------------------
StegaFusion Configuration Module
------------------------------------------------------------
This module stores all global configuration settings required
throughout the StegaFusion project.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path


# ==========================================================
# PROJECT INFORMATION
# ==========================================================

class ProjectConfig:
    """Stores project information."""

    PROJECT_NAME: str = "StegaFusion"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Yashwanth Gowda M"

    DESCRIPTION: str = (
        "A Multi-Domain Hybrid Framework for Secure Video Data Hiding "
        "using AES Encryption, DWT, and Adaptive LSB Steganography."
    )


# ==========================================================
# PROJECT PATHS
# ==========================================================

class PathConfig:
    """Stores all directory paths used in the project."""

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    CONFIG_DIR: Path = PROJECT_ROOT / "config"
    DOCS_DIR: Path = PROJECT_ROOT / "docs"

    INPUT_DIR: Path = PROJECT_ROOT / "input"
    OUTPUT_DIR: Path = PROJECT_ROOT / "output"
    TEMP_DIR: Path = PROJECT_ROOT / "temp"

    MODULES_DIR: Path = PROJECT_ROOT / "modules"
    UTILS_DIR: Path = PROJECT_ROOT / "utils"
    TESTS_DIR: Path = PROJECT_ROOT / "tests"

    # Input folders
    COVER_VIDEO_DIR: Path = INPUT_DIR / "cover_video"
    SECRET_DATA_DIR: Path = INPUT_DIR / "secret_data"
    KEY_DIR: Path = INPUT_DIR / "keys"

    # Output folders
    STEGO_VIDEO_DIR: Path = OUTPUT_DIR / "stego_video"
    RECOVERED_DATA_DIR: Path = OUTPUT_DIR / "recovered_data"
    REPORTS_DIR: Path = OUTPUT_DIR / "reports"

    # Temporary folders
    FRAME_DIR: Path = TEMP_DIR / "frames"
    STEGO_FRAME_DIR: Path = TEMP_DIR / "stego_frames"


# ==========================================================
# AES CONFIGURATION
# ==========================================================

class AESConfig:
    """AES Encryption Settings."""

    KEY_SIZE: int = 32          # AES-256
    BLOCK_SIZE: int = 16
    MODE: str = "CBC"
    IV_SIZE: int = 16
    PADDING: str = "PKCS7"


# ==========================================================
# VIDEO CONFIGURATION
# ==========================================================

class VideoConfig:
    """Video Processing Settings."""

    SUPPORTED_FORMATS = [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    ]

    FRAME_FORMAT: str = ".png"
    VIDEO_CODEC: str = "mp4v"


# ==========================================================
# DWT CONFIGURATION
# ==========================================================

class DWTConfig:
    """Discrete Wavelet Transform Settings."""

    WAVELET: str = "haar"
    LEVEL: int = 1


# ==========================================================
# STEGANOGRAPHY CONFIGURATION
# ==========================================================

class EmbeddingConfig:
    """Adaptive LSB Embedding Settings."""

    LSB_BITS: int = 1
    MAX_PAYLOAD_PERCENTAGE: float = 0.30


# ==========================================================
# DEBUG CONFIGURATION
# ==========================================================

class DebugConfig:
    """Debugging Options."""

    DEBUG: bool = True
    VERBOSE: bool = True
    SAVE_INTERMEDIATE_FRAMES: bool = True