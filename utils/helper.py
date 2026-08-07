"""
------------------------------------------------------------
StegaFusion Helper Module
------------------------------------------------------------
This module contains reusable utility functions used
throughout the StegaFusion project.

Author      : Yashwanth Gowda M
Version     : 1.0.0
------------------------------------------------------------
"""

from pathlib import Path
from datetime import datetime
import time
import shutil

from config.config import PathConfig


# ==========================================================
# DIRECTORY MANAGEMENT
# ==========================================================

def create_project_directories() -> None:
    """
    Creates all required project directories if they do not exist.
    """

    directories = [
        PathConfig.COVER_VIDEO_DIR,
        PathConfig.SECRET_DATA_DIR,
        PathConfig.KEY_DIR,
        PathConfig.STEGO_VIDEO_DIR,
        PathConfig.RECOVERED_DATA_DIR,
        PathConfig.REPORTS_DIR,
        PathConfig.FRAME_DIR,
        PathConfig.STEGO_FRAME_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ==========================================================
# FILE VALIDATION
# ==========================================================

def file_exists(file_path: Path) -> bool:
    """
    Checks whether a file exists.

    Args:
        file_path (Path): Path of the file.

    Returns:
        bool
    """
    return file_path.exists()


# ==========================================================
# TEXT FILE OPERATIONS
# ==========================================================

def read_text_file(file_path: Path) -> str:
    """
    Reads a text file.

    Args:
        file_path (Path)

    Returns:
        str
    """

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def write_text_file(file_path: Path, data: str) -> None:
    """
    Writes data into a text file.
    """

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)


# ==========================================================
# BINARY FILE OPERATIONS
# ==========================================================

def read_binary_file(file_path: Path) -> bytes:
    """
    Reads a binary file.
    """

    with open(file_path, "rb") as file:
        return file.read()


def write_binary_file(file_path: Path, data: bytes) -> None:
    """
    Writes binary data.
    """

    with open(file_path, "wb") as file:
        file.write(data)


# ==========================================================
# TIME UTILITIES
# ==========================================================

def current_timestamp() -> str:
    """
    Returns current timestamp.
    """

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def start_timer() -> float:
    """
    Starts execution timer.
    """

    return time.perf_counter()


def stop_timer(start_time: float) -> float:
    """
    Stops execution timer.

    Returns:
        Execution time in seconds.
    """

    return time.perf_counter() - start_time


# ==========================================================
# TEMP DIRECTORY MANAGEMENT
# ==========================================================

def clear_directory(directory: Path) -> None:
    """
    Deletes all files inside a directory.
    """

    if not directory.exists():
        return

    for item in directory.iterdir():

        if item.is_file():
            item.unlink()

        elif item.is_dir():
            shutil.rmtree(item)


# ==========================================================
# FILE INFORMATION
# ==========================================================

def get_file_extension(file_path: Path) -> str:
    """
    Returns file extension.
    """

    return file_path.suffix.lower()


def get_file_size(file_path: Path) -> float:
    """
    Returns file size in KB.
    """

    return file_path.stat().st_size / 1024


# ==========================================================
# CONSOLE UTILITIES
# ==========================================================

def print_header(title: str) -> None:
    """
    Prints formatted section header.
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_success(message: str) -> None:
    """
    Prints success message.
    """

    print(f"[SUCCESS] {message}")


def print_error(message: str) -> None:
    """
    Prints error message.
    """

    print(f"[ERROR] {message}")


def print_warning(message: str) -> None:
    """
    Prints warning message.
    """

    print(f"[WARNING] {message}")