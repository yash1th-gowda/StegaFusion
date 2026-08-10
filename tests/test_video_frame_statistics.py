"""
StegaFusion Video Frame Statistics Diagnostic

Examines the actual source video frames to determine whether
the extremely small blue-channel range is present throughout
the video or only in the first frame.
"""

from pathlib import Path

import cv2
import numpy as np

from config.config import PathConfig


def main():

    print("=" * 70)
    print("StegaFusion Video Frame Statistics")
    print("=" * 70)

    video = (
        PathConfig.COVER_VIDEO_DIR /
        "sample.mp4"
    )

    if not video.exists():
        raise FileNotFoundError(
            f"Video not found: {video}"
        )

    cap = cv2.VideoCapture(
        str(video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video}"
        )

    frame_count = 0

    blue_mins = []
    blue_maxs = []
    blue_means = []
    blue_unique_counts = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        blue = frame[:, :, 0]

        blue_mins.append(
            int(blue.min())
        )

        blue_maxs.append(
            int(blue.max())
        )

        blue_means.append(
            float(blue.mean())
        )

        blue_unique_counts.append(
            int(np.unique(blue).size)
        )

        frame_count += 1

    cap.release()

    if frame_count == 0:
        raise RuntimeError(
            "No frames were read."
        )

    blue_mins = np.array(
        blue_mins
    )

    blue_maxs = np.array(
        blue_maxs
    )

    blue_means = np.array(
        blue_means
    )

    blue_unique_counts = np.array(
        blue_unique_counts
    )

    print()
    print(
        f"Video : {video}"
    )

    print(
        f"Frames : {frame_count}"
    )

    print()
    print("BLUE CHANNEL STATISTICS")
    print("-" * 70)

    print(
        f"Global Minimum : "
        f"{blue_mins.min()}"
    )

    print(
        f"Global Maximum : "
        f"{blue_maxs.max()}"
    )

    print(
        f"Mean of Frame Means : "
        f"{blue_means.mean():.6f}"
    )

    print(
        f"Minimum Frame Mean : "
        f"{blue_means.min():.6f}"
    )

    print(
        f"Maximum Frame Mean : "
        f"{blue_means.max():.6f}"
    )

    print(
        f"Minimum Unique Count : "
        f"{blue_unique_counts.min()}"
    )

    print(
        f"Maximum Unique Count : "
        f"{blue_unique_counts.max()}"
    )

    print()
    print("PER-FRAME SUMMARY")
    print("-" * 70)

    print(
        f"{'Frame':>7} "
        f"{'Min':>5} "
        f"{'Max':>5} "
        f"{'Mean':>12} "
        f"{'Unique':>8}"
    )

    for index in range(frame_count):

        print(
            f"{index:7d} "
            f"{blue_mins[index]:5d} "
            f"{blue_maxs[index]:5d} "
            f"{blue_means[index]:12.6f} "
            f"{blue_unique_counts[index]:8d}"
        )

    print()
    print("=" * 70)

    if (
        blue_mins.max() <= 10
        and
        blue_maxs.max() <= 10
    ):

        print(
            "WARNING: The entire video has an "
            "extremely low blue-channel range."
        )

    else:

        print(
            "The blue-channel range varies "
            "across the video."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()