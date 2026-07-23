"""File, video, and overlay helper functions for Task A."""

from pathlib import Path

import cv2
import numpy as np


# ------------------------------ OVERLAY SETTINGS -----------------------------

TALKING_WIDTH_RATIO = 0.28
OVERLAY_MARGIN = 20
WATERMARK_OPACITY = 0.75


# Purpose: Confirm that a required file exists before processing starts.
def require_existing_file(file_path, label):
    """Return the file as a Path or raise a clear error if it is missing."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found: {path}")
    return path


# Purpose: Open a video and stop with a clear error if OpenCV cannot read it.
def open_video(video_path, label):
    """Return a validated OpenCV VideoCapture object."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {label}: {video_path}")
    return capture


# Purpose: Prevent an old result from being overwritten by adding _v2, _v3, etc.
def create_available_output_path(output_directory, input_path):
    """Return the first unused output filename for an input video."""
    output_path = output_directory / f"{input_path.stem}_processed.avi"
    version = 2

    while output_path.exists():
        output_path = output_directory / f"{input_path.stem}_processed_v{version}.avi"
        version += 1

    return output_path


# Purpose: Read the next talking-video frame and restart it when the clip ends.
def read_looping_frame(capture):
    """Return the next frame, the first frame after looping, or None."""
    success, frame = capture.read()
    if success:
        return frame

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, frame = capture.read()
    return frame if success else None


# Purpose: Resize and place the talking video at the top-left of the main frame.
def overlay_talking_video(main_frame, talking_frame):
    """Add a proportional picture-in-picture inset with a white border."""
    main_height, main_width = main_frame.shape[:2]
    talking_height, talking_width = talking_frame.shape[:2]

    inset_width = max(1, int(main_width * TALKING_WIDTH_RATIO))
    inset_height = max(1, int(inset_width * talking_height / talking_width))

    # Keep the inset inside unusually small or narrow main-video frames.
    maximum_height = max(1, main_height - 2 * OVERLAY_MARGIN)
    if inset_height > maximum_height:
        inset_height = maximum_height
        inset_width = max(1, int(inset_height * talking_width / talking_height))

    inset = cv2.resize(
        talking_frame,
        (inset_width, inset_height),
        interpolation=cv2.INTER_AREA,
    )

    x1, y1 = OVERLAY_MARGIN, OVERLAY_MARGIN
    x2, y2 = x1 + inset_width, y1 + inset_height
    main_frame[y1:y2, x1:x2] = inset

    # The border keeps the inset visible on both dark and bright backgrounds.
    cv2.rectangle(main_frame, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)
    return main_frame


# Purpose: Load and resize a watermark once so it matches the output video.
def prepare_watermark(watermark_path, output_size):
    """Return a full-frame BGR watermark at the requested size."""
    watermark = cv2.imread(str(watermark_path), cv2.IMREAD_COLOR)
    if watermark is None:
        raise RuntimeError(f"Could not read watermark image: {watermark_path}")

    width, height = output_size
    return cv2.resize(watermark, (width, height), interpolation=cv2.INTER_AREA)


# Purpose: Blend the visible watermark while treating black pixels as transparent.
def overlay_black_key_watermark(frame, watermark):
    """Apply the three-channel PNG watermark using a black-key mask."""
    non_black_mask = np.max(watermark, axis=2) > 3
    blended_frame = cv2.addWeighted(
        frame,
        1.0 - WATERMARK_OPACITY,
        watermark,
        WATERMARK_OPACITY,
        0,
    )
    frame[non_black_mask] = blended_frame[non_black_mask]
    return frame
