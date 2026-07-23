"""Brightness, face detection, and complete video-processing functions."""

import cv2
import numpy as np

from video_utils import (
    open_video,
    overlay_black_key_watermark,
    overlay_talking_video,
    prepare_watermark,
    read_looping_frame,
)


# ---------------------------- PROCESSING SETTINGS ----------------------------

TARGET_NIGHT_BRIGHTNESS = 115.0
MAX_BRIGHTNESS_INCREASE = 60
FALLBACK_FPS = 30.0

# Create CLAHE once because the same settings are used for every night frame.
NIGHT_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# Purpose: Sample frames across a video to estimate its overall brightness.
def estimate_video_brightness(video_path, sample_count=60):
    """Return the average greyscale intensity of representative frames."""
    capture = open_video(video_path, "input video")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    brightness_values = []

    if total_frames > 0:
        sample_total = min(sample_count, total_frames)
        frame_indices = np.linspace(
            0, total_frames - 1, sample_total, dtype=np.int32
        )

        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if success:
                grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness_values.append(float(np.mean(grey_frame)))
    else:
        # Some video containers do not report their total number of frames.
        frame_index = 0
        while len(brightness_values) < sample_count:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % 30 == 0:
                grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness_values.append(float(np.mean(grey_frame)))
            frame_index += 1

    capture.release()

    if not brightness_values:
        raise RuntimeError(f"No readable frames were found in {video_path}")

    return float(np.mean(brightness_values))


# Purpose: Calculate how much a dark video should be brightened within safe limits.
def calculate_brightness_increase(mean_brightness):
    """Return a limited additive brightness increase for a nighttime video."""
    needed_increase = int(round(TARGET_NIGHT_BRIGHTNESS - mean_brightness))
    return max(0, min(MAX_BRIGHTNESS_INCREASE, needed_increase))


# Purpose: Brighten one BGR frame without allowing colour values to exceed 255.
def brighten_frame(frame, increase):
    """Return the original frame or a safely brightened copy."""
    if increase <= 0:
        return frame
    return cv2.convertScaleAbs(frame, alpha=1.0, beta=increase)


# Purpose: Load and check the Haar cascade used for frontal-face detection.
def create_face_detector(cascade_path):
    """Return a ready-to-use OpenCV CascadeClassifier."""
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
    return detector


# Purpose: Detect frontal faces and blur every detected face for privacy.
def blur_frontal_faces(frame, detector, is_night):
    """Return the processed frame and the number of faces detected."""
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE improves local contrast at night; plain greyscale is used by day.
    detection_image = NIGHT_CLAHE.apply(grey_frame) if is_night else grey_frame

    faces = detector.detectMultiScale(
        detection_image,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(30, 30),
    )

    for x, y, width, height in faces:
        face_region = frame[y : y + height, x : x + width]

        # Gaussian kernels must be positive, odd, and fit inside the face area.
        kernel_size = min(width, height)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel_size = max(3, kernel_size)

        frame[y : y + height, x : x + width] = cv2.GaussianBlur(
            face_region,
            (kernel_size, kernel_size),
            sigmaX=0,
        )

    return frame, len(faces)


# Purpose: Resize and append the full end-screen clip after the main video.
def append_end_screen(writer, endscreen_path, output_size):
    """Write every readable end-screen frame and return the frame count."""
    capture = open_video(endscreen_path, "end-screen video")
    appended_frames = 0

    while True:
        success, frame = capture.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
        writer.write(resized_frame)
        appended_frames += 1

    capture.release()

    if appended_frames == 0:
        raise RuntimeError(f"No frames could be read from {endscreen_path}")
    return appended_frames


# Purpose: Apply all five assignment operations to one complete input video.
def process_one_video(
    input_path,
    output_path,
    talking_path,
    endscreen_path,
    watermark_path,
    face_detector,
    night_threshold,
):
    """Process one source video and save the completed AVI result."""
    mean_brightness = estimate_video_brightness(input_path)
    is_night = mean_brightness < night_threshold
    brightness_increase = (
        calculate_brightness_increase(mean_brightness) if is_night else 0
    )

    time_label = "NIGHTTIME" if is_night else "DAYTIME"
    print(
        f"\n{input_path.name}: mean brightness={mean_brightness:.2f} -> {time_label}"
    )
    if is_night:
        print(f"Brightness increase: +{brightness_increase}")

    main_capture = open_video(input_path, "input video")
    talking_capture = open_video(talking_path, "talking video")

    width = int(main_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(main_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(main_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(main_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        main_capture.release()
        talking_capture.release()
        raise RuntimeError(f"Invalid video dimensions in {input_path}")
    if not np.isfinite(fps) or fps <= 0:
        fps = FALLBACK_FPS

    output_size = (width, height)
    watermark = prepare_watermark(watermark_path, output_size)

    # MJPG in an AVI container is supported by standard OpenCV builds.
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        main_capture.release()
        talking_capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    processed_frames = 0
    detected_faces = 0
    progress_interval = max(1, int(round(fps * 5)))

    try:
        while True:
            success, frame = main_capture.read()
            if not success:
                break

            # Preserve the original Task A(1) to A(4) processing order.
            frame = brighten_frame(frame, brightness_increase)
            frame, face_count = blur_frontal_faces(frame, face_detector, is_night)
            detected_faces += face_count

            talking_frame = read_looping_frame(talking_capture)
            if talking_frame is None:
                raise RuntimeError(f"No frames could be read from {talking_path}")

            frame = overlay_talking_video(frame, talking_frame)
            frame = overlay_black_key_watermark(frame, watermark)

            writer.write(frame)
            processed_frames += 1

            if processed_frames % progress_interval == 0:
                if total_frames > 0:
                    percentage = 100.0 * processed_frames / total_frames
                    print(f"  Processing: {percentage:5.1f}%", end="\r", flush=True)
                else:
                    print(
                        f"  Processed frames: {processed_frames}",
                        end="\r",
                        flush=True,
                    )

        if processed_frames == 0:
            raise RuntimeError(f"No frames could be read from {input_path}")

        # Task A(5) starts after all main-video frames have been written.
        end_frames = append_end_screen(writer, endscreen_path, output_size)
    finally:
        main_capture.release()
        talking_capture.release()
        writer.release()

    print(
        f"  Done: {processed_frames} main frames, {end_frames} end-screen frames, "
        f"{detected_faces} face detections"
    )
    print(f"  Saved to: {output_path}")
