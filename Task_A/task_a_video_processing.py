"""CSC2014 Digital Image Processing - Task A.

This program processes one or more YouTube videos by:
1. classifying each video as daytime/nighttime and brightening nighttime video;
2. detecting and blurring frontal faces;
3. placing a looping talking video at the top-left corner;
4. adding one of two supplied watermarks; and
5. appending an end-screen video.

Only Python's standard library, OpenCV, and NumPy are used.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# ---------------------------- Tunable parameters ----------------------------

# Videos with sampled mean greyscale intensity below this value are nighttime.
NIGHT_THRESHOLD = 85.0

# Nighttime frames are raised toward this brightness, subject to the cap below.
TARGET_NIGHT_BRIGHTNESS = 115.0
MAX_BRIGHTNESS_INCREASE = 60

# Size and placement of the talking-video picture-in-picture window.
TALKING_WIDTH_RATIO = 0.28
OVERLAY_MARGIN = 20

# Watermark opacity. Black watermark pixels are treated as transparent.
WATERMARK_OPACITY = 0.75

# Assignment videos are specified as 30 FPS; this is a safe metadata fallback.
FALLBACK_FPS = 30.0


def parse_arguments():
    """Read file paths and optional settings from the command line."""
    parser = argparse.ArgumentParser(
        description="Complete all CSC2014 Task A video-processing operations."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="The four main videos to process, in the desired order.",
    )
    parser.add_argument(
        "--talking",
        default="talking.mp4",
        help="Talking-video path (default: talking.mp4).",
    )
    parser.add_argument(
        "--endscreen",
        default="endscreen.mp4",
        help="End-screen path (default: endscreen.mp4).",
    )
    parser.add_argument(
        "--watermarks",
        nargs=2,
        default=["watermark1.png", "watermark2.png"],
        metavar=("WATERMARK1", "WATERMARK2"),
        help="Exactly two watermark images; they are alternated across inputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="task_a_outputs",
        help="Directory for processed AVI files (default: task_a_outputs).",
    )
    parser.add_argument(
        "--night-threshold",
        type=float,
        default=NIGHT_THRESHOLD,
        help=f"Mean brightness below which a video is nighttime (default: {NIGHT_THRESHOLD}).",
    )
    return parser.parse_args()


def require_existing_file(file_path, label):
    """Stop early with a clear message when a required file is missing."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found: {path}")
    return path


def open_video(video_path, label):
    """Open a video and verify that OpenCV can decode it."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {label}: {video_path}")
    return capture


def estimate_video_brightness(video_path, sample_count=60):
    """Estimate overall brightness using evenly spaced greyscale frames.

    Sampling gives a representative value without decoding the video twice in
    full. Pixel intensities range from 0 (black) to 255 (white).
    """
    capture = open_video(video_path, "input video")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    brightness_values = []

    if total_frames > 0:
        # linspace selects frames from the beginning, middle, and end.
        sample_total = min(sample_count, total_frames)
        indices = np.linspace(0, total_frames - 1, sample_total, dtype=np.int32)
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if success:
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness_values.append(float(np.mean(grey)))
    else:
        # Fallback for containers that do not report their number of frames.
        frame_index = 0
        while len(brightness_values) < sample_count:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % 30 == 0:
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness_values.append(float(np.mean(grey)))
            frame_index += 1

    capture.release()

    if not brightness_values:
        raise RuntimeError(f"No readable frames were found in {video_path}")

    return float(np.mean(brightness_values))


def calculate_brightness_increase(mean_brightness):
    """Choose a bounded additive increase for a nighttime video."""
    needed_increase = int(round(TARGET_NIGHT_BRIGHTNESS - mean_brightness))
    return max(0, min(MAX_BRIGHTNESS_INCREASE, needed_increase))


def brighten_frame(frame, increase):
    """Increase every BGR channel with saturation at 255."""
    if increase <= 0:
        return frame
    return cv2.convertScaleAbs(frame, alpha=1.0, beta=increase)


def create_face_detector():
    """Load OpenCV's pre-trained frontal-face Haar cascade."""
    # cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade_path = Path(__file__).resolve().parent / "face_detector.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
    return detector


def blur_frontal_faces(frame, detector, is_night):
    """Detect frontal faces and replace each face region with Gaussian blur."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Equalisation improves face detection when lighting is uneven.
    equalised = cv2.equalizeHist(grey)
    
    clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8))
    
    # If the video is classified as nighttime, use CLAHE for better face detection; 
    # otherwise, use the original greyscale image.
    if is_night:
        detection_image = clahe.apply(grey)
    else:
        detection_image = grey
    
    faces = detector.detectMultiScale(
        detection_image,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(30, 30),
    )
    
    
    for x, y, width, height in faces:
        face_region = frame[y : y + height, x : x + width]

        # The kernel must be positive, odd, and no larger than the face region.
        kernel = min(width, height)
        if kernel % 2 == 0:
            kernel -= 1
        kernel = max(3, kernel)

        frame[y : y + height, x : x + width] = cv2.GaussianBlur(
            face_region, (kernel, kernel), sigmaX=0
        )

    return frame, len(faces)


def read_looping_frame(capture):
    """Read the next talking frame and restart from frame zero at its end."""
    success, frame = capture.read()
    if success:
        return frame

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, frame = capture.read()
    return frame if success else None


def overlay_talking_video(main_frame, talking_frame):
    """Resize the talking frame proportionally and place it at top-left."""
    main_height, main_width = main_frame.shape[:2]
    talking_height, talking_width = talking_frame.shape[:2]

    inset_width = max(1, int(main_width * TALKING_WIDTH_RATIO))
    inset_height = max(1, int(inset_width * talking_height / talking_width))

    # Prevent the inset from exceeding the main frame on unusual resolutions.
    max_height = max(1, main_height - 2 * OVERLAY_MARGIN)
    if inset_height > max_height:
        inset_height = max_height
        inset_width = max(1, int(inset_height * talking_width / talking_height))

    inset = cv2.resize(
        talking_frame, (inset_width, inset_height), interpolation=cv2.INTER_AREA
    )

    x1, y1 = OVERLAY_MARGIN, OVERLAY_MARGIN
    x2, y2 = x1 + inset_width, y1 + inset_height
    main_frame[y1:y2, x1:x2] = inset

    # A white border keeps the inset visible over both dark and bright scenes.
    cv2.rectangle(main_frame, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)
    return main_frame


def prepare_watermark(watermark_path, output_size):
    """Load a supplied full-frame watermark and resize it to the video."""
    watermark = cv2.imread(str(watermark_path), cv2.IMREAD_COLOR)
    if watermark is None:
        raise RuntimeError(f"Could not read watermark image: {watermark_path}")

    width, height = output_size
    return cv2.resize(watermark, (width, height), interpolation=cv2.INTER_AREA)


def overlay_black_key_watermark(frame, watermark):
    """Blend non-black watermark pixels while leaving black pixels transparent.

    The supplied PNG files have three RGB channels and a black background; they
    do not contain an alpha channel. Therefore, a black-key mask is necessary.
    """
    non_black_mask = np.max(watermark, axis=2) > 3
    blended = cv2.addWeighted(
        frame, 1.0 - WATERMARK_OPACITY, watermark, WATERMARK_OPACITY, 0
    )
    frame[non_black_mask] = blended[non_black_mask]
    return frame


def append_end_screen(writer, endscreen_path, output_size):
    """Resize and append every frame from the end-screen video."""
    capture = open_video(endscreen_path, "end-screen video")
    appended_frames = 0

    while True:
        success, frame = capture.read()
        if not success:
            break
        frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        appended_frames += 1

    capture.release()
    if appended_frames == 0:
        raise RuntimeError(f"No frames could be read from {endscreen_path}")
    return appended_frames


def process_one_video(
    input_path,
    output_path,
    talking_path,
    endscreen_path,
    watermark_path,
    face_detector,
    night_threshold,
):
    """Apply all five assignment operations to one input video."""
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

    # MJPG in an AVI container is widely supported by standard OpenCV builds.
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, output_size
    )
    if not writer.isOpened():
        main_capture.release()
        talking_capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    processed_frames = 0
    detected_faces = 0

    try:
        while True:
            success, frame = main_capture.read()
            if not success:
                break

            # The order below follows Tasks A(1) to A(4).
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

            # Print progress roughly every five seconds of source footage.
            progress_interval = max(1, int(round(fps * 5)))
            if processed_frames % progress_interval == 0:
                if total_frames > 0:
                    percentage = 100.0 * processed_frames / total_frames
                    print(f"  Processing: {percentage:5.1f}%", end="\r", flush=True)
                else:
                    print(f"  Processed frames: {processed_frames}", end="\r", flush=True)

        if processed_frames == 0:
            raise RuntimeError(f"No frames could be read from {input_path}")

        # Task A(5): add the end screen only after the main video is complete.
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


def main():
    """Validate inputs and process each video with alternating watermarks."""
    args = parse_arguments()

    input_paths = [require_existing_file(path, "Input video") for path in args.inputs]
    talking_path = require_existing_file(args.talking, "Talking video")
    endscreen_path = require_existing_file(args.endscreen, "End-screen video")
    watermark_paths = [
        require_existing_file(path, "Watermark image") for path in args.watermarks
    ]

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    face_detector = create_face_detector()

    for index, input_path in enumerate(input_paths):
        # Videos 1/3 use watermark1; videos 2/4 use watermark2.
        watermark_path = watermark_paths[index % len(watermark_paths)]
        output_path = output_directory / f"{input_path.stem}_processed.avi"
        
        version = 2

        while output_path.exists():
            output_path = (
                output_directory /
                f"{input_path.stem}_processed_v{version}.avi"
            )
            version += 1

        process_one_video(
            input_path=input_path,
            output_path=output_path,
            talking_path=talking_path,
            endscreen_path=endscreen_path,
            watermark_path=watermark_path,
            face_detector=face_detector,
            night_threshold=args.night_threshold,
        )

    print("\nAll videos were processed successfully.")


if __name__ == "__main__":
    main()
