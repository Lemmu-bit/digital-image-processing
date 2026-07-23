"""CSC2014 Digital Image Processing - Task A.

Install the required packages before running:
    pip install opencv-python numpy

Example command:
    python main.py --inputs video1.mp4 video2.mp4 video3.mp4 video4.mp4

Optional arguments can be used to change the talking video, end screen,
watermarks, face detector, output folder, and nighttime threshold.
"""

import argparse
from pathlib import Path

from processing import create_face_detector, process_one_video
from video_utils import create_available_output_path, require_existing_file


# ----------------------------- PROGRAM SETTINGS -----------------------------

PROJECT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_FACE_DETECTOR_PATH = PROJECT_DIRECTORY / "face_detector.xml"

# Videos below this average greyscale brightness are treated as nighttime.
NIGHT_THRESHOLD = 85.0


# Purpose: Read all required file paths and optional settings from the command line.
def parse_arguments():
    """Return the command-line arguments entered by the user."""
    parser = argparse.ArgumentParser(
        description="Complete all CSC2014 Task A video-processing operations."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="The main videos to process, in the desired order.",
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
        help="Two watermark images that alternate across the input videos.",
    )
    parser.add_argument(
        "--face-detector",
        default=str(DEFAULT_FACE_DETECTOR_PATH),
        help="Haar cascade XML path (default: face_detector.xml).",
    )
    parser.add_argument(
        "--output-dir",
        default="task_a_outputs",
        help="Output directory (default: task_a_outputs).",
    )
    parser.add_argument(
        "--night-threshold",
        type=float,
        default=NIGHT_THRESHOLD,
        help=f"Brightness below which a video is nighttime (default: {NIGHT_THRESHOLD}).",
    )
    return parser.parse_args()


# Purpose: Validate the files, prepare shared objects, and process every input video.
def main():
    """Run the complete Task A workflow."""
    args = parse_arguments()

    input_paths = [
        require_existing_file(path, "Input video") for path in args.inputs
    ]
    talking_path = require_existing_file(args.talking, "Talking video")
    endscreen_path = require_existing_file(args.endscreen, "End-screen video")
    watermark_paths = [
        require_existing_file(path, "Watermark image") for path in args.watermarks
    ]
    face_detector_path = require_existing_file(args.face_detector, "Face detector")

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    face_detector = create_face_detector(face_detector_path)

    for index, input_path in enumerate(input_paths):
        # Videos 1/3 use watermark 1; videos 2/4 use watermark 2.
        watermark_path = watermark_paths[index % len(watermark_paths)]
        output_path = create_available_output_path(output_directory, input_path)

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
