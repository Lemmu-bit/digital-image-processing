# Task A: Video Processing Guide

This folder contains the Task A video-processing script for applying brightness adjustment, face blurring, picture-in-picture overlay, watermarking, and end-screen appending.

## Requirements

First, create and activate a Python virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the required packages:

```bash
pip install opencv-python numpy
```

You should also have these files in the same folder as the script:

- street.mp4
- talking.mp4
- endscreen.mp4
- watermark1.png
- watermark2.png
- face_detector.xml

## How to Run

Open a terminal in the Task A folder and make sure your virtual environment is active. Then run:

```bash
python task_a_video_processing.py --inputs street.mp4 --talking talking.mp4 --endscreen endscreen.mp4 --watermarks watermark1.png watermark2.png
```

If you want to confirm the environment is active, you can check:

```bash
python --version
```

This command will:

- process the input video named street.mp4
- use talking.mp4 as the looping inset video
- append endscreen.mp4 at the end
- apply watermark1.png and watermark2.png as the alternating watermarks

## Output

Processed videos will be saved in the folder:

```bash
task_a_outputs
```

The output files will be named like:

```bash
street_processed.avi
```

If the file already exists, the script will create a new version such as:

```bash
street_processed_v2.avi
```

## Notes

- The script uses OpenCV and NumPy only.
- If you want to process more than one input video, add more file names after --inputs.
- The script alternates between the two supplied watermarks across the input videos.
