# Enhanced Face Analysis Tool

This project provides a comprehensive face analysis tool with numerous features, built using OpenCV and Python. It is designed to be compatible with various Python versions, including 3.13.3, by providing fallbacks for GUI components.

## Features

- **Face Detection**: Detects faces in images and webcam video.
- **Age and Gender Prediction**: Predicts age range and gender using pre-trained models.
- **Basic Emotion Recognition**: Detects happy/neutral based on smile intensity.
- **Smile Intensity**: Classifies smiles as Slight, Moderate, or Broad.
- **Head Pose Estimation**: Estimates if the head is looking left/right/straight and if it's tilted.
- **Face Shape Estimation**: Classifies face shape as oval, round, or square/oblong based on aspect ratio.
- **Glasses Detection**: Detects if a person is wearing glasses.
- **Facial Symmetry Analysis**: Calculates a percentage score for facial symmetry.
- **Image Filters**: Applies Sepia, Black & White, or Cartoon filters to detected faces.
- **Face Tracking**: Tracks faces in video mode with trajectory visualization.
- **Multiple Face Comparison**: (Basic implementation using ORB features - for demonstration).
- **Face Orientation Correction**: Automatically rotates the image to align faces horizontally.
- **Basic Face Recognition**: Simple recognition to assign IDs to faces across frames/images.
- **Face Blurring**: Option to blur detected faces for privacy.
- **Batch Processing**: Process multiple images in a directory.
- **Face Measurements**: Measures eye distance, face width, and height.
- **GUI Interface**: A simple Tkinter-based GUI for easier interaction (requires Tkinter and Pillow).
- **Command-Line Interface**: Full functionality available via command-line arguments.
- **Cross-Platform Compatibility**: Works on Windows, Linux, macOS.

## Project Structure

```
FACE_DETECTION_PROJECT/
│── data/                # Haarcascade XML files
│── models/              # Trained/saved models
│── samples/             # Sample images/videos
│── src/                 # Python scripts
│   ├── face_detection.py
│   ├── simplified_face_detection.py
│   ├── enhanced_face_detection.py
│   ├── enhanced_face_detection_emotion.py
│   ├── face_analyzer.py
│── get_xml_files.py
│── requirements.txt
│── README.md
│── .gitignore

```

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- **Optional (for GUI)**: Tkinter (usually included with Python), Pillow (`pillow`)

## Installation

1.  Extract this zip file.
2.  Install required packages:
    ```bash
    pip install opencv-python numpy pillow
    ```
    *Note: Tkinter might need separate installation depending on your Python distribution (e.g., `sudo apt-get install python3-tk` on Debian/Ubuntu).* 

## Usage

The primary script is `src/face_analyzer.py`.

### GUI Mode (Recommended)

```bash
python src/face_analyzer.py --mode gui 
```
*(Requires Tkinter and Pillow)*

- Use the buttons to load images, start the webcam, process images/batches, and save results.
- Select options like filters, blurring, recognition, and orientation correction before processing.

### Command-Line Modes

**1. Image Processing:**

```bash
python src/face_analyzer.py --mode image --input <image_path> [--output <save_path>] [options...]
```
Example:
```bash
python src/face_analyzer.py --mode image --input samples/lena.jpg --output samples/lena_processed.jpg --filter cartoon --recognize --orient
```

**2. Video Processing:**

```bash
python src/face_analyzer.py --mode video [--video_source <index>] [options...]
```
Example (using default webcam 0):
```bash
python src/face_analyzer.py --mode video --blur --track
```
*(Press 'q' in the video window to quit)*

**3. Batch Processing:**

```bash
python src/face_analyzer.py --mode batch --input <input_dir> --output <output_dir> [options...]
```
Example:
```bash
python src/face_analyzer.py --mode batch --input samples/ --output samples/processed_batch/ --filter sepia --blur
```

### Command-Line Options

- `--mode`: `gui`, `image`, `video`, `batch` (default: `gui`)
- `--input`: Path to input image or directory.
- `--output`: Path to save output image or directory.
- `--video_source`: Webcam index (default: 0).
- `--filter`: `none`, `sepia`, `bw`, `cartoon` (default: `none`).
- `--blur`: Blur detected faces.
- `--recognize`: Enable basic face recognition.
- `--track`: Enable face tracking in video mode (default: True for video).
- `--orient`: Correct face orientation.
- `--data-dir`: Custom path for Haar cascades.
- `--models-dir`: Custom path for DNN models.

## Troubleshooting

- **Tkinter/Pillow Not Found**: If you see a warning about missing GUI libraries, you can still use the command-line modes (`image`, `video`, `batch`). To use the GUI, install Tkinter and Pillow.
- **Cascade/Model Not Found**: Ensure the `data` and `models` directories are in the correct location relative to the script, or use `--data-dir` and `--models-dir` to specify their paths.
- **Webcam Issues**: Ensure your webcam is connected and drivers are installed. Try a different `--video_source` index if needed.
- **Slow Tracking**: CSRT tracker is robust but can be slow. For faster tracking, you could modify the code to use `cv2.TrackerKCF_create()` (commented out in the code).

## License

This project is provided for educational purposes.
