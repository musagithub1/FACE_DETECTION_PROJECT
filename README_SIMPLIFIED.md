# Simplified Face Detection Project for Python 3.13.3

This simplified version of the face detection project is designed to be compatible with Python 3.13.3 on Windows. It avoids dependencies on MediaPipe and DeepFace while still providing core functionality.

## Features

- **Face detection** in images and webcam video (real-time)
- **Basic emotion recognition** (happy/neutral) using smile detection
- **Age and gender prediction** using pre-trained models
- **Head pose estimation** (looking left/right/straight, tilted)
- **Simple facial structure analysis** and face shape classification
- **Haircut recommendations** based on facial structure and gender
- **Cross-platform compatibility** (Windows, Linux, macOS)
- **Python 3.13.3 compatibility**

## Project Structure

```
face_detection_project/
├── data/               # Contains Haar cascade XML files
├── models/             # Contains age and gender prediction models
├── samples/            # Sample images for testing
└── src/                # Source code
    ├── face_detection.py                # Original face detection script
    ├── enhanced_face_detection.py       # Enhanced version with facial structure
    ├── enhanced_face_detection_emotions.py # Advanced version (requires Python 3.7-3.10)
    └── simplified_face_detection.py     # Simplified version for Python 3.13.3
```

## Requirements

- Python 3.13.3 (or any version)
- OpenCV (cv2)
- NumPy
- Requests (optional)

## Installation

1. Extract this zip file to a location of your choice
2. Install required packages:
   ```
   pip install opencv-python numpy requests
   ```

## Usage

### For Image Processing

```bash
python src/simplified_face_detection.py --image samples/lena.jpg --output result.jpg
```

### For Webcam Processing

```bash
python src/simplified_face_detection.py --webcam
```

### Headless Mode (No GUI)

```bash
python src/simplified_face_detection.py --image samples/lena.jpg --output result.jpg --headless
```

### Custom Data Directory

If you need to specify a custom location for the Haar cascade XML files:

```bash
python src/simplified_face_detection.py --image samples/lena.jpg --data-dir path/to/xml/files
```

### Custom Models Directory

If you need to specify a custom location for the age and gender models:

```bash
python src/simplified_face_detection.py --image samples/lena.jpg --models-dir path/to/models
```

## How It Works

### Face Detection
The application uses Haar cascade classifiers to detect faces in images or video streams.

### Basic Emotion Recognition
The system uses the Haar cascade smile detector to classify emotions as:
- Happy (when smile is detected)
- Neutral (when no smile is detected)

### Age and Gender Prediction
Pre-trained Caffe models are used to predict age and gender:
- Age ranges: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
- Gender: Male, Female

### Head Pose Estimation
Head pose is estimated using eye positions relative to the face:
- Looking left/right is determined by the position of eyes within the face
- Tilted left/right is determined by the angle between the eyes

### Simple Facial Structure Analysis
Basic facial structure analysis uses width-to-height ratio to classify face shapes:
- Oval: Width-to-height ratio between 0.85 and 0.95
- Round: Width-to-height ratio greater than 0.95
- Square: Width-to-height ratio less than 0.85

### Haircut Recommendations
The application provides personalized haircut recommendations based on:
- Detected facial structure/shape
- Gender

## Upgrading to Full Features

If you want to use the full-featured version with advanced emotion recognition and stress detection:

1. Install Python 3.7-3.10 (MediaPipe and DeepFace are not compatible with Python 3.13.3)
2. Create a new environment:
   ```
   # Using conda
   conda create -n face_detection python=3.9
   conda activate face_detection
   
   # Or using Python's venv
   python -m venv face_detection_env --python=3.9
   # Activate on Windows
   face_detection_env\Scripts\activate
   ```
3. Install all dependencies:
   ```
   pip install opencv-python numpy mediapipe deepface openai tf-keras requests
   ```
4. Use the enhanced version:
   ```
   python src/enhanced_face_detection_emotions.py --image samples/lena.jpg
   ```

## Troubleshooting

- **XML File Not Found**: The application uses relative paths to locate XML files. If you still encounter issues, use the `--data-dir` parameter to specify the exact location of your XML files.
- **Model Files Not Found**: Use the `--models-dir` parameter to specify the location of your age and gender models.
- **Webcam Not Available**: Ensure your webcam is properly connected and not being used by another application.

## License

This project is provided for educational purposes.
