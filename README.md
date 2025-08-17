# 🔍 Enhanced Face Analysis Tool

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=for-the-badge" alt="Platform">
  <img src="https://img.shields.io/badge/License-Educational-orange?style=for-the-badge" alt="License">
</div>

<div align="center">
  <h3>🚀 A comprehensive face analysis tool with AI-powered features</h3>
  <p>Built with OpenCV and Python for advanced computer vision applications</p>
</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Core Detection
- **Face Detection** - Real-time face detection in images/video
- **Age & Gender Prediction** - AI-powered demographic analysis
- **Emotion Recognition** - Happy/neutral emotion detection
- **Smile Analysis** - Intensity classification (Slight/Moderate/Broad)

</td>
<td width="50%">

### 🎨 Advanced Analysis
- **Head Pose Estimation** - Direction and tilt analysis
- **Face Shape Classification** - Oval, round, square detection
- **Glasses Detection** - Automated eyewear identification
- **Facial Symmetry** - Percentage-based symmetry scoring

</td>
</tr>
<tr>
<td width="50%">

### 🛠️ Processing Tools
- **Image Filters** - Sepia, B&W, Cartoon effects
- **Face Tracking** - Video trajectory visualization
- **Orientation Correction** - Auto-alignment features
- **Privacy Blurring** - Face anonymization

</td>
<td width="50%">

### 📊 Utilities
- **Batch Processing** - Multiple image handling
- **Face Measurements** - Dimensional analysis
- **GUI Interface** - User-friendly Tkinter interface
- **CLI Support** - Full command-line functionality

</td>
</tr>
</table>

---

## 📁 Project Structure

```
FACE_DETECTION_PROJECT/
├── 📂 data/                    # Haarcascade XML files
├── 📂 models/                  # Trained/saved AI models
├── 📂 samples/                 # Sample images/videos
├── 📂 src/                     # Python source code
│   ├── 🐍 face_detection.py
│   ├── 🐍 simplified_face_detection.py
│   ├── 🐍 enhanced_face_detection.py
│   ├── 🐍 enhanced_face_detection_emotion.py
│   └── 🐍 face_analyzer.py    # Main application
├── 🐍 get_xml_files.py
├── 📋 requirements.txt
├── 📖 README.md
└── 🚫 .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

<div align="center">

![Python](https://img.shields.io/badge/python-v3.x+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-v4.x+-green.svg)
![NumPy](https://img.shields.io/badge/numpy-latest-orange.svg)

</div>

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/face-analysis-tool.git
   cd face-analysis-tool
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python numpy pillow
   ```

3. **Optional GUI dependencies** (Ubuntu/Debian)
   ```bash
   sudo apt-get install python3-tk
   ```

---

## 💻 Usage Guide

### 🖥️ GUI Mode (Recommended)

<div align="center">
  <img src="https://img.shields.io/badge/🎨-GUI%20Interface-success?style=for-the-badge" alt="GUI Mode">
</div>

```bash
python src/face_analyzer.py --mode gui
```

**Features:**
- 🖱️ Point-and-click interface
- 📂 Easy file loading
- 🎥 Webcam integration
- ⚙️ Real-time settings adjustment
- 💾 One-click save functionality

### 📷 Image Processing

```bash
python src/face_analyzer.py --mode image --input <image_path> [options]
```

**Example:**
```bash
python src/face_analyzer.py --mode image \
  --input samples/portrait.jpg \
  --output samples/analyzed_portrait.jpg \
  --filter cartoon --recognize --orient
```

### 🎥 Video Processing

```bash
python src/face_analyzer.py --mode video [options]
```

**Example:**
```bash
python src/face_analyzer.py --mode video --blur --track
```

> 💡 **Tip:** Press `q` in the video window to quit

### 📁 Batch Processing

```bash
python src/face_analyzer.py --mode batch \
  --input <input_directory> \
  --output <output_directory> [options]
```

**Example:**
```bash
python src/face_analyzer.py --mode batch \
  --input samples/ \
  --output processed_batch/ \
  --filter sepia --blur
```

---

## ⚙️ Configuration Options

<details>
<summary><b>🔧 Command Line Arguments</b></summary>

| Option | Values | Description |
|--------|--------|-------------|
| `--mode` | `gui`, `image`, `video`, `batch` | Processing mode |
| `--input` | `<path>` | Input file/directory path |
| `--output` | `<path>` | Output file/directory path |
| `--video_source` | `<index>` | Webcam index (default: 0) |
| `--filter` | `none`, `sepia`, `bw`, `cartoon` | Image filter type |
| `--blur` | - | Enable face blurring |
| `--recognize` | - | Enable face recognition |
| `--track` | - | Enable face tracking |
| `--orient` | - | Auto-correct orientation |
| `--data-dir` | `<path>` | Custom Haar cascade path |
| `--models-dir` | `<path>` | Custom models directory |

</details>

---

## 🛠️ Troubleshooting

<details>
<summary><b>🚨 Common Issues & Solutions</b></summary>

### GUI Not Available
```
⚠️ Warning: Tkinter/Pillow not found
```
**Solution:** Install GUI dependencies or use CLI modes
```bash
pip install pillow
# Ubuntu/Debian:
sudo apt-get install python3-tk
```

### Missing Cascade Files
```
❌ Error: Cascade/Model not found
```
**Solution:** Verify file paths or use custom directories
```bash
python src/face_analyzer.py --data-dir /custom/path --models-dir /model/path
```

### Webcam Issues
```
❌ Error: Cannot access camera
```
**Solutions:**
- Check camera connections and drivers
- Try different video source index: `--video_source 1`
- Verify camera permissions

### Slow Performance
```
⏳ Slow tracking performance
```
**Solutions:**
- Switch to KCF tracker (modify code)
- Reduce video resolution
- Close unnecessary applications

</details>

---

## 🤝 Contributing

<div align="center">

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](contributing.md)
[![GitHub issues](https://img.shields.io/github/issues/yourusername/face-analysis-tool?style=flat-square)](issues)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/face-analysis-tool?style=flat-square)](stargazers)

</div>

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

---

## 📊 Performance & Compatibility

<div align="center">

| Platform | Status | Python Versions |
|----------|--------|-----------------|
| 🪟 Windows | ✅ Fully Supported | 3.7+ |
| 🐧 Linux | ✅ Fully Supported | 3.7+ |
| 🍎 macOS | ✅ Fully Supported | 3.7+ |

</div>

---

## 📄 License

<div align="center">
  <img src="https://img.shields.io/badge/License-Educational%20Use-orange?style=for-the-badge" alt="License">
</div>

This project is provided for **educational purposes** only. Please respect privacy and ethical guidelines when using face analysis technology.

---

<div align="center">
  <h3>⭐ If you found this project helpful, please give it a star! ⭐</h3>
  
  <p>
    <a href="https://github.com/yourusername/face-analysis-tool">🏠 Home</a> •
    <a href="#-features">📋 Features</a> •
    <a href="#-installation">💿 Install</a> •
    <a href="#-usage-guide">📖 Docs</a> •
    <a href="#-contributing">🤝 Contribute</a>
  </p>
  
  <p><sub>Built with ❤️ for the computer vision community</sub></p>
</div>
