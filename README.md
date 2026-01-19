# NGT Sign Language Recognition System

> 🚧 **Work in Progress** — This project is currently under active development

## Project Overview

This project aims to develop a real-time computer vision system for recognizing Dutch Sign Language (Nederlandse Gebarentaal - NGT) fingerspelling. The system captures hand gestures via camera input, processes them using machine learning techniques, and outputs the corresponding letters or words.

### Goal

Create an accessible learning tool that enables users to practice NGT fingerspelling and receive real-time feedback on their hand movements and positions. This tool is designed to support the approximately 30,000 people in the Netherlands who use NGT as their primary language.

### Key Features (Planned)

- ✅ Real-time hand detection using MediaPipe Hand Landmarks
- ✅ Face mesh detection for body-relative hand positioning
- ✅ Signing zone validation (ensures hand is at proper height)
- 🔲 Letter recognition for static and dynamic NGT alphabet signs
- 🔲 Feedback system to indicate correctness of signing
- 🔲 Data collection module for building training dataset
- 🔲 User-friendly interface (Streamlit-based UI perhaps?)

## Technical Approach

### Detection Pipeline

The system uses a two-model approach:

1. **MediaPipe Hands** — Detects 21 hand landmarks per hand (wrist, knuckles, fingertips)
2. **MediaPipe Face Mesh** — Detects face landmarks to establish body reference points

### Why Face Detection?

Proper NGT fingerspelling requires the hand to be positioned at shoulder/face height. By tracking the face, we can:

- Calculate hand position **relative to the body** (not just the camera frame)
- Validate that the user is signing in the correct zone
- Normalize measurements so they work at any distance from camera

### Hand Position Normalization

Hand coordinates are normalized in two ways:

| Normalization | Purpose |
|---------------|---------|
| Relative to wrist | Hand shape recognition (independent of position) |
| Relative to face | Hand position validation (proper signing posture) |

### Classification Approach (Planned)

Landmark sequences will be classified using traditional ML algorithms:

- **Random Forest** — Primary approach (robust, provides feature importance)
- **K-Nearest Neighbors** — Alternative for small datasets
- **FastDTW** — For dynamic letters requiring movement (H, J, Z)

## Project Structure

```
ngt-sign-language/
├── main.py  # Hand + face detection with signing zone
├── requirements.txt        # Python dependencies
├── README.md
├── data/                   # Training data (to be created)
├── models/                 # Trained models (to be created)
└── notebooks/              # Experimentation (to be created)
```

## Current Files

### `main.py`

Basic hand + face detection script that:
- Opens webcam and detects hands AND face simultaneously
- Visualizes 21 MediaPipe landmarks on each hand
- Displays hand label (Left/Right)
- Supports mirror mode toggle
- Calculates hand position relative to face (nose as origin)
- Displays signing zone (yellow rectangle)
- Shows whether hand is IN or OUT of proper signing zone
- Normalizes landmarks relative to wrist (for shape recognition)
- Normalizes position relative to face size (works at any distance)

**Controls:** `q` = quit, `m` = toggle mirror, `z` = toggle zone visibility

## Installation

### Prerequisites

- Python 3.8+
- Webcam

### Setup

```bash
# Clone the repository
git clone https://github.com/[username]/ngt-sign-language.git
cd ngt-sign-language

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.9.0.80 | Webcam access, image processing, display |
| mediapipe | 0.10.9 | Hand and face landmark detection |
| scikit-learn | 1.4.0 | ML classifiers (Random Forest, KNN, SVM) |
| numpy | 1.26.3 | Numerical operations |
| pandas | 2.1.4 | Data handling |
| fastdtw | 0.3.4 | Dynamic time warping for movement letters |
| scipy | 1.12.0 | Scientific computing, DTW dependency |
| matplotlib | 3.8.2 | Visualization and analysis |
| seaborn | 0.13.1 | Statistical visualization |
| joblib | 1.3.2 | Model serialization |
| tqdm | 4.66.1 | Progress bars |

## Usage

### Basic Hand Detection

```bash
python hand_detection.py
```

### Hand + Face Detection (with signing zone)

```bash
python hand_face_detection.py
```

## NGT Alphabet Reference

The Dutch Sign Language alphabet includes 26 letters:

- **Static signs** — Hand held in fixed position (e.g., A, B, C, D, E)
- **Dynamic signs** — Hand moves during the sign (e.g., H, J, Z)

## MediaPipe Landmarks Reference

### Hand Landmarks (21 points) (this is not a pencil)

```
        8   12  16  20
        |   |   |   |
        7   11  15  19
        |   |   |   |
        6   10  14  18
        |   |   |   |
        5---9---13--17
         \         /
      4   \       /
      |    \     /
      3     \   /
      |      \ /
      2       0 (wrist)
      |
      1
```

| ID | Landmark | ID | Landmark |
|----|----------|-----|----------|
| 0 | Wrist | 11 | Middle MCP |
| 1 | Thumb CMC | 12 | Middle PIP |
| 2 | Thumb MCP | 13 | Middle DIP |
| 3 | Thumb IP | 14 | Middle Tip |
| 4 | Thumb Tip | 15 | Ring MCP |
| 5 | Index MCP | 16 | Ring PIP |
| 6 | Index PIP | 17 | Ring DIP |
| 7 | Index DIP | 18 | Ring Tip |
| 8 | Index Tip | 19 | Pinky MCP |
| 9 | Middle MCP | 20 | Pinky PIP |
| 10 | Middle DIP | 21 | Pinky Tip |

### Face Landmarks Used (from 468 total)

| ID | Landmark | Purpose |
|----|----------|---------|
| 1 | Nose tip | Center reference point |
| 10 | Forehead | Upper face boundary |
| 152 | Chin | Lower face boundary |
| 234 | Left ear | Face width measurement |
| 454 | Right ear | Face width measurement |

## Roadmap

- [x] Basic hand detection with MediaPipe
- [x] Face mesh integration for body-relative positioning
- [x] Signing zone validation
- [ ] Data collection script
- [ ] Dataset creation (landmark sequences for each letter)
- [ ] Model training (Random Forest classifier)
- [ ] Real-time letter recognition
- [ ] Dynamic letter support (FastDTW)
- [ ] Feedback system (correct/incorrect indicators)
- [ ] Streamlit UI..?


## License

*To be determined...perhaps*

---

*Part of the ADS&AI Sign Language Challenge at Breda University of Applied Sciences*