# Real-Time Face Recognition System using FaceNet

A Deep Learning-based facial recognition system that identifies registered individuals in real time using a webcam. Built with pretrained FaceNet, OpenCV, Python and Flask.

## Project Overview

This project implements a real-time face recognition system that extracts facial embeddings using a pretrained FaceNet model and compares them with stored embeddings using cosine similarity.

The application includes a Flask-based web dashboard that displays live recognition results, registered individuals and recognition history.

## Key Features

- Real-time face detection and recognition.
- Pretrained FaceNet model for facial feature extraction.
- 512-dimensional facial embeddings.
- Cosine similarity-based identity matching.
- Threshold-based unknown-person classification.
- Flask web dashboard.
- Live webcam integration using OpenCV.
- Recognition history and registered-user display.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| FaceNet | Facial feature extraction |
| OpenCV | Image processing and webcam access |
| NumPy | Numerical operations |
| Scikit-learn | Cosine similarity |
| Flask | Backend web framework |
| HTML and CSS | Dashboard interface |

## System Workflow

1. Collect facial images of registered individuals.
2. Organize images into training, validation and test datasets.
3. Apply data augmentation to training images.
4. Extract facial embeddings using pretrained FaceNet.
5. Store embeddings with corresponding identity labels.
6. Capture live webcam frames.
7. Extract embeddings from detected faces.
8. Compare live embeddings with stored embeddings using cosine similarity.
9. Identify the closest matching person if the similarity score meets the threshold; otherwise, classify the face as unrecognized.

## Model Evaluation

The FaceNet implementation was evaluated on a small held-out test dataset.

| Metric | Result |
|---|---:|
| Test images evaluated | 18 |
| Correct predictions | 18 |
| Incorrect predictions | 0 |
| Observed test accuracy | 100% |
| Skipped images | 1 |

**Note:** This is a preliminary result based on a limited test dataset containing four registered identities. Unknown-person rejection and real-world generalization require further evaluation.

## Project Structure

```text
Face-Recognition-Using-FaceNet/
│
├── app.py
├── generate_embeddings.py
├── recognize_face.py
├── webcam_recognition.py
├── requirements.txt
│
└── templates/
    └── index.html
```

The personal face dataset and generated embeddings are excluded from this public repository for privacy.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ziyapatel0169/Face-Recognization-Dashboard.git
cd Face-Recognization-Dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your dataset

Create a local dataset with training images organized by identity.

### 4. Generate embeddings

```bash
python generate_embeddings.py
```

The script generates the embeddings and labels required by the recognition system.

### 5. Run the Flask application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

**Note:** The local dataset must be placed in the expected `data/train` structure. Generated files must be available in `model_data` before running the application.

## Project Team

**Lead Developer**
- Ziya Patel — FaceNet integration, recognition pipeline, Flask backend and dashboard implementation.

**Group Members**
- Saloni Malviya
- Ujas Patel
- Sneh Parmar

Developed as a B.Tech Computer Science mini-project at Karnavati University, Unitedworld Institute of Technology.

## Future Improvements

- Calibrate the unknown-person detection threshold.
- Evaluate using a larger and more diverse dataset.
- Improve recognition under varying lighting and facial angles.
- Add secure user registration and persistent recognition history.

---

**Developed using Python, Deep Learning and Computer Vision.**
