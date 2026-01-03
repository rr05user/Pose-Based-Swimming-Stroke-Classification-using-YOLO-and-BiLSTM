# Pose-Based-Swimming-Stroke-Classification-using-YOLO-and-BiLSTM
Computer vision pipeline for swimming action recognition using YOLO pose estimation and BiLSTM sequence modeling. Converts video keypoints into temporal features with tracking, normalization, and deep learning-based classification.

# Frontstroke vs Non-Frontstroke (YOLO Pose + LSTM)

This project classifies swimming videos into **Frontstroke (Freestyle/Crawl)** vs **Non-Frontstroke**.

Pipeline:
1. Use your own dataset to test this model and set the filepath as vids for your dataset
2. Run **YOLO pose** on frames to extract 17 keypoints per person
3. Track the correct swimmer using an optional CSV point (centroid matching) and fallback tracking
4. Convert keypoints → biomechanics-inspired features + velocities
5. Pad to `(150 frames, 100 features)`
6. Train a **BiLSTM + attention** classifier and export a `.keras` model

> The code uses a local workspace structure:
- Optional tracking CSV: `./csvPoints.csv`
- YOLO weights file: `./yolo11n-pose.pt`
(See paths in code.)

## Repo Structure.
├── frontstroke_with_centroid.py
├── frontstroke_lstm_model.keras
├── csvPoints.csv
├── Requirements
├── README.md
└── LICENSE

📄 File Overview

frontstroke_with_centroid.py
Main training pipeline. Extracts pose keypoints using YOLO, performs swimmer tracking, engineers temporal features, and trains a BiLSTM-based classifier.

frontstroke_lstm_model.keras
Pretrained BiLSTM model for inference and architecture reference (trained on a private dataset).

csvPoints.csv
Optional centroid reference file used to stabilize swimmer tracking in multi-person scenes.

Requirements
Python dependencies required to run the pipeline.

README.md
Project overview, methodology, and usage instructions.

LICENSE
MIT license for open-source distribution.



