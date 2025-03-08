# Gesture Recorder

## Overview
The Gesture Recorder project is designed to capture and analyze facial gestures using a webcam. It utilizes the MediaPipe library for facial landmark detection and provides functionality for recording, saving, and playing back gesture videos.

## Project Structure
```
gesture-recorder
├── src
│   ├── analyzeMovement.py
│   ├── parseVideo.py
│   ├── mediapipe_extract.py
│   ├── videoProcessingTools.py
│   └── record_samples.py
├── requirements.txt
└── README.md
```

## File Descriptions
- **src/analyzeMovement.py**: Contains functions for analyzing movement data, extracting features from video, and saving the processed data to CSV files. It includes functionality for plotting features over time.

- **src/parseVideo.py**: Handles video processing, extracting features from video frames, and saving the extracted features to a CSV file. It includes functionality for reading video files and processing frames to gather relevant data.

- **src/mediapipe_extract.py**: Utilizes the MediaPipe library to extract facial landmarks and features from video frames. It includes functions for feature extraction and normalization.

- **src/videoProcessingTools.py**: Provides utility functions for playing video, extracting features for gestures, and calculating angles based on facial landmarks. It includes functions for handling video playback and gesture extraction.

- **src/record_samples.py**: Implements the main functionality of the project. It continuously plays a webcam stream, displays a play/pause button, cycles through a list of gestures, records video associated with each gesture, and implements playback mode with gesture selection and saving functionality.

## Requirements
To run this project, you need to install the following dependencies:

- OpenCV
- NumPy
- Matplotlib
- MediaPipe

You can install the required packages using the following command:

```
pip install -r requirements.txt
```

## Usage
1. Connect your webcam.
2. Run the `record_samples.py` script to start the webcam stream.
3. Use the interface to cycle through gestures, record videos, and save them for later analysis.
4. Playback recorded gestures as needed.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.