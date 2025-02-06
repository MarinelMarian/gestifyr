# Python library install
pip install opencv-python
pip install mediapipe
pip install matplotlib

# Main scipt 
1. --> parseVideo.py to analyze movement based on feature extraction
For visual tracking points of interest, uncomment line with `showPointsOfInterest(frame, allFeatures[frameNr, :, :], pointsOfInterest)`

2. --> analyzeMovement.py to crop video and save features