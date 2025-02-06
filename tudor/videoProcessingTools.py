import cv2
from mediapipe_extract import extractFeatures, featureNormalization , extractFeaturesv2
import numpy as np

def playVideoAndExtract(cap, startSec, stopSec):
    fps = cap.get(cv2.CAP_PROP_FPS)   
    if not cap.isOpened():
        print("Error: Cannot open video file")
        exit()

    start_frame = int(startSec *fps)  # Define start frame
    end_frame = int(stopSec *fps)    # Define end frame

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame number
        if current_frame > end_frame:  # Stop when reaching the end frame
            break
        ret, frame = cap.read()
        if not ret:
            break  # Exit if video ends or there's an error
        cv2.imshow("Video Playback", frame)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord("q"):  # Press 'q' to exit
            break   


def extractFeatures4gesture(cap, startSec, stopSec):
    fps = cap.get(cv2.CAP_PROP_FPS)   
    if not cap.isOpened():
        print("Error: Cannot open video file")
        exit()

    start_frame = int(startSec *fps)  # Define start frame
    end_frame = int(stopSec *fps)    # Define end frame
    rawFeatures = []
    processedFeatures = []
    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while True:
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame number
        if current_frame > end_frame:  # Stop when reaching the end frame
            break
        ret, frame = cap.read()
        if not ret:
            break  # Exit if video ends or there's an error
        result_features = extractFeaturesv2(frame)
        rawArray = featureNormalization(np.array([[el.x, el.y, el.z] for el in result_features.face_landmarks[0]]))
        rawFeatures.append(list(rawArray.reshape(-1)))
        processedFeatures.append([c.score for c in result_features.face_blendshapes[0]])
    return (rawFeatures, processedFeatures)

        

        


