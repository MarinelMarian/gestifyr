import cv2
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
from mediapipe_extract import extract_features_v2
from videoProcessingTools import write_to_csv

# Define gestures
gestures = ["Smile", "Blink", "Head Nod", "Shake Head"]
current_gesture_index = 0
recording = False
video_writer = None

# Setup webcam
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a directory to save recordings
output_dir = "recordings"
os.makedirs(output_dir, exist_ok=True)

def toggle_recording():
    global recording, video_writer
    if recording:
        video_writer.release()
        print("Recording stopped.")
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_file = os.path.join(output_dir, f"{gestures[current_gesture_index]}_{timestamp}.avi")
        video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
        print(f"Recording {gestures[current_gesture_index]}...")
    recording = not recording

def next_gesture():
    global current_gesture_index
    current_gesture_index = (current_gesture_index + 1) % len(gestures)
    print(f"Current gesture: {gestures[current_gesture_index]}")

# Create a simple GUI for play/pause and gesture selection
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if recording:
        video_writer.write(frame)

    # Display gesture and recording status
    cv2.putText(frame, f"Gesture: {gestures[current_gesture_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'r' to toggle recording, 'n' for next gesture, 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Webcam Stream", frame)

    key = cv2.waitKey(1)
    if key == ord('r'):
        toggle_recording()
    elif key == ord('n'):
        next_gesture()
    elif key == ord('q'):
        break

# Cleanup
if recording:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()