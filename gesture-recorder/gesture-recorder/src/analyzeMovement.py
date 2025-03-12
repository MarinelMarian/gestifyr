import cv2
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
from videoProcessingTools import extract_features_for_gesture
from tools import write_to_csv

# Define the gestures
gestures = ["Smile", "Blink", "Nod", "Shake", "Surprised"]
current_gesture_index = 0
recording = False
cap = cv2.VideoCapture(0)

# Create a directory to save recordings
output_folder = "recordings"
os.makedirs(output_folder, exist_ok=True)

def play_video_stream():
    global recording, current_gesture_index
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the current gesture on the frame
        cv2.putText(frame, f"Gesture: {gestures[current_gesture_index]}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow("Webcam Stream", frame)

        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):  # Pause/Resume recording
            recording = not recording
            if recording:
                print(f"Recording gesture: {gestures[current_gesture_index]}")
                timestamp = int(dt.datetime.now().timestamp())
                video_filename = f"{output_folder}/gesture_{gestures[current_gesture_index]}_{timestamp}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
            else:
                print("Paused recording.")
                out.release()

        if recording:
            out.write(frame)

        if key == ord('n'):  # Next gesture
            if recording:
                out.release()
            current_gesture_index = (current_gesture_index + 1) % len(gestures)
            print(f"Switched to gesture: {gestures[current_gesture_index]}")

        if key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_video_stream()