import cv2
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt

# List of gestures to cycle through
gestures = ["Smile", "Blink", "Nod", "Shake", "Surprised"]
current_gesture_index = 0
recording = False
video_writer = None

# Function to start recording
def start_recording(cap, gesture):
    global video_writer
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"gesture_{gesture}_{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

# Function to stop recording
def stop_recording():
    global video_writer
    if video_writer is not None:
        video_writer.release()
        video_writer = None

# Function to display the current gesture
def display_gesture(frame, gesture):
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Main function to run the webcam stream
def run_webcam():
    global recording, current_gesture_index
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the current gesture
        display_gesture(frame, gestures[current_gesture_index])

        # Record video if recording is active
        if recording:
            video_writer.write(frame)

        cv2.imshow("Webcam Stream", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):  # Toggle play/pause
            recording = not recording
            if recording:
                start_recording(cap, gestures[current_gesture_index])
            else:
                stop_recording()
        elif key == ord('n'):  # Next gesture
            current_gesture_index = (current_gesture_index + 1) % len(gestures)
        elif key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()