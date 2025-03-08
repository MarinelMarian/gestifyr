import cv2
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
from videoProcessingTools import play_video_and_extract
from mediapipe_extract import extract_features_v2

# Define gestures
gestures = ["Smile", "Blink", "Head Nod", "Shake Head"]
current_gesture_index = 0
recording = False
cap = cv2.VideoCapture(0)

def toggle_recording():
    global recording
    recording = not recording
    if recording:
        print("Recording started...")
    else:
        print("Recording stopped.")

def record_gesture(gesture):
    timestamp = int(dt.datetime.now().timestamp())
    output_file = f"gesture_{gesture}_{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

    while recording:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('Recording Gesture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    print(f"Gesture recorded: {output_file}")

def display_gesture_selection():
    global current_gesture_index
    plt.clf()
    plt.title("Gesture Selection")
    plt.bar(gestures, [1] * len(gestures))
    plt.xticks(rotation=45)
    plt.ylim(0, 1.5)
    plt.pause(0.1)

def main():
    global current_gesture_index
    plt.ion()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Webcam Stream', frame)
        display_gesture_selection()

        key = cv2.waitKey(1)
        if key == ord('p'):  # Play/Pause
            toggle_recording()
        elif key == ord('n'):  # Next gesture
            current_gesture_index = (current_gesture_index + 1) % len(gestures)
            print(f"Current Gesture: {gestures[current_gesture_index]}")
        elif key == ord('r'):  # Record current gesture
            record_gesture(gestures[current_gesture_index])
        elif key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.close()

if __name__ == "__main__":
    main()