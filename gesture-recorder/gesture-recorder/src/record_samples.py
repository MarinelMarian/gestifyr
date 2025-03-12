import cv2
import numpy as np
import datetime as dt
import os
import time
import matplotlib.pyplot as plt

# Define the gestures
gestures = ["Smile", "Blink", "Head Nod", "Shake Head", "Raise Eyebrows"]
current_gesture_index = 0
# We'll remove manual recording toggling as recording is automatic per gesture
# is_recording = False
video_writer = None
gesture_frame_count = 0

# Create a directory to save recorded videos
output_dir = "recorded_gestures"
os.makedirs(output_dir, exist_ok=True)

# Recording parameters
frame_width, frame_height = 1920, 1080
fps = 30.0

# Gesture cycling parameters
gesture_display_duration = 3.0  # seconds to display the gesture and record
gesture_wait_duration = 2.0     # seconds to wait without record before next gesture
gesture_phase = "display"       # "display" or "wait"
gesture_phase_start = time.time()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Use Unicode symbols for Play and Pause
# When video is paused, show PLAY symbol and state is_paused True;
# when playing, show PAUSE symbol.
is_paused = True

cv2.namedWindow("Webcam Feed")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame to our desired resolution if needed
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Check pause state. When paused, we simply freeze the current frame.
    if is_paused:
        # Do nothing extra; you could add additional overlay if desired.
        pass
    else:
        # If not paused, update gesture phase and handle automatic recording.
        now = time.time()
        elapsed = now - gesture_phase_start

        if gesture_phase == "display":
            # Start automatic recording on the first frame of display phase.
            if video_writer is None:
                # Start recording: use a temporary filename.
                temp_timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M--%S")
                temp_filename = os.path.join(output_dir, f"temp_gesture_{current_gesture_index}_{temp_timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(temp_filename, fourcc, fps, (frame_width, frame_height))
                gesture_frame_count = 0
                print(f"Recording gesture {gestures[current_gesture_index]}...")

            # Write the current frame to video and count frames.
            if video_writer is not None:
                video_writer.write(frame)
                gesture_frame_count += 1

            # Draw progress bar at top center during the display phase.
            pb_total_width = 300
            pb_height = 5
            pb_x = (frame_width - pb_total_width) // 2
            pb_y = 10
            pb_progress = int((elapsed / gesture_display_duration) * pb_total_width)
            cv2.rectangle(frame, (pb_x, pb_y), (pb_x + pb_total_width, pb_y + pb_height), (200,200,200), 1)
            cv2.rectangle(frame, (pb_x, pb_y), (pb_x + pb_progress, pb_y + pb_height), (255,0,0), -1)

            # Display the current gesture name at the center.
            gesture_text = gestures[current_gesture_index]
            gesture_font = cv2.FONT_HERSHEY_SIMPLEX
            gesture_font_scale = 2
            gesture_thickness = 3
            gesture_text_size, _ = cv2.getTextSize(gesture_text, gesture_font, gesture_font_scale, gesture_thickness)
            gesture_text_x = (frame_width - gesture_text_size[0]) // 2
            gesture_text_y = frame_height // 2
            cv2.putText(frame, gesture_text, (gesture_text_x, gesture_text_y), gesture_font, gesture_font_scale, (0,255,0), gesture_thickness)

            # If display duration elapsed, transition to wait phase.
            if elapsed >= gesture_display_duration:
                gesture_phase = "wait"
                gesture_phase_start = now
                # Stop recording and finalize the file.
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    # Build final filename with gesture index, frame count and timestamp.
                    final_timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    # Assume gesture_frame_count is the count of frames recorded.
                    temp_file = os.path.join(output_dir, f"temp_gesture_{current_gesture_index}_{temp_timestamp}.mp4")
                    final_file = os.path.join(output_dir, f"gesture_{current_gesture_index}_{gesture_frame_count}_{final_timestamp}.mp4")
                    os.rename(temp_file, final_file)
                    print(f"Saved recording: {final_file}")

        else:  # wait phase
            # Do nothing special during wait phase (could add a pause indicator)
            if elapsed >= gesture_wait_duration:
                # Move on to the next gesture.
                current_gesture_index = (current_gesture_index + 1) % len(gestures)
                gesture_phase = "display"
                gesture_phase_start = now

    # Draw play/pause overlay at the bottom.
    overlay_height = 50
    overlay_y = frame_height - overlay_height
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, overlay_y), (frame_width, frame_height), (50, 50, 50), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Instead of "Play" and "Pause" words, use symbols: ▶ for play, ⏸ for pause.
    button_symbol = "▶" if is_paused else "⏸"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size, _ = cv2.getTextSize(button_symbol, font, font_scale, thickness)
    text_x = (frame_width - text_size[0]) // 2
    text_y = overlay_y + (overlay_height + text_size[1]) // 2
    cv2.putText(frame, button_symbol, (text_x, text_y), font, font_scale, (255,255,255), thickness)

    # Also display current gesture on the frame (for debugging, along the top-left corner)
    cv2.putText(frame, f"Gesture: {gestures[current_gesture_index]}", (10, frame_height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Webcam Feed", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord(" "):  # toggle play/pause with space bar
        is_paused = not is_paused
        print("Paused." if is_paused else "Resumed.")
    elif key == ord("q"):
        break

# Clean up
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()