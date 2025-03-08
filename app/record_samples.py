from contextlib import redirect_stderr
import cv2
import numpy as np
import datetime as dt
import os
import time
import matplotlib.pyplot as plt

from videoProcessingTools import get_angles
from mediapipe_extract import extract_features_v2
from tools import write_to_csv
from dotenv import load_dotenv

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")
APP_REL_PATH = os.getenv("APP_REL_PATH")

# Set current working directory to the app folder
os.chdir(BASE_PATH + APP_REL_PATH)

os.environ["OPENCV_FFMPEG_DEBUG"] = "0"  # Disable FFMPEG debug logging
# Try to suppress logging if supported.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except AttributeError:
    pass

# Define the gestures
GESTURES = [
    "Nod",
    "Shake",
    "Mouth open",
    "Eyebrows raise",
    "Eyes blink long",
    "Smile",
    "None",
]
current_gesture_index = 0
# We'll remove manual recording toggling as recording is automatic per gesture
video_writer = None
gesture_frame_count = 0

# Create a directory to save recorded videos
output_dir = "recorded_gestures"
os.makedirs(output_dir, exist_ok=True)

# Gesture cycling parameters
GESTURE_DISPLAY_DURATION = 3.0  # seconds to display the gesture and record
GESTURE_WAIT_DURATION = 2.0  # seconds to wait without record before next gesture
gesture_phase = "display"  # "display" or "wait"
gesture_phase_start = time.time()

# Initialize webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get recording parameters from the webcam
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp42")  # avc1 works on MacOS, mp4v works on Windows

# When video is paused, show PLAY symbol and state is_paused True;
# when playing, show PAUSE symbol.
is_paused = True
is_playback = False

font = cv2.FONT_HERSHEY_SIMPLEX
gesture_font = cv2.FONT_HERSHEY_SIMPLEX

cv2.namedWindow("Webcam Feed")


def process_video_file(video_path):
    """
    Process a recorded gesture video file and generate a CSV file with selected features.
    The CSV file will be named according to the pattern:
        <video filename without extension>.csv
    For inspiration, this function reuses similar processing as seen in parseVideo.py.
    """
    cap_vid = cv2.VideoCapture(video_path)
    if not cap_vid.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = int(cap_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid = cap_vid.get(cv2.CAP_PROP_FPS)
    # Use the first frame to get dimensions
    ret, first_frame = cap_vid.read()
    if not ret:
        print(f"Error reading first frame of {video_path}")
        cap_vid.release()
        return
    img_h, img_w = first_frame.shape[:2]
    # Reset capture pointer to start
    cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Define points to extract; same indices as in your parseVideo.py for inspiration.
    points_to_extract = [4, 5, 25, 9, 10, 44, 45]

    all_features = []
    frame_nr = 0
    while True:
        ret, frame = cap_vid.read()
        if not ret:
            break
        result_features = extract_features_v2(frame)
        # Get scores from face blendshapes.
        processed_features_row = [c.score for c in result_features.face_blendshapes[0]]
        # Select features of interest.
        reduced_features_row = [processed_features_row[i] for i in points_to_extract]
        # Process face landmarks into a flat array.
        raw_array = np.array(
            [[el.x, el.y, el.z] for el in result_features.face_landmarks[0]]
        )
        raw_array_row = raw_array.reshape(-1)
        # Get angles
        x, y = get_angles(raw_array_row, img_w, img_h)
        all_features.append([y, x, *reduced_features_row])
        frame_nr += 1

    cap_vid.release()

    # Build CSV filename: remove the extension from the video filename.
    csv_filename = os.path.splitext(video_path)[0] + ".csv"
    header = [
        "angleUpDown",
        "angleLeftRight",
        "browOuterUpLeft",
        "browOuterUpRight",
        "jawOpen",
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "smileLeft",
        "smileRight",
    ]
    write_to_csv(csv_filename, [header] + all_features)
    print(f"Processed {video_path}\nSaved CSV: {csv_filename}")


while True:
    ret, frame = cap.read()
    # Flip frame horizontally for a mirror effect.
    frame = cv2.flip(frame, 1)

    if not ret:
        print("Error: Could not read frame.")
        break

    # Normal mode (not paused / not in playback)
    if not is_playback:
        # Check pause state. When paused, simply use the frozen frame.
        if is_paused:
            pass  # do nothing extra for paused state
        else:
            # If not paused, update gesture phase and handle automatic recording.
            now = time.time()
            elapsed = now - gesture_phase_start

            if gesture_phase == "display":
                # Start automatic recording on the first frame of display phase.
                if video_writer is None:
                    temp_timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    temp_filename = os.path.join(
                        output_dir,
                        f"temp_gesture_{current_gesture_index}_{temp_timestamp}.mp4",
                    )
                    with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
                        video_writer = cv2.VideoWriter(
                            temp_filename, fourcc, fps, (frame_width, frame_height)
                        )
                    gesture_frame_count = 0
                    print(f"Recording gesture {GESTURES[current_gesture_index]}...")

                if video_writer is not None:
                    video_writer.write(frame)
                    gesture_frame_count += 1

                # Draw progress bar at top center.
                pb_total_width = 300
                pb_height = 5
                pb_x = (frame_width - pb_total_width) // 2
                pb_y = 10
                pb_progress = int((elapsed / GESTURE_DISPLAY_DURATION) * pb_total_width)
                cv2.rectangle(
                    frame,
                    (pb_x, pb_y),
                    (pb_x + pb_total_width, pb_y + pb_height),
                    (200, 200, 200),
                    1,
                )
                cv2.rectangle(
                    frame,
                    (pb_x, pb_y),
                    (pb_x + pb_progress, pb_y + pb_height),
                    (255, 0, 0),
                    -1,
                )

                # Display current gesture name.
                gesture_text = GESTURES[current_gesture_index]
                gesture_font_scale = 2
                gesture_thickness = 3
                text_size, _ = cv2.getTextSize(
                    gesture_text, gesture_font, gesture_font_scale, gesture_thickness
                )
                gesture_text_x = (frame_width - text_size[0]) // 2
                gesture_text_y = frame_height // 2
                cv2.putText(
                    frame,
                    gesture_text,
                    (gesture_text_x, gesture_text_y),
                    gesture_font,
                    gesture_font_scale,
                    (0, 255, 0),
                    gesture_thickness,
                )

                if elapsed >= GESTURE_DISPLAY_DURATION:
                    gesture_phase = "wait"
                    gesture_phase_start = now
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        final_timestamp = dt.datetime.now().strftime(
                            "%Y-%m-%d_%H-%M-%S"
                        )
                        temp_file = os.path.join(
                            output_dir,
                            f"temp_gesture_{current_gesture_index}_{temp_timestamp}.mp4",
                        )
                        final_file = os.path.join(
                            output_dir,
                            f"gesture_{current_gesture_index}_{gesture_frame_count}_{final_timestamp}.mp4",
                        )
                        os.rename(temp_file, final_file)
                        print(f"Saved recording: {final_file}")

            else:  # wait phase
                if elapsed >= GESTURE_WAIT_DURATION:
                    current_gesture_index = (current_gesture_index + 1) % len(GESTURES)
                    gesture_phase = "display"
                    gesture_phase_start = now

    # Overlay: play/pause button at bottom.
    overlay_height = 50
    overlay_y = frame_height - overlay_height
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (0, overlay_y), (frame_width, frame_height), (50, 50, 50), -1
    )
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Display symbol: using ">" for play and "||" for pause.
    button_symbol = ">" if is_paused else "||"
    font_scale_disp = 1.5 if is_paused else 1
    thickness_disp = 3
    text_size, _ = cv2.getTextSize(button_symbol, font, font_scale_disp, thickness_disp)
    text_x = (frame_width - text_size[0]) // 2
    text_y = overlay_y + (overlay_height + text_size[1]) // 2
    cv2.putText(
        frame,
        button_symbol,
        (text_x, text_y),
        font,
        font_scale_disp,
        (255, 255, 255),
        thickness_disp,
    )

    # Display current gesture for debugging.
    cv2.putText(
        frame,
        f"Gesture: {GESTURES[current_gesture_index]}",
        (10, frame_height - 60),
        font,
        1,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Webcam Feed", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord(" "):  # Toggle play/pause.
        is_paused = not is_paused
        if not is_paused:
            gesture_phase = "display"
            gesture_phase_start = time.time()
        print("Paused." if is_paused else "Resumed.")
    elif key == ord("p"):
        is_paused = True
        print("Paused.")
        # Enter playback mode. In playback mode, process each gesture video file.
        is_playback = True
        print("Entering Playback Mode...")
        for f in os.listdir(output_dir):
            if f.startswith("gesture_") and f.endswith(".mp4"):
                video_path = os.path.join(output_dir, f)
                process_video_file(video_path)
        print("Finished processing all gesture videos. Exiting Playback Mode.")
        is_playback = False
    elif key == ord("q"):
        break

# Clean up
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
