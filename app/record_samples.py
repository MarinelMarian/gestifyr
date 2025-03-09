from contextlib import redirect_stderr
import cv2
import numpy as np
import datetime as dt
import os
import time
import matplotlib.pyplot as plt
import csv

from videoProcessingTools import get_angles
from mediapipe_extract import extract_features_v2
from tools import write_to_csv
from dotenv import load_dotenv

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")
APP_REL_PATH = os.getenv("APP_REL_PATH")
WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX")) or 0

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
cap = cv2.VideoCapture(WEBCAM_INDEX)
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
        all_features.append([y / 90, x / 90, *reduced_features_row])
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


def show_timeline_and_features():
    # Parameters for timeline images.
    target_img_height = 100  # fixed image height
    header_height = 10  # header space for gesture name
    composite_height = header_height + target_img_height
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.33
    thickness = 1
    text_color = (0, 0, 0)  # black text

    # Lists to store composite image info and CSV features.
    composite_info = []  # Each entry: (composite image, video_frame_count)
    csv_info = []  # Each entry: (header, data_rows)

    # Process video files in output_dir (sorted order).
    video_files = sorted(
        [
            f
            for f in os.listdir(output_dir)
            if f.startswith("gesture_") and f.endswith(".mp4")
        ]
    )
    for f in video_files:
        video_path = os.path.join(output_dir, f)
        cap_vid = cv2.VideoCapture(video_path)
        if not cap_vid.isOpened():
            continue
        total_frames = int(cap_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame_index = total_frames // 2
        cap_vid.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
        ret, frame = cap_vid.read()
        cap_vid.release()
        if ret:
            # Resize frame to fixed target_img_height.
            h, w = frame.shape[:2]
            scale = target_img_height / h
            new_w = int(w * scale)
            resized_frame = cv2.resize(frame, (new_w, target_img_height))
            # Create composite image: header on top and resized frame below.
            composite = np.full((composite_height, new_w, 3), 255, dtype=np.uint8)
            # Determine gesture name from filename.
            base = os.path.basename(video_path)
            parts = base.split("_")
            if len(parts) >= 2:
                try:
                    gesture_index = int(parts[1])
                    gesture_label = GESTURES[gesture_index]
                except Exception:
                    gesture_label = "Unknown"
            else:
                gesture_label = "Unknown"
            # Center gesture name in header.
            (text_w, text_h), _ = cv2.getTextSize(
                gesture_label, font_face, font_scale, thickness
            )
            text_x = (new_w - text_w) // 2
            text_y = (header_height + text_h) // 2
            cv2.putText(
                composite,
                gesture_label,
                (text_x, text_y),
                font_face,
                font_scale,
                text_color,
                thickness,
            )
            # Place the resized frame below the header.
            composite[header_height:composite_height, 0:new_w, :] = resized_frame
            composite_info.append((composite, total_frames))
        # Process corresponding CSV file.
        csv_path = os.path.splitext(video_path)[0] + ".csv"
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r") as f_csv:
                    reader = csv.reader(f_csv)
                    rows = list(reader)
                    # Skip if CSV has no data rows.
                    if len(rows) < 2:
                        continue
                    data_rows = []
                    for row in rows[1:]:
                        if row:
                            data_rows.append([float(val) for val in row])
                    csv_info.append((rows[0], data_rows))
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")

    # Build timeline image.
    # Sum total frames of all videos.
    total_video_frames = sum(frames for (_, frames) in composite_info)
    # Set desired overall timeline width (in pixels).
    timeline_total_width = 1000
    timeline_parts = []
    for composite, frames in composite_info:
        # Resize each composite image horizontally in proportion to its video frame count.
        target_width = int((frames / total_video_frames) * timeline_total_width)
        composite_resized = cv2.resize(composite, (target_width, composite_height))
        timeline_parts.append(composite_resized)
    if timeline_parts:
        timeline = np.hstack(timeline_parts)
    else:
        timeline = None

    # Aggregate feature data from CSV files in sorted order.
    all_features = []
    header = None
    total_feature_frames = 0
    for csv_header, data_rows in csv_info:
        all_features.extend(data_rows)
        total_feature_frames += len(data_rows)
        if header is None:
            header = csv_header
    all_features = np.array(all_features) if all_features else None

    # Scale x-axis for features so that total time equals timeline_total_width.
    factor = timeline_total_width / total_feature_frames if total_feature_frames else 1
    x_vals_scaled = (
        [x * factor for x in range(total_feature_frames)]
        if total_feature_frames
        else []
    )

    # Create figure and maximize window (Windows-specific).
    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [1, 2]}, figsize=(14, 8)
    )
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state("zoomed")
    except Exception:
        mng.resize(1400, 1000)

    # Plot timeline (top axes):
    if timeline is not None:
        # Use extent so that the timeline image covers x from 0 to timeline_total_width.
        timeline_rgb = cv2.cvtColor(timeline, cv2.COLOR_BGR2RGB)
        ax1.imshow(timeline_rgb, extent=[0, timeline_total_width, 0, composite_height])
        ax1.set_xlim([0, timeline_total_width])
        ax1.axis("off")
        ax1.set_title("Timeline of Recorded Gestures")
    else:
        ax1.text(0.5, 0.5, "No timeline available", ha="center", va="center")
        ax1.axis("off")

    # Plot monitored features (bottom axes) using the scaled x-values.
    if all_features is not None:
        num_features = all_features.shape[1]
        for i in range(num_features):
            ax2.plot(
                x_vals_scaled,
                all_features[:, i],
                label=header[i] if header is not None else f"F{i}",
            )
        ax2.set_xlabel("Time (scaled to timeline)")
        ax2.set_title("Monitored Features")
        ax2.legend(loc="lower right")
        ax2.set_xlim(0, timeline_total_width)
    else:
        ax2.text(0.5, 0.5, "No feature data available", ha="center", va="center")
        ax2.axis("off")

    plt.tight_layout()
    plt.show()


# Main loop
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
        # Enter playback mode. In playback mode, process each gesture video file.
        is_playback = not is_playback
        is_paused = True
        if is_playback:
            print("Paused.")
            print("Entering Playback Mode...")
            for f in os.listdir(output_dir):
                if f.startswith("gesture_") and f.endswith(".mp4"):
                    video_path = os.path.join(output_dir, f)
                    csv_path = os.path.splitext(video_path)[0] + ".csv"
                    if not os.path.exists(csv_path):
                        process_video_file(video_path)
            print("Finished processing all gesture videos. Exiting Playback Mode.")

            # After processing, display timeline and feature plot.
            show_timeline_and_features()

    elif key == ord("q") or key == 27:  # 'q' or ESC to quit
        break

# Clean up
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
