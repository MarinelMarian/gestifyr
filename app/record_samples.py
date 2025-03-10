import json
from contextlib import redirect_stderr
import cv2
import numpy as np
import datetime as dt
import os
import time
import matplotlib.pyplot as plt
import csv
from PIL import ImageFont, ImageDraw, Image

from videoProcessingTools import get_angles
from mediapipe_extract import extract_features_v2
from tools import write_to_csv
from dotenv import load_dotenv

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")
APP_REL_PATH = os.getenv("APP_REL_PATH")
WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX") or 0) 
DETECTION_INTERVAL_SEC = float(os.getenv("DETECTION_INTERVAL_SEC") or 0.25)

# Set current working directory to the app folder
os.chdir(BASE_PATH + APP_REL_PATH)

os.environ["OPENCV_FFMPEG_DEBUG"] = "0"  # Disable FFMPEG debug logging
# Try to suppress logging if supported.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except AttributeError:
    pass

# Load gestures from file (replace the GESTURES list).
with open("gestures.json", "r") as f:
    gestures_dict = json.load(f)

# Define gesture order (so that filenames use gesture index consistently).
GESTURES = ["nod", "shake", "mouth", "eyebrows", "blink", "smile", "none"]

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

def detect_fluctuations_for_feature(feature_data, fps, interval_sec, threshold):
    """
    Given a 1D numpy array 'feature_data' for one feature in a gesture segment,
    split it into intervals of duration 'interval_sec' (in seconds) given the 'fps',
    and compute the mean of the absolute values in each interval.

    Determine the baseline as the minimum of these means and then mark any interval
    where (mean - baseline) > threshold as significant.

    The detection start is moved back to include the previous interval (if available)
    and the detection end is moved forward to include the next interval (if available),
    so that we capture the state change from baseline to significant and back.

    Returns:
        A list containing one detection tuple (start_idx, end_idx) or an empty list.
    """
    interval_len = max(1, int(fps * interval_sec))
    n = len(feature_data)
    means = []
    intervals = []
    for i in range(0, n, interval_len):
        block = feature_data[i: i + interval_len]
        mean_val = np.mean(np.abs(block))
        means.append(mean_val)
        intervals.append((i, i + len(block) - 1))
    
    if not means:
        return []
    
    baseline = min(means)
    significant_flags = [(m - baseline) > threshold for m in means]
    
    detections = []
    i = 0
    while i < len(significant_flags):
        if significant_flags[i]:
            # Extend start: include the previous interval (if available)
            start_interval = intervals[i][0]
            if i > 0:
                start_interval = intervals[i - 1][0]
            # Process contiguous significant intervals
            while i < len(significant_flags) and significant_flags[i]:
                end_interval = intervals[i][1]
                i += 1
            # Extend end: include the next interval (if available)
            if i < len(intervals):
                end_interval = intervals[i][1]
            detections.append((start_interval, end_interval))
        else:
            i += 1

    # Merge all detections into a single detection if more than one is found.
    if detections:
        overall_start = min(d[0] for d in detections)
        overall_end = max(d[1] for d in detections)
        return [(overall_start, overall_end)]
    else:
        return []    
def merge_intervals(intervals):
    """
    Given a list of intervals (start, end), merge overlapping or adjacent intervals.
    """
    if not intervals:
        return []
    # Sort intervals by start index.
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        # Consider overlapping or adjacent intervals as mergeable.
        if current[0] <= prev[1] + 1:
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)
    return merged

def show_timeline_and_features():
    # Parameters for timeline images.
    target_img_height = 80  # fixed image height
    header_height = 10       # header space for gesture name
    composite_height = header_height + target_img_height
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.33
    thickness = 1
    text_color = (0, 0, 0)   # black text

    # Lists to store composite image info and CSV features.
    composite_info = []      # Each entry: (composite image, video_frame_count, gesture_key)
    csv_info = []            # Each entry: (header, data_rows)
    segment_feature_counts = []  # Number of CSV rows per segment

    # Process video files in output_dir (sorted order).
    video_files = get_sorted_video_file_list(output_dir)
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
            # Create composite image: header on top and resized frame below; fill background with grey.
            composite = np.full((composite_height, new_w, 3), 200, dtype=np.uint8)
            # Determine gesture from filename.
            base = os.path.basename(video_path)
            parts = base.split("_")
            if len(parts) >= 2:
                try:
                    gesture_index = int(parts[1])
                    gesture_key = GESTURES[gesture_index]
                    gesture_label = gestures_dict[gesture_key]["name"]
                except Exception:
                    gesture_key = "unknown"
                    gesture_label = "Unknown"
            else:
                gesture_key = "unknown"
                gesture_label = "Unknown"
            # # Write the gesture label in header.
            # (text_w, text_h), _ = cv2.getTextSize(gesture_label, font_face, font_scale, thickness)
            # text_x = (new_w - text_w) // 2
            # text_y = text_h
            # cv2.putText(
            #     composite,
            #     gesture_label,
            #     (text_x, text_y),
            #     font_face,
            #     font_scale,
            #     text_color,
            #     thickness,
            #     lineType=cv2.LINE_AA,
            # )
            # Place the resized frame below the header.
            composite[header_height:composite_height, 0:new_w, :] = resized_frame
            composite_info.append((composite, total_frames, gesture_key))
        # Process corresponding CSV file.
        csv_path = os.path.splitext(video_path)[0] + ".csv"
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r") as f_csv:
                    reader = csv.reader(f_csv)
                    rows = list(reader)
                    if len(rows) < 2:
                        continue
                    data_rows = []
                    for row in rows[1:]:
                        if row:
                            data_rows.append([float(val) for val in row])
                    csv_info.append((rows[0], data_rows))
                    segment_feature_counts.append(len(data_rows))
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")

    # Build timeline image.
    timeline_total_width = 100  # overall desired minimum width
    timeline_parts = []
    boundaries = []   # left boundary x positions (in pixels)
    cumulative = 0
    # Use composite_info (from video files) to set timeline width.
    total_video_frames = sum(frames for (_, frames, _) in composite_info)
    timeline_total_width = max(timeline_total_width, total_video_frames)
    for composite, frames, gesture_key in composite_info:
        # Compute target width proportional to video frame count.
        target_width = int((frames / total_video_frames) * timeline_total_width)
        boundaries.append(cumulative)
        current_width = composite.shape[1]
        if current_width < target_width:
            total_pad = target_width - current_width
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            left_pad = np.full((composite.shape[0], pad_left, 3), 200, dtype=np.uint8)
            right_pad = np.full((composite.shape[0], pad_right, 3), 200, dtype=np.uint8)
            composite_resized = np.hstack((left_pad, composite, right_pad))
        else:
            excess = current_width - target_width
            crop_left = excess // 2
            composite_resized = composite[:, crop_left:crop_left + target_width]
        # Re-add gesture label (if cropping changed it)
        gesture_label = gestures_dict[gesture_key]["name"]
        (text_w, text_h), _ = cv2.getTextSize(gesture_label, font_face, font_scale, thickness)
        text_x = (target_width - text_w) // 2
        text_y = text_h
        cv2.putText(
            composite_resized,
            gesture_label,
            (text_x, text_y),
            font_face,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        timeline_parts.append(composite_resized)
        cumulative += target_width

    if timeline_parts:
        current_width = sum(part.shape[1] for part in timeline_parts)
        if current_width < timeline_total_width:
            filler = np.full((composite_height, timeline_total_width - current_width, 3), 200, dtype=np.uint8)
            timeline_parts.append(filler)
        timeline = np.hstack(timeline_parts)
    else:
        timeline = None

    # Aggregate feature data from CSV files in sorted order.
    all_features = []
    header = None
    segment_boundaries = []  # cumulative boundaries of feature rows per segment
    cumulative_feat = 0
    for i, (csv_header, data_rows) in enumerate(csv_info):
        all_features.extend(data_rows)
        cumulative_feat += len(data_rows)
        segment_boundaries.append(cumulative_feat)
        if header is None:
            header = csv_header
    all_features = np.array(all_features) if all_features else None

    # Scale x-axis for features so that total time equals timeline_total_width.
    all_features_count = len(all_features)
    factor = timeline_total_width / (all_features_count) if all_features_count else 1
    x_vals_scaled = [x * factor for x in range(all_features_count)] if all_features_count else []

    # --- Detection of significant feature fluctuations ---
    # For each gesture segment we now compute means over 0.5-second intervals.
    detections = []  # Each detection is a tuple: (global_x_start, global_x_end)
    cumulative_feat_prev = 0
    for i, (csv_header, data_rows) in enumerate(csv_info):
        seg_feat = np.array(data_rows)  # shape: (num_rows, num_features)
        seg_length = seg_feat.shape[0]
        # Compute global x offset for this segment.
        x_offset = cumulative_feat_prev * factor
        # Get gesture key for this segment from composite_info; assume same order.
        if i < len(composite_info):
            _, _, gesture_key = composite_info[i]
        else:
            gesture_key = "unknown"
        segment_detections = []
        if gesture_key in gestures_dict:
            features_of_interest = gestures_dict[gesture_key]["features"]
            threshold = gestures_dict[gesture_key]["threshold"]
            # Check each feature of interest.
            for feat_name in features_of_interest:
                if header and feat_name in header:
                    col = header.index(feat_name)
                    subdata = seg_feat[:, col]
                    # Get detections for this feature over DETECTION_INTERVAL_SEC intervals.
                    det = detect_fluctuations_for_feature(subdata, fps, DETECTION_INTERVAL_SEC, threshold)
                    segment_detections.extend(det)
        # Merge detections from all features in this segment.
        segment_detections = merge_intervals(segment_detections)
        # Convert detection indices to global x-axis positions.
        for (start_idx, end_idx) in segment_detections:
            global_x0 = x_offset + start_idx * factor
            global_x1 = x_offset + end_idx * factor
            detections.append((global_x0, global_x1))
        cumulative_feat_prev += seg_length


    # --- Plotting: Create figure and maximize window (Windows-specific). ---
    fig, (ax1, ax2) = plt.subplots(2, 1, 
        gridspec_kw={"height_ratios": [1, 2]}, figsize=(14, 8))
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state("zoomed")
    except Exception:
        mng.resize(1400, 1000)

    # Plot timeline (top axes):
    if timeline is not None:
        timeline_rgb = cv2.cvtColor(timeline, cv2.COLOR_BGR2RGB)
        ax1.imshow(timeline_rgb, extent=[0, timeline_total_width, 0, composite_height])
        ax1.set_xlim([0, timeline_total_width])
        ax1.axis("off")
        ax1.set_title("Timeline of Recorded Gestures")
        for b in boundaries:
            ax1.axvline(x=b, color="grey", linewidth=0.5)
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
        # Draw vertical grey delimiters on the feature plot as well.
        for b in boundaries:
            ax2.axvline(x=b, color="grey", linewidth=0.5)
        # Draw blue overlay rectangles corresponding to detections.
        for (x0, x1) in detections:
            ax2.axvspan(x0, x1, color="blue", alpha=0.2)
    else:
        ax2.text(0.5, 0.5, "No feature data available", ha="center", va="center")
        ax2.axis("off")

    plt.tight_layout()
    plt.show()

def draw_play_pause_symbol(frame, is_paused):
    """
    Draws the play/pause symbol on the provided frame using the arial.ttf font.
    - When is_paused is True, display the play symbol (►).
    - When is_paused is False, display the pause symbol (‖).
    """
    # Convert BGR frame to RGB PIL image.
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)
    
    # Choose symbol and font properties.
    symbol = "►" if is_paused else "‖"
    font_size = 40  # adjust as needed
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    bbox = font.getbbox(symbol)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1] + font_size // 2  # adjust for better centering
    
    h, w = frame.shape[:2]
    overlay_height = 50  # same as your overlay height
    x = (w - text_w) // 2
    y = h - overlay_height + (overlay_height - text_h) // 2

    # Draw the symbol with white color.
    draw.text((x, y), symbol, font=font, fill=(255, 255, 255))
    
    # Convert back to OpenCV BGR image.
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

def get_sort_key(f):
    parts = f.rsplit('_', 2)
    key = (parts[1] + "_" + parts[2].split('.')[0], f)
    return key

def get_sorted_video_file_list(output_dir, get_sort_key_func=get_sort_key):
    return sorted([f for f in os.listdir(output_dir) if f.endswith(".mp4")], key=get_sort_key_func)

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
                    print(f"Recording gesture {gestures_dict[GESTURES[current_gesture_index]]["name"]}...")

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
                gesture_text = gestures_dict[GESTURES[current_gesture_index]]["name"].capitalize()
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

    # After drawing the overlay, replace the play/pause symbol with a PIL-rendered version.
    frame = draw_play_pause_symbol(frame, is_paused)

    # Display current gesture for debugging.
    cv2.putText(
        frame,
        f"Gesture: {gestures_dict[GESTURES[current_gesture_index]]["name"]}",
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
            file_list = get_sorted_video_file_list(output_dir)
            for f in file_list:
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
