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
import shutil

from videoProcessingTools import get_angles
from mediapipe_extract import extract_features_v2
from tools import write_to_csv
from dotenv import load_dotenv

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")
APP_REL_PATH = os.getenv("APP_REL_PATH") or "app/"
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

# Create a directory to save detected gesture samples
SAMPLES_FOLDER = "samples"
os.makedirs(SAMPLES_FOLDER, exist_ok=True)

# Global state for interactive review.
# List of detections; each detection is (global_x0, global_x1, csv_filename)
# obtained previously from show_timeline_and_features.
# Here we also maintain a dictionary mapping detection index to a decision:
#   "none" (default), "save", or "skip".
detection_states = {}  # detection index -> state string.
selected_detection_idx = None

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

def get_composite_info(video_files, target_img_height, header_height, font_face, font_scale, thickness, text_color):
    """
    Process video files to create composite images.
    Returns a list of tuples:
      (composite image, video_frame_count, gesture_key, filename)
    """
    composite_info = []
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
            h, w = frame.shape[:2]
            scale = target_img_height / h
            new_w = int(w * scale)
            resized_frame = cv2.resize(frame, (new_w, target_img_height))
            composite = np.full((header_height + target_img_height, new_w, 3), 200, dtype=np.uint8)
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
            # Write the gesture label in the header.
            (text_w, text_h), _ = cv2.getTextSize(gesture_label, font_face, font_scale, thickness)
            text_x = (new_w - text_w) // 2
            text_y = text_h
            cv2.putText(composite, gesture_label, (text_x, text_y),
                        font_face, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
            # Place the resized frame below the header.
            composite[header_height:header_height+target_img_height, 0:new_w, :] = resized_frame
            composite_info.append((composite, total_frames, gesture_key, f))
    return composite_info

def get_csv_info(video_files):
    """
    For each video file, if an accompanying CSV exists, load it.
    Returns a list of tuples:
      (csv_header, data_rows, filename)
    """
    csv_info = []
    for f in video_files:
        video_path = os.path.join(output_dir, f)
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
                    csv_info.append((rows[0], data_rows, f))
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
    return csv_info

def build_timeline(composite_info, timeline_total_width, composite_height):
    """
    Build the timeline image and determine vertical boundaries.
    Returns (timeline, boundaries)
    """
    timeline_parts = []
    boundaries = []
    cumulative = 0
    total_video_frames = sum(frames for (_, frames, _, _) in composite_info)
    timeline_total_width = max(timeline_total_width, total_video_frames)
    for composite, frames, _, _ in composite_info:
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
        # Re-add gesture label if needed.
        # (Optional step if cropping alters the header.)
        timeline_parts.append(composite_resized)
        cumulative += target_width

    current_width = sum(part.shape[1] for part in timeline_parts)
    if current_width < timeline_total_width:
        filler = np.full((composite_height, timeline_total_width - current_width, 3), 200, dtype=np.uint8)
        timeline_parts.append(filler)
    timeline = np.hstack(timeline_parts) if timeline_parts else None
    return timeline, boundaries

def aggregate_feature_data(csv_info):
    """
    Aggregate CSV feature data from all segments.
    Returns all_features (numpy array), header (list), and segment_boundaries (list).
    """
    all_features = []
    header = None
    segment_boundaries = []
    cumulative_feat = 0
    for csv_header, data_rows, _ in csv_info:
        all_features.extend(data_rows)
        cumulative_feat += len(data_rows)
        segment_boundaries.append(cumulative_feat)
        if header is None:
            header = csv_header
    if all_features:
        all_features = np.array(all_features)
    else:
        all_features = None
    return all_features, header, segment_boundaries

def plot_timeline_and_features(timeline, boundaries, composite_height, all_features, header, x_vals_scaled, detections, timeline_total_width):
    """
    Plot the timeline and monitored features with detections.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]}, figsize=(14, 8))
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state("zoomed")
    except Exception:
        mng.resize(1400, 1000)

    # Plot timeline
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

    # Plot features
    if all_features is not None:
        num_features = all_features.shape[1]
        for i in range(num_features):
            ax2.plot(x_vals_scaled, all_features[:, i],
                     label=header[i] if header is not None else f"F{i}")
        ax2.set_xlabel("Time (scaled to timeline)")
        ax2.set_title("Monitored Features")
        ax2.legend(loc="lower right")
        ax2.set_xlim(0, timeline_total_width)
        for b in boundaries:
            ax2.axvline(x=b, color="grey", linewidth=0.5)
        # Draw blue overlay for detections.
        for (x0, x1, fname) in detections:
            ax2.axvspan(x0, x1, color="blue", alpha=0.2)
    else:
        ax2.text(0.5, 0.5, "No feature data available", ha="center", va="center")
        ax2.axis("off")

    plt.tight_layout()
    plt.show()

def build_detections(csv_info, composite_info, fps, detection_interval):
    """
    For each CSV segment, detect significant fluctuations from the features of
    interest defined in gestures_dict. Each detection tuple (in CSV row indices)
    is augmented with the originating gesture file name and the segment's offset.
    Returns a list of detections:
         (csv_filename, start_idx, end_idx, seg_offset)
    """
    detections = []
    cumulative_feat_prev = 0
    for i, (csv_header, data_rows, csv_filename) in enumerate(csv_info):
        seg_feat = np.array(data_rows)
        seg_length = seg_feat.shape[0]
        if i < len(composite_info):
            _, _, gesture_key, file_name = composite_info[i]
        else:
            gesture_key = "unknown"
            file_name = "unknown"
        segment_detections = []
        if gesture_key in gestures_dict:
            features_of_interest = gestures_dict[gesture_key]["features"]
            threshold = gestures_dict[gesture_key]["threshold"]
            for feat_name in features_of_interest:
                if csv_header and feat_name in csv_header:
                    col = csv_header.index(feat_name)
                    subdata = seg_feat[:, col]
                    det = detect_fluctuations_for_feature(subdata, fps, detection_interval, threshold)
                    segment_detections.extend(det)
        segment_detections = merge_intervals(segment_detections)
        for (start_idx, end_idx) in segment_detections:
            detections.append((csv_filename, start_idx, end_idx, cumulative_feat_prev))
        cumulative_feat_prev += seg_length
    return detections

def save_detections(detections, output_dir):
    """
    Given a list of detections (each a tuple: (csv_filename, start_idx, end_idx, seg_offset)),
    for each detection, crop the corresponding CSV and MP4 files and save
    them to a folder called "detections". The cropped files use the same base filename
    as the original gesture files with an added _detection marker.
    """
    det_folder = "detections"
    os.makedirs(det_folder, exist_ok=True)
    for (csv_fname, start_idx, end_idx, seg_offset) in detections:
        # The original CSV and video files are in the output directory.
        base = os.path.splitext(csv_fname)[0]
        csv_path = os.path.join(output_dir, base + ".csv")
        video_path = os.path.join(output_dir, base + ".mp4")
        # Crop the CSV.
        try:
            with open(csv_path, "r") as f_csv:
                reader = list(csv.reader(f_csv))
            header_line = reader[0]
            data_rows = reader[1:]
            cropped_data = data_rows[start_idx : end_idx + 1]
            cropped_csv_path = os.path.join(det_folder, base + f"_detection_{start_idx}_{end_idx}.csv")
            with open(cropped_csv_path, "w", newline="") as f_out:
                writer = csv.writer(f_out)
                writer.writerow(header_line)
                writer.writerows(cropped_data)
            print(f"Saved detection CSV: {cropped_csv_path}")
        except Exception as e:
            print(f"Error cropping CSV {csv_path}: {e}")
        # Crop the video.
        cap_vid = cv2.VideoCapture(video_path)
        if not cap_vid.isOpened():
            print(f"Error opening video for detection: {video_path}")
            continue
        total_frames = int(cap_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        # We assume here that the number of CSV rows equals the number of frames.
        start_frame = start_idx
        end_frame = min(end_idx, total_frames - 1)
        ret, frame = cap_vid.read()
        if not ret:
            cap_vid.release()
            continue
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp42")
        fps_vid = cap_vid.get(cv2.CAP_PROP_FPS)
        cropped_video_path = os.path.join(det_folder, base + f"_detection_{start_idx}_{end_idx}.mp4")
        writer = cv2.VideoWriter(cropped_video_path, fourcc, fps_vid, (w, h))
        cap_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for fnum in range(start_frame, end_frame + 1):
            ret, frame = cap_vid.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()
        cap_vid.release()
        print(f"Saved detection video: {cropped_video_path}")

def build_detections(csv_info, composite_info, fps, detection_interval):
    """
    For each CSV segment, detect significant fluctuations from the features of
    interest defined in gestures_dict. Each detection tuple (in CSV row indices)
    is augmented with the originating gesture file name and the segment's offset.
    Returns a list of detections:
         (csv_filename, start_idx, end_idx, seg_offset)
    """
    detections = []
    cumulative_feat_prev = 0
    for i, (csv_header, data_rows, csv_filename) in enumerate(csv_info):
        seg_feat = np.array(data_rows)
        seg_length = seg_feat.shape[0]
        if i < len(composite_info):
            _, _, gesture_key, file_name = composite_info[i]
        else:
            gesture_key = "unknown"
            file_name = "unknown"
        segment_detections = []
        if gesture_key in gestures_dict:
            features_of_interest = gestures_dict[gesture_key]["features"]
            threshold = gestures_dict[gesture_key]["threshold"]
            for feat_name in features_of_interest:
                if csv_header and feat_name in csv_header:
                    col = csv_header.index(feat_name)
                    subdata = seg_feat[:, col]
                    det = detect_fluctuations_for_feature(subdata, fps, detection_interval, threshold)
                    segment_detections.extend(det)
        segment_detections = merge_intervals(segment_detections)
        for (start_idx, end_idx) in segment_detections:
            detections.append((csv_filename, start_idx, end_idx, cumulative_feat_prev))
        cumulative_feat_prev += seg_length
    return detections

def save_detections(detections, output_dir):
    """
    Given a list of detections (each a tuple: (csv_filename, start_idx, end_idx, seg_offset)),
    for each detection, crop the corresponding CSV and MP4 files and save
    them to a folder called "detections". The cropped files use the same base filename
    as the original gesture files with an added _detection marker.
    """
    det_folder = "detections"
    os.makedirs(det_folder, exist_ok=True)
    for (csv_fname, start_idx, end_idx, seg_offset) in detections:
        # The original CSV and video files are in the output directory.
        base = os.path.splitext(csv_fname)[0]
        csv_path = os.path.join(output_dir, base + ".csv")
        video_path = os.path.join(output_dir, base + ".mp4")

        if not os.path.exists(csv_path):
            # Crop the CSV.
            try:
                with open(csv_path, "r") as f_csv:
                    reader = list(csv.reader(f_csv))
                header_line = reader[0]
                data_rows = reader[1:]
                cropped_data = data_rows[start_idx : end_idx + 1]
                cropped_csv_path = os.path.join(det_folder, base + f"_detection_{start_idx}_{end_idx}.csv")
                with open(cropped_csv_path, "w", newline="") as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(header_line)
                    writer.writerows(cropped_data)
                print(f"Saved detection CSV: {cropped_csv_path}")
            except Exception as e:
                print(f"Error cropping CSV {csv_path}: {e}")

        if not os.path.exists(video_path):
            # Crop the video.
            cap_vid = cv2.VideoCapture(video_path)
            if not cap_vid.isOpened():
                print(f"Error opening video for detection: {video_path}")
                continue
            total_frames = int(cap_vid.get(cv2.CAP_PROP_FRAME_COUNT))
            # We assume here that the number of CSV rows equals the number of frames.
            start_frame = start_idx
            end_frame = min(end_idx, total_frames - 1)
            ret, frame = cap_vid.read()
            if not ret:
                cap_vid.release()
                continue
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp42")
            fps_vid = cap_vid.get(cv2.CAP_PROP_FPS)
            cropped_video_path = os.path.join(det_folder, base + f"_detection_{start_idx}_{end_idx}.mp4")
            writer = cv2.VideoWriter(cropped_video_path, fourcc, fps_vid, (w, h))
            cap_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for fnum in range(start_frame, end_frame + 1):
                ret, frame = cap_vid.read()
                if not ret:
                    break
                writer.write(frame)
            writer.release()
            cap_vid.release()
            print(f"Saved detection video: {cropped_video_path}")

def playback_detection_video(detection, output_dir, fps):
    """
    Given a detection tuple (global_x0, global_x1, csv_filename) play
    the corresponding video (derived from the CSV filename) in a separate window.
    The video playback auto-closes 1 second after the video ends or if space is pressed.
    """
    # Derive video filename; assume CSV filename has the same base name.
    csv_fname = detection[2]
    base = os.path.splitext(csv_fname)[0]
    video_path = os.path.join(output_dir, base + ".mp4")
    cap_vid = cv2.VideoCapture(video_path)
    if not cap_vid.isOpened():
        print("Error opening video for detection:", video_path)
        return
    window_name = "Detection Playback"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Playback loop.
    while True:
        ret, frame = cap_vid.read()
        if not ret:
            break
        cv2.imshow(window_name, frame)
        # Wait 30ms per frame; if space is pressed, break playback early.
        key = cv2.waitKey(30) & 0xFF
        if key == ord(" "):
            break
    # Auto-close window 1 second after playback stops.
    cv2.waitKey(1000)
    cap_vid.release()
    cv2.destroyWindow(window_name)

# New helper: update the color for a given state.
def get_detection_color(state):
    return {"none": "blue", "save": "green", "skip": "grey"}.get(state, "blue")

# New helper: (re)draw all detection overlays based on detection_states.
def draw_detection_overlays(ax, global_detections, detection_states):
    patches = {}  # mapping detection index -> patch artist.
    for idx, (x0, x1, fname) in enumerate(global_detections):
        state = detection_states.get(idx, "none")
        color = get_detection_color(state)
        patch = ax.axvspan(x0, x1, color=color, alpha=0.2)
        patches[idx] = patch
    return patches

# New helper: update all detection overlays (remove old ones and redraw).
def update_all_detection_overlays(ax, global_detections, detection_states, detection_patches):
    # Remove previous patches.
    for patch in detection_patches.values():
        try:
            patch.remove()
        except Exception:
            pass
    # Redraw patches.
    new_patches = draw_detection_overlays(ax, global_detections, detection_states)
    return new_patches

# New helper: process detection states to copy files.
def process_detection_states(global_detections, detection_states, output_dir):
    samples_folder = "samples"
    skipped_folder = "skipped"
    os.makedirs(samples_folder, exist_ok=True)
    os.makedirs(skipped_folder, exist_ok=True)
    
    for idx, (_, _, csv_fname) in enumerate(global_detections):
        state = detection_states.get(idx, "none")
        base = os.path.splitext(csv_fname)[0]
        src_csv = os.path.join(output_dir, base + ".csv")
        src_video = os.path.join(output_dir, base + ".mp4")
        if state == "save":
            dst_csv = os.path.join(samples_folder, base + ".csv")
            dst_video = os.path.join(samples_folder, base + ".mp4")
        elif state == "skip" or state == "none":
            dst_csv = os.path.join(skipped_folder, base + ".csv")
            dst_video = os.path.join(skipped_folder, base + ".mp4")
        else:
            continue
        try:
            if not os.path.exists(dst_csv):
                shutil.copy2(src_csv, dst_csv)
                print(f"Copied CSV to {dst_csv}")
        except Exception as e:
            print(f"Error copying CSV {src_csv}: {e}")
        try:
            if not os.path.exists(dst_video):
                shutil.copy2(src_video, dst_video)
                print(f"Copied video to {dst_video}")
        except Exception as e:
            print(f"Error copying video {src_video}: {e}")

def interactive_feature_plot(timeline, boundaries, composite_height, all_features, header, 
                               x_vals_scaled, global_detections, timeline_total_width, fps, output_dir):
    """
    Creates the interactive figure with timeline (ax1) and monitored features (ax2).
    Allows mouse click selection and keyboard events:
      - Left/right arrow keys: change selected detection.
      - Down arrow: mark selected detection as "save" (green tint).
      - Up arrow: mark selected detection as "skip" (grey tint).
      - 'a': mark all detections as "save".
      - 'w': process all detections—copy files to folder "samples" or "skipped".
      - 'c': clear all detections (reset to "none").
      - Space: play the video for the selected detection.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, 
          gridspec_kw={"height_ratios": [1, 2]}, figsize=(14, 8))
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state("zoomed")
    except Exception:
        mng.resize(1400, 1000)

    # Plot timeline (ax1) as before.
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

    # Plot monitored features (ax2)
    if all_features is not None:
        num_features = all_features.shape[1]
        for i in range(num_features):
            ax2.plot(x_vals_scaled, all_features[:, i],
                     label=header[i] if header is not None else f"F{i}")
        ax2.set_xlabel("Time (scaled to timeline)")
        ax2.set_title("Monitored Features (Click/Arrow to select, Space to playback)")
        ax2.legend(loc="lower right")
        ax2.set_xlim(0, timeline_total_width)
        for b in boundaries:
            ax2.axvline(x=b, color="grey", linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, "No feature data available", ha="center", va="center")
        ax2.axis("off")

    # Draw detection overlays based on detection_states.
    # detection_states is a global dict: detection index -> state ("none", "save", "skip").
    global detection_states
    for idx in range(len(global_detections)):
        if idx not in detection_states:
            detection_states[idx] = "none"
    detection_patches = draw_detection_overlays(ax2, global_detections, detection_states)

    # Selected detection index.
    selected_idx = [0]
    # Highlight selection with an extra black border (we draw an extra patch).
    highlight_patch = [None]
    def update_selection_highlight():
        if highlight_patch[0] is not None:
            try:
                highlight_patch[0].remove()
            except Exception:
                pass
        if global_detections and 0 <= selected_idx[0] < len(global_detections):
            x0, x1, _ = global_detections[selected_idx[0]]
            highlight_patch[0] = ax2.axvspan(x0, x1, color="black", alpha=0.3)
        fig.canvas.draw_idle()

    def refresh_overlays():
        nonlocal detection_patches
        detection_patches = update_all_detection_overlays(ax2, global_detections, detection_states, detection_patches)
        fig.canvas.draw_idle()

    def on_key(event):
        # Left/Right: change selection.
        if not global_detections:
            return
        if event.key == "left":
            selected_idx[0] = max(0, selected_idx[0] - 1)
            update_selection_highlight()
        elif event.key == "right":
            selected_idx[0] = min(len(global_detections) - 1, selected_idx[0] + 1)
            update_selection_highlight()
        elif event.key == "down":
            # Mark selected detection as "save".
            detection_states[selected_idx[0]] = "save"
            refresh_overlays()
        elif event.key == "up":
            # Mark selected detection as "skip".
            detection_states[selected_idx[0]] = "skip"
            refresh_overlays()
        elif event.key == "a":
            # Mark all detections as "save".
            for idx in range(len(global_detections)):
                detection_states[idx] = "save"
            refresh_overlays()
        elif event.key == "c":
            # Clear all detections.
            for idx in range(len(global_detections)):
                detection_states[idx] = "none"
            refresh_overlays()
        elif event.key == "w":
            # Process: copy all detections marked for save/skip.
            process_detection_states(global_detections, detection_states, output_dir)
        elif event.key == " ":
            # Playback selected detection.
            detection = global_detections[selected_idx[0]]
            playback_detection_video(detection, output_dir, fps)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax2:
            return
        click_x = event.xdata
        for idx, (x0, x1, _) in enumerate(global_detections):
            if x0 <= click_x <= x1:
                selected_idx[0] = idx
                update_selection_highlight()
                break

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    # Initialize selection.
    if global_detections:
        selected_idx[0] = 0
        update_selection_highlight()
    plt.tight_layout()
    plt.show()

def interactive_feature_wrapper(timeline, boundaries, composite_height, all_features, header, 
                                 x_vals_scaled, global_detections, timeline_total_width, fps, output_dir):
    interactive_feature_plot(timeline, boundaries, composite_height, all_features, header, 
                              x_vals_scaled, global_detections, timeline_total_width, fps, output_dir)
    
def show_timeline_and_features():
    """
    Orchestrates the building and plotting of the timeline and feature graph.
    CSV info tuples now include the originating gesture file name, and
    detection tuples also include that name.
    BEFORE plotting, call save_detections to save cropped CSV and video files of
    each detection in a folder called "detections".
    Then display an interactive feature plot that allows selection and playback.
    """
    target_img_height = 80
    header_height = 10
    composite_height = header_height + target_img_height
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_val = 0.33
    thickness = 1
    text_color = (0, 0, 0)
    timeline_min_width = 100

    video_files = get_sorted_video_file_list(output_dir)
    composite_info = get_composite_info(video_files, target_img_height, header_height,
                                        font_face, font_scale_val, thickness, text_color)
    csv_info = get_csv_info(video_files)
    
    # Build timeline image.
    total_video_frames = sum(frames for (_, frames, _, _) in composite_info)
    timeline_total_width = max(timeline_min_width, total_video_frames)
    timeline, boundaries = build_timeline(composite_info, timeline_total_width, composite_height)
    
    # Aggregate feature data.
    all_features, header, segment_boundaries = aggregate_feature_data(csv_info)
    all_features_count = len(all_features) if all_features is not None else 0
    factor = timeline_total_width / all_features_count if all_features_count else 1
    x_vals_scaled = [x * factor for x in range(all_features_count)] if all_features_count else []
    
    # Detect fluctuations in each segment.
    detection_interval = DETECTION_INTERVAL_SEC  # e.g., 0.5 sec
    raw_detections = build_detections(csv_info, composite_info, fps, detection_interval)
    # Save the detections (cropped CSV & video) in folder "detections".
    save_detections(raw_detections, output_dir)
    
    # Convert raw_detections (with CSV row indices) into global x–coordinates for plotting.
    global_detections = []
    for (csv_fname, start_idx, end_idx, seg_offset) in raw_detections:
        global_x0 = (seg_offset + start_idx) * factor
        global_x1 = (seg_offset + end_idx) * factor
        global_detections.append((global_x0, global_x1, csv_fname))
    
    # Now, call the interactive feature plot that enables selection and playback.
    interactive_feature_wrapper(timeline, boundaries, composite_height, all_features, header, x_vals_scaled, global_detections, timeline_total_width, fps, output_dir)    

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
