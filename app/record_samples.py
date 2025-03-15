import json
from contextlib import redirect_stderr
import PySide6
import cv2
import numpy as np
import datetime as dt
import os
import time
import matplotlib.pyplot as plt
import csv
from PIL import ImageFont, ImageDraw, Image
import shutil
import glob
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QListWidget,
    QPushButton,
)
import sys
import threading
from playsound import playsound
import pygame


from videoProcessingTools import get_angles
from mediapipe_extract import extract_features_v2
from tools import write_to_csv
from dotenv import load_dotenv

# Initialize pygame mixer once (perhaps near your other global initializations)
pygame.mixer.init()

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")
APP_REL_PATH = os.getenv("APP_REL_PATH") or "app/"
SOUNDS_DIR = os.getenv("SOUNDS_FOLDER") or "sounds"
WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX") or 0)
DETECTION_INTERVAL_SEC = float(os.getenv("DETECTION_INTERVAL_SEC") or 0.25)
VIDEO_ENCODER_FOURCC = (
    os.getenv("VIDEO_ENCODER_FOURCC") or "mp42"
)  # "mp4v" for Windows, "avc1" for MacOS
SKIP_PROGRESSBAR_WINDOWS = int(os.getenv("SKIP_PROGRESSBAR_WINDOWS") or 0)

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
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)
DETECTIONS_DIR = "detections"
os.makedirs(DETECTIONS_DIR, exist_ok=True)
# Create a directory to save detected gesture samples
SAMPLES_DIR = "samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)
SKIPPED_DIR = "skipped"
os.makedirs(SKIPPED_DIR, exist_ok=True)

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get recording parameters from the webcam
recording_frame_width = int(cap.get(3))
recording_frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(
    *VIDEO_ENCODER_FOURCC
)  # avc1 works on MacOS, mp4v works on Windows
recording_in_progress_thread = None
recording_in_progress_stop_event = None

# Define display settings
display_frame_width = 640
display_frame_height = int(
    recording_frame_height * display_frame_width / recording_frame_width
)
font = cv2.FONT_HERSHEY_SIMPLEX
gesture_font = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "Webcam Feed"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, display_frame_width, display_frame_height)


# When video is paused, show PLAY symbol and state is_paused True;
# when playing, show PAUSE symbol.
is_paused = True
is_review = False

def play_recording_in_progress(stop_event):
    """
    Loops playing recording_in_progress.mp3 until stop_event is set.
    """
    sound_path = f"{BASE_PATH}{APP_REL_PATH}{SOUNDS_DIR}/recording_in_progress.mp3"
    while not stop_event.is_set():
        try:
            playsound(sound_path)
        except Exception as e:
            print(f"Error playing recording_in_progress sound: {e}")
            break
        
def show_progress_modal(total, operation_text):
    """
    Create a modal progress window using PySide6 that:
      - Displays the current operation (operation_text)
      - Shows a count label "x/y"
      - Contains a progress bar (maximum set to 'total')
    """
    # Ensure a QApplication instance exists.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    dialog = QDialog()
    dialog.setWindowTitle("Processing...")
    dialog.setModal(True)
    layout = QVBoxLayout()

    label_op = QLabel(operation_text)
    layout.addWidget(label_op)

    label_count = QLabel("0/{}".format(total))
    layout.addWidget(label_count)

    progress_bar = QProgressBar()
    progress_bar.setOrientation(PySide6.QtCore.Qt.Horizontal)  # Horizontal
    progress_bar.setMaximum(total)
    progress_bar.setValue(0)
    layout.addWidget(progress_bar)

    dialog.setLayout(layout)
    dialog.resize(400, 150)
    dialog.show()
    app.processEvents()  # Ensure the dialog appears
    return dialog, label_op, label_count, progress_bar


def update_progress(progress_bar, label_count, current, total):
    """Update the progress bar value and count label using PySide6."""
    progress_bar.setValue(current)
    label_count.setText("{}/{}".format(current, total))
    QApplication.processEvents()


def show_help_window():
    global is_paused
    is_paused = True
    help_text = (
        "Happy flow:\n"
        "  Webcam Window: \n"
        "  -> space: Start recording\n"
        "         [Do the gestures]\n"
        "  -> space: Stop recording\n"
        "  -> r: Switch to Review Mode \n"
        "  Review Window: \n"
        "  -> left/right or click: Select detection in review window\n"
        "  -> a: Mark ALL detections as SAVE\n"
        "  -> w: Display saved/skipped samples window\n"
        "  -> x: Clean-up detections & recorded gestures,\n"
        "        return to Webcam Window\n"
        "  [Repeat / End]\n"
        "    \n"
        "-------------------------------------\n"
        "    \n"
        "Help - Available Keys:\n"
        "  Webcam Window: \n"
        "    space: Toggle recording\n"
        "    r: Switch to Review Mode \n"
        "    h: Show this help window\n"
        "    q / ESC: Exit\n"
        "    \n"
        "  Review Window: \n"
        "    left/right or click: Select detection in review window\n"
        "    space: Playback the selected detection video\n"
        "    down: Mark selected detection as SAVE (green tint)\n"
        "    up: Mark selected detection as SKIP (grey tint)\n"
        "    a: Mark ALL detections as SAVE\n"
        "    c: Mark ALL detections as NONE\n"
        "    q: Close to Webcam Window (without clean-up)\n"
        "    w: Display saved/skipped samples window\n"
        "    x: Clean-up detections & recorded gestures,\n"
        "       return to Webcam Window"
    )

    help_img = np.ones((560, 500, 3), dtype=np.uint8) * 230
    y0, dy = 16, 16
    for i, line in enumerate(help_text.split("\n")):
        cv2.putText(
            help_img,
            line,
            (10, y0 + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imshow("Help", help_img)

    cv2.waitKey(1000)  # Wait 1 second to ensure the window is displayed.

    # Continuously check if the window is closed or a key is pressed.
    while True:
        # wait 10ms for key press.
        key = cv2.waitKey(10)
        if key != -1 or cv2.getWindowProperty("Help", cv2.WND_PROP_VISIBLE) < 1:
            break
    try:
        cv2.destroyWindow("Help")
    except cv2.error:
        pass


def show_saved_samples_window(samples_folder, skipped_folder):
    """
    Create a PySide6 modal dialog titled "Saved Samples" that lists the saved samples
    (files in the 'samples' folder) and skipped samples (files in the 'skipped' folder).
    The dialog closes when the user clicks the OK button.
    """
    # Ensure a QApplication instance exists.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = QDialog()
    dialog.setWindowTitle("Saved Samples")
    layout = QVBoxLayout()

    # Saved Samples list.
    saved_label = QLabel("Saved Samples")
    layout.addWidget(saved_label)
    saved_list = QListWidget()
    saved_files = (
        sorted(os.listdir(samples_folder)) if os.path.exists(samples_folder) else []
    )
    for file in saved_files:
        saved_list.addItem(file)
    layout.addWidget(saved_list)

    # Skipped Samples list.
    skipped_label = QLabel("Skipped Samples")
    layout.addWidget(skipped_label)
    skipped_list = QListWidget()
    skipped_files = (
        sorted(os.listdir(skipped_folder)) if os.path.exists(skipped_folder) else []
    )
    for file in skipped_files:
        skipped_list.addItem(file)
    layout.addWidget(skipped_list)

    # OK button to close the dialog.
    btn = QPushButton("OK")
    btn.clicked.connect(dialog.accept)
    layout.addWidget(btn)

    dialog.setLayout(layout)
    dialog.resize(600, 400)
    dialog.exec()


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
        block = feature_data[i : i + interval_len]
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


def get_composite_info(
    video_files,
    target_img_height,
    header_height,
    font_face,
    font_scale,
    thickness,
    text_color,
):
    """
    Process video files to create composite images.
    The header height, target image height and font_scale will be reduced
    proportional to the number of video_files.
    Returns a tuple:
      (composite_info, new_target_img_height, new_header_height, new_font_scale)
    where composite_info is a list of tuples:
      (composite image, video_frame_count, gesture_key, filename)
    """
    num_files = len(video_files)
    # Calculate an adjustment factor (for example, 1 divided by number of files)
    # You might modify the formula as desired.
    adjust_factor = (1.0 / num_files if num_files > 0 else 1.0) * num_files
    new_target_img_height = max(1, int(target_img_height * adjust_factor))  # //3+1
    new_header_height = max(1, int(header_height * adjust_factor))
    new_font_scale = font_scale * adjust_factor

    composite_info = []
    for f in video_files:
        video_path = os.path.join(RECORDINGS_DIR, f)
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
            scale = new_target_img_height / h
            new_w = int(w * scale)
            resized_frame = cv2.resize(frame, (new_w, new_target_img_height))
            composite = np.full(
                (new_header_height + new_target_img_height, new_w, 3),
                200,
                dtype=np.uint8,
            )
            # Determine gesture from filename.
            base = os.path.basename(video_path)
            parts = base.split("_")
            try:
                gesture_index = int(parts[1])
                gesture_key = GESTURES[gesture_index]
                gesture_label = gestures_dict[gesture_key]["name"]
            except Exception:
                gesture_key = "unknown"
                gesture_label = "Unknown"
            # Write the gesture label in the header using the new font scale.
            (text_w, text_h), _ = cv2.getTextSize(
                gesture_label, font_face, new_font_scale, thickness
            )
            text_x = (new_w - text_w) // 2
            text_y = text_h
            cv2.putText(
                composite,
                gesture_label,
                (text_x, text_y),
                font_face,
                new_font_scale,
                text_color,
                thickness,
                lineType=cv2.LINE_AA,
            )
            # Place the resized frame below the header.
            composite[
                new_header_height : new_header_height + new_target_img_height,
                0:new_w,
                :,
            ] = resized_frame
            composite_info.append((composite, total_frames, gesture_key, f))
    return composite_info, new_target_img_height, new_header_height, new_font_scale


def get_csv_info(video_files):
    """
    For each video file, if an accompanying CSV exists, load it.
    Returns a list of tuples:
      (csv_header, data_rows, filename)
    """
    csv_info = []
    for f in video_files:
        video_path = os.path.join(RECORDINGS_DIR, f)
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
    For each composite image (representing a gesture), the image is centered
    horizontally within its target width on the timeline.

    Returns:
        timeline: the final timeline image.
        boundaries: a list of left-boundary x positions in pixels.
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
            # Pad equally on both sides.
            total_pad = target_width - current_width
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            left_pad = np.full((composite.shape[0], pad_left, 3), 200, dtype=np.uint8)
            right_pad = np.full((composite.shape[0], pad_right, 3), 200, dtype=np.uint8)
            composite_resized = np.hstack((left_pad, composite, right_pad))
        else:
            # Crop equally on both sides.
            excess = current_width - target_width
            crop_left = excess // 2
            composite_resized = composite[:, crop_left : crop_left + target_width]
        timeline_parts.append(composite_resized)
        cumulative += target_width

    current_width = sum(part.shape[1] for part in timeline_parts)
    if current_width < timeline_total_width:
        filler = np.full(
            (composite_height, timeline_total_width - current_width, 3),
            200,
            dtype=np.uint8,
        )
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


def plot_timeline_and_features(
    timeline,
    boundaries,
    composite_height,
    all_features,
    header,
    x_vals_scaled,
    detections,
    timeline_total_width,
):
    """
    Plot the timeline and monitored features with detections.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [1, 2]}, figsize=(14, 8)
    )
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
            ax2.plot(
                x_vals_scaled,
                all_features[:, i],
                label=header[i] if header is not None else f"F{i}",
            )
        ax2.set_xlabel("Time (scaled to timeline)")
        ax2.set_title("Monitored Features")
        ax2.legend(loc="lower right")
        ax2.set_xlim(0, timeline_total_width)
        for b in boundaries:
            ax2.axvline(x=b, color="grey", linewidth=0.5)
        # Draw blue overlay for detections.
        for x0, x1, fname in detections:
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
                    det = detect_fluctuations_for_feature(
                        subdata, fps, detection_interval, threshold
                    )
                    segment_detections.extend(det)
        segment_detections = merge_intervals(segment_detections)
        for start_idx, end_idx in segment_detections:
            detections.append((csv_filename, start_idx, end_idx, cumulative_feat_prev))
        cumulative_feat_prev += seg_length
    return detections


def save_detections(detections, output_dir, det_folder):
    """
    Given a list of detections (each a tuple: (csv_filename, start_idx, end_idx, seg_offset)),
    for each detection, crop the corresponding CSV and MP4 files and save
    them to a folder called "detections". The cropped files use the same base filename
    as the original gesture files with an added _detection marker.
    """

    for csv_fname, start_idx, end_idx, seg_offset in detections:
        # The original CSV and video files are in the output directory.
        base = os.path.splitext(csv_fname)[0]
        csv_path = os.path.join(output_dir, base + ".csv")
        video_path = os.path.join(output_dir, base + ".mp4")
        dest_csv_path = os.path.join(det_folder, base + ".csv")
        dest_video_path = os.path.join(det_folder, base + ".mp4")

        if os.path.exists(csv_path) and not os.path.exists(dest_csv_path):
            # Crop the CSV.
            try:
                with open(csv_path, "r") as f_csv:
                    reader = list(csv.reader(f_csv))
                header_line = reader[0]
                data_rows = reader[1:]
                cropped_data = data_rows[start_idx : end_idx + 1]

                with open(dest_csv_path, "w", newline="") as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(header_line)
                    writer.writerows(cropped_data)
                print(f"Saved detection CSV: {dest_csv_path}")
            except Exception as e:
                print(f"Error cropping CSV {csv_path}: {e}")

        if os.path.exists(video_path) and not os.path.exists(dest_video_path):
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
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_ENCODER_FOURCC)
            fps_vid = cap_vid.get(cv2.CAP_PROP_FPS)

            writer = cv2.VideoWriter(dest_video_path, fourcc, fps_vid, (w, h))
            cap_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for fnum in range(start_frame, end_frame + 1):
                ret, frame = cap_vid.read()
                if not ret:
                    break
                writer.write(frame)
            writer.release()
            cap_vid.release()
            print(f"Saved detection video: {dest_video_path}")


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
                    det = detect_fluctuations_for_feature(
                        subdata, fps, detection_interval, threshold
                    )
                    segment_detections.extend(det)
        segment_detections = merge_intervals(segment_detections)
        for start_idx, end_idx in segment_detections:
            detections.append((csv_filename, start_idx, end_idx, cumulative_feat_prev))
        cumulative_feat_prev += seg_length
    return detections


def playback_detection_video(detection, detections_dir, width, height):
    """
    Given a detection tuple (csv_filename, ...), play the corresponding video (derived from the CSV filename)
    in a separate window. The window size is set to the display dimensions. Playback occurs at the video's realtime frame rate.
    The playback auto-closes 1 second after the video ends or if space is pressed.
    """
    csv_fname = detection[2]
    base = os.path.splitext(csv_fname)[0]
    video_path = os.path.join(detections_dir, base + ".mp4")
    cap_vid = cv2.VideoCapture(video_path)
    if not cap_vid.isOpened():
        print("Error opening video for detection:", video_path)
        return
    window_name = "Detection Playback"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    # Get video's FPS and compute frame duration.
    fps_vid = cap_vid.get(cv2.CAP_PROP_FPS)
    if fps_vid <= 0:
        fps_vid = 30
    frame_duration = 1.0 / fps_vid

    # Reset playback position to the beginning.
    cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        start_time = time.time()
        ret, frame = cap_vid.read()
        if not ret:
            break
        # Resize the frame for display.
        frame = cv2.resize(frame, (width, height))
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break
        # Ensure correct playback speed.
        elapsed = time.time() - start_time
        remaining = frame_duration - elapsed
        if remaining > 0:
            time.sleep(remaining)
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
def update_all_detection_overlays(
    ax, global_detections, detection_states, detection_patches
):
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
def process_detection_states(global_detections, detection_states):
    for idx, (_, _, csv_fname) in enumerate(global_detections):
        state = detection_states.get(idx, "none")
        base = os.path.splitext(csv_fname)[0]
        src_csv = os.path.join(DETECTIONS_DIR, base + ".csv")
        src_video = os.path.join(DETECTIONS_DIR, base + ".mp4")
        if state == "save":
            dst_csv = os.path.join(SAMPLES_DIR, base + ".csv")
            dst_video = os.path.join(SAMPLES_DIR, base + ".mp4")
        elif state == "skip" or state == "none":
            dst_csv = os.path.join(SKIPPED_DIR, base + ".csv")
            dst_video = os.path.join(SKIPPED_DIR, base + ".mp4")
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


def interactive_feature_plot(
    timeline,
    boundaries,
    composite_height,
    all_features,
    header,
    x_vals_scaled,
    global_detections,
    timeline_total_width,
    fps,
    output_dir,
):
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
    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [1, 2]}, figsize=(14, 8)
    )

    # Set the window title to "Review"
    fig.canvas.manager.set_window_title("Review")
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
            ax2.plot(
                x_vals_scaled,
                all_features[:, i],
                label=header[i] if header is not None else f"F{i}",
            )
        ax2.set_xlabel("Time (scaled to timeline)")
        ax2.set_title("Monitored Features")
        ax2.legend(loc="lower right")
        ax2.set_xlim(0, timeline_total_width)
        for b in boundaries:
            ax2.axvline(x=b, color="grey", linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, "No feature data available", ha="center", va="center")
        ax2.axis("off")

    # Add text box listing key actions.
    instructions = (
        "Keys: Left/Right: Select | Space: Playback | Down: Save | Up: Skip | "
        "'a': Save All | 'w': Write | 'x': Cleanup"
    )
    ax2.text(
        0.5,
        -0.1,
        instructions,
        transform=ax2.transAxes,
        ha="center",
        fontsize=10,
        color="black",
    )

    # Draw detection overlays based on detection_states.
    # detection_states is a global dict: detection index -> state ("none", "save", "skip").
    global detection_states
    for idx in range(len(global_detections)):
        if idx not in detection_states:
            detection_states[idx] = "none"
    detection_patches = draw_detection_overlays(
        ax2, global_detections, detection_states
    )

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
        detection_patches = update_all_detection_overlays(
            ax2, global_detections, detection_states, detection_patches
        )
        fig.canvas.draw_idle()

    def on_key(event):
        global is_review
        if not global_detections:
            return
        if event.key == "left":
            selected_idx[0] = max(0, selected_idx[0] - 1)
            update_selection_highlight()
        elif event.key == "right":
            selected_idx[0] = min(len(global_detections) - 1, selected_idx[0] + 1)
            update_selection_highlight()
        elif event.key == "down":
            detection_states[selected_idx[0]] = "save"
            refresh_overlays()
        elif event.key == "up":
            detection_states[selected_idx[0]] = "skip"
            refresh_overlays()
        elif event.key == "a":
            for idx in range(len(global_detections)):
                detection_states[idx] = "save"
            refresh_overlays()
        elif event.key == "w":
            # Process: copy all detections marked save or skip
            total = len(global_detections)
            if not SKIP_PROGRESSBAR_WINDOWS:
                progress_win, label_op, label_count, progress_bar = show_progress_modal(
                    total, "Copying detections..."
                )
            for idx, detection in enumerate(global_detections):
                state = detection_states.get(idx, "none")
                base = os.path.splitext(detection[2])[0]
                src_csv = os.path.join(DETECTIONS_DIR, base + ".csv")
                src_video = os.path.join(DETECTIONS_DIR, base + ".mp4")
                if state == "save":
                    dst_csv = os.path.join(SAMPLES_DIR, base + ".csv")
                    dst_video = os.path.join(SAMPLES_DIR, base + ".mp4")
                elif state in ["skip", "none"]:
                    dst_csv = os.path.join(SKIPPED_DIR, base + ".csv")
                    dst_video = os.path.join(SKIPPED_DIR, base + ".mp4")
                else:
                    continue
                try:
                    if os.path.exists(src_csv) and not os.path.exists(dst_csv):
                        shutil.copy2(src_csv, dst_csv)
                        print(f"Copied CSV: {dst_csv}")
                except Exception as e:
                    print(f"Error copying CSV {src_csv}: {e}")
                try:
                    if os.path.exists(src_video) and not os.path.exists(dst_video):
                        shutil.copy2(src_video, dst_video)
                        print(f"Copied video: {dst_video}")
                except Exception as e:
                    print(f"Error copying video {src_video}: {e}")
                if not SKIP_PROGRESSBAR_WINDOWS:
                    update_progress(progress_bar, label_count, idx + 1, total)
                    QApplication.processEvents()
            if not SKIP_PROGRESSBAR_WINDOWS:
                progress_win.close()
            show_saved_samples_window(SAMPLES_DIR, SKIPPED_DIR)
        elif event.key == "x":
            # Cleanup operation: delete files from "detections" and the output directory.
            folders = [DETECTIONS_DIR, output_dir]
            all_files = []
            for folder in folders:
                all_files.extend(glob.glob(os.path.join(folder, "*")))
            total = len(all_files)
            if not SKIP_PROGRESSBAR_WINDOWS:
                progress_win, label_op, label_count, progress_bar = show_progress_modal(
                    total, "Cleaning up files..."
                )
            current = 0
            for f in all_files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Error deleting {f}: {e}")
                current += 1
                if not SKIP_PROGRESSBAR_WINDOWS:
                    update_progress(progress_bar, label_count, current, total)
                    QApplication.processEvents()
            if not SKIP_PROGRESSBAR_WINDOWS:
                progress_win.close()
            detection_states.clear()
            del global_detections[:]
            selected_idx[0] = 0
            plt.close(fig)
            is_review = False
            print("Cleanup complete. Ready for new recordings.")
        elif event.key == " ":
            detection = global_detections[selected_idx[0]]
            playback_detection_video(
                detection, DETECTIONS_DIR, display_frame_width, display_frame_height
            )
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


def interactive_feature_wrapper(
    timeline,
    boundaries,
    composite_height,
    all_features,
    header,
    x_vals_scaled,
    global_detections,
    timeline_total_width,
    fps,
    output_dir,
):
    interactive_feature_plot(
        timeline,
        boundaries,
        composite_height,
        all_features,
        header,
        x_vals_scaled,
        global_detections,
        timeline_total_width,
        fps,
        output_dir,
    )


def show_timeline_and_features():
    """
    Orchestrates the building and plotting of the timeline and feature graph.
    Uses composite_info computed with adjusted target and header heights and font scale.
    """
    # Initial (base) parameters.
    base_target_img_height = 200
    base_header_height = 15
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    base_font_scale = 0.5
    thickness = 1
    text_color = (0, 0, 0)
    timeline_min_width = 2000

    video_files = get_sorted_video_file_list(RECORDINGS_DIR)
    # Obtain composite info while scaling down the sizes according to the number of video_files.
    (
        composite_info,
        adjusted_target_img_height,
        adjusted_header_height,
        adjusted_font_scale,
    ) = get_composite_info(
        video_files,
        base_target_img_height,
        base_header_height,
        font_face,
        base_font_scale,
        thickness,
        text_color,
    )

    composite_height = adjusted_header_height + adjusted_target_img_height
    total_video_frames = sum(frames for (_, frames, _, _) in composite_info)
    timeline_total_width = max(timeline_min_width, total_video_frames)
    timeline, boundaries = build_timeline(
        composite_info, timeline_total_width, composite_height
    )

    # Continue as before...
    all_features, header, segment_boundaries = aggregate_feature_data(
        get_csv_info(video_files)
    )
    all_features_count = len(all_features) if all_features is not None else 0
    factor = timeline_total_width / all_features_count if all_features_count else 1
    x_vals_scaled = (
        [x * factor for x in range(all_features_count)] if all_features_count else []
    )

    detection_interval = DETECTION_INTERVAL_SEC  # e.g., 0.5 sec
    raw_detections = build_detections(
        get_csv_info(video_files), composite_info, fps, detection_interval
    )
    save_detections(raw_detections, RECORDINGS_DIR, DETECTIONS_DIR)

    global_detections = []
    for csv_fname, start_idx, end_idx, seg_offset in raw_detections:
        global_x0 = (seg_offset + start_idx) * factor
        global_x1 = (seg_offset + end_idx) * factor
        global_detections.append((global_x0, global_x1, csv_fname))

    interactive_feature_wrapper(
        timeline,
        boundaries,
        composite_height,
        all_features,
        header,
        x_vals_scaled,
        global_detections,
        timeline_total_width,
        fps,
        RECORDINGS_DIR,
    )


def draw_record_pause_symbol(frame, is_paused):
    """
    Instead of a play symbol, when recording (i.e. not paused) display a full red circle.
    When paused, display the pause symbol as before.
    """
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)

    # If recording is active (not paused), display a red full circle; else, display pause symbol.
    if not is_paused:
        symbol = "●"  # Unicode full circle (U+25CF)
        color = (255, 0, 0)  # Red color
    else:
        symbol = "●"  # Pause symbol
        color = (128, 128, 128)  # Gray color

    font_size = 40
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    bbox = font.getbbox(symbol)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1] + font_size  # adjust for better centering
    h, w = frame.shape[:2]
    overlay_height = 50
    x = (w - text_w) // 2
    y = h - overlay_height + (overlay_height - text_h) // 2

    draw.text((x, y), symbol, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


def get_sort_key(f):
    parts = f.rsplit("_", 2)
    key = (parts[1] + "_" + parts[2].split(".")[0], f)
    return key


def get_sorted_video_file_list(output_dir, get_sort_key_func=get_sort_key):
    return sorted(
        [f for f in os.listdir(output_dir) if f.endswith(".mp4")], key=get_sort_key_func
    )


# Global flag to ensure we only play the gesture sound once per gesture.
gesture_sound_played = False

# Main loop
while True:
    ret, original_frame = cap.read()
    # Flip frame horizontally for mirror effect.
    original_frame = cv2.flip(original_frame, 1)
    # Resize the displayed frame to the display dimensions.
    frame = cv2.resize(original_frame, (display_frame_width, display_frame_height))

    if not ret:
        print("Error: Could not read frame.")
        break

    if not is_review:
        if is_paused:
            pass  # paused state
        else:
            now = time.time()
            elapsed = now - gesture_phase_start

            if gesture_phase == "display":
                # If it's a new gesture, play its gesture sound once.
                if not gesture_sound_played:
                    gesture_name = gestures_dict[GESTURES[current_gesture_index]]["name"]
                    threading.Thread(
                        target=playsound,
                        args=(f"{BASE_PATH}{APP_REL_PATH}{SOUNDS_DIR}/{gesture_name}.mp3",),
                        daemon=True,
                    ).start()
                    gesture_sound_played = True

                # Start recording_in_progress sound if not already playing.
                if recording_in_progress_thread is None:
                    recording_in_progress_stop_event = threading.Event()
                    recording_in_progress_thread = threading.Thread(
                        target=play_recording_in_progress,
                        args=(recording_in_progress_stop_event,),
                        daemon=True,
                    )
                    recording_in_progress_thread.start()

                if video_writer is None:
                    temp_timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    temp_filename = os.path.join(
                        RECORDINGS_DIR,
                        f"temp_gesture_{current_gesture_index}_{round(fps)}_{temp_timestamp}.mp4",
                    )
                    with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
                        video_writer = cv2.VideoWriter(
                            temp_filename,
                            fourcc,
                            fps,
                            (recording_frame_width, recording_frame_height),
                        )
                    gesture_frame_count = 0
                    print(
                        f"Recording gesture {gestures_dict[GESTURES[current_gesture_index]]['name']}..."
                    )

                if video_writer is not None:
                    video_writer.write(original_frame)
                    gesture_frame_count += 1

                # Draw progress bar ...
                pb_total_width = 300
                pb_height = 5
                pb_x = (display_frame_width - pb_total_width) // 2
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

                # Display gesture name centered on displayed frame.
                gesture_text = gestures_dict[GESTURES[current_gesture_index]][
                    "name"
                ].capitalize()
                gesture_font_scale = 2
                gesture_thickness = 3
                text_size, _ = cv2.getTextSize(
                    gesture_text, gesture_font, gesture_font_scale, gesture_thickness
                )
                gesture_text_x = (display_frame_width - text_size[0]) // 2
                gesture_text_y = display_frame_height // 2
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
                    # Reset the gesture sound flag for the next gesture.
                    gesture_sound_played = False

                    # Stop recording_in_progress sound.
                    if recording_in_progress_thread is not None:
                        recording_in_progress_stop_event.set()
                        recording_in_progress_thread.join()
                        recording_in_progress_thread = None
                        recording_in_progress_stop_event = None

                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        final_timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        temp_file = os.path.join(
                            RECORDINGS_DIR,
                            f"temp_gesture_{current_gesture_index}_{round(fps)}_{temp_timestamp}.mp4",
                        )
                        final_file = os.path.join(
                            RECORDINGS_DIR,
                            f"gesture_{current_gesture_index}_{round(fps)}_{final_timestamp}.mp4",
                        )
                        os.rename(temp_file, final_file)
                        print(f"Saved recording: {final_file}")

            else:  # wait phase
                if elapsed >= GESTURE_WAIT_DURATION:
                    current_gesture_index = (current_gesture_index + 1) % len(GESTURES)
                    gesture_phase = "display"
                    gesture_phase_start = now
                    # Reset flag so new gesture sound will play.
                    gesture_sound_played = False

    # Overlay: play/pause button at bottom.
    overlay_height = 50
    overlay_y = display_frame_height - overlay_height
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, overlay_y),
        (display_frame_width, display_frame_height),
        (50, 50, 50),
        -1,
    )
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # After drawing the overlay, replace the play/pause symbol with a PIL-rendered version.
    frame = draw_record_pause_symbol(frame, is_paused)

    # Debug: display current gesture.
    cv2.putText(
        frame,
        f"Gesture: {gestures_dict[GESTURES[current_gesture_index]]['name']}",
        (10, display_frame_height - 60),
        font,
        1,
        (255, 255, 255),
        2,
    )

    # Draw help prompt in top-right corner.
    help_text = "Press 'h' for help"
    (ts_width, ts_height), _ = cv2.getTextSize(help_text, font, 0.33, 1)
    margin = 10
    text_x = display_frame_width - ts_width - margin
    text_y = ts_height + margin
    cv2.putText(
        frame, help_text, (text_x, text_y), font, 0.33, (0, 0, 0), 1, cv2.LINE_AA
    )

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    # If help key is pressed, show help window.
    if key == ord("h"):
        is_paused = True
        show_help_window()
        # Returning from the help window will resume the previous window (which now displays the frozen frame).
    elif key == ord(" "):  # Toggle pause/recording.
        is_paused = not is_paused
        if is_paused:
            # If pausing during recording, stop the recording_in_progress sound.
            if recording_in_progress_thread is not None:
                recording_in_progress_stop_event.set()
                recording_in_progress_thread.join()
                recording_in_progress_thread = None
                recording_in_progress_stop_event = None
            print("Paused.")
        else:
            gesture_phase = "display"
            gesture_phase_start = time.time()
            print("Recording...")
    elif key == ord("r"):
        # Enter review mode.
        is_review = True
        is_paused = True
        print("Entering Review Mode...")
        file_list = get_sorted_video_file_list(RECORDINGS_DIR)
        if not SKIP_PROGRESSBAR_WINDOWS:
            progress_win, label_op, label_count, progress_bar = show_progress_modal(
                len(file_list), "Generating recording CSVs"
            )
        current = 0
        for f in file_list:
            if f.startswith("gesture_") and f.endswith(".mp4"):
                video_path = os.path.join(RECORDINGS_DIR, f)
                csv_path = os.path.splitext(video_path)[0] + ".csv"
                if not os.path.exists(csv_path):
                    process_video_file(video_path)
            current += 1
            if not SKIP_PROGRESSBAR_WINDOWS:
                update_progress(progress_bar, label_count, current, len(file_list))
                QApplication.processEvents()
        if not SKIP_PROGRESSBAR_WINDOWS:
            progress_win.close()
        print("Finished processing all gesture videos. Launching review window...")
        show_timeline_and_features()  # review window
        is_review = False
    elif key == ord("q") or key == 27:
        break

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

# Clean up
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
