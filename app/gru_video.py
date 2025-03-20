import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import joblib
import cv2
from collections import deque
from audio_tools import check_trigger_and_play_sound
from mediapipe_extract import draw_landmarks_on_image, extract_features_v2
from tools import clear_terminal
import json
import numpy as np
from videoProcessingTools import get_angles, overlayBar, overlayRoundedSquare, points_to_extract
from dotenv import load_dotenv
import icons as icons
from tkinter import filedialog
import datetime as dt
from contextlib import redirect_stderr, redirect_stdout


load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")
APP_REL_PATH = os.getenv("APP_REL_PATH")
VIDEO_ENCODER_FOURCC = (
    os.getenv("VIDEO_ENCODER_FOURCC") or "mp42"
)  # "mp4v" for Windows, "avc1" for MacOS

# ----- user params --------
queue_frame_size_sec = (
    0.6  # window length in seconds. This window will be the input to prediction
)
analyse_video_step_sec = (
    0.2  # in seconds, how much time should pass until next window is analyzed
)
output_file_base_name = "output_model"
trigger_release_time = 2  # in seconds, how much time should pass until next trigger
not_threshold = 0.8  # threshold for not gesture
shake_threshold = 0.8  # threshold for shake gesture
mouth_threshold = 0.8  # threshold for open mouth gesture
eyebrows_threshold = 0.8  # threshold for raise eyebrows gesture
blink_threshold = 0.8  # threshold for blink gesture
smile_threshold = 0.8  # threshold for smile gesture
# ------------------------

def stitch_frames(frame1, frame2):
    """ Resize two frames to 50% and stitch them side by side. """
    # Resize frames to 50%
    frame_height, frame_width, _ = frame1.shape
    new_w, new_h = frame_width // 2, frame_height // 2
    frame1_resized = cv2.resize(frame1, (new_w, new_h))
    frame2_resized = cv2.resize(frame2, (new_w, new_h))

    # Create black frame (original size)
    stitched_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Calculate positions to center them
    x_offset = (frame_width - (new_w * 2)) // 2
    y_offset = (frame_height - new_h) // 2

    # Place the resized frames side by side
    stitched_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame1_resized
    stitched_frame[y_offset:y_offset+new_h, x_offset+new_w:x_offset+(2*new_w)] = frame2_resized

    return stitched_frame

file_path = filedialog.askopenfilename(title="Select a file",
                                       filetypes=[("All Files", "*.*")])

if not file_path:
    exit()




clear_terminal()


# Model parameters
metadata_filename = (
    f"{BASE_PATH}{APP_REL_PATH}model/gru_128_4_0.0001_5/info_model.txt"
)
with open(metadata_filename, "r") as file:
    data = json.load(file)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
num_layers = data["num_layers"]
saved_model_file_name = f"{BASE_PATH}{data['model_file']}"
saved_scaler_file_name = f"{BASE_PATH}{data['scaler_file']}"
labels = ["da", "nu ", "openMouth", "eyebrows up", "blink", "smile"," nimic"]
print(
    f"Loaded input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, num_layers={num_layers}, model_file={saved_model_file_name}"
)


# ----------- 
# Define GRU Model
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Take last output
        return out


# Recreate the model (use the same architecture as before)
loaded_model = GRUNet(input_size, hidden_size, output_size, num_layers)

# Load the saved weights
loaded_model.load_state_dict(torch.load(saved_model_file_name))
loaded_model.eval()  # Set to evaluation mode
label_encoder = LabelEncoder()
scaler = joblib.load(saved_scaler_file_name)

print("Model loaded successfully!")

cap = cv2.VideoCapture(file_path)
if not cap.isOpened():
    print("Error: Could not open file.")
# ~~~~~~ init ~~~~~


fps = cap.get(cv2.CAP_PROP_FPS)
queue_frame_size = int(fps * queue_frame_size_sec)
analyse_video_step = int(fps * analyse_video_step_sec)
frame_queue = deque(maxlen=queue_frame_size)
recording_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
recording_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


idx = 0
mask_on = False
sound_on = False
probValues = [0,0,0,0,0,0,0]
sound_triggered = False
trigger_release_counter = int(fps * trigger_release_time)
probabilities_thresholds = [not_threshold, shake_threshold, mouth_threshold, eyebrows_threshold, blink_threshold, smile_threshold]

temp_timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
temp_filename = f"{BASE_PATH}{APP_REL_PATH}{output_file_base_name}_{temp_timestamp}.mp4"
fourcc = cv2.VideoWriter_fourcc(*VIDEO_ENCODER_FOURCC)
video_writer = None
with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
    video_writer = cv2.VideoWriter(
        temp_filename,
        fourcc,
        fps,
        (recording_frame_width, recording_frame_height),
    )

while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:
        img_h, img_w, _ = frame.shape
        original_frame = frame.copy()
        result_features = extract_features_v2(frame)
        frame = draw_landmarks_on_image(frame,result_features) if mask_on else frame

        if len(result_features.face_blendshapes) == 0:
            continue
        raw_features = [
            coord
            for point in result_features.face_landmarks[0]
            for coord in (point.x, point.y, point.z)
        ]
        processed_features = [c.score for c in result_features.face_blendshapes[0]]
        reduced_features_row = [processed_features[i] for i in points_to_extract]

        x, y = get_angles(raw_features, img_w, img_h)
        features_reduced = ([y / 90, x / 90, *reduced_features_row])
        if mask_on:
            nose_2d = raw_features[3:6]
            p1 = (int(nose_2d[0]*img_w), int(nose_2d[1]*img_h))
            p2 = (int(nose_2d[0]*img_w + x * 20) , int(nose_2d[1]*img_h - y * 20))
            cv2.line(frame, p1, p2, (0, 0, 255), 3)
            frame = cv2.circle(frame, p1, 10, (0, 0, 255), -1)
        frame_queue.append(features_reduced)
        frame = overlayRoundedSquare(frame, (50, 50), (120, 100), "Mask ON", "Mask OFF", mask_on)
        frame = overlayRoundedSquare(frame, (50, 170), (120, 100), "Sound ON", "Sound OFF", sound_on)
        
        idx = idx + 1
    
        if idx % analyse_video_step == 0:

            features_to_test = torch.tensor(
                scaler.transform(frame_queue), dtype=torch.float32
            ).unsqueeze(0)
            # predict
            with torch.no_grad():

                # inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
                outputs = loaded_model(features_to_test)
                # Apply softmax to get probabilities
                probabilities = F.softmax(outputs, dim=1)
                # print(probabilities)
                # Get predicted class and confidence
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1).values

                predicted_label = labels[predicted_class.item()]

                # if confidence.item() > 0.8:
                print(
                    f"Predicted Gesture: {predicted_label}, Confidence: {confidence.item():.4f}"
                )
                probValues = probabilities.tolist()[0]
        frame = overlayBar( frame, position_idx = 1, value = probValues[0], icon_image = icons.icon_nod , threshold = not_threshold)
        frame = overlayBar( frame, position_idx = 2, value = probValues[1], icon_image = icons.icon_shake , threshold = shake_threshold)
        frame = overlayBar( frame, position_idx = 3, value = probValues[2], icon_image = icons.icon_open_mouth , threshold = mouth_threshold)
        frame = overlayBar( frame, position_idx = 4, value = probValues[3], icon_image = icons.icon_raise_eyebrows , threshold = eyebrows_threshold)
        frame = overlayBar( frame, position_idx = 5, value = probValues[4], icon_image = icons.icon_eyes_shut , threshold = blink_threshold)  
        frame = overlayBar( frame, position_idx = 6, value = probValues[5], icon_image = icons.icon_smile , threshold = smile_threshold)
    cv2.imshow("Capturing Frames", frame)
    if video_writer is not None:
        frame_to_write = stitch_frames(original_frame, frame)

        with open(os.devnull, "w") as devnull, redirect_stderr(devnull), redirect_stdout(devnull):
            video_writer.write(frame_to_write)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        mask_on = not mask_on  # Toggle mask_on flag
    elif key == ord('s'):
        sound_on = not sound_on
        trigger_release_counter = int(fps * trigger_release_time)
        sound_triggered = False

    if sound_on:
        if not sound_triggered:
            sound_triggered = check_trigger_and_play_sound(probValues[0:6], probabilities_thresholds)
        else:
            trigger_release_counter -= 1
            if trigger_release_counter <= 0:
                sound_triggered = False
                trigger_release_counter = int(fps * trigger_release_time)

cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
