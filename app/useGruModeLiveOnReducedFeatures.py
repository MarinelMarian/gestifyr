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

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")
APP_REL_PATH = os.getenv("APP_REL_PATH")

# ----- user params --------
queue_frame_size_sec = (
    0.8  # window length in seconds. This window will be the input to prediction
)
analyse_video_step_sec = (
    0.3  # in seconds, how much time should pass until next window is analyzed
)
trigger_release_time = 2  # in seconds, how much time should pass until next trigger
not_threshold = 0.8  # threshold for not gesture
shake_threshold = 0.8  # threshold for shake gesture
mouth_threshold = 0.8  # threshold for open mouth gesture
eyebrows_threshold = 0.8  # threshold for raise eyebrows gesture
blink_threshold = 0.8  # threshold for blink gesture
smile_threshold = 0.8  # threshold for smile gesture
# ------------------------






clear_terminal()


# Model parameters
metadata_filename = (
    f"{BASE_PATH}{APP_REL_PATH}model/gru_128_3_0.002_5/info_model.txt"
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

def initialize_webcam(camera_device_id):
    cap = cv2.VideoCapture(camera_device_id)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
    # ~~~~~~ init ~~~~~
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
    # Get the frames per second (FPS) of the video
    # cap.set(cv2.CAP_PROP_EXPOSURE, 40)
    fps = cap.get(cv2.CAP_PROP_FPS)
    queue_frame_size = int(fps * queue_frame_size_sec)
    analyse_video_step = int(fps * analyse_video_step_sec)
    frame_queue = deque(maxlen=queue_frame_size)
    return (cap, fps, queue_frame_size, analyse_video_step, frame_queue)

camera_device_id = 0
# Initialize webcam
(cap, fps, queue_frame_size, analyse_video_step, frame_queue) = initialize_webcam(camera_device_id)

idx = 0
mask_on = False
sound_on = False
probValues = [0,0,0,0,0,0,0]
sound_triggered = False
trigger_release_counter = int(fps * trigger_release_time)
probabilities_thresholds = [not_threshold, shake_threshold, mouth_threshold, eyebrows_threshold, blink_threshold, smile_threshold]

while True:
    ret, frame = cap.read()
    if not ret:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = overlayRoundedSquare(frame, (50, 290), (120, 100), "", f"Camera {camera_device_id}", isActive = False)
        text = "Camera not detected, Select another camera (0-6)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1] + 100) // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    else:
        img_h, img_w, _ = frame.shape

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
        frame = overlayRoundedSquare(frame, (50, 290), (120, 100), f"Camera {camera_device_id}")
        
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
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        mask_on = not mask_on  # Toggle mask_on flag
    elif key == ord('s'):
        sound_on = not sound_on
        trigger_release_counter = int(fps * trigger_release_time)
        sound_triggered = False
    elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
        camera_device_id = int(chr(key))
        cap.release()  # Release the current capture
        (cap, fps, queue_frame_size, analyse_video_step, frame_queue) = initialize_webcam(camera_device_id)  # Switch to the new camera
    if sound_on:
        if not sound_triggered:
            sound_triggered = check_trigger_and_play_sound(probValues[0:6], probabilities_thresholds)
        else:
            trigger_release_counter -= 1
            if trigger_release_counter <= 0:
                sound_triggered = False
                trigger_release_counter = int(fps * trigger_release_time)

cap.release()
cv2.destroyAllWindows()
