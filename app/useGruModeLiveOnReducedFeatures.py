import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import joblib
import cv2
from collections import deque
from mediapipe_extract import extract_features_v2, get_distance_between_corner_eyes
from tools import clear_terminal
import json
import numpy as np
from videoProcessingTools import get_angles, points_to_extract
from dotenv import load_dotenv

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")

# ----- user params --------
queue_frame_size_sec = (
    1.5  # window length in seconds. This window will be the input to prediction
)
analyse_video_step_sec = (
    1  # in seconds, how much time should pass until next window is analyzed
)
# ------------------------



# ~~~~~~ init ~~~~~
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
# Get the frames per second (FPS) of the video
# cap.set(cv2.CAP_PROP_EXPOSURE, 40)
fps = cap.get(cv2.CAP_PROP_FPS)
queue_frame_size = int(fps * queue_frame_size_sec)
analyse_video_step = int(fps * analyse_video_step_sec)
frame_queue = deque(maxlen=queue_frame_size)


clear_terminal()


# Model parameters
metadata_filename = (
    f"{BASE_PATH}tudor/model/processed_trimmed/gru_1741016164/info_model.txt"
)
with open(metadata_filename, "r") as file:
    data = json.load(file)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
num_layers = data["num_layers"]
saved_model_file_name = f"{BASE_PATH}{data['modelFile']}"
saved_scaler_file_name = f"{BASE_PATH}{data['scalerFile']}"
labels = ["da", "nu ", "gura casca", "ridicat", "nimic"]
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

cap = cv2.VideoCapture(1)  # Open the default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
idx = 0
while True:
    ret, frame = cap.read()
    idx = idx + 1
    # print(idx)
    if not ret:
        print("Error: Could not read frame.")
        break
    result_features = extract_features_v2(frame)
    if len(result_features.face_blendshapes) == 0:
        continue
    raw_features = [
        coord
        for point in result_features.face_landmarks[0]
        for coord in (point.x, point.y, point.z)
    ]
    processed_features = [c.score for c in result_features.face_blendshapes[0]]
    features_reduced = [processed_features[i] for i in points_to_extract]
    x, y = get_angles(raw_features)
    features_reduced.append(x)
    features_reduced.append(y)

    frame_queue.append(features_reduced)

    # Optional: Display the frame
    cv2.imshow("Capturing Frames", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
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
            # print(f'propabilities:{probabilities}')

            predicted_label = labels[predicted_class.item()]

            # if confidence.item() > 0.8:
            print(
                f"Predicted Gesture: {predicted_label}, Confidence: {confidence.item():.4f}"
            )


cap.release()
cv2.destroyAllWindows()
