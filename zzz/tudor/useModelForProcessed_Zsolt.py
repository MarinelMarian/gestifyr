import cv2
from collections import deque
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# === Added by Zsolt ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
# === END added by Zsolt ===




# ----- user params --------
queueFrameSizeSec = 2.0 #window length in seconds. This window will be the input to prediction
analyseVideoStepSec = 0.5 #in seconds, how much time should pass until next window is analyzed
#------------------------


# ~~~~~~ init ~~~~~
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
queueFrameSize = int(fps * queueFrameSizeSec)
analyseVideoStep = int(fps * analyseVideoStepSec)
frameQueue = deque(maxlen=queueFrameSize)
# ~~~~~~~~~~~~~~~~~~

# ====== mediapipe init =========
base_options = python.BaseOptions(model_asset_path='app/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)
def extractFeaturesv2(frame:cv2.typing.MatLike):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    return detection_result
# ============================

# ============================
# Code added by Zsolt

## PREDICTION
trained_model_file_path = "app/trained_model_OK26_e8877_p100__1740745288.pth"

# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[-1])  # Take last output
        return out.squeeze()

# Define the model
input_size = 52
hidden_size = 52
num_layers = 2
model = RNNModel(input_size, hidden_size, num_layers)

# load the trained model
model_state = torch.load(trained_model_file_path)
model.load_state_dict(model_state)
model.eval()

# gesture prediction by a sample input
# default input_size, hidden_size and num_layers are the values this model was trained with
# return value is a number from 1 to 5, referring to the gesture number that is predicts
def predictLabel(frameQueue) -> int:
    print(f"Frame rows: {len(frameQueue)}")
    frameQueue = np.array(frameQueue)

    frameQueue = torch.tensor(frameQueue, dtype=torch.float32)

    # generate a prediction
    with torch.no_grad():
        prediction = model(frameQueue)

    return round(float(prediction) * 10), 0
# End of code added by Zsolt
# ============================

idx = 0
while True:    
    ret, frame = cap.read()
    idx += 1
    if not ret:
        print("End of video ? can't read new frame.")
        break
    
    result_features = extractFeaturesv2(frame) # extract frame features
    if len(result_features.face_blendshapes) == 0:
        continue
    frameQueue.append([c.score for c in result_features.face_blendshapes[0]])

    # Optional: Display the frame
    cv2.imshow('Capturing Frames', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if idx % analyseVideoStep == 0: #stop to analyze when step time has completed
        predictedLabel, confidence  = predictLabel(frameQueue) # <-------- add your model prediciton method
        print(f"Predicted Gesture    -->    {predictedLabel},    Confidence: {confidence}\n")

