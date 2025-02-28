import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import cv2
from collections import deque
from mediapipe_extract import extractFeaturesv2, extractFeatures, featureNormalization, getDistanteBetweenCornerEyes
from tools import clearTerminal
import json
import numpy as np

points_to_get_angles = [1, 61 , 291, 33, 263, 199]
img_w,img_h = 1980,1080
def getAngles(result_features):
    sublist = [result_features[i] for i in points_to_get_angles]
    face_3d = [ [int(e[0]*img_w), int(e[1]*img_h), e[2] ] for e in sublist] 
    face_3d = np.array(face_3d, dtype=np.float64)
    face_2d = np.array(face_3d[:, 0:2],  dtype=np.float64)
    focal_length = 1 * img_w

    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]])

    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
                # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    return x,y,z

clearTerminal()

# ----- init params --------

# Model parameters
metadataFilename = "tudor/model/raw_normalized/gru_1740421523/info_model.txt"
with open(metadataFilename, "r") as file:
    data = json.load(file)
input_size = data['input_size']  # Feature size 478 points of x,y,z
hidden_size = data['hidden_size']
output_size = data['output_size']
num_layers = data['num_layers']
savedModelFileName = data['modelFile']
savedScalerFileName = data['scalerFile']
labels = ['da','nu ','gura casca','ridicat', 'nimic']
print(f'Loaded inputSize={input_size}, hiddenSize={hidden_size}, outputSize={output_size}, numLayers={num_layers}, modelFile={savedModelFileName}')
windowStepInFrames = 60
windowSizeInFrames = 45
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
loaded_model.load_state_dict(torch.load(savedModelFileName))
loaded_model.eval()  # Set to evaluation mode
label_encoder = LabelEncoder()
scaler = joblib.load(savedScalerFileName)

print("Model loaded successfully!")

cap = cv2.VideoCapture(0)  # Open the default webcam
frameQueue = deque(maxlen=windowSizeInFrames)
shouldExit = False
if not cap.isOpened():
    print("Error: Could not open webcam.")
    shouldExit = True
idx = 0
while not shouldExit:    
    ret, frame = cap.read()
    idx = idx + 1
    # print(idx)
    if not ret:
        print("Error: Could not read frame.")
        break
    # result_features = extractFeaturesv2(frame)
    # if len(result_features.face_blendshapes) == 0:
    #     continue
    # frameQueue.append([c.score for c in result_features.face_blendshapes[0]])

    result_features = extractFeatures(frame)
    if len(result_features) == 0:
        continue
    frameQueue.append(list(featureNormalization(result_features).reshape(-1)))

    # Optional: Display the frame
    cv2.imshow('Capturing Frames', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if idx % windowStepInFrames == 0: 
  

        
        features2test = torch.tensor(scaler.transform(frameQueue), dtype=torch.float32).unsqueeze(0)
        #predict
        with torch.no_grad():
        
            # inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
            outputs = loaded_model(features2test)
            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            # print(probabilities)
                # Get predicted class and confidence
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1).values
            # print(f'propabilities:{probabilities}')

            predicted_label = labels[predicted_class.item()]

            # if confidence.item() > 0.8:
            print(f"Predicted Gesture: {predicted_label}, Confidence: {confidence.item():.4f}")



    
   
# #load data
# test_file = "tudor/processed/gesture_7__1739197934.csv"
# df = pd.read_csv(test_file, skiprows=1)  # Ignore first row
# features2test = torch.tensor(scaler.transform(df.values), dtype=torch.float32).unsqueeze(0)



# #predict
# with torch.no_grad():
    
#     # inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
#     outputs = loaded_model(features2test)
#     # Apply softmax to get probabilities
#     probabilities = F.softmax(outputs, dim=1)
#         # Get predicted class and confidence
#     predicted_class = torch.argmax(probabilities, dim=1)
#     confidence = torch.max(probabilities, dim=1).values


#     predicted_label = llabels[predicted_class.item()]

#     print(f"Predicted Gesture: {predicted_label}, Confidence: {confidence.item():.4f}")


cap.release()
cv2.destroyAllWindows()