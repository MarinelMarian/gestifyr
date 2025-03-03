
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import joblib
import cv2
from collections import deque
from mediapipe_extract import extractFeaturesv2, getDistanteBetweenCornerEyes
from tools import clearTerminal
import json
import numpy as np
from videoProcessingTools import getAngles, points_to_extract

# ----- user params --------
queueFrameSizeSec = 1.5 #window length in seconds. This window will be the input to prediction
analyseVideoStepSec = 1 #in seconds, how much time should pass until next window is analyzed
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


clearTerminal()



# Model parameters
metadataFilename = "tudor/model/processed_trimmed/gru_1741016164/info_model.txt"
with open(metadataFilename, "r") as file:
    data = json.load(file)
input_size = data['input_size']  
hidden_size = data['hidden_size']
output_size = data['output_size']
num_layers = data['num_layers']
savedModelFileName = data['modelFile']
savedScalerFileName = data['scalerFile']
labels = ['da','nu ','gura casca','ridicat', 'nimic']
print(f'Loaded inputSize={input_size}, hiddenSize={hidden_size}, outputSize={output_size}, numLayers={num_layers}, modelFile={savedModelFileName}')

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
    result_features = extractFeaturesv2(frame)
    if len(result_features.face_blendshapes) == 0:
        continue
    rawFeatures = [coord for point in result_features.face_landmarks[0] for coord in (point.x, point.y, point.z)]
    processedFeatures = ([c.score for c in result_features.face_blendshapes[0]])
    featuresReduced = [processedFeatures[i] for i in points_to_extract]
    featuresReduced.append(getDistanteBetweenCornerEyes(np.array(rawFeatures)))
    x,y = getAngles(rawFeatures)
    featuresReduced.append(x)
    featuresReduced.append(y)

    frameQueue.append(featuresReduced)

    
    # Optional: Display the frame
    cv2.imshow('Capturing Frames', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if idx % analyseVideoStep == 0: 
  

        
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


cap.release()
cv2.destroyAllWindows()