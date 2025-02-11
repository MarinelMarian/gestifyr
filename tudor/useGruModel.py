import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import cv2
from collections import deque
from mediapipe_extract import extractFeaturesv2, extractFeatures, featureNormalization
from tools import clearTerminal
clearTerminal()

# ----- init params --------
# Model parameters
input_size = 1434  # Feature size 478 points of x,y,z
hidden_size = 128
output_size = 5
num_layers = 2
savedModelFileName = "gru_model_raw_normalized_features.pth"
savedScalerFileName = "scaler_raw_normalized_features.pkl"
labels = ['da','nu ','gura casca','ridicat', 'nimic']

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
frameQueue = deque(maxlen=60)
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
    if idx % 10 == 0: 
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
            print(f'propabilities:{probabilities}')

            predicted_label = labels[predicted_class.item()]

            if confidence.item() > 0.94:
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