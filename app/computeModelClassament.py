import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import cv2
from collections import deque
from tools import clear_terminal
import json
import os
import glob
import csv


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

clear_terminal()

# ----- init params --------

modelBaseFolderPath = "app/model/for_final"
data_folder = "app/forTesting/for_final"
outputResultsFile = "app/forTesting/results/for_final.csv"
folders = [x[0] for x in os.walk(modelBaseFolderPath)]
modelResults = {}
for folder in folders:
    print(f"---- Starting new run for model {folder}")
    if folder == modelBaseFolderPath:
        continue
    metadataFilename = f'{folder}/info_model.txt'

    # Model parameters
    with open(metadataFilename, "r") as file:
        data = json.load(file)
    input_size = data['input_size']  # Feature size 478 points of x,y,z
    hidden_size = data['hidden_size']
    output_size = data['output_size']
    num_layers = data['num_layers']
    savedModelFileName = data['model_file']
    savedModelFileName = savedModelFileName.replace('model/', 'model/for_final/')
    savedScalerFileName = data['scaler_file']
    savedScalerFileName = savedScalerFileName.replace('model/', 'model/for_final/')
    labels = ['da','nu ','gura casca','ridicat', 'blink', 'smile','nimic']
    print(f'Loaded inputSize={input_size}, hiddenSize={hidden_size}, outputSize={output_size}, numLayers={num_layers}, modelFile={savedModelFileName}')
    # -----------
    # Recreate the model (use the same architecture as before)
    loaded_model = GRUNet(input_size, hidden_size, output_size, num_layers)

    # Load the saved weights
    loaded_model.load_state_dict(torch.load(savedModelFileName))
    loaded_model.eval()  # Set to evaluation mode
    label_encoder = LabelEncoder()
    scaler = joblib.load(savedScalerFileName)
    print(f"Model from folder {folder} loaded successfully!")
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))  # Find all CSV files
    fileResults = {}
    for file in csv_files:
        print(f'Reading file for testing {file}')
        df = pd.read_csv(file, skiprows=1, header=None)       
        features2test = torch.tensor(scaler.transform(df), dtype=torch.float32).unsqueeze(0)
        #predict
        with torch.no_grad():
        
            # inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
            outputs = loaded_model(features2test)
            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
                # Get predicted class and confidence
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1).values
            print(f'propabilities:{probabilities}')

            predicted_label = labels[predicted_class.item()]

            # if confidence.item() > 0.8:
            print(f"Predicted Gesture: {predicted_label}, Confidence: {confidence.item():.4f}")
            fileName = file.split('/')[-1]
            corectGesture = fileName[8]
            # fileResults[file.split('/')[-1]] = fileResults.get(file.split('/')[-1], 1) * (confidence.item() if corectGesture == predicted_class.item() else 0)
            fileResults[file.split('/')[-1]] = fileResults.get(file.split('/')[-1], 1) * (confidence.item() )

    modelResults[folder] = fileResults

print(modelResults)

dataOutputFormat = []
# allFiles = modelResults[folders[1]].keys()
# for f in allFiles:
#     for model in modelResults.keys():
#         dataOutputFormat.append([f, model, modelResults[model][f]])

for modelName in modelResults.keys():
    tmp = 0
    for fileName in modelResults[modelName].keys():
        tmp = tmp + modelResults[modelName][fileName]
    dataOutputFormat.append([modelName, tmp])

with open(outputResultsFile, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(dataOutputFormat)  # Writes multiple rows

print(f"CSV file saved as {outputResultsFile}")




