import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import joblib
import json
import os
import datetime as dt



# --------- Init Params -------
#Model params
# hidden_size = 128
# num_layers = 4
epochs = 1000

epochThresh = 0.1
# ---------------------
modelInputs = { 'filePath':['app/samples2/processed_trimmed'], # list of folders
               'hiddenSize':[16, 32, 64, 128, 256, 512],
               'numLayers' : [2, 3, 4],
               'epochThresh' :[0.5, 0.1, 0.05]
               } 

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

# Custom Dataset
class GestureDataset(Dataset):
    def __init__(self, file_paths, label_encoder, scaler, train=True):
        self.data = []
        self.labels = []
        self.train = train
        for file_path in file_paths:
            df = pd.read_csv(file_path, skiprows=1)  # Ignore first row
            features = df.values
            if train:
                scaler.partial_fit(features)  # Fit scaler on training data
            features = scaler.transform(features)  # Normalize
            self.data.append(features)
            if train:
                gesture_name = os.path.basename(file_path).split('_')[1]  # Extract label from file name
                self.labels.append(label_encoder.transform([gesture_name])[0])
        self.data = [torch.tensor(d, dtype=torch.float32) for d in self.data]
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        return self.data[idx], [] if self.train == False else self.labels[idx]
    
def collate_fn(batch):
    sequences, labels = zip(*batch)  # Unzip batch
    sequences_padded = pad_sequence(sequences, batch_first=True)  # Pad sequences
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences_padded, labels


# Load and process data
def calculateModel(**kwargs):
    print('\n\n\n-------------------')
    print(f"Starting new model calculation with params :{kwargs}")
    outputFolder = kwargs["filePath"].split('/')[-1]

    modelName = f'gru_{int(dt.datetime.now().timestamp())}'
    os.makedirs(f'app/model/{outputFolder}/{modelName}', exist_ok=True)
    saveModelFileName = f"app/model/{outputFolder}/{modelName}/gru_model.pth"
    saveScalerFileName = f"app/model/{outputFolder}/{modelName}/scaler.pkl"
    modelMetadataFile = f"app/model/{outputFolder}/{modelName}/info_model.txt"
    train_files = glob.glob(f'{kwargs["filePath"]}/*.csv')
    labels = sorted(set(os.path.basename(f).split('_')[1] for f in train_files))
    print(f'Labels found: {labels}')
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    scaler = StandardScaler()

    train_dataset = GestureDataset(train_files, label_encoder, scaler, train=True)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)


    # Model parameters
    input_size = next(iter(train_dataset))[0].shape[1]  # Feature size
    output_size = len(labels)


    model = GRUNet(input_size, kwargs['hiddenSize'], output_size, kwargs['numLayers'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
            targets = targets
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        if total_loss/len(train_loader) < kwargs['epochThresh']:
            break
    torch.save(model.state_dict(), saveModelFileName)
    joblib.dump(scaler, saveScalerFileName)
    metadata = {
        'modelFile': saveModelFileName,
        'scalerFile': saveScalerFileName,
        'hidden_size' : kwargs['hiddenSize'],
        "num_layers" : kwargs['numLayers'],
        "labels": labels,
        "input_size": input_size,
        "output_size": output_size
    }
    with open(modelMetadataFile, "w") as file:
        json.dump(metadata, file, indent=4)
        print(f"MetaData saved successfully! to file {file}")
    print("Model saved successfully!")
for filePath in modelInputs['filePath']:
    for hiddenSize in modelInputs['hiddenSize']:
        for numLayers in modelInputs['numLayers']:
            for epochThresh in modelInputs['epochThresh']:
                calculateModel(filePath=filePath, hiddenSize=hiddenSize, numLayers=numLayers, epochThresh=epochThresh)